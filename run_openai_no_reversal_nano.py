from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
PROVIDER = "openai"
DESIGN = "openai_fixed_probe_20"
MODELS = (
    "gpt-5-nano",
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "gpt-5.4-nano",
    "gpt-5.6-luna",
    "gpt-3.5-turbo",
    "gpt-4.1-mini",
    "gpt-5.4-mini",
    "gpt-4o",
    "gpt-5.6-terra",
)
TASK_MODE = "no_reversal"
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 64
N_TRIALS_TOLD = 20
N_TRIALS_ACTUAL = 20
INITIAL_CONTINGENCY = {"B": 1, "A": 0}
PROBE_CONTINGENCY = {"A": 1, "B": 0}
PROBE_TRIALS = frozenset({4, 7, 14, 16})
CHOICES = ("A", "B")

CHOICE_FORMAT = {
    "type": "json_schema",
    "name": "bandit_choice",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": list(CHOICES)}},
        "required": ["choice"],
        "additionalProperties": False,
    },
}


class BehavioralFormatError(ValueError):
    """The model response violates the required one-key choice schema."""


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def verify_models(client: OpenAI, models: list[str] | tuple[str, ...] = MODELS) -> None:
    live_ids = {model.id for model in client.models.list().data}
    missing = [model for model in models if model not in live_ids]
    print("LIVE MODEL VERIFICATION", flush=True)
    for model in models:
        status = "available" if model in live_ids else "MISSING"
        print(f"  {status}: {model}", flush=True)
    if missing:
        raise RuntimeError(f"OpenAI live model list is missing: {', '.join(missing)}")


def init_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(
        """
        CREATE TABLE experiment (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            design TEXT NOT NULL,
            provider TEXT NOT NULL,
            requested_model TEXT NOT NULL,
            model TEXT NOT NULL,
            substitution_reason TEXT,
            task_mode TEXT NOT NULL,
            temperature REAL NOT NULL,
            reasoning_effort TEXT NOT NULL,
            max_output_tokens INTEGER NOT NULL,
            games_requested INTEGER NOT NULL,
            n_trials_told INTEGER NOT NULL,
            n_trials_actual INTEGER NOT NULL,
            intentional_trial_count_mismatch INTEGER NOT NULL,
            initial_contingency_json TEXT NOT NULL,
            prompts_json TEXT NOT NULL,
            config_json TEXT NOT NULL
        );
        CREATE TABLE game (
            id INTEGER PRIMARY KEY,
            experiment_id INTEGER NOT NULL REFERENCES experiment(id),
            game_number INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            total_reward INTEGER NOT NULL DEFAULT 0,
            error_type TEXT,
            error_message TEXT,
            UNIQUE(experiment_id, game_number)
        );
        CREATE TABLE trial (
            id INTEGER PRIMARY KEY,
            game_id INTEGER NOT NULL REFERENCES game(id),
            trial_number INTEGER NOT NULL,
            request_json TEXT NOT NULL,
            response_json TEXT,
            response_id TEXT,
            raw_output TEXT,
            choice TEXT,
            contingency_json TEXT,
            is_probe INTEGER NOT NULL DEFAULT 0,
            reward INTEGER,
            cumulative_reward INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            reasoning_tokens INTEGER,
            latency_ms REAL,
            error_type TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(game_id, trial_number)
        );
        """
    )
    return db


def parse_choice(raw_output: str) -> str:
    payload = json.loads(raw_output)
    if not isinstance(payload, dict) or set(payload) != {"choice"}:
        raise BehavioralFormatError(
            f"Expected exactly one 'choice' field, received: {payload!r}"
        )
    choice = payload["choice"]
    if choice not in CHOICES:
        raise BehavioralFormatError(f"Choice must be A or B, received: {choice!r}")
    return choice


def usage_values(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    details = getattr(usage, "output_tokens_details", None)
    return (
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
        getattr(details, "reasoning_tokens", None) if details else None,
    )


def run_game(
    db: sqlite3.Connection,
    client: OpenAI,
    experiment_id: int,
    game_number: int,
    games_requested: int,
    model: str,
    system_prompt: str,
    prompts: dict[str, str],
) -> list[str]:
    game_id = db.execute(
        """INSERT INTO game
           (experiment_id, game_number, started_at, status)
           VALUES (?, ?, ?, 'running')""",
        (experiment_id, game_number, now_iso()),
    ).lastrowid
    db.commit()

    print(
        f"\nGAME provider={PROVIDER} model={model} game={game_number}/{games_requested}",
        flush=True,
    )
    print(
        "  TRIAL COUNTS: "
        f"n_trials_told={N_TRIALS_TOLD}; n_trials_actual={N_TRIALS_ACTUAL}",
        flush=True,
    )

    answers: list[str] = []
    total_reward = 0
    previous_response_id: str | None = None
    next_input = prompts["start"].format(
        n_trials=N_TRIALS_TOLD,
        choice_a=CHOICES[0],
        choice_b=CHOICES[1],
    )

    for trial_number in range(1, N_TRIALS_ACTUAL + 1):
        request: dict[str, Any] = {
            "model": model,
            "instructions": system_prompt,
            "input": (
                f"Respond in JSON. {next_input}"
                if model == "gpt-3.5-turbo"
                else next_input
            ),
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "text": {
                "format": (
                    {"type": "json_object"}
                    if model == "gpt-3.5-turbo"
                    else CHOICE_FORMAT
                )
            },
            "store": True,
        }
        if model.startswith(("gpt-5.4-", "gpt-5.6-")):
            request["reasoning"] = {"effort": "none"}
        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        trial_id = db.execute(
            """INSERT INTO trial
               (game_id, trial_number, request_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (game_id, trial_number, as_json(request), now_iso()),
        ).lastrowid
        db.commit()
        print(
            f"  SEND actual_trial={trial_number}/{N_TRIALS_ACTUAL}: {next_input}",
            flush=True,
        )

        began = time.perf_counter()
        try:
            response = client.responses.create(**request)
            latency_ms = (time.perf_counter() - began) * 1000
            raw_output = response.output_text
            response_json = as_json(response.model_dump(mode="json", exclude_none=False))
            input_tokens, output_tokens, reasoning_tokens = usage_values(response)
            db.execute(
                """UPDATE trial SET response_json=?, response_id=?, raw_output=?,
                   input_tokens=?, output_tokens=?, reasoning_tokens=?, latency_ms=?
                   WHERE id=?""",
                (
                    response_json, response.id, raw_output, input_tokens,
                    output_tokens, reasoning_tokens, latency_ms, trial_id,
                ),
            )
            db.commit()

            choice = parse_choice(raw_output)
            is_probe = trial_number in PROBE_TRIALS
            contingency = PROBE_CONTINGENCY if is_probe else INITIAL_CONTINGENCY
            reward = contingency[choice]
            total_reward += reward
            answers.append(choice)
            db.execute(
                """UPDATE trial SET choice=?, contingency_json=?, is_probe=?,
                   reward=?, cumulative_reward=? WHERE id=?""",
                (choice, as_json(contingency), int(is_probe), reward, total_reward, trial_id),
            )
            db.commit()
            print(
                f"  RECV actual_trial={trial_number}/{N_TRIALS_ACTUAL}: "
                f"choice={choice} reward={reward} total={total_reward} "
                f"probe={is_probe} contingency={contingency}",
                flush=True,
            )
            previous_response_id = response.id
            if trial_number < N_TRIALS_ACTUAL:
                next_input = prompts["feedback"].format(
                    previous_trial=trial_number,
                    previous_choice=choice,
                    reward=reward,
                    total_reward=total_reward,
                    trial=trial_number + 1,
                )
        except Exception as exc:
            latency_ms = (time.perf_counter() - began) * 1000
            error_type = (
                "behavioral_format"
                if isinstance(exc, (json.JSONDecodeError, BehavioralFormatError))
                else type(exc).__name__
            )
            error_message = f"{type(exc).__name__}: {exc}"
            db.execute(
                """UPDATE trial SET latency_ms=?, error_type=?, error_message=?
                   WHERE id=?""",
                (latency_ms, error_type, error_message, trial_id),
            )
            db.execute(
                """UPDATE game SET finished_at=?, status='error', total_reward=?,
                   error_type=?, error_message=? WHERE id=?""",
                (now_iso(), total_reward, error_type, error_message, game_id),
            )
            db.commit()
            print(
                f"  ERROR actual_trial={trial_number}: {error_type}: {error_message}",
                flush=True,
            )
            return answers

    db.execute(
        "UPDATE game SET finished_at=?, status='completed', total_reward=? WHERE id=?",
        (now_iso(), total_reward, game_id),
    )
    db.commit()
    return answers


def write_and_print_wide_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["provider", "model", "game_number"] + [
        f"answer_{trial}" for trial in range(1, N_TRIALS_ACTUAL + 1)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\nWIDE SUMMARY TABLE", flush=True)
    print(",".join(columns), flush=True)
    for row in rows:
        print(",".join(str(row.get(column, "")) for column in columns), flush=True)
    print(f"WIDE SUMMARY CSV {path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OpenAI 20-trial fixed-probe experiment."
    )
    parser.add_argument("--games", type=int, default=10, help="Games per model.")
    parser.add_argument("--dry-run", action="store_true", help="Run one game per model.")
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS),
        help="Models to run (default: all configured models).",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    games_per_model = 1 if args.dry_run else args.games

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing", file=sys.stderr)
        return 2
    client = OpenAI(timeout=120, max_retries=2)
    try:
        verify_models(client, args.models)
    except Exception as exc:
        print(f"Live model verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    prompts = json.loads((PROJECT_ROOT / "prompts.json").read_text(encoding="utf-8"))
    start_prompt = prompts["start"].format(
        n_trials=N_TRIALS_TOLD,
        choice_a=CHOICES[0],
        choice_b=CHOICES[1],
    )
    format_instruction = (
        'Respond with a JSON object containing exactly one key, "choice", with '
        'no other keys. The value must be "A" or "B". Do not include explanations '
        'or other text. Example exact response: {"choice":"A"}.'
    )
    system_prompt = f"{prompts['system']} {format_instruction} {start_prompt}"
    stored_prompts = dict(prompts)
    stored_prompts["system_actual"] = system_prompt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = "dry_run" if args.dry_run else "full_run"
    stamp = utc_stamp()
    db_path = args.output_dir / f"openai_fixed_probe_20_{label}_{len(args.models)}models_{stamp}.sql"
    csv_path = args.output_dir / f"openai_fixed_probe_20_{label}_{len(args.models)}models_{stamp}_wide.csv"
    db = init_db(db_path)
    rows: list[dict[str, Any]] = []
    failures = 0

    print(f"DATABASE {db_path}", flush=True)
    print(f"TASK_MODE {TASK_MODE}", flush=True)
    print(
        f"CONTINGENCY {INITIAL_CONTINGENCY}; isolated probe trials "
        f"{sorted(PROBE_TRIALS)} use {PROBE_CONTINGENCY}; no permanent reversal",
        flush=True,
    )
    print(
        "TRIAL COUNTS "
        f"n_trials_told={N_TRIALS_TOLD} n_trials_actual={N_TRIALS_ACTUAL}",
        flush=True,
    )

    for model in args.models:
        requested_model = model
        substitution_reason = None
        reasoning_effort = (
            "none" if model.startswith(("gpt-5.4-", "gpt-5.6-"))
            else "model_default"
        )
        config = {
            "design": DESIGN,
            "provider": PROVIDER,
            "requested_model": requested_model,
            "model": model,
            "substitution_reason": substitution_reason,
            "task_mode": TASK_MODE,
            "temperature": TEMPERATURE,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "games": games_per_model,
            "n_trials_told": N_TRIALS_TOLD,
            "n_trials_actual": N_TRIALS_ACTUAL,
            "intentional_trial_count_mismatch": False,
            "initial_contingency": INITIAL_CONTINGENCY,
            "post_reversal_contingency": None,
            "probe_trials": sorted(PROBE_TRIALS),
            "probe_contingency": PROBE_CONTINGENCY,
            "permanent_reversal": False,
            "reward_size": 1,
        }
        experiment_id = db.execute(
            """INSERT INTO experiment
               (created_at, design, provider, requested_model, model,
                substitution_reason, task_mode, temperature, reasoning_effort,
                max_output_tokens, games_requested, n_trials_told,
                n_trials_actual, intentional_trial_count_mismatch,
                initial_contingency_json, prompts_json, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now_iso(), DESIGN, PROVIDER, requested_model, model,
                substitution_reason, TASK_MODE, TEMPERATURE, reasoning_effort,
                MAX_OUTPUT_TOKENS, games_per_model, N_TRIALS_TOLD,
                N_TRIALS_ACTUAL, 0, as_json(INITIAL_CONTINGENCY),
                as_json(stored_prompts), as_json(config),
            ),
        ).lastrowid
        db.commit()

        for game_number in range(1, games_per_model + 1):
            answers = run_game(
                db, client, experiment_id, game_number, games_per_model,
                model, system_prompt, prompts,
            )
            row: dict[str, Any] = {
                "provider": PROVIDER,
                "model": model,
                "game_number": game_number,
            }
            row.update(
                {
                    f"answer_{trial}": answers[trial - 1] if trial <= len(answers) else ""
                    for trial in range(1, N_TRIALS_ACTUAL + 1)
                }
            )
            rows.append(row)
            if len(answers) != N_TRIALS_ACTUAL:
                failures += 1

    write_and_print_wide_summary(csv_path, rows)
    db.close()
    print(
        f"\nFinished: models={len(args.models)} games_per_model={games_per_model} "
        f"games_total={len(rows)} failures={failures}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
