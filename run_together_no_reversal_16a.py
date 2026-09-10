from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
import httpx
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
PROVIDER = "together"
BASE_URL = "https://api.together.xyz/v1"
MODELS = (
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3.5-9B",
)
TASK_MODE = "no_reversal"
TEMPERATURE = 0.7
MAX_TOKENS = 512
N_TRIALS_TOLD = 20
N_TRIALS_ACTUAL = 10
INITIAL_CONTINGENCY = {"B": 1, "A": 0}
PROBE_CONTINGENCY = {"A": 1, "B": 0}
PROBE_TRIALS = frozenset({4, 7})
CHOICES = ("A", "B")
FORMAT_INSTRUCTION = (
    'Respond with a JSON object containing exactly one key, "choice", with no '
    'other keys. The value must be either "A" or "B". Do not include reasoning '
    'or any text outside the JSON object. Example exact response: {"choice":"A"}. '
    'Required JSON Schema: {"type":"object","properties":{"choice":'
    '{"type":"string","enum":["A","B"]}},"required":["choice"],'
    '"additionalProperties":false}.'
)
CHOICE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "bandit_choice",
        "schema": {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
            "required": ["choice"],
            "additionalProperties": False,
        },
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


def together_api_key() -> str | None:
    return os.getenv("TOGETHERAI_API_KEY") or os.getenv("TOGETHER_API_KEY")


def live_model_ids(api_key: str) -> set[str]:
    response = httpx.get(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    response.raise_for_status()
    return {model["id"] for model in response.json()}


def verify_models(api_key: str) -> None:
    live_ids = live_model_ids(api_key)
    missing = [model for model in MODELS if model not in live_ids]
    print("LIVE MODEL VERIFICATION", flush=True)
    for model in MODELS:
        status = "available" if model in live_ids else "MISSING"
        print(f"  {status}: {model}", flush=True)
    if missing:
        raise RuntimeError(f"Together live model list is missing: {', '.join(missing)}")


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
            model TEXT NOT NULL,
            task_mode TEXT NOT NULL,
            temperature REAL NOT NULL,
            max_tokens INTEGER NOT NULL,
            games_requested INTEGER NOT NULL,
            n_trials_told INTEGER NOT NULL,
            n_trials_actual INTEGER NOT NULL,
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
            raw_output TEXT,
            choice TEXT,
            contingency_json TEXT,
            is_probe INTEGER NOT NULL DEFAULT 0,
            reward INTEGER,
            cumulative_reward INTEGER,
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
        raise BehavioralFormatError(f"Expected exactly one 'choice' field, received: {payload!r}")
    choice = payload["choice"]
    if choice not in CHOICES:
        raise BehavioralFormatError(f"Choice must be A or B, received: {choice!r}")
    return choice


def run_game(
    db: sqlite3.Connection,
    client: OpenAI,
    experiment_id: int,
    game_number: int,
    games_requested: int,
    model: str,
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
        "  INTENTIONAL TRIAL-COUNT MISMATCH: "
        f"n_trials_told={N_TRIALS_TOLD}; n_trials_actual={N_TRIALS_ACTUAL}",
        flush=True,
    )

    start_prompt = prompts["start"].format(
        n_trials=N_TRIALS_TOLD,
        choice_a=CHOICES[0],
        choice_b=CHOICES[1],
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": f"{prompts['system']} {start_prompt}"},
        {"role": "user", "content": start_prompt},
    ]
    answers: list[str] = []
    total_reward = 0

    for trial_number in range(1, N_TRIALS_ACTUAL + 1):
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "response_format": CHOICE_RESPONSE_FORMAT,
            "extra_body": {"reasoning": {"enabled": False}},
        }
        if model == "openai/gpt-oss-20b":
            request.pop("extra_body")
            request["reasoning_effort"] = "low"
        trial_id = db.execute(
            """INSERT INTO trial
               (game_id, trial_number, request_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (game_id, trial_number, as_json(request), now_iso()),
        ).lastrowid
        db.commit()
        print(f"  SEND actual_trial={trial_number}/{N_TRIALS_ACTUAL}: {messages[-1]['content']}", flush=True)

        began = time.perf_counter()
        try:
            response = client.chat.completions.create(**request)
            latency_ms = (time.perf_counter() - began) * 1000
            raw_output = response.choices[0].message.content or ""
            response_json = as_json(response.model_dump(mode="json", exclude_none=False))
            db.execute(
                "UPDATE trial SET response_json=?, raw_output=?, latency_ms=? WHERE id=?",
                (response_json, raw_output, latency_ms, trial_id),
            )
            db.commit()
            choice = parse_choice(raw_output)
            is_probe = trial_number in PROBE_TRIALS
            contingency = PROBE_CONTINGENCY if is_probe else INITIAL_CONTINGENCY
            reward = contingency[choice]
            total_reward += reward
            answers.append(choice)
            db.execute(
                """UPDATE trial SET response_json=?, raw_output=?, choice=?,
                   contingency_json=?, is_probe=?, reward=?, cumulative_reward=?,
                   latency_ms=? WHERE id=?""",
                (
                    as_json(response.model_dump(mode="json", exclude_none=False)),
                    raw_output,
                    choice,
                    as_json(contingency),
                    int(is_probe),
                    reward,
                    total_reward,
                    latency_ms,
                    trial_id,
                ),
            )
            db.commit()
            print(
                f"  RECV actual_trial={trial_number}/{N_TRIALS_ACTUAL}: "
                f"choice={choice} reward={reward} total={total_reward} "
                f"probe={is_probe} contingency={contingency}",
                flush=True,
            )
            messages.append({"role": "assistant", "content": raw_output})
            if trial_number < N_TRIALS_ACTUAL:
                feedback = prompts["feedback"].format(
                    previous_trial=trial_number,
                    previous_choice=choice,
                    reward=reward,
                    total_reward=total_reward,
                    trial=trial_number + 1,
                )
                messages.append({"role": "user", "content": feedback})
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
            print(f"  ERROR actual_trial={trial_number}: {error_type}: {error_message}", flush=True)
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
    parser = argparse.ArgumentParser(description="Run standalone Together no-reversal experiment 16a.")
    parser.add_argument("--games", type=int, default=10, help="Games per model (default: 10).")
    parser.add_argument("--dry-run", action="store_true", help="Run exactly one game per model.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    games_per_model = 1 if args.dry_run else args.games

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = together_api_key()
    if not api_key:
        print("TOGETHERAI_API_KEY (or TOGETHER_API_KEY) is missing", file=sys.stderr)
        return 2

    try:
        verify_models(api_key)
    except Exception as exc:
        print(f"Live model verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    prompts = json.loads((PROJECT_ROOT / "prompts.json").read_text(encoding="utf-8"))
    prompts["system"] = f"{prompts['system']} {FORMAT_INSTRUCTION}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = "dry_run" if args.dry_run else "full_run"
    stamp = utc_stamp()
    db_path = args.output_dir / f"together_no_reversal_16a_{label}_{stamp}.sql"
    csv_path = args.output_dir / f"together_no_reversal_16a_{label}_{stamp}_wide.csv"
    db = init_db(db_path)
    client = OpenAI(api_key=api_key, base_url=BASE_URL, max_retries=0)
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
        "INTENTIONAL TRIAL-COUNT MISMATCH "
        f"n_trials_told={N_TRIALS_TOLD} n_trials_actual={N_TRIALS_ACTUAL}",
        flush=True,
    )

    for model in MODELS:
        config = {
            "design": "16a",
            "provider": PROVIDER,
            "model": model,
            "task_mode": TASK_MODE,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "games": games_per_model,
            "n_trials_told": N_TRIALS_TOLD,
            "n_trials_actual": N_TRIALS_ACTUAL,
            "intentional_trial_count_mismatch": True,
            "initial_contingency": INITIAL_CONTINGENCY,
            "post_reversal_contingency": None,
            "probe_trials": sorted(PROBE_TRIALS),
            "probe_contingency": PROBE_CONTINGENCY,
            "permanent_reversal": False,
            "reward_size": 1,
        }
        experiment_id = db.execute(
            """INSERT INTO experiment
               (created_at, design, provider, model, task_mode, temperature,
                max_tokens, games_requested, n_trials_told, n_trials_actual,
                initial_contingency_json, prompts_json, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now_iso(), "16a", PROVIDER, model, TASK_MODE, TEMPERATURE,
                MAX_TOKENS, games_per_model, N_TRIALS_TOLD, N_TRIALS_ACTUAL,
                as_json(INITIAL_CONTINGENCY), as_json(prompts), as_json(config),
            ),
        ).lastrowid
        db.commit()

        for game_number in range(1, games_per_model + 1):
            answers = run_game(
                db, client, experiment_id, game_number, games_per_model, model, prompts
            )
            row: dict[str, Any] = {
                "provider": PROVIDER,
                "model": model,
                "game_number": game_number,
            }
            row.update(
                {f"answer_{trial}": answers[trial - 1] if trial <= len(answers) else ""
                 for trial in range(1, N_TRIALS_ACTUAL + 1)}
            )
            rows.append(row)
            if len(answers) != N_TRIALS_ACTUAL:
                failures += 1

    write_and_print_wide_summary(csv_path, rows)
    db.close()
    print(
        f"\nFinished: models={len(MODELS)} games_per_model={games_per_model} "
        f"games_total={len(rows)} failures={failures}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
