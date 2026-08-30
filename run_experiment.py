from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
CHOICES = ("A", "B")
CONFIG_NUMBER = 1
REWARD_SIZE = 1
MAX_OUTPUT_TOKENS = 64
REASONING_EFFORT = "none"
TEMPERATURE = None  # Intentionally omitted; recorded as the API default.


SCHEMA = {
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


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def init_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(
        """
        CREATE TABLE experiment (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            config_number INTEGER NOT NULL,
            temperature REAL,
            max_output_tokens INTEGER NOT NULL,
            reasoning_effort TEXT NOT NULL,
            games_requested INTEGER NOT NULL,
            choice_a TEXT NOT NULL,
            choice_b TEXT NOT NULL,
            initial_contingency_json TEXT NOT NULL,
            post_contingency_json TEXT NOT NULL,
            reward_size INTEGER NOT NULL,
            prompts_json TEXT NOT NULL,
            config_json TEXT NOT NULL
        );
        CREATE TABLE game (
            id INTEGER PRIMARY KEY,
            experiment_id INTEGER NOT NULL REFERENCES experiment(id),
            game_number INTEGER NOT NULL,
            seed TEXT NOT NULL,
            reversal_trial INTEGER NOT NULL,
            post_reversal_trials INTEGER NOT NULL,
            n_trials INTEGER NOT NULL,
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
            phase INTEGER NOT NULL,
            request_json TEXT NOT NULL,
            response_json TEXT,
            response_id TEXT,
            raw_output TEXT,
            choice TEXT,
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


def response_dict(response: Any) -> dict[str, Any]:
    return response.model_dump(mode="json", exclude_none=False)


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
    model: str,
    prompts: dict[str, str],
    seed: str,
) -> bool:
    rng = random.Random(seed)
    reversal_trial = rng.randint(11, 16)
    post_trials = rng.randint(10, 15)
    n_trials = reversal_trial - 1 + post_trials
    started_at = now_iso()
    cursor = db.execute(
        """INSERT INTO game
           (experiment_id, game_number, seed, reversal_trial, post_reversal_trials,
            n_trials, started_at, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'running')""",
        (experiment_id, game_number, seed, reversal_trial, post_trials, n_trials, started_at),
    )
    game_id = cursor.lastrowid
    db.commit()

    total_reward = 0
    previous_response_id: str | None = None
    next_input = prompts["start"].format(
        n_trials=n_trials, choice_a=CHOICES[0], choice_b=CHOICES[1]
    )

    print(
        f"\n[{model}] game={game_number}/20 seed={seed} n={n_trials} "
        f"reversal={reversal_trial}", flush=True
    )

    for trial_number in range(1, n_trials + 1):
        phase = 1 if trial_number < reversal_trial else 2
        request: dict[str, Any] = {
            "model": model,
            "instructions": prompts["system"],
            "input": next_input,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "reasoning": {"effort": REASONING_EFFORT},
            "text": {"format": SCHEMA},
            "store": True,
        }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        trial_cursor = db.execute(
            """INSERT INTO trial
               (game_id, trial_number, phase, request_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (game_id, trial_number, phase, as_json(request), now_iso()),
        )
        trial_id = trial_cursor.lastrowid
        db.commit()
        print(f"  SEND trial={trial_number}: {next_input}", flush=True)

        began = time.perf_counter()
        try:
            response = client.responses.create(**request)
            latency_ms = (time.perf_counter() - began) * 1000
            raw_output = response.output_text
            payload = json.loads(raw_output)
            if set(payload) != {"choice"} or payload["choice"] not in CHOICES:
                raise ValueError(f"Invalid choice payload: {payload!r}")
            choice = payload["choice"]
            reward = REWARD_SIZE if (
                (phase == 1 and choice == "B") or (phase == 2 and choice == "A")
            ) else 0
            total_reward += reward
            input_tokens, output_tokens, reasoning_tokens = usage_values(response)
            response_json = as_json(response_dict(response))
            db.execute(
                """UPDATE trial SET response_json=?, response_id=?, raw_output=?, choice=?,
                   reward=?, cumulative_reward=?, input_tokens=?, output_tokens=?,
                   reasoning_tokens=?, latency_ms=? WHERE id=?""",
                (response_json, response.id, raw_output, choice, reward, total_reward,
                 input_tokens, output_tokens, reasoning_tokens, latency_ms, trial_id),
            )
            db.commit()
            print(
                f"  RECV trial={trial_number}: {raw_output} reward={reward} "
                f"total={total_reward}", flush=True
            )
            previous_response_id = response.id
            next_input = prompts["feedback"].format(
                previous_trial=trial_number,
                previous_choice=choice,
                reward=reward,
                total_reward=total_reward,
                trial=trial_number + 1,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - began) * 1000
            error_type = type(exc).__name__
            error_message = str(exc)
            db.execute(
                """UPDATE trial SET latency_ms=?, error_type=?, error_message=? WHERE id=?""",
                (latency_ms, error_type, error_message, trial_id),
            )
            db.execute(
                """UPDATE game SET finished_at=?, status='error', total_reward=?,
                   error_type=?, error_message=? WHERE id=?""",
                (now_iso(), total_reward, error_type, error_message, game_id),
            )
            db.commit()
            print(f"  ERROR trial={trial_number}: {error_type}: {error_message}", flush=True)
            return False

    db.execute(
        "UPDATE game SET finished_at=?, status='completed', total_reward=? WHERE id=?",
        (now_iso(), total_reward, game_id),
    )
    db.commit()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hidden-reversal bandit experiments.")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing", file=sys.stderr)
        return 2
    prompts = json.loads((PROJECT_ROOT / "prompts.json").read_text(encoding="utf-8"))
    client = OpenAI()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for model_index, model in enumerate(args.models):
        stamp = utc_stamp()
        db_path = args.output_dir / f"{model}_config_{CONFIG_NUMBER}_{stamp}.sql"
        db = init_db(db_path)
        config = {
            "provider": "OpenAI", "model": model, "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "reasoning_effort": REASONING_EFFORT, "games": args.games,
            "reversal_trial_min": 11, "reversal_trial_max": 16,
            "post_reversal_trials_min": 10, "post_reversal_trials_max": 15,
            "choice_a": "A", "choice_b": "B",
            "initial_contingency": {"A": 0, "B": 1},
            "post_reversal_contingency": {"A": 1, "B": 0},
            "reward_size": REWARD_SIZE,
        }
        experiment_id = db.execute(
            """INSERT INTO experiment
               (created_at, provider, model, config_number, temperature, max_output_tokens,
                reasoning_effort, games_requested, choice_a, choice_b,
                initial_contingency_json, post_contingency_json, reward_size,
                prompts_json, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now_iso(), "OpenAI", model, CONFIG_NUMBER, TEMPERATURE, MAX_OUTPUT_TOKENS,
             REASONING_EFFORT, args.games, "A", "B", as_json({"A": 0, "B": 1}),
             as_json({"A": 1, "B": 0}), REWARD_SIZE, as_json(prompts), as_json(config)),
        ).lastrowid
        db.commit()
        print(f"DATABASE {db_path}", flush=True)
        for game_number in range(1, args.games + 1):
            # A fresh UTC timestamp with microseconds is the default seed. Keep it
            # as text because its 20 digits exceed SQLite's signed INTEGER range.
            game_seed = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
            if not run_game(db, client, experiment_id, game_number, model, prompts, game_seed):
                failures += 1
        db.close()

    print(f"\nFinished: {len(args.models) * args.games} games, failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
