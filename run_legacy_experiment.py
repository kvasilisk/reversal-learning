from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, APIStatusError, OpenAI

from run_experiment import (
    CHOICES,
    CONFIG_NUMBER,
    MAX_OUTPUT_TOKENS,
    PROJECT_ROOT,
    REWARD_SIZE,
    as_json,
    init_db,
    now_iso,
    utc_stamp,
)


MODELS = ("gpt-3.5-turbo", "gpt-4", "gpt-4o-mini")
TEMPERATURE = None  # Omitted, matching the previous experiment's API-default setting.
API_STYLE = "chat_completions"
MAX_API_ATTEMPTS = 5
INITIAL_BACKOFF_SECONDS = 1.0
DEFAULT_GPT4_CALL_DELAY_SECONDS = 1.5


class BehavioralError(ValueError):
    """A model answer violates the experiment's required response contract."""


def is_infrastructure_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    return isinstance(exc, APIStatusError) and (
        exc.status_code == 429 or exc.status_code >= 500
    )


def ensure_legacy_schema(db) -> None:
    """Add retry/top-up metadata and normalize historical error categories."""
    game_columns = {row[1] for row in db.execute("PRAGMA table_info(game)")}
    if "run_id" not in game_columns:
        db.execute("ALTER TABLE game ADD COLUMN run_id TEXT")
    if "is_topup" not in game_columns:
        db.execute("ALTER TABLE game ADD COLUMN is_topup INTEGER NOT NULL DEFAULT 0")
    db.execute(
        """CREATE TABLE IF NOT EXISTS api_attempt (
            id INTEGER PRIMARY KEY,
            trial_id INTEGER NOT NULL REFERENCES trial(id),
            attempt_number INTEGER NOT NULL,
            request_json TEXT NOT NULL,
            response_json TEXT,
            response_id TEXT,
            raw_output TEXT,
            latency_ms REAL,
            status TEXT NOT NULL,
            error_class TEXT,
            exception_type TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(trial_id, attempt_number)
        )"""
    )
    # Preserve the original exception name in error_message while making the
    # machine-readable error_type conform to the two required categories.
    for table in ("trial", "game"):
        db.execute(
            f"""UPDATE {table} SET error_message='Original exception: ' || error_type || '. ' || error_message,
                error_type='infrastructure'
                WHERE error_type IN ('RateLimitError','InternalServerError','APIConnectionError','APITimeoutError')"""
        )
        db.execute(
            f"""UPDATE {table} SET error_message='Original exception: ' || error_type || '. ' || error_message,
                error_type='behavioral'
                WHERE error_type IN ('ValueError','JSONDecodeError','BehavioralError')"""
        )
    db.commit()


def chat_response_dict(response: Any) -> dict[str, Any]:
    return response.model_dump(mode="json", exclude_none=False)


def run_game(
    db,
    client: OpenAI,
    experiment_id: int,
    game_number: int,
    games_requested: int,
    model: str,
    prompts: dict[str, str],
    seed: str,
    run_id: str,
    is_topup: bool = False,
) -> bool:
    rng = random.Random(seed)
    reversal_trial = rng.randint(11, 16)
    post_trials = rng.randint(10, 15)
    n_trials = reversal_trial - 1 + post_trials
    cursor = db.execute(
        """INSERT INTO game
           (experiment_id, game_number, seed, reversal_trial, post_reversal_trials,
            n_trials, started_at, status, run_id, is_topup)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)""",
        (experiment_id, game_number, seed, reversal_trial, post_trials, n_trials,
         now_iso(), run_id, int(is_topup)),
    )
    game_id = cursor.lastrowid
    db.commit()

    total_reward = 0
    start = prompts["start"].format(
        n_trials=n_trials, choice_a=CHOICES[0], choice_b=CHOICES[1]
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": start},
    ]
    print(
        f"\n[{model}] game={game_number}/{games_requested} seed={seed} "
        f"n={n_trials} reversal={reversal_trial}", flush=True
    )

    for trial_number in range(1, n_trials + 1):
        phase = 1 if trial_number < reversal_trial else 2
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        trial_cursor = db.execute(
            """INSERT INTO trial
               (game_id, trial_number, phase, request_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (game_id, trial_number, phase, as_json(request), now_iso()),
        )
        trial_id = trial_cursor.lastrowid
        db.commit()
        print(f"  SEND trial={trial_number}: {messages[-1]['content']}", flush=True)
        response_json: str | None = None
        response_id: str | None = None
        raw_output: str | None = None
        response = None
        last_infrastructure_error: Exception | None = None
        total_latency_ms = 0.0
        for attempt_number in range(1, MAX_API_ATTEMPTS + 1):
            attempt_cursor = db.execute(
                """INSERT INTO api_attempt
                   (trial_id, attempt_number, request_json, status, created_at)
                   VALUES (?, ?, ?, 'running', ?)""",
                (trial_id, attempt_number, as_json(request), now_iso()),
            )
            attempt_id = attempt_cursor.lastrowid
            db.commit()
            began = time.perf_counter()
            try:
                response = client.chat.completions.create(**request)
                latency_ms = (time.perf_counter() - began) * 1000
                total_latency_ms += latency_ms
                response_json = as_json(chat_response_dict(response))
                response_id = response.id
                raw_output = response.choices[0].message.content or ""
                db.execute(
                    """UPDATE api_attempt SET response_json=?, response_id=?, raw_output=?,
                       latency_ms=?, status='success' WHERE id=?""",
                    (response_json, response_id, raw_output, latency_ms, attempt_id),
                )
                db.commit()
                break
            except Exception as exc:
                latency_ms = (time.perf_counter() - began) * 1000
                total_latency_ms += latency_ms
                if not is_infrastructure_error(exc):
                    # Unexpected local/programming exceptions are not mislabeled
                    # as model behavior and are allowed to surface.
                    raise
                last_infrastructure_error = exc
                db.execute(
                    """UPDATE api_attempt SET latency_ms=?, status='error',
                       error_class='infrastructure', exception_type=?, error_message=?
                       WHERE id=?""",
                    (latency_ms, type(exc).__name__, str(exc), attempt_id),
                )
                db.commit()
                print(
                    f"  RETRY trial={trial_number} attempt={attempt_number}/{MAX_API_ATTEMPTS}: "
                    f"{type(exc).__name__}: {exc}", flush=True
                )
                if attempt_number < MAX_API_ATTEMPTS:
                    delay = INITIAL_BACKOFF_SECONDS * (2 ** (attempt_number - 1))
                    time.sleep(delay)

        if response is None:
            assert last_infrastructure_error is not None
            error_message = (
                f"Retries exhausted after {MAX_API_ATTEMPTS} attempts. "
                f"Last exception: {type(last_infrastructure_error).__name__}: "
                f"{last_infrastructure_error}"
            )
            db.execute(
                """UPDATE trial SET latency_ms=?, error_type='infrastructure',
                   error_message=? WHERE id=?""",
                (total_latency_ms, error_message, trial_id),
            )
            db.execute(
                """UPDATE game SET finished_at=?, status='error', total_reward=?,
                   error_type='infrastructure', error_message=? WHERE id=?""",
                (now_iso(), total_reward, error_message, game_id),
            )
            db.commit()
            print(f"  ERROR trial={trial_number}: infrastructure: {error_message}", flush=True)
            return False

        try:
            payload = json.loads(raw_output)
            if (
                not isinstance(payload, dict)
                or set(payload) != {"choice"}
                or payload["choice"] not in CHOICES
            ):
                raise BehavioralError(f"Invalid choice payload: {payload!r}")
            choice = payload["choice"]
            reward = REWARD_SIZE if (
                (phase == 1 and choice == "B") or (phase == 2 and choice == "A")
            ) else 0
            total_reward += reward
            usage = response.usage
            db.execute(
                """UPDATE trial SET response_json=?, response_id=?, raw_output=?, choice=?,
                   reward=?, cumulative_reward=?, input_tokens=?, output_tokens=?,
                   reasoning_tokens=?, latency_ms=? WHERE id=?""",
                (response_json, response_id, raw_output, choice,
                 reward, total_reward, getattr(usage, "prompt_tokens", None),
                 getattr(usage, "completion_tokens", None), None, total_latency_ms, trial_id),
            )
            db.commit()
            print(
                f"  RECV trial={trial_number}: {raw_output} reward={reward} "
                f"total={total_reward}", flush=True
            )
            messages.append({"role": "assistant", "content": raw_output})
            if trial_number < n_trials:
                feedback = prompts["feedback"].format(
                    previous_trial=trial_number,
                    previous_choice=choice,
                    reward=reward,
                    total_reward=total_reward,
                    trial=trial_number + 1,
                )
                messages.append({"role": "user", "content": feedback})
        except (json.JSONDecodeError, BehavioralError) as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            db.execute(
                """UPDATE trial SET response_json=?, response_id=?, raw_output=?,
                   latency_ms=?, error_type='behavioral', error_message=? WHERE id=?""",
                (response_json, response_id, raw_output, total_latency_ms,
                 error_message, trial_id),
            )
            db.execute(
                """UPDATE game SET finished_at=?, status='error', total_reward=?,
                   error_type='behavioral', error_message=? WHERE id=?""",
                (now_iso(), total_reward, error_message, game_id),
            )
            db.commit()
            print(f"  ERROR trial={trial_number}: behavioral: {error_message}", flush=True)
            return False

    db.execute(
        "UPDATE game SET finished_at=?, status='completed', total_reward=? WHERE id=?",
        (now_iso(), total_reward, game_id),
    )
    db.commit()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run legacy-model bandit experiments.")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--top-up-db", type=Path,
        help="Append games to an existing database until --target-completed is reached.",
    )
    parser.add_argument("--target-completed", type=int, default=20)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing", file=sys.stderr)
        return 2

    prompts = json.loads((PROJECT_ROOT / "prompts.json").read_text(encoding="utf-8"))
    # Disable SDK-level retries so every physical API attempt is controlled and
    # recorded by this runner's explicit five-attempt policy.
    client = OpenAI(max_retries=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    if args.top_up_db:
        if not args.top_up_db.exists():
            parser.error(f"top-up database does not exist: {args.top_up_db}")
        import sqlite3
        db = sqlite3.connect(args.top_up_db)
        db.execute("PRAGMA foreign_keys = ON")
        ensure_legacy_schema(db)
        experiment = db.execute(
            "SELECT id, model, prompts_json, config_json FROM experiment ORDER BY id LIMIT 1"
        ).fetchone()
        if experiment is None:
            parser.error("top-up database has no experiment row")
        experiment_id, model, prompts_json, config_json = experiment
        if model not in MODELS:
            parser.error(f"unsupported model in top-up database: {model}")
        prompts = json.loads(prompts_json)
        config = json.loads(config_json)
        if (
            config.get("reversal_trial_min") != 11
            or config.get("reversal_trial_max") != 16
            or config.get("post_reversal_trials_min") != 10
            or config.get("post_reversal_trials_max") != 15
        ):
            parser.error("top-up database configuration does not match this runner")
        run_id = f"topup_{utc_stamp()}"
        next_game_number = db.execute(
            "SELECT COALESCE(MAX(game_number), 0) + 1 FROM game WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()[0]
        completed = db.execute(
            "SELECT COUNT(*) FROM game WHERE experiment_id=? AND status='completed'",
            (experiment_id,),
        ).fetchone()[0]
        print(
            f"TOP-UP {args.top_up_db} run_id={run_id} model={model} "
            f"completed={completed} target={args.target_completed}", flush=True
        )
        while completed < args.target_completed:
            seed = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
            succeeded = run_game(
                db, client, experiment_id, next_game_number,
                args.target_completed, model, prompts, seed, run_id, True
            )
            next_game_number += 1
            if succeeded:
                completed += 1
            else:
                failures += 1
        db.close()
        print(
            f"\nTop-up finished: completed={completed}, new_failures={failures}, "
            f"run_id={run_id}"
        )
        return 1 if failures else 0

    for model in args.models:
        db_path = args.output_dir / f"{model}_config_{CONFIG_NUMBER}_{utc_stamp()}.sql"
        db = init_db(db_path)
        ensure_legacy_schema(db)
        config = {
            "provider": "OpenAI", "model": model, "api_style": API_STYLE,
            "temperature": TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS,
            "reasoning_effort": "not_applicable", "games": args.games,
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
             "not_applicable", args.games, "A", "B", as_json({"A": 0, "B": 1}),
             as_json({"A": 1, "B": 0}), REWARD_SIZE, as_json(prompts), as_json(config)),
        ).lastrowid
        db.commit()
        print(f"DATABASE {db_path}", flush=True)
        run_id = f"initial_{utc_stamp()}"
        for game_number in range(1, args.games + 1):
            seed = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
            if not run_game(
                db, client, experiment_id, game_number, args.games, model, prompts, seed,
                run_id, False
            ):
                failures += 1
        db.close()
    print(f"\nFinished: {len(args.models) * args.games} games, failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
