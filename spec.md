# Original Specification

Measure how an LLM adapts its choice behavior in a two-armed bandit task after a hidden, one-time reversal of the reward contingency. The model must track feedback across trials and update its policy without ever being told the underlying rule.

Two actions: A, B.

One trial = one action choice by the model, followed by a binary reward (0 or 1) reported back to it.

Contingency, phase 1 (trials 0 .. r-1): B -> 1, A -> 0.

Hidden reversal at trial r, drawn once per run.

Contingency, phase 2 (trials r .. n-1): A -> 1, B -> 0.

Reward is deterministic given the active contingency (no reward noise).

The model is never told r, never told which contingency is active, and never told the reversal occurred. It only sees its own choice and the resulting reward, each trial, for the whole run.

A "run" (a.k.a. "game") is one full sequence of n_trials trials with one independently drawn reversal point. Runs are independent: no state, RNG, or conversation history crosses runs. The model knows the number of trials and has a task to maximize its score. Everything should be stored in a database. Database's name should go like model_name_config_number_date_time.sql. The date and time (with microseconds) of the run goes as a seed by default. The model has to answer in a json format ONLY, without any additional reasonings. If any error occurs, including wrong answer's format, the run has to be stopped and recorded to the db. The error's description goes to the db too. Everything which is sent to and received from the mode and errors should be displayed on the screen.

Information which has to go to the config:

* provider  - Anthropic, OpenAI, Together.ai
* model name
* Temperature
* Max tokens
* Seed #optional
* N trials
* Reversal trial
* Choice A: <string> #Choices may be A and B or Left and Right or Red and Blue and so on.
* Choice B: <string>
* Initial contingency, for example: {"B": 1, "A": 0}
* Post reversal contingency, for example: {"B": 0, "A": 1}
* Reward size

You are welcome to propose more.

There should be a separate file with prompts. Knowing everything above, fill it with preliminary versions.

Config, all the prompts, all models answers go to the db. If you have any questions and suggestions on how to improve the work flow, please, tell me. Let's discuss them before implementation.
## 16a. Together.ai standalone no-reversal run with intentional trial-count mismatch

This is an additional standalone design and does not replace the earlier section
16 sweep. Use `task_mode = "no_reversal"` with a constant deterministic
contingency of `{"B": 1, "A": 0}`. Run 10 independent games for each of three
Together.ai models at temperature 0.7:

- `deepseek-ai/DeepSeek-V4-Flash-0731`
- `openai/gpt-oss-20b`
- `Qwen/Qwen3.5-9B`

The originally planned `meta-llama/Meta-Llama-3-8B-Instruct-Lite` and
`Qwen/Qwen2.5-7B-Instruct-Turbo` were replaced because Together required
hourly billed dedicated endpoints. Their serverless substitutes are
`openai/gpt-oss-20b` and `Qwen/Qwen3.5-9B`, respectively. All three final IDs
were verified through standard serverless chat completions; no dedicated
endpoint may be created or started for this run.

The prompt tells each model that the game contains `n_trials_told = 20`, while
the runner intentionally stops after `n_trials_actual = 10`; both values must
be stored and clearly labeled. Save the normal database records plus a wide CSV
with columns `provider`, `model`, `game_number`, and `answer_1` through
`answer_10`. Enforce the exact one-key `{"choice":"A"}` or `{"choice":"B"}`
contract with JSON Schema and explicit prompt text. Any malformed response is a
behavioral/format error and must not be silently normalized.

For the requested run, trials 4 and 7 are fixed, isolated probe trials using `{"A": 1, "B": 0}`. Every other actual trial uses `{"B": 1, "A": 0}`. After each probe, the next trial immediately returns to the initial contingency; this is not a permanent reversal. Store the applied contingency and probe flag on every trial.
