# Hidden-reversal two-armed bandit experiment

The runner tests `gpt-3.5-turbo`, `gpt-4`, `gpt-4o-mini`, `gpt-5.6-sol`, `gpt-5.6-terra`, and `gpt-5.6-luna` over 20
independent games each. Reversal occurs uniformly on trial 11 through 16, and
each game has 10 through 15 trials after reversal.

```powershell
python run_experiment.py
```

Results are SQLite databases in `results/`. Despite the `.sql` suffix required
by the specification, these are binary SQLite database files, not SQL text dumps.
Every game stops at its first API or format error; other games continue.

The comparable legacy-model batch uses Chat Completions because GPT-3.5 Turbo
and GPT-4 do not support strict Structured Outputs:

```powershell
python run_legacy_experiment.py
```
