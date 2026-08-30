# Hidden-reversal two-armed bandit experiment

The two runners test `gpt-3.5-turbo`, `gpt-4`, `gpt-4o-mini`, `gpt-5.6-sol`,
`gpt-5.6-terra`, and `gpt-5.6-luna`. By default, each selected model runs 20
independent games. Reversal occurs uniformly on trial 11 through 16, and each
game has 10 through 15 trials after reversal.

Results are stored in SQLite databases under `results/`. Files use a `.sql`
suffix to match the experiment specification, but they are binary SQLite
databases rather than SQL text dumps.

## Setup

1. Clone the repository and enter it:

   ```bash
   git clone <repository-url>
   cd bandit-test
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   PowerShell: `.\.venv\Scripts\Activate.ps1`

   macOS/Linux: `source .venv/bin/activate`

3. Install the pinned dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and replace the placeholder with your key:

   PowerShell: `Copy-Item .env.example .env`

   macOS/Linux: `cp .env.example .env`

   The required variable is `OPENAI_API_KEY`. The `.env` file is ignored by Git.
   You can instead export the same variable in your shell; existing environment
   variables take precedence over values in `.env`.

5. Optionally verify the key and list the models available to it:

   ```bash
   python check_models.py
   ```

## Run an experiment

Start with a one-game smoke test:

```bash
python run_experiment.py --games 1 --models gpt-5.6-sol
```

Run the complete default batch (20 games for each configured model):

```bash
python run_experiment.py
```

The modern runner uses OpenAI's Responses API, structured JSON output, and
`previous_response_id` to continue each game. This is why its requests differ
from a basic Chat Completions integration. Every game stops at its first API or
response-format error; remaining games continue.

The legacy-model batch uses Chat Completions:

```bash
python run_legacy_experiment.py
```

Useful options include `--games N`, `--models MODEL [MODEL ...]`, and
`--output-dir PATH`. Run either script with `--help` for all options. Project
configuration and the default output directory are resolved from the cloned
repository, so the scripts can be launched from any directory.
