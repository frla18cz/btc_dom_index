# Contributing

Thanks for your interest in improving this project! Please read AGENTS.md for the complete contributor guide. This file summarizes how to get started and open effective pull requests.

## Setup
- Python 3.10+ recommended.
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Playwright: `python -m playwright install-deps firefox && python -m playwright install firefox`
- Run app: `streamlit run app.py`

## Tests
- Script-style tests: `python test_chart_fix.py` and `python test_benchmark_chart.py`
- Keep tests deterministic and headless (use `MPLBACKEND=Agg` if plotting).

## Workflow
- Branch from `main`: `feature/<short-topic>` or `fix/<short-topic>`
- Conventional commits preferred: `feat(core): ...`, `fix(ui): ...`, `refactor(bench): ...`

## Pull Requests
- Describe what and why; link issues.
- Include before/after notes (screenshots for Streamlit charts).
- Update docs when behavior changes: `README.md`, `AGENTS.md`, and config defaults.
- Ensure tests/scripts pass locally; avoid committing large data or secrets.

## Issues
- Provide steps to reproduce, expected/actual behavior, logs, and environment.

For detailed structure, commands, style, and security notes, see AGENTS.md.
