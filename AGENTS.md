# Repository Guidelines

## Project Structure & Module Organization
- `app.py`: Streamlit UI entrypoint (web app).
- `analyzer_weekly.py`: Core backtester and plotting utilities.
- `benchmark_analyzer.py`: Benchmark calculation and comparison charts.
- `fetcher.py`: Weekly data fetching (Playwright-driven scraping).
- `config/config.py`: Tunable parameters (weights, exclusions, dates).
- `utils/`, `indicators/`, `scripts/`: Helpers, metrics, and dev scripts.
- `data/`, `reports/`, `snapshots/`: Local artifacts (CSV, figures, diagnostics).
- `test_benchmark_chart.py`, `test_chart_fix.py`: Sanity tests for charts/baselines.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Playwright setup: `python -m playwright install-deps firefox && python -m playwright install firefox`
- Run app: `streamlit run app.py`
- Fetch data: `python fetcher.py [--add-historical]`
- Backtest (CLI): `python analyzer_weekly.py`
- Verify calcs: `python verify_calculations.py`
- Tests (scripts): `python test_chart_fix.py` and `python test_benchmark_chart.py`

## Coding Style & Naming Conventions
- Python, PEP 8, 4‑space indentation; line length ~100.
- Functions/vars: `snake_case`; Classes: `CamelCase`; constants: `UPPER_SNAKE_CASE` (see `config.py`).
- Prefer type hints for public functions and clear docstrings.
- Keep plotting headless-safe (set `MPLBACKEND=Agg` when applicable).

## Testing Guidelines
- Tests are executable scripts that print diagnostics and surface errors.
- Naming: `test_*.py` at repo root; keep tests deterministic and file-path agnostic.
- Run selectively: `python test_benchmark_chart.py`.
- Expected artifacts: figures are created in-memory; do not rely on GUI.

## Commit & Pull Request Guidelines
- Conventional commits preferred: `feat(scope): …`, `fix(ui): …`, `refactor(core): …` (see `git log`).
- PRs must include: clear description, rationale, before/after notes (screenshots for Streamlit charts), and linked issues.
- Update docs when changing behavior: `README.md`, `config/config.py` defaults, and tests.
- Keep diffs focused; include small test or script update validating the change.

## Security & Configuration Tips
- Do not commit secrets; local datasets may be large—avoid adding new binaries.
- Scraping: respect timeouts and sleep intervals (`config/config.py`); avoid unnecessary traffic.
- Deployment (Procfile): the `release` step installs Playwright; mirror locally when debugging.
