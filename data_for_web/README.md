# `data_for_web` quick guide

This folder stores the JSON payload that feeds the public web viz together with
the helper script that rebuilds it from the backtest outputs.

## Refresh workflow

1. Ensure `top100_weekly_data.csv` is up to date (either via `fetcher.py` or the Streamlit UI).
2. Run the updater script with the desired window:
   ```bash
   python data_for_web/update_data.py --start-date 2023-01-02 --end-date <latest_monday>
   ```
   - Omit the flags to fall back to config defaults.
   - Add `--benchmark-policy` or `--include-fng` when you need non-default behaviour.
   - Set `MPLCONFIGDIR=/tmp/.matplotlib` if matplotlib cannot write to `$HOME`.
3. Inspect `data_for_web/data.json` (start/end dates + last weeks) and commit/push.

## LLM playbook

- Use `python data_for_web/update_data.py --start-date 2023-01-02 --end-date <YYYY-MM-DD>` with `<YYYY-MM-DD>` equal to the latest available Monday in the snapshots.
- After running, summarise the new `summary` block (final value, benchmark value, max drawdowns, range).
- If `.gitignore` changes hide the JSON, remember to stage both `data.json` and `update_data.py`.
- When the run emits a matplotlib cache warning, export `MPLCONFIGDIR` to a writable scratch path before retrying.
- Always push changes to a feature branch and open a PR unless told otherwise.
