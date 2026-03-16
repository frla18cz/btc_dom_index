# `data_for_web` quick guide

This folder stores the JSON payload that feeds the public web viz together with
the helper script that rebuilds it from the backtest outputs.

## Multi-phase strategy

The strategy parameters changed over time.  `update_data.py` runs each phase
separately and stitches the equity curves so that the historical portion is
always reproduced identically.

| | Phase 1 (Legacy) | Phase 2 (Current) |
|---|---|---|
| **Dates** | 2023-01-02 → 2025-10-20 (weeks 0–146) | 2025-10-27 → ongoing (weeks 147+) |
| **BTC long weight** | 175 % | 150 % |
| **ALT short weight** | 75 % | 25 % |
| **Rebalance** | weekly | weekly |
| **Top N / Excluded** | 10 / default | 10 / default |
| **Benchmark** | 50/50 BTC/ETH, weekly rebalance | 50/50 BTC/ETH, monthly rebalance |

The transition point (2025-10-20) is the last week of phase 1 and the first
week of phase 2.  Phase 1 final equity becomes phase 2 starting capital; same
for the benchmark.  This is defined in `STRATEGY_PHASES` at the top of
`update_data.py`.

### Adding a new phase

To add a future parameter change:

1. Set the current last phase's `end_date` to the transition Monday.
2. Append a new dict to `STRATEGY_PHASES` with `start_date` equal to that same
   Monday and `end_date: None`.
3. Run the update and verify the transition week is seamless.

## Refresh workflow

1. Ensure `top100_weekly_data.csv` is up to date (either via `fetcher.py` or the Streamlit UI).
2. Run the updater script:
   ```bash
   python data_for_web/update_data.py \
       --start-date 2023-01-02 \
       --end-date <latest_monday>
   ```
   `--start-date 2023-01-02` is required to capture all phases.  `--end-date`
   should be the latest available Monday in the CSV (omitting it falls back to
   the config default which may be outdated).
3. Inspect `data_for_web/data.json` (start/end dates + last weeks) and commit/push.

### Optional flags

| Flag | Purpose |
|---|---|
| `--benchmark-policy` | Override benchmark rebalance for **all** phases (normally per-phase from `STRATEGY_PHASES`) |
| `--include-fng` | Enable Fear & Greed enrichment |
| `--verbose` | Print full backtest logs |
| `--start-cap` | Override initial capital (default 100 000) |

## Reproducibility guarantee

Running the script on the same CSV with `--start-date 2023-01-02` always
reproduces the historical weeks identically because:

- Each phase uses fixed `btc_w`, `alt_w`, and `benchmark_rebalance` from
  `STRATEGY_PHASES` — not from `config.py` defaults, which may change.
- Phase equity is carried forward deterministically (phase N final equity →
  phase N+1 starting capital).
- Adding new data (future weeks) only extends the last open-ended phase; it
  never alters earlier phases.

**Reference data** (`data_for_web/test/data.json`, 163 weeks through 2026-02-09)
is verified to match at $0.00 difference on both strategy and benchmark.

## Verification

After any code change, run:

```bash
python data_for_web/update_data.py \
    --start-date 2023-01-02 \
    --end-date 2026-02-09 \
    --output /tmp/verify.json

# Compare against reference (all 163 weeks should show $0.00 diff)
python3 -c "
import json
ref = json.load(open('data_for_web/test/data.json'))
out = json.load(open('/tmp/verify.json'))
max_ds = max_db = 0
for i in range(len(ref['performanceData'])):
    r, o = ref['performanceData'][i], out['performanceData'][i]
    max_ds = max(max_ds, abs(r['strategie'] - o['strategie']))
    max_db = max(max_db, abs(r['benchmark'] - o['benchmark']))
print(f'Strategy max diff: \${max_ds:.2f}')
print(f'Benchmark max diff: \${max_db:.2f}')
"
```

## JSON schema

```
{
  "meta":             { lastUpdate, description },
  "summary":          { finalValue, benchmarkFinalValue, startValue, maxDrawdownAI, maxDrawdownBenchmark, totalWeeks, startDate, endDate },
  "performanceData":  [ { week, strategie, benchmark, date }, ... ],
  "monthlyComparison": [ { month, strategie, benchmark, weeks, weekDates }, ... ]
}
```

## LLM playbook

- Always use `--start-date 2023-01-02` to include all phases.
- Set `--end-date` to the latest Monday in the CSV snapshots.
- After running, summarise the new `summary` block (final value, benchmark
  value, max drawdowns, range).
- If `.gitignore` changes hide the JSON, remember to stage both `data.json`
  and `update_data.py`.
- When the run emits a matplotlib cache warning, export `MPLCONFIGDIR` to a
  writable scratch path before retrying.
- Always push changes to a feature branch and open a PR unless told otherwise.
