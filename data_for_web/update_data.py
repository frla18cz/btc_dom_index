#!/usr/bin/env python3
"""
Utility script that reruns the weekly backtest and refreshes the
`data_for_web/data.json` payload used by the marketing site.

The script:
  1. Loads the historical weekly snapshot CSV (same pipeline as `analyzer_weekly.py`)
  2. Executes the strategy backtest and benchmark comparison
  3. Serialises the key outputs (meta, summary, performance series, quarterly stats)
     into the JSON schema consumed by the web front-end

Example:
    python data_for_web/update_data.py --output data_for_web/data.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Mapping

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from analyzer_weekly import (
    CSV_PATH as DEFAULT_CSV_PATH,
    END_DATE as DEFAULT_END_DATE,
    START_CAP as DEFAULT_START_CAP,
    START_DATE as DEFAULT_START_DATE,
    backtest_rank_altbtc_short,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    load_and_prepare,
)
from benchmark_analyzer import (
    calculate_benchmark_performance,
    compare_strategy_vs_benchmark,
)
from config.config import (
    BENCHMARK_REBALANCE_DEFAULT,
    BENCHMARK_REBALANCE_WEEKLY,
    DEFAULT_BENCHMARK_WEIGHTS,
)

# -----------------------------------------------------------------------------
# Multi-phase strategy configuration
# -----------------------------------------------------------------------------
# Each phase defines the strategy parameters for a date range.
# end_date of one phase should equal start_date of the next (overlap week).
# end_date=None means "run to the end of available data".
STRATEGY_PHASES = [
    {
        "start_date": "2023-01-02",
        "end_date": "2025-10-20",
        "btc_w": 1.75,
        "alt_w": 0.75,
        "benchmark_rebalance": "weekly",
    },
    {
        "start_date": "2025-10-20",
        "end_date": None,
        "btc_w": 1.50,
        "alt_w": 0.25,
        "benchmark_rebalance": "monthly",
    },
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _resolve_dates(value: str | None, fallback: dt.date | dt.datetime | str | None) -> dt.date | None:
    """Parse CLI dates while accepting the config defaults (string or datetime)."""
    if value:
        return dt.date.fromisoformat(value)
    if fallback is None:
        return None
    if isinstance(fallback, (dt.date, dt.datetime)):
        return fallback.date() if isinstance(fallback, dt.datetime) else fallback
    return dt.date.fromisoformat(str(fallback))


def _determine_benchmark_policy(cli_value: str | None) -> str | bool | None:
    """
    Resolve benchmark rebalance policy to the format expected by the analyzer.
    Priority: CLI flag -> config default -> weekly toggle fallback.
    """
    if cli_value:
        return cli_value
    if BENCHMARK_REBALANCE_DEFAULT is not None:
        return BENCHMARK_REBALANCE_DEFAULT
    return "weekly" if BENCHMARK_REBALANCE_WEEKLY else "none"


def _build_performance_series(perf_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge strategy and benchmark equity curves on date and add a sequential week index.
    """
    strategy = perf_df[["Date", "Equity_USD"]].copy()
    strategy["Date"] = pd.to_datetime(strategy["Date"])

    benchmark = benchmark_df[["Date", "Portfolio_Value"]].copy()
    benchmark["Date"] = pd.to_datetime(benchmark["Date"])

    merged = strategy.merge(benchmark, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    merged.rename(
        columns={
            "Equity_USD": "strategie",
            "Portfolio_Value": "benchmark",
        },
        inplace=True,
    )
    merged.insert(0, "week", merged.index.astype(int))
    return merged


def _quarterly_comparison(perf_merged: pd.DataFrame) -> list[dict[str, object]]:
    """
    Aggregate quarter-over-quarter returns for strategy vs benchmark.
    """
    perf_merged["Date"] = pd.to_datetime(perf_merged["Date"])
    perf_merged["Period"] = perf_merged["Date"].dt.to_period("Q")

    comparison: list[dict[str, object]] = []
    for period, group in perf_merged.groupby("Period"):
        group = group.sort_values("Date")
        if group.empty:
            continue

        start = group.iloc[0]
        end = group.iloc[-1]

        strategy_return = (end["strategie"] / start["strategie"] - 1.0) * 100 if start["strategie"] else 0.0
        benchmark_return = (end["benchmark"] / start["benchmark"] - 1.0) * 100 if start["benchmark"] else 0.0

        week_min = int(group["week"].min())
        week_max = int(group["week"].max())
        week_dates = [d.strftime("%Y-%m-%d") for d in group["Date"]]

        comparison.append(
            {
                "month": f"Q{period.quarter} {period.year}",
                "strategie": round(strategy_return, 1),
                "benchmark": round(benchmark_return, 1),
                "weeks": f"{week_min}-{week_max}",
                "weekDates": week_dates,
            }
        )
    return comparison


def _meta_block(description: str) -> dict[str, object]:
    """Build the metadata portion for the JSON payload."""
    return {
        "lastUpdate": dt.date.today().isoformat(),
        "description": description,
    }


def _summary_block(
    summary: Mapping[str, float],
    benchmark_df: pd.DataFrame,
    benchmark_metrics: Mapping[str, float],
    perf_merged: pd.DataFrame,
    start_cap: float,
) -> dict[str, object]:
    """Derive summary numbers for the landing page."""
    start_value = start_cap
    final_equity = float(summary.get("final_equity", 0.0))
    benchmark_final = float(benchmark_df["Portfolio_Value"].iloc[-1]) if not benchmark_df.empty else 0.0

    start_date = pd.to_datetime(perf_merged["Date"].iloc[0])
    end_date = pd.to_datetime(perf_merged["Date"].iloc[-1])

    def _fmt_date(value: pd.Timestamp) -> str:
        # Example format: "1. 1. 2023" (avoids platform-dependent strftime specifiers)
        return f"{value.day}. {value.month}. {value.year}"

    return {
        "finalValue": round(final_equity),
        "benchmarkFinalValue": round(benchmark_final),
        "startValue": round(start_value),
        "maxDrawdownAI": round(float(summary.get("max_drawdown", 0.0)), 2),
        "maxDrawdownBenchmark": round(float(benchmark_metrics.get("benchmark_max_drawdown", 0.0)), 2),
        "totalWeeks": int(len(perf_merged)),
        "startDate": _fmt_date(start_date),
        "endDate": _fmt_date(end_date),
    }


def _performance_payload(perf_merged: pd.DataFrame) -> list[dict[str, object]]:
    """Convert merged weekly data into the array expected by the web front-end."""
    records = []
    for row in perf_merged.itertuples(index=False):
        records.append(
            {
                "week": int(row.week),
                "strategie": round(float(row.strategie), 2),
                "benchmark": round(float(row.benchmark), 2),
                "date": pd.Timestamp(row.Date).strftime("%Y-%m-%d"),
            }
        )
    return records


def _run_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, dict]:
    """Execute the strategy backtest with optional stdout suppression."""
    df = load_and_prepare(
        Path(args.csv),
        start_date=args.start_date,
        end_date=args.end_date,
        include_fng=args.include_fng,
    )

    benchmark_policy = _determine_benchmark_policy(args.benchmark_policy)

    stream = nullcontext()
    buffer: io.StringIO | None = None
    if not args.verbose:
        buffer = io.StringIO()
        stream = redirect_stdout(buffer)

    with stream:
        perf_df, summary, _, benchmark_df, benchmark_metrics = backtest_rank_altbtc_short(
            df,
            start_cap=args.start_cap,
            detailed_output=args.verbose,
            benchmark_weights=DEFAULT_BENCHMARK_WEIGHTS,
            benchmark_rebalance=benchmark_policy,
        )

    if perf_df.empty or benchmark_df.empty:
        raise RuntimeError("Backtest did not produce performance data; aborting JSON refresh.")

    return perf_df, summary, benchmark_df, df, benchmark_metrics


def _run_multiphase_backtest(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the strategy backtest in multiple phases with different parameters,
    then stitch the equity curves together.  The benchmark is run once over
    the full date range (same params throughout).
    """
    benchmark_policy = _determine_benchmark_policy(args.benchmark_policy)

    # Determine the overall date range from CLI / config
    overall_start = args.start_date
    overall_end = args.end_date

    # ----- Run each phase sequentially (strategy + benchmark) -----
    phase_perf_dfs: list[pd.DataFrame] = []
    phase_bench_dfs: list[pd.DataFrame] = []
    strategy_carry_cap = args.start_cap
    benchmark_carry_cap = args.start_cap

    for idx, phase in enumerate(STRATEGY_PHASES):
        # Resolve phase date boundaries, clipped to the overall window
        p_start = phase["start_date"]
        p_end = phase["end_date"]  # None ⇒ run to end of data

        # Apply overall bounds
        if overall_start and (p_end is not None) and str(p_end) < str(overall_start):
            continue  # entire phase is before our window
        if overall_end and str(p_start) > str(overall_end):
            continue  # entire phase is after our window

        effective_start = max(str(p_start), str(overall_start)) if overall_start else p_start
        effective_end = min(str(p_end), str(overall_end)) if (p_end and overall_end) else (p_end or overall_end)

        # Resolve benchmark rebalance policy for this phase
        phase_bench_policy = phase.get("benchmark_rebalance") or benchmark_policy

        # Load phase-specific slice
        buf = io.StringIO()
        with redirect_stdout(buf) if not args.verbose else nullcontext():
            df_phase = load_and_prepare(
                Path(args.csv),
                start_date=effective_start,
                end_date=effective_end,
                include_fng=args.include_fng,
            )
            perf_df_i, summary_i, _, _, _ = backtest_rank_altbtc_short(
                df_phase,
                btc_w=phase["btc_w"],
                alt_w=phase["alt_w"],
                start_cap=strategy_carry_cap,
                detailed_output=False,
            )
            bench_df_i = calculate_benchmark_performance(
                df_phase,
                DEFAULT_BENCHMARK_WEIGHTS,
                start_cap=benchmark_carry_cap,
                rebalance_weekly=phase_bench_policy,
            )

        if perf_df_i.empty:
            raise RuntimeError(f"Phase {idx} backtest produced no data (dates {effective_start}–{effective_end}).")

        phase_perf_dfs.append(perf_df_i)
        phase_bench_dfs.append(bench_df_i)
        strategy_carry_cap = summary_i["final_equity"]
        benchmark_carry_cap = float(bench_df_i["Portfolio_Value"].iloc[-1])

    # ----- Stitch strategy phases -----
    if not phase_perf_dfs:
        raise RuntimeError("No strategy phases produced data.")

    stitched = phase_perf_dfs[0]
    for subsequent in phase_perf_dfs[1:]:
        # Skip the first row of subsequent phases (overlap / transition point)
        stitched = pd.concat([stitched, subsequent.iloc[1:]], ignore_index=True)

    # ----- Stitch benchmark phases -----
    benchmark_df = phase_bench_dfs[0]
    for subsequent in phase_bench_dfs[1:]:
        benchmark_df = pd.concat([benchmark_df, subsequent.iloc[1:]], ignore_index=True)

    # ----- Recalculate summary metrics from stitched curve -----
    final_equity = float(stitched["Equity_USD"].iloc[-1])
    total_return_pct = ((final_equity - args.start_cap) / args.start_cap) * 100
    weekly_returns = stitched["Weekly_Return_Pct"].iloc[1:].dropna().values  # skip t0 row
    num_weeks = len(weekly_returns) if len(weekly_returns) > 0 else 1
    annualized_return = total_return_pct * (52 / num_weeks)
    max_dd = calculate_max_drawdown(stitched)

    summary = {
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "annualized_return": annualized_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": calculate_sharpe_ratio(weekly_returns) if len(weekly_returns) > 1 else 0,
        "sortino_ratio": calculate_sortino_ratio(weekly_returns) if len(weekly_returns) > 1 else 0,
        "win_rate": float(np.mean(weekly_returns > 0) * 100) if len(weekly_returns) > 0 else 0,
    }

    # ----- Compute benchmark comparison metrics -----
    buf = io.StringIO()
    with redirect_stdout(buf) if not args.verbose else nullcontext():
        benchmark_metrics = compare_strategy_vs_benchmark(
            stitched, benchmark_df, summary, args.start_cap,
        )

    if stitched.empty or benchmark_df.empty:
        raise RuntimeError("Multi-phase backtest did not produce performance data; aborting.")

    return stitched, summary, benchmark_df, stitched, benchmark_metrics


def parse_args() -> argparse.Namespace:
    """CLI arg parser."""
    parser = argparse.ArgumentParser(description="Rebuild data_for_web/data.json from the latest backtest output.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="Path to the weekly snapshot CSV.")
    parser.add_argument("--output", default="data_for_web/data.json", help="Target JSON file for the web payload.")
    parser.add_argument("--start-date", dest="start_date", help="Optional ISO start date override (YYYY-MM-DD).")
    parser.add_argument("--end-date", dest="end_date", help="Optional ISO end date override (YYYY-MM-DD).")
    parser.add_argument("--start-cap", dest="start_cap", type=float, default=DEFAULT_START_CAP, help="Initial capital.")
    parser.add_argument(
        "--benchmark-policy",
        choices=["none", "weekly", "monthly"],
        help="Override benchmark rebalance policy.",
    )
    parser.add_argument(
        "--include-fng",
        action="store_true",
        help="Enable Fear & Greed (FNG) enrichment before running the backtest.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full backtest logs instead of running quietly.")
    parser.add_argument(
        "--description",
        default="Performance Data",
        help="Text for the meta.description field in the JSON payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.start_date = _resolve_dates(args.start_date, DEFAULT_START_DATE)
    args.end_date = _resolve_dates(args.end_date, DEFAULT_END_DATE)

    perf_df, summary, benchmark_df, _, benchmark_metrics = _run_multiphase_backtest(args)

    perf_merged = _build_performance_series(perf_df, benchmark_df)

    payload = {
        "meta": _meta_block(description=args.description),
        "summary": _summary_block(summary, benchmark_df, benchmark_metrics, perf_merged, args.start_cap),
        "performanceData": _performance_payload(perf_merged),
        "monthlyComparison": _quarterly_comparison(perf_merged),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote updated dataset to {output_path}")


if __name__ == "__main__":
    main()

