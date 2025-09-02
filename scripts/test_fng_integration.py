#!/usr/bin/env python3
"""
Quick validation for Fear & Greed integration without committing changes.

Checks performed:
1) Ensure FNG daily CSV can be fetched/loaded and shows sane date range
2) Aggregate to weekly (Monday) and preview a few rows
3) If top100_weekly_data.csv exists, load prepared dataset with include_fng=True and
   report join coverage and a few sample weeks

Run:
    python scripts/test_fng_integration.py
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path('.').resolve()))

from indicators.fng import load_fng_daily_csv, daily_to_weekly
from analyzer_weekly import load_and_prepare


def main():
    ok = True

    print("[1/3] Loading FNG daily CSV (will auto-fetch if missing)...")
    fng_daily = load_fng_daily_csv(Path("data/fng_daily.csv"))
    if fng_daily.empty:
        print("  ERROR: FNG daily dataframe is empty")
        ok = False
    else:
        print(f"  OK: {len(fng_daily)} rows | range: {fng_daily['date'].min().date()} -> {fng_daily['date'].max().date()}")
        print(fng_daily.tail(3))

    print("\n[2/3] Aggregating to weekly (policies: monday, sunday, tuesday_lookahead) and preview...")
    weekly_mon = daily_to_weekly(fng_daily, policy="monday") if not fng_daily.empty else pd.DataFrame()
    weekly_sun = daily_to_weekly(fng_daily, policy="sunday") if not fng_daily.empty else pd.DataFrame()
    weekly_tue = daily_to_weekly(fng_daily, policy="tuesday_lookahead") if not fng_daily.empty else pd.DataFrame()
    if not weekly_mon.empty:
        print(f"  Monday policy: {len(weekly_mon)} weeks | sample tail:")
        print(weekly_mon.tail(3))
    else:
        print("  WARNING: weekly (monday) is empty")
    if not weekly_sun.empty:
        print(f"  Sunday policy: {len(weekly_sun)} weeks | sample tail:")
        print(weekly_sun.tail(3))
    if not weekly_tue.empty:
        print(f"  Tuesday LOOKAHEAD policy: {len(weekly_tue)} weeks | sample tail:")
        print(weekly_tue.tail(3))

    data_path = Path("top100_weekly_data.csv")
    if not data_path.exists():
        print("\n[3/3] Skipping dataset join check (top100_weekly_data.csv not found)")
        return 0 if ok else 1

    print("\n[3/3] Loading crypto dataset and merging FNG (include_fng=True, policy=tuesday_lookahead)...")
    df = load_and_prepare(data_path, include_fng=True, fng_csv_path=Path("data/fng_daily.csv"), fng_policy="tuesday_lookahead")
    if df.empty:
        print("  ERROR: Prepared dataset is empty after load")
        return 1

    # Coverage of FNG values over distinct weeks
    weeks = df["rebalance_ts"].drop_duplicates()
    fng_present = df.dropna(subset=["fng_value"])["rebalance_ts"].drop_duplicates()
    coverage = (len(fng_present) / len(weeks)) * 100 if len(weeks) else 0
    print(f"  OK: {len(weeks)} distinct weeks; FNG coverage: {coverage:.1f}% (non-null)")

    # Show sample of 5 weeks with fng_value
    preview_cols = ["rebalance_ts", "sym", "rank", "price_usd", "fng_value", "fng_classification"]
    have_cols = [c for c in preview_cols if c in df.columns]
    sample = df.dropna(subset=["fng_value"]).sort_values("rebalance_ts").groupby("rebalance_ts").head(1)[have_cols].tail(5)
    print("\n  Sample of weekly join (1 row per week):")
    print(sample)

    print("\nAll checks completed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

