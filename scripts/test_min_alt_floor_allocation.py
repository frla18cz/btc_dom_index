#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on sys.path so we can import analyzer_weekly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analyzer_weekly import load_and_prepare, backtest_rank_altbtc_short
from config.config import EXCLUDED_SYMBOLS

def assert_allclose(a, b, tol=1e-9):
    if abs(a - b) > tol:
        raise AssertionError(f"Expected {a} ~ {b} within tol {tol}")

def run_case(df, alt_w, top_n, min_share, title):
    import io, sys as _sys
    _old = _sys.stdout
    _buf = io.StringIO()
    _sys.stdout = _buf  # suppress verbose analyzer prints
    try:
        perf, summary, detailed, *_ = backtest_rank_altbtc_short(
            df,
            btc_w=1.0,
            alt_w=alt_w,
            top_n=top_n,
            excluded=EXCLUDED_SYMBOLS,
            start_cap=100_000.0,
            detailed_output=True,
            benchmark_weights=None,
            alt_min_share_per_alt=min_share,
        )
    finally:
        _sys.stdout = _old

    if detailed is None or detailed.empty:
        raise AssertionError("Detailed positions empty")

    # Take first SHORT snapshot (initial week)
    shorts = detailed[detailed["Type"] == "SHORT"].copy()
    if shorts.empty:
        raise AssertionError("No ALT short positions present in detailed output")
    first_date = shorts["Date"].min()
    z = shorts[shorts["Date"] == first_date]
    symbols = z["Symbol"].tolist()
    weights = z["Weight"].to_numpy(dtype=float)

    # Invariants
    n = len(symbols)
    if n == 0:
        raise AssertionError("No alt positions on first week")

    total_w = weights.sum()
    assert_allclose(total_w, alt_w, 1e-8)

    scaled_min = min(alt_w, min_share * n) / n
    min_weight = weights.min()
    if min_weight + 1e-12 < scaled_min:
        raise AssertionError(
            f"Min weight {min_weight:.8f} < scaled_min {scaled_min:.8f}"
        )

    # If remainder exists, check weights beyond floor correlate with market caps
    remainder = alt_w - scaled_min * n

    # Get mcap from df at first_date (rebalance_ts)
    w0 = df[df["rebalance_ts"] == pd.Timestamp(first_date)].set_index("sym")
    mcaps = []
    for s in symbols:
        if s not in w0.index:
            raise AssertionError(f"{s} not in data for {first_date}")
        mcaps.append(float(w0.loc[s, "mcap_btc"]))
    mcaps = np.array(mcaps)

    if remainder > 1e-12 and mcaps.sum() > 0:
        excess = weights - scaled_min
        # Spearman-like rank correlation without scipy
        ranks_weights = np.argsort(np.argsort(excess))
        ranks_mcap = np.argsort(np.argsort(mcaps))
        tau = np.corrcoef(ranks_weights, ranks_mcap)[0, 1]
        if tau < 0.9:
            raise AssertionError(
                f"Rank correlation between (weight-floor) and mcap too low: {tau:.3f}"
            )

    print(
        f"[OK] {title}: N={n}, alt_w={alt_w:.3f}, min={min_share:.3f}, "
        f"scaled_min={scaled_min:.5f}, sum_w={total_w:.5f}, min_w={min_weight:.5f}"
    )


def main():
    csv_path = Path("top100_weekly_data.csv")
    if not csv_path.exists():
        print(
            "ERROR: top100_weekly_data.csv not found. Run `python fetcher.py` or use the Streamlit 'Update Data' button.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load full dataset for robustness
    df = load_and_prepare(csv_path)
    weeks = sorted(df["rebalance_ts"].unique())
    if len(weeks) < 2:
        print("ERROR: Not enough weekly snapshots to run backtest", file=sys.stderr)
        sys.exit(1)

    # Case 1: User example
    run_case(df, alt_w=0.75, top_n=10, min_share=0.05, title="Example 75% ALT, TOP10, min 5%")
    # Case 2: Scaling boundary (min too high → scales to ALT/N)
    run_case(
        df,
        alt_w=0.75,
        top_n=10,
        min_share=0.15,
        title="Scaling case 75% ALT, TOP10, min 15% (scale to 7.5%)",
    )
    # Case 3: Zero min (back-compat)
    run_case(df, alt_w=0.75, top_n=10, min_share=0.0, title="Zero min (cap-weight only)")


if __name__ == "__main__":
    main()

