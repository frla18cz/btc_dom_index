#!/usr/bin/env python3
"""
Verify UI parameters match the config defaults.
"""
import datetime as dt
import os
os.environ['MPLBACKEND'] = 'Agg'

from analyzer_weekly import load_and_prepare, backtest_rank_altbtc_short
from config.config import BACKTEST_BTC_WEIGHT, BACKTEST_ALT_WEIGHT

def main():
    # Same period as UI test
    start_date = dt.datetime(2025, 6, 2)  
    end_date = dt.datetime(2025, 6, 30)   

    print('=== VERIFYING UI PARAMETERS ===')
    print(f'Config BTC Weight: {BACKTEST_BTC_WEIGHT}')
    print(f'Config ALT Weight: {BACKTEST_ALT_WEIGHT}')
    print(f'Total Leverage: {BACKTEST_BTC_WEIGHT + BACKTEST_ALT_WEIGHT}')

    print('\nLoading data...')
    df = load_and_prepare('top100_weekly_data.csv', start_date, end_date)

    print(f'Data loaded: {len(df)} rows over {len(df["rebalance_ts"].unique())} snapshots')

    # Run with UI/config default parameters  
    perf_df, summary, detailed_df, benchmark_df, benchmark_comparison = backtest_rank_altbtc_short(
        df,
        btc_w=BACKTEST_BTC_WEIGHT,    # Use config defaults (1.75)
        alt_w=BACKTEST_ALT_WEIGHT,    # Use config defaults (0.75)
        top_n=10,
        start_cap=100000.0,
        benchmark_weights={'BTC': 1.0},
        benchmark_rebalance_weekly=False,
        detailed_output=False
    )

    print('\n=== COMPARISON WITH UI RESULTS ===')
    print('Expected UI Results:')
    print('  Total Return: +4.65%')
    print('  Final Equity: $104,651.09')
    print('  BTC P/L: +$2,089.11')
    print('  ALT P/L: +$2,561.97')
    
    print('\nActual Script Results (with UI params):')
    print(f'  Total Return: {summary["total_return_pct"]:+.2f}%')
    print(f'  Final Equity: ${summary["final_equity"]:,.2f}')
    print(f'  BTC P/L: +${summary["cum_btc_pnl"]:,.2f}')
    print(f'  ALT P/L: +${summary["cum_alt_pnl"]:,.2f}')

    # Calculate differences
    total_return_diff = abs(4.65 - summary["total_return_pct"])
    final_equity_diff = abs(104651.09 - summary["final_equity"])
    btc_pnl_diff = abs(2089.11 - summary["cum_btc_pnl"])
    alt_pnl_diff = abs(2561.97 - summary["cum_alt_pnl"])

    print('\nDifferences:')
    print(f'  Total Return: {total_return_diff:.3f}%')
    print(f'  Final Equity: ${final_equity_diff:,.2f}')
    print(f'  BTC P/L: ${btc_pnl_diff:,.2f}')
    print(f'  ALT P/L: ${alt_pnl_diff:,.2f}')

    # Check if results match (within reasonable tolerance)
    tolerance_pct = 0.01  # 0.01% tolerance
    tolerance_usd = 1.0   # $1 tolerance

    matches = (
        total_return_diff <= tolerance_pct and
        final_equity_diff <= tolerance_usd and
        btc_pnl_diff <= tolerance_usd and
        alt_pnl_diff <= tolerance_usd
    )

    if matches:
        print('\n✅ RESULTS MATCH! UI and script are consistent.')
    else:
        print('\n⚠️  RESULTS DO NOT MATCH. Investigation needed.')

    print('\n=== DETAILED METRICS ===')
    for key, value in summary.items():
        if isinstance(value, float):
            print(f'{key}: {value:.2f}')
        else:
            print(f'{key}: {value}')

if __name__ == "__main__":
    main()