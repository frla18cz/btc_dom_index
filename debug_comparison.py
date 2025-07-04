#!/usr/bin/env python3
"""
Debug script to compare UI results with direct script execution.
"""
import datetime as dt
from analyzer_weekly import load_and_prepare, backtest_rank_altbtc_short

def main():
    # Exact same period as UI test
    start_date = dt.datetime(2025, 6, 2)  
    end_date = dt.datetime(2025, 6, 30)   

    print('Loading data with same parameters as UI...')
    df = load_and_prepare('top100_weekly_data.csv', start_date, end_date)

    print(f'Data loaded: {len(df)} rows over {len(df["rebalance_ts"].unique())} snapshots')
    print(f'Date range: {df["rebalance_ts"].min().date()} to {df["rebalance_ts"].max().date()}')

    # Run with same default parameters as UI (no custom benchmark)
    perf_df, summary, detailed_df, benchmark_df, benchmark_comparison = backtest_rank_altbtc_short(
        df,
        btc_w=0.5,
        alt_w=0.5,
        top_n=10,
        start_cap=100000.0,
        benchmark_weights={'BTC': 1.0},
        benchmark_rebalance_weekly=False,
        detailed_output=False
    )

    print()
    print('=== DIRECT COMPARISON ===')
    print('UI Total Return: +4.65%')
    print(f'Script Total Return: {summary["total_return_pct"]:.2f}%')
    print(f'Difference: {4.65 - summary["total_return_pct"]:.2f}%')

    print()
    print('UI Final Equity: $104,651.09')
    print(f'Script Final Equity: ${summary["final_equity"]:,.2f}')
    print(f'Difference: ${104651.09 - summary["final_equity"]:,.2f}')

    print()
    print('UI BTC P/L: +$2,089.11')
    print(f'Script BTC P/L: +${summary["cum_btc_pnl"]:,.2f}')
    print(f'BTC P/L Difference: ${2089.11 - summary["cum_btc_pnl"]:,.2f}')

    print()
    print('UI ALT P/L: +$2,561.97') 
    print(f'Script ALT P/L: +${summary["cum_alt_pnl"]:,.2f}')
    print(f'ALT P/L Difference: ${2561.97 - summary["cum_alt_pnl"]:,.2f}')

    print()
    print('=== DETAILED BREAKDOWN ===')
    for key, value in summary.items():
        if isinstance(value, float):
            print(f'{key}: {value:.2f}')
        else:
            print(f'{key}: {value}')

    print()
    print('=== WEEKLY PERFORMANCE COMPARISON ===')
    print('Week-by-week breakdown from script:')
    for i, row in perf_df.iterrows():
        print(f"Week {i+1} ({row['Date'].strftime('%Y-%m-%d')}): {row['Weekly_Return_Pct']:+.2f}%")

if __name__ == "__main__":
    main()