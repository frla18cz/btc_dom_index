#!/usr/bin/env python3
"""
Verification script for checking backtest calculations.
"""
import datetime as dt
import os
os.environ['MPLBACKEND'] = 'Agg'

from analyzer_weekly import load_and_prepare, backtest_rank_altbtc_short

def main():
    # Set date range for last month (June 2025)
    start_date = dt.datetime(2025, 6, 2)  
    end_date = dt.datetime(2025, 6, 30)   

    print("Loading data...")
    df = load_and_prepare('top100_weekly_data.csv', start_date, end_date)

    # Define benchmark weights (100% BTC for simple verification)
    benchmark_weights = {'BTC': 1.0}

    print("Running backtest...")
    perf_df, summary, detailed_df, benchmark_df, benchmark_comparison = backtest_rank_altbtc_short(
        df,
        btc_w=0.5,
        alt_w=0.5,
        top_n=10,
        start_cap=100000.0,
        benchmark_weights=benchmark_weights,
        benchmark_rebalance_weekly=False,
        detailed_output=False
    )

    print("\n" + "="*80)
    print("FINAL VERIFICATION RESULTS")
    print("="*80)

    # Print final verification data
    print(f"Strategy periods analyzed: {len(perf_df)}")
    print(f"Benchmark periods analyzed: {len(benchmark_df)}")

    print("\nFinal comparison:")
    print(f"Strategy total return: {summary['total_return_pct']:.2f}%")
    print(f"Benchmark total return: {benchmark_comparison.get('benchmark_total_return', 0):.2f}%")
    print(f"Alpha (excess return): {benchmark_comparison.get('alpha', 0):.2f}%")

    print("\nDate alignment check:")
    print(f"Strategy first date: {perf_df['Date'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"Benchmark first date: {benchmark_df['Date'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"First dates match: {perf_df['Date'].iloc[0] == benchmark_df['Date'].iloc[0]}")

    print(f"Strategy last date: {perf_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Benchmark last date: {benchmark_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Last dates match: {perf_df['Date'].iloc[-1] == benchmark_df['Date'].iloc[-1]}")

    # Manual BTC calculation check
    print("\n" + "="*50)
    print("MANUAL CALCULATION VERIFICATION")
    print("="*50)

    # Get actual BTC prices from data
    weeks = sorted(df['rebalance_ts'].unique())
    btc_start_data = df[(df['rebalance_ts'] == weeks[0]) & (df['sym'] == 'BTC')]['price_usd'].iloc[0]
    btc_end_data = df[(df['rebalance_ts'] == weeks[-1]) & (df['sym'] == 'BTC')]['price_usd'].iloc[0]

    print(f"BTC price start (from data): ${btc_start_data:,.2f}")
    print(f"BTC price end (from data): ${btc_end_data:,.2f}")

    btc_return_manual = ((btc_end_data - btc_start_data) / btc_start_data) * 100
    print(f"Manual BTC return: {btc_return_manual:.2f}%")

    bench_return_from_data = benchmark_comparison.get('benchmark_total_return', 0)
    print(f"Benchmark return (from calc): {bench_return_from_data:.2f}%")
    print(f"Difference: {abs(btc_return_manual - bench_return_from_data):.4f}%")

    if abs(btc_return_manual - bench_return_from_data) < 0.01:
        print("\n✅ Benchmark calculation is CORRECT!")
    else:
        print("\n⚠️  WARNING: Benchmark calculation mismatch!")

    print("\n" + "="*50)
    print("WEEKLY RETURNS CHECK")
    print("="*50)

    print("Week        Strategy%  Benchmark%  Difference")
    print("-" * 45)
    for i in range(min(len(perf_df), len(benchmark_df))):
        strat_ret = perf_df.iloc[i]['Weekly_Return_Pct']
        bench_ret = benchmark_df.iloc[i]['Weekly_Return_Pct']
        diff = strat_ret - bench_ret
        date_str = perf_df.iloc[i]['Date'].strftime('%Y-%m-%d')
        print(f"{date_str}  {strat_ret:8.2f}  {bench_ret:9.2f}  {diff:8.2f}")

    print("\n" + "="*50)
    print("STRATEGY SUMMARY")
    print("="*50)
    
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()