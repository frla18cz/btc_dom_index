#!/usr/bin/env python3
"""
Verification script for checking backtest calculations.
"""
import datetime as dt
import os
os.environ['MPLBACKEND'] = 'Agg'

from analyzer_weekly import load_and_prepare, backtest_rank_altbtc_short

# Pretty printing helpers
try:
    from utils.pretty import section, kv_table, table, fmt_pct, fmt_money
except Exception:
    section = kv_table = table = None
    def fmt_pct(x, places=2, sign=False):
        return f"{x:+.{places}f}%" if sign else f"{x:.{places}f}%"
    def fmt_money(x, currency="$"):
        return f"{currency}{x:,.2f}"

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

    if section:
        section("FINAL VERIFICATION RESULTS")
    else:
        print("\n" + "="*80)
        print("FINAL VERIFICATION RESULTS")
        print("="*80)

    # Print final verification data
    if kv_table:
        kv_table({
            "Strategy periods analyzed": len(perf_df),
            "Benchmark periods analyzed": len(benchmark_df)
        }, title="Counts")
        
        kv_table({
            "Strategy total return": fmt_pct(summary['total_return_pct'], 2),
            "Benchmark total return": fmt_pct(benchmark_comparison.get('benchmark_total_return', 0), 2),
            "Alpha (excess return)": fmt_pct(benchmark_comparison.get('alpha', 0), 2)
        }, title="Final comparison")
        
        kv_table({
            "Strategy first date": perf_df['Date'].iloc[0].strftime('%Y-%m-%d'),
            "Benchmark first date": benchmark_df['Date'].iloc[0].strftime('%Y-%m-%d'),
            "First dates match": perf_df['Date'].iloc[0] == benchmark_df['Date'].iloc[0],
            "Strategy last date": perf_df['Date'].iloc[-1].strftime('%Y-%m-%d'),
            "Benchmark last date": benchmark_df['Date'].iloc[-1].strftime('%Y-%m-%d'),
            "Last dates match": perf_df['Date'].iloc[-1] == benchmark_df['Date'].iloc[-1]
        }, title="Date alignment check")
    else:
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
    if section:
        section("MANUAL CALCULATION VERIFICATION", width=50)
    else:
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

    if section:
        section("WEEKLY RETURNS CHECK", width=50)
    else:
        print("\n" + "="*50)
        print("WEEKLY RETURNS CHECK")
        print("="*50)

    rows = []
    for i in range(min(len(perf_df), len(benchmark_df))):
        strat_ret = perf_df.iloc[i]['Weekly_Return_Pct']
        bench_ret = benchmark_df.iloc[i]['Weekly_Return_Pct']
        diff = strat_ret - bench_ret
        date_str = perf_df.iloc[i]['Date'].strftime('%Y-%m-%d')
        rows.append([date_str, f"{strat_ret:+.2f}%", f"{bench_ret:+.2f}%", f"{diff:+.2f}%"])    
    if table:
        table(rows, headers=["Week", "Strategy%", "Benchmark%", "Difference"]) 
    else:
        print("Week        Strategy%  Benchmark%  Difference")
        print("-" * 45)
        for r in rows:
            print(f"{r[0]}  {r[1]:>10}  {r[2]:>11}  {r[3]:>9}")

    if kv_table:
        # Order subset of most relevant metrics first
        ordered = {
            "Final Equity": fmt_money(summary.get('final_equity', 0.0)),
            "Total Return": fmt_pct(summary.get('total_return_pct', 0.0), 2, sign=True),
            "Annualized Return": fmt_pct(summary.get('annualized_return', 0.0), 2, sign=True),
            "Max Drawdown": fmt_pct(summary.get('max_drawdown', 0.0), 2),
            "Sharpe Ratio": f"{summary.get('sharpe_ratio', 0.0):.2f}",
            "Sortino Ratio": f"{summary.get('sortino_ratio', 0.0):.2f}",
            "Win Rate": fmt_pct(summary.get('win_rate', 0.0), 1),
            "BTC P&L": fmt_money(summary.get('cum_btc_pnl', 0.0)),
            "ALT P&L": fmt_money(summary.get('cum_alt_pnl', 0.0)),
        }
        kv_table(ordered, title="STRATEGY SUMMARY", width=50)
    else:
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