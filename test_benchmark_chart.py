#!/usr/bin/env python3
"""
Test script to verify benchmark chart baseline fix.
"""
import datetime as dt
import os
os.environ['MPLBACKEND'] = 'Agg'

from analyzer_weekly import load_and_prepare, backtest_rank_altbtc_short
from benchmark_analyzer import calculate_benchmark_performance, plot_strategy_vs_benchmark

def test_benchmark_chart():
    """Test benchmark chart baseline with BTC benchmark."""
    
    print("=== TESTING BENCHMARK CHART BASELINE FIX ===")
    
    # Test data for June 2025
    start_date = dt.datetime(2025, 6, 2)
    end_date = dt.datetime(2025, 6, 30)
    initial_capital = 100000.0
    
    # Load data
    df = load_and_prepare('top100_weekly_data.csv', start_date, end_date)
    
    print(f"Testing with period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${initial_capital:,}")
    print()
    
    # Run strategy backtest
    strategy_perf, strategy_summary, _, _, _ = backtest_rank_altbtc_short(
        df,
        btc_w=1.75,
        alt_w=0.75,
        top_n=10,
        start_cap=initial_capital,
        benchmark_weights={'BTC': 1.0},
        benchmark_rebalance=False,
        detailed_output=False
    )
    
    print("=== STRATEGY DATA ===")
    print(f"Strategy DataFrame shape: {strategy_perf.shape}")
    if not strategy_perf.empty:
        print("Strategy equity values:")
        for i, row in strategy_perf.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            equity = row['Equity_USD']
            print(f"  {date_str}: ${equity:,.2f}")
        print(f"Strategy total return: {strategy_summary.get('total_return_pct', 0):+.2f}%")
    print()
    
    # Run benchmark calculation (100% BTC)
    benchmark_perf = calculate_benchmark_performance(
        df,
        benchmark_weights={'BTC': 1.0},
        start_cap=initial_capital,
        rebalance_weekly=False
    )
    
    print("=== BENCHMARK DATA ===")
    print(f"Benchmark DataFrame shape: {benchmark_perf.shape}")
    if not benchmark_perf.empty:
        print("Benchmark portfolio values:")
        for i, row in benchmark_perf.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            portfolio_value = row['Portfolio_Value']
            weekly_ret = row['Weekly_Return_Pct']
            print(f"  {date_str}: ${portfolio_value:,.2f} ({weekly_ret:+.2f}%)")
        
        # Calculate benchmark return manually
        first_value = benchmark_perf['Portfolio_Value'].iloc[0]
        last_value = benchmark_perf['Portfolio_Value'].iloc[-1]
        manual_return = ((last_value - initial_capital) / initial_capital) * 100
        
        print(f"First benchmark value: ${first_value:,.2f}")
        print(f"Last benchmark value: ${last_value:,.2f}")
        print(f"Manual benchmark return: {manual_return:+.2f}%")
        
        # Check if first value equals initial capital
        if abs(first_value - initial_capital) < 0.01:
            print("✅ Benchmark starts at initial capital")
        else:
            print(f"❌ Benchmark starts at ${first_value:,.2f} instead of ${initial_capital:,.2f}")
            print(f"   Difference: ${first_value - initial_capital:+,.2f}")
    
    print()
    
    # Test chart generation
    print("=== CHART GENERATION TEST ===")
    try:
        from benchmark_analyzer import compare_strategy_vs_benchmark
        
        comparison = compare_strategy_vs_benchmark(
            strategy_perf, benchmark_perf, strategy_summary, initial_capital
        )
        
        fig = plot_strategy_vs_benchmark(
            strategy_perf, benchmark_perf, strategy_summary, comparison,
            start_date, end_date, {'BTC': 1.0}
        )
        
        if fig:
            print("✅ Benchmark chart generation: Success")
            
            # Check normalization
            strategy_normalized = (strategy_perf["Equity_USD"] / strategy_perf["Equity_USD"].iloc[0]) * 100
            benchmark_normalized = (benchmark_perf["Portfolio_Value"] / benchmark_perf["Portfolio_Value"].iloc[0]) * 100
            
            print(f"Strategy normalization:")
            print(f"  First point: {strategy_normalized.iloc[0]:.1f}%")
            print(f"  Last point: {strategy_normalized.iloc[-1]:.1f}%")
            
            print(f"Benchmark normalization:")
            print(f"  First point: {benchmark_normalized.iloc[0]:.1f}%")
            print(f"  Last point: {benchmark_normalized.iloc[-1]:.1f}%")
            
            # Check if both start at 100%
            if abs(strategy_normalized.iloc[0] - 100.0) < 0.01 and abs(benchmark_normalized.iloc[0] - 100.0) < 0.01:
                print("✅ Both charts start at 100% baseline")
            else:
                print("❌ Charts don't start at 100% baseline")
                
        else:
            print("❌ Benchmark chart generation: Failed")
            
    except Exception as e:
        print(f"❌ Chart generation error: {e}")
    
    print("\n=== BENCHMARK CHART TEST COMPLETE ===")

if __name__ == "__main__":
    test_benchmark_chart()
