#!/usr/bin/env python3
"""
Test to verify chart baseline fix.
"""
import datetime as dt
import os
os.environ['MPLBACKEND'] = 'Agg'

from analyzer_weekly import load_and_prepare, backtest_rank_altbtc_short, plot_equity_curve

def test_chart_baseline():
    """Test chart baseline with different initial capital values."""
    
    # Test data for June 2025
    start_date = dt.datetime(2025, 6, 2)
    end_date = dt.datetime(2025, 6, 30)
    
    print("=== TESTING CHART BASELINE FIX ===")
    
    df = load_and_prepare('top100_weekly_data.csv', start_date, end_date)
    
    # Test with different initial capital values
    test_capitals = [50000, 100000, 200000]
    
    for initial_capital in test_capitals:
        print(f"\n--- Testing with Initial Capital: ${initial_capital:,} ---")
        
        # Run backtest
        perf_df, summary, detailed_df, benchmark_df, benchmark_comparison = backtest_rank_altbtc_short(
            df,
            btc_w=1.75,  # Use aggressive settings from UI
            alt_w=0.75,
            top_n=10,
            start_cap=initial_capital,
            benchmark_weights={'BTC': 1.0},
            benchmark_rebalance=False,
            detailed_output=False
        )
        
        print(f"Initial Capital Used: ${initial_capital:,}")
        print(f"Final Equity: ${summary['final_equity']:,.2f}")
        print(f"Total Return: {summary['total_return_pct']:+.2f}%")
        
        # Calculate expected final equity
        expected_final = initial_capital * (1 + summary['total_return_pct'] / 100)
        print(f"Expected Final (calc): ${expected_final:,.2f}")
        print(f"Match: {'✅' if abs(expected_final - summary['final_equity']) < 0.01 else '❌'}")
        
        # Test chart generation (no display, just check it doesn't error)
        try:
            equity_fig = plot_equity_curve(perf_df, summary, start_date, end_date, initial_capital)
            if equity_fig:
                print("Chart Generation: ✅ Success")
                
                # Get chart y-axis limits to verify baseline
                ax = equity_fig.get_axes()[0]
                ymin, ymax = ax.get_ylim()
                
                # Check if baseline is within chart range
                baseline_visible = ymin <= initial_capital <= ymax
                print(f"Baseline ${initial_capital:,} in range [${ymin:,.0f}, ${ymax:,.0f}]: {'✅' if baseline_visible else '❌'}")
                
                # Check if first equity point is reasonable
                first_equity = perf_df['Equity_USD'].iloc[0] if not perf_df.empty else 0
                baseline_close = abs(first_equity - initial_capital) / initial_capital < 0.1  # Within 10%
                print(f"First equity ${first_equity:,.0f} close to baseline: {'✅' if baseline_close else '❌'}")
                
            else:
                print("Chart Generation: ❌ Failed")
        except Exception as e:
            print(f"Chart Generation: ❌ Error - {e}")
    
    print(f"\n=== CHART BASELINE TEST COMPLETE ===")

if __name__ == "__main__":
    test_chart_baseline()
