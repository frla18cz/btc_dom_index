#!/usr/bin/env python3
"""
Debug script to understand what's happening with the chart display in Streamlit.
"""
import datetime as dt
import os
os.environ['MPLBACKEND'] = 'Agg'

from analyzer_weekly import load_and_prepare, backtest_rank_altbtc_short, plot_equity_curve

def debug_chart_data():
    """Debug chart data to understand the issue."""
    
    print("=== DEBUGGING CHART DISPLAY ISSUE ===")
    
    # Same parameters as your UI test
    start_date = dt.datetime(2025, 6, 2)
    end_date = dt.datetime(2025, 6, 30)
    initial_capital = 100000.0  # UI default
    btc_weight = 1.75  # UI default (175%)
    alt_weight = 0.75  # UI default (75%)
    
    print(f"Parameters:")
    print(f"  Initial Capital: ${initial_capital:,}")
    print(f"  BTC Weight: {btc_weight} ({btc_weight*100}%)")
    print(f"  ALT Weight: {alt_weight} ({alt_weight*100}%)")
    print(f"  Total Leverage: {btc_weight + alt_weight}x")
    print()
    
    # Load data
    df = load_and_prepare('top100_weekly_data.csv', start_date, end_date)
    
    # Run backtest
    perf_df, summary, detailed_df, benchmark_df, benchmark_comparison = backtest_rank_altbtc_short(
        df,
        btc_w=btc_weight,
        alt_w=alt_weight,
        top_n=10,
        start_cap=initial_capital,
        benchmark_weights={'BTC': 1.0},
        benchmark_rebalance_weekly=False,
        detailed_output=False
    )
    
    print("=== PERFORMANCE DATA ANALYSIS ===")
    print(f"Performance DataFrame shape: {perf_df.shape}")
    print(f"Columns: {list(perf_df.columns)}")
    print()
    
    if not perf_df.empty:
        print("Raw Equity Values:")
        for i, row in perf_df.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            equity = row['Equity_USD']
            weekly_ret = row['Weekly_Return_Pct']
            print(f"  {date_str}: ${equity:,.2f} ({weekly_ret:+.2f}%)")
        
        print()
        print("=== CHART DATA ANALYSIS ===")
        
        # What the chart should show
        first_equity = perf_df['Equity_USD'].iloc[0]
        last_equity = perf_df['Equity_USD'].iloc[-1]
        
        print(f"First Equity Value: ${first_equity:,.2f}")
        print(f"Last Equity Value: ${last_equity:,.2f}")
        print(f"Initial Capital (baseline): ${initial_capital:,.2f}")
        print()
        
        # Calculate what percentage change this represents visually
        if first_equity > 0:
            visual_change_from_first = ((last_equity - first_equity) / first_equity) * 100
            visual_change_from_baseline = ((last_equity - initial_capital) / initial_capital) * 100
            
            print(f"Visual change from first point: {visual_change_from_first:+.2f}%")
            print(f"Visual change from baseline: {visual_change_from_baseline:+.2f}%")
            print(f"Summary total return: {summary['total_return_pct']:+.2f}%")
            print()
        
        # Check if first equity is close to initial capital
        if abs(first_equity - initial_capital) / initial_capital > 0.05:  # More than 5% difference
            print("⚠️  WARNING: First equity point is NOT close to initial capital!")
            print(f"   This means the chart starts from ${first_equity:,.2f} instead of ${initial_capital:,.2f}")
            print(f"   Difference: ${first_equity - initial_capital:+,.2f} ({((first_equity - initial_capital) / initial_capital) * 100:+.2f}%)")
            print()
        
        # Analyze the trend
        if perf_df.shape[0] >= 2:
            equity_values = perf_df['Equity_USD'].values
            trend_up = equity_values[-1] > equity_values[0]
            
            print(f"Chart trend from first to last point: {'📈 UP' if trend_up else '📉 DOWN'}")
            print(f"Expected trend (positive return): 📈 UP")
            
            if not trend_up and summary['total_return_pct'] > 0:
                print("❌ MISMATCH: Positive return but chart shows downward trend!")
                print("   This indicates the chart baseline or data issue")
            elif trend_up and summary['total_return_pct'] > 0:
                print("✅ MATCH: Positive return and chart shows upward trend")
            print()
        
        # Analyze weekly progression
        print("=== WEEKLY PROGRESSION ANALYSIS ===")
        if perf_df.shape[0] >= 2:
            for i in range(len(perf_df)):
                row = perf_df.iloc[i]
                date_str = row['Date'].strftime('%Y-%m-%d')
                equity = row['Equity_USD']
                weekly_ret = row['Weekly_Return_Pct']
                
                if i == 0:
                    from_baseline = ((equity - initial_capital) / initial_capital) * 100
                    print(f"Week {i+1} ({date_str}): ${equity:,.2f} ({weekly_ret:+.2f}%) [From baseline: {from_baseline:+.2f}%]")
                else:
                    prev_equity = perf_df.iloc[i-1]['Equity_USD']
                    from_baseline = ((equity - initial_capital) / initial_capital) * 100
                    print(f"Week {i+1} ({date_str}): ${equity:,.2f} ({weekly_ret:+.2f}%) [From baseline: {from_baseline:+.2f}%]")
        
        print()
        print("=== SUMMARY VERIFICATION ===")
        manual_return = ((last_equity - initial_capital) / initial_capital) * 100
        print(f"Manual calculation: {manual_return:+.2f}%")
        print(f"Summary return: {summary['total_return_pct']:+.2f}%")
        print(f"Match: {'✅' if abs(manual_return - summary['total_return_pct']) < 0.01 else '❌'}")
        
        # Generate chart and check settings
        print()
        print("=== CHART GENERATION TEST ===")
        try:
            equity_fig = plot_equity_curve(perf_df, summary, start_date, end_date, initial_capital)
            if equity_fig:
                ax = equity_fig.get_axes()[0]
                ymin, ymax = ax.get_ylim()
                
                print(f"Chart Y-axis range: [${ymin:,.0f}, ${ymax:,.0f}]")
                print(f"Baseline position: ${initial_capital:,}")
                print(f"Baseline visible: {'✅' if ymin <= initial_capital <= ymax else '❌'}")
                
                # Check if equity curve goes up or down
                lines = ax.get_lines()
                if lines:
                    line = lines[0]  # First line should be equity curve
                    y_data = line.get_ydata()
                    if len(y_data) >= 2:
                        visual_trend = y_data[-1] > y_data[0]
                        print(f"Visual chart trend: {'📈 UP' if visual_trend else '📉 DOWN'}")
                        
                        if not visual_trend and summary['total_return_pct'] > 0:
                            print("❌ CHART ISSUE CONFIRMED: Positive return shows as downward trend")
                        else:
                            print("✅ Chart trend matches return direction")
                
            else:
                print("❌ Chart generation failed")
        except Exception as e:
            print(f"❌ Chart generation error: {e}")

if __name__ == "__main__":
    debug_chart_data()