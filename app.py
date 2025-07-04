#!/usr/bin/env python3
"""
Streamlit app for Bitcoin Dominance Index backtest analyzer.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
import sys
import io
import numpy as np
from tabulate import tabulate
import asyncio
import threading
import os

# Import the analyzer functionality directly from analyzer_weekly
from analyzer_weekly import (
    load_and_prepare,
    backtest_rank_altbtc_short,
    plot_equity_curve,
    plot_btc_vs_alts
)

# Import benchmark functionality
from benchmark_analyzer import (
    get_available_assets_from_data,
    validate_benchmark_weights,
    plot_strategy_vs_benchmark,
    plot_rolling_correlation
)

# Import default configuration
from config.config import (
    EXCLUDED_SYMBOLS,
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_BTC_WEIGHT,
    BACKTEST_ALT_WEIGHT,
    BACKTEST_TOP_N_ALTS,
    BENCHMARK_AVAILABLE_ASSETS,
    DEFAULT_BENCHMARK_WEIGHTS,
    BENCHMARK_REBALANCE_WEEKLY
)

# Import fetcher functions
from fetcher import get_last_date_from_csv, scrape_historical, mondays

def check_for_new_data(csv_file):
    """Check if new data is available since the last date in CSV."""
    last_date = get_last_date_from_csv(csv_file)
    if not last_date:
        return None, None
    
    # Calculate next Monday after last date
    next_monday = last_date + dt.timedelta(days=(7 - last_date.weekday()) % 7)
    if next_monday == last_date:  # if last_date is already Monday
        next_monday += dt.timedelta(days=7)
    
    today = dt.date.today()
    if next_monday <= today:
        # Check how many new snapshots would be available
        new_mondays = list(mondays(next_monday, today))
        return next_monday, len(new_mondays)
    
    return None, 0

def update_data_with_progress(csv_file, start_date):
    """Update data with progress tracking."""
    try:
        # Scrape new data with proper error handling for Playwright
        import subprocess
        import sys
        
        # Create a temporary script to run the fetcher
        temp_script = """
import sys
sys.path.insert(0, '.')
import subprocess
import os

# First, try to install Playwright browsers if needed
try:
    # Check if Firefox is available
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            # Test if Firefox is available
            browser = p.firefox.launch(headless=True)
            browser.close()
            print("Firefox browser is available")
        except Exception as e:
            print(f"Firefox not available: {{e}}")
            print("Attempting to install Playwright browsers and dependencies...")
            
            # First try to install system dependencies
            deps_result = subprocess.run([
                sys.executable, "-m", "playwright", "install-deps", "firefox"
            ], capture_output=True, text=True, timeout=300)
            
            if deps_result.returncode != 0:
                print(f"Warning: Could not install system dependencies: {{deps_result.stderr}}")
                print("Continuing with browser installation...")
            else:
                print("System dependencies installed successfully")
            
            # Try to install browsers
            install_result = subprocess.run([
                sys.executable, "-m", "playwright", "install", "firefox"
            ], capture_output=True, text=True, timeout=300)
            
            if install_result.returncode != 0:
                print(f"Failed to install Firefox: {{install_result.stderr}}")
                raise Exception("Cannot install Firefox browser")
            else:
                print("Firefox installed successfully")
except Exception as e:
    print(f"Error setting up Playwright: {{e}}")
    sys.exit(1)

# Now try to scrape the data
from fetcher import scrape_historical
import pandas as pd
import datetime as dt

start_date = dt.date({}, {}, {})
new_df = scrape_historical(start_date=start_date)
new_df.to_csv('temp_new_data.csv', index=False)
print(f"Downloaded {{len(new_df)}} rows")
""".format(start_date.year, start_date.month, start_date.day)
        
        # Write temp script
        with open('temp_fetcher.py', 'w') as f:
            f.write(temp_script)
        
        # Run the script in a subprocess (increased timeout for browser installation)
        result = subprocess.run([sys.executable, 'temp_fetcher.py'], 
                              capture_output=True, text=True, timeout=600)
        
        # Clean up temp script
        try:
            os.remove('temp_fetcher.py')
        except:
            pass
        
        if result.returncode == 0:
            # Load the scraped data
            if os.path.exists('temp_new_data.csv'):
                new_df = pd.read_csv('temp_new_data.csv')
                new_df['snapshot_date'] = pd.to_datetime(new_df['snapshot_date'])
                
                # Clean up temp data file
                try:
                    os.remove('temp_new_data.csv')
                except:
                    pass
                
                if not new_df.empty:
                    # Load existing data and append new data
                    existing_df = pd.read_csv(csv_file)
                    existing_df['snapshot_date'] = pd.to_datetime(existing_df['snapshot_date'])
                    
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df = combined_df.sort_values('snapshot_date').reset_index(drop=True)
                    
                    # Save combined data
                    combined_df.to_csv(csv_file, index=False)
                    return len(new_df), len(combined_df)
                
                return 0, 0
            else:
                st.error("Failed to create temporary data file")
                return None, None
        else:
            error_msg = result.stderr or result.stdout
            st.error(f"Error running fetcher: {error_msg}")
            
            # Show helpful message for common issues
            if any(phrase in error_msg for phrase in ["Executable doesn't exist", "playwright install", "missing dependencies", "Host system is missing"]):
                st.info("💡 **Alternative solution**: If you're running this locally, you can:")
                st.code("python fetcher.py", language="bash")
                st.write("This will download the missing data directly.")
                
                if "missing dependencies" in error_msg or "Host system is missing" in error_msg:
                    st.warning("⚠️ **System dependencies issue**: The hosting environment may not support browser automation. This is a limitation of the current hosting setup.")
            
            return None, None
        
    except subprocess.TimeoutExpired:
        st.error("Data fetching timed out after 10 minutes")
        return None, None
    except Exception as e:
        st.error(f"Error updating data: {e}")
        return None, None

# Configure page settings
st.set_page_config(
    page_title="BTC Dominance Index Backtest",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header and description
st.title("Bitcoin Dominance Index Backtest")
st.markdown("""
This app lets you backtest a strategy of going long on Bitcoin (BTC) while shorting a basket of altcoins.
The strategy implements weekly rebalancing to maintain target weights and supports leveraged positions.

### Strategy Logic
1. Go long on Bitcoin (BTC) with a configurable portion of equity (0% to 300%)
2. Go short on a basket of top altcoins with a configurable portion of equity (0% to 300%)
3. Rebalance weekly to maintain target weights
4. Track performance in USD terms

### Leverage Options
- **Standard (1.0x)**: BTC weight + ALT weight = 1.0 (no leverage, fully invested)
- **Leveraged (>1.0x)**: BTC weight + ALT weight > 1.0 (e.g., 1.5x BTC + 1.5x ALT = 3.0x leverage)
- **Partial (<1.0x)**: BTC weight + ALT weight < 1.0 (keeps portion in cash)

### Data Management
The app automatically checks for new weekly data and allows you to update it directly from the sidebar.
""")

# Sidebar configuration
st.sidebar.header("Strategy Configuration")

# Load data to determine available date range
csv_path = Path("top100_weekly_data.csv")
available_start_date = dt.date(2021, 1, 1)
available_end_date = dt.date.today()

if csv_path.exists():
    try:
        # Read just the date column to determine range
        df_dates = pd.read_csv(csv_path, usecols=['snapshot_date'])
        df_dates['snapshot_date'] = pd.to_datetime(df_dates['snapshot_date'])
        available_start_date = df_dates['snapshot_date'].min().date()
        available_end_date = df_dates['snapshot_date'].max().date()
        
        # Show available data range
        st.sidebar.success(f"📊 **Available Data Range:**\n{available_start_date} to {available_end_date}")
        st.sidebar.info(f"Total snapshots: {len(df_dates)}")
        
        # Check for new data availability
        next_date, new_count = check_for_new_data(csv_path)
        if new_count > 0:
            st.sidebar.warning(f"📥 **{new_count} new snapshots available** from {next_date}")
            
            # Add update button
            if st.sidebar.button("🔄 Update Data", type="secondary", help="Download missing data"):
                with st.sidebar:
                    st.info("Downloading new data...")
                    progress_bar = st.progress(0, text="Initializing...")
                    
                    # Update progress text
                    progress_bar.progress(25, text="Setting up browser...")
                    st.caption("This may take a few minutes on first run...")
                    
                    # Perform the update
                    new_rows, total_rows = update_data_with_progress(csv_path, next_date)
                    
                    if new_rows is not None:
                        progress_bar.progress(100, text="Complete!")
                        st.success(f"✅ Added {new_rows} new rows! Total: {total_rows}")
                        st.rerun()  # Refresh the app to show new data
                    else:
                        st.error("❌ Failed to update data")
        else:
            st.sidebar.info("✅ Data is up to date")
            
    except Exception as e:
        st.sidebar.warning(f"Could not read data file: {e}")

# Date range selector with actual data bounds
st.sidebar.subheader("📅 Backtest Period")
col1, col2 = st.sidebar.columns(2)

# Handle default values
default_start_2025 = dt.date(2025, 1, 6)  # First Monday of 2025
if available_start_date <= default_start_2025 <= available_end_date:
    config_default_start = default_start_2025
else:
    config_default_start = max(BACKTEST_START_DATE.date(), available_start_date)

with col1:
    # Use persisted values if they exist, otherwise use config defaults
    default_start = st.session_state.get('selected_start_date', config_default_start)
    
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        min_value=available_start_date,
        max_value=available_end_date,
        help=f"Available data starts from {available_start_date}",
        key="start_date_widget"
    )
    
    # Store the selected value
    st.session_state.selected_start_date = start_date

with col2:
    # Use persisted values if they exist, otherwise use config defaults
    default_end = st.session_state.get('selected_end_date', available_end_date)
    
    end_date = st.date_input(
        "End Date",
        value=default_end,
        min_value=available_start_date,
        max_value=available_end_date,
        help=f"Available data ends on {available_end_date}",
        key="end_date_widget"
    )
    
    # Store the selected value
    st.session_state.selected_end_date = end_date

# Add a Run Backtest button
run_backtest = st.sidebar.button("▶️ Run Backtest", type="primary", use_container_width=True)

# Quick date selection buttons
st.sidebar.write("**Quick Selection:**")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("📅 Latest Week", help="Set to most recent Monday → Monday period"):
        # Find the last available Monday data and set Monday to Monday period
        if csv_path.exists():
            try:
                # Read the data to find available dates
                temp_df = pd.read_csv(csv_path, usecols=['snapshot_date'])
                temp_df['snapshot_date'] = pd.to_datetime(temp_df['snapshot_date'])
                
                # Get all available dates sorted
                all_dates = temp_df['snapshot_date'].dt.date.unique()
                all_dates = sorted(all_dates, reverse=True)
                
                # Find the most recent Monday in the data
                end_monday = None
                for date in all_dates:
                    if date.weekday() == 0:  # Monday
                        end_monday = date
                        break
                
                if end_monday is not None:
                    # Find the previous Monday (7 days earlier)
                    start_monday = end_monday - dt.timedelta(days=7)
                    
                    # Make sure start_monday is within available data
                    if start_monday >= available_start_date:
                        st.session_state.selected_start_date = start_monday
                        st.session_state.selected_end_date = end_monday
                        st.rerun()
                    else:
                        st.sidebar.warning("Not enough historical data for Monday-to-Monday period")
                else:
                    st.sidebar.warning("No Monday data found in dataset")
            except Exception as e:
                st.sidebar.error(f"Error accessing data: {e}")
        else:
            st.sidebar.warning("No data file found")

with col2:
    if st.button("📊 Last Month", help="Set to last 4 weeks"):
        end_date_month = available_end_date
        start_date_month = end_date_month - dt.timedelta(days=28)  # 4 weeks
        
        # Make sure start date is within available range
        if start_date_month >= available_start_date:
            st.session_state.selected_start_date = start_date_month
            st.session_state.selected_end_date = end_date_month
            st.rerun()
        else:
            st.sidebar.warning("Not enough historical data for 4-week period")


# Check if the selected date range is valid
if start_date >= end_date:
    st.sidebar.error("❌ End date must be after start date")

# Show selected period length with actual snapshot count
if start_date < end_date:
    period_days = (end_date - start_date).days
    period_weeks = period_days // 7
    
    # If we have data, calculate actual snapshot count in the selected range
    if csv_path.exists():
        try:
            temp_df = pd.read_csv(csv_path, usecols=['snapshot_date'])
            temp_df['snapshot_date'] = pd.to_datetime(temp_df['snapshot_date'])
            
            # Filter for selected date range
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            filtered = temp_df[
                (temp_df['snapshot_date'] >= start_dt) & 
                (temp_df['snapshot_date'] <= end_dt)
            ]
            
            actual_snapshots = len(filtered['snapshot_date'].unique())
            if actual_snapshots > 0:
                st.sidebar.info(f"📊 Selected period: {period_days} days | {actual_snapshots} actual snapshots")
                if actual_snapshots == 1:
                    st.sidebar.warning("⚠️ Single snapshot selected - limited statistical analysis")
            else:
                st.sidebar.warning(f"⚠️ No data available for {start_date} to {end_date}")
        except Exception:
            st.sidebar.info(f"📊 Selected period: {period_days} days (~{period_weeks} weeks)")
    else:
        st.sidebar.info(f"📊 Selected period: {period_days} days (~{period_weeks} weeks)")

# Warning if no data file exists
if not csv_path.exists():
    st.sidebar.error("❌ **Data file not found!**")
    st.sidebar.info("Run `python fetcher.py` to download data first.")

# Initial capital input
initial_capital = st.sidebar.number_input(
    "Initial Capital (USD)",
    min_value=1000.0,
    max_value=10000000.0,
    value=float(BACKTEST_INITIAL_CAPITAL),
    step=1000.0,
    format="%.2f",
)

# Weight sliders
st.sidebar.subheader("Portfolio Allocation")
btc_weight = st.sidebar.slider(
    "BTC Long Weight",
    min_value=0.0,
    max_value=3.0,
    value=BACKTEST_BTC_WEIGHT,
    step=0.1,
    format="%.2f",
)
alt_weight = st.sidebar.slider(
    "ALT Short Weight",
    min_value=0.0,
    max_value=3.0,
    value=BACKTEST_ALT_WEIGHT,
    step=0.1,
    format="%.2f",
)

# Check if total leverage exceeds 3.0 (300%)
total_leverage = btc_weight + alt_weight
if total_leverage > 3.0:
    st.sidebar.warning(f"Warning: Total leverage ({total_leverage:.2f}) exceeds 3.0. This may result in excessive risk.")

# Show current leverage information
st.sidebar.info(f"Total leverage: {total_leverage:.2f}x")

# If weights don't sum to 1.0, inform the user this is intentional
if abs(total_leverage - 1.0) > 0.01:
    st.sidebar.info("Note: Weights don't sum to 1.0, which means you're using leverage or partial capital allocation.")

# Number of altcoins in the short basket
top_n_alts = st.sidebar.slider(
    "Number of ALTs in Short Basket",
    min_value=1,
    max_value=50,
    value=BACKTEST_TOP_N_ALTS,
    step=1,
)

# Benchmark Configuration
st.sidebar.subheader("📊 Benchmark Configuration")

# Load available assets for benchmark from data if possible
available_assets = BENCHMARK_AVAILABLE_ASSETS.copy()
if csv_path.exists():
    try:
        sample_df = pd.read_csv(csv_path, nrows=1000)  # Sample to get available assets
        df_prepared = load_and_prepare(csv_path)
        if not df_prepared.empty:
            available_assets = get_available_assets_from_data(df_prepared)
    except Exception:
        pass  # Use default assets if data loading fails

# Benchmark weights input
benchmark_weights = {}
use_benchmark = st.sidebar.checkbox("Enable Benchmark Comparison", value=True)

if use_benchmark:
    # Benchmark rebalancing strategy selection
    benchmark_rebalance = st.sidebar.radio(
        "**Benchmark Strategy:**",
        options=[False, True],
        format_func=lambda x: "🏠 Buy & Hold (weights drift)" if not x else "⚖️ Weekly Rebalanced (maintain weights)",
        index=0,  # Default to Buy & Hold
        help="Buy & Hold: Initial weights drift over time based on performance.\nWeekly Rebalanced: Weights are reset to targets each week."
    )
    
    st.sidebar.write("**Select Assets and Weights:**")
    
    # Quick preset buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("💰 100% BTC", help="Set 100% BTC (default)", type="primary"):
            st.session_state.benchmark_btc = 100.0
            for asset in available_assets[1:]:  # Reset others
                st.session_state[f"benchmark_{asset.lower()}"] = 0.0
    
    with col2:
        if st.button("⚖️ 50/50 BTC/ETH", help="Set 50% BTC, 50% ETH"):
            st.session_state.benchmark_btc = 50.0
            st.session_state.benchmark_eth = 50.0
            for asset in available_assets[2:]:  # Reset others
                st.session_state[f"benchmark_{asset.lower()}"] = 0.0
    
    # Individual asset weight sliders
    total_weight = 0.0
    for asset in available_assets[:10]:  # Limit to top 10 for UI
        key = f"benchmark_{asset.lower()}"
        default_val = DEFAULT_BENCHMARK_WEIGHTS.get(asset, 0.0) * 100
        
        weight = st.sidebar.slider(
            f"{asset} Weight (%)",
            min_value=0.0,
            max_value=100.0,
            value=default_val,
            step=5.0,
            key=key
        )
        
        if weight > 0:
            benchmark_weights[asset] = weight / 100.0
            total_weight += weight
    
    # Weight validation
    if total_weight > 0:
        if abs(total_weight - 100.0) > 0.1:
            if total_weight > 100.0:
                st.sidebar.error(f"❌ Total weight: {total_weight:.1f}% (exceeds 100%)")
            else:
                st.sidebar.warning(f"⚠️ Total weight: {total_weight:.1f}% (below 100%)")
                
                # Auto-normalize option
                if st.sidebar.button("📐 Auto-normalize to 100%"):
                    for asset in benchmark_weights:
                        key = f"benchmark_{asset.lower()}"
                        current_val = st.session_state.get(key, 0.0)
                        st.session_state[key] = (current_val / total_weight) * 100.0
                    st.rerun()
        else:
            st.sidebar.success(f"✅ Total weight: {total_weight:.1f}%")
            
        # Show benchmark composition
        if benchmark_weights:
            st.sidebar.write("**Benchmark Composition:**")
            for asset, weight in sorted(benchmark_weights.items(), key=lambda x: x[1], reverse=True):
                if weight > 0:
                    st.sidebar.write(f"• {asset}: {weight*100:.1f}%")

# Excluded tokens
show_excluded = st.sidebar.checkbox("Show Excluded Tokens")
excluded_tokens = EXCLUDED_SYMBOLS.copy()

if show_excluded:
    st.sidebar.subheader("Excluded Tokens")
    
    # Excluded tokens input
    excluded_tokens_input = st.sidebar.text_area(
        "Excluded Tokens (comma-separated)",
        value=", ".join(EXCLUDED_SYMBOLS),
        height=100,
    )
    excluded_tokens = [token.strip() for token in excluded_tokens_input.split(",") if token.strip()]


# Main area for displaying results
if run_backtest:
    # Show progress while loading and preparing data
    csv_path = Path("top100_weekly_data.csv")
    
    if not csv_path.exists():
        st.error(f"Error: Data file not found at {csv_path}")
        st.info("Please run fetcher.py to generate the data file first.")
    else:
        progress_bar = st.progress(0)
        st.info("Loading data...")
        
        # Convert date inputs to datetime
        start_dt = dt.datetime.combine(start_date, dt.time(0, 0))
        end_dt = dt.datetime.combine(end_date, dt.time(23, 59))
        
        # Load data
        df = load_and_prepare(csv_path, start_dt, end_dt)
        progress_bar.progress(30)
        
        if df.empty:
            st.error("❌ No data available for the selected date range.")
            st.warning(f"Selected range: {start_date} to {end_date}")
            st.info("Try selecting dates within the available data range shown in the sidebar.")
        else:
            actual_start = df['rebalance_ts'].min().date()
            actual_end = df['rebalance_ts'].max().date()
            weeks_count = df['rebalance_ts'].nunique()
            
            # Calculate the analysis period more accurately
            if weeks_count >= 2:
                analysis_periods = weeks_count - 1  # Number of week-to-week transitions analyzed
                st.success(f"✅ Loaded {len(df)} rows over {weeks_count} snapshots ({actual_start} to {actual_end})")
                st.info(f"📈 Analysis covers {analysis_periods} weekly transitions for performance calculation")
            else:
                st.success(f"✅ Loaded {len(df)} rows over {weeks_count} snapshots ({actual_start} to {actual_end})")
                st.warning("⚠️ Single snapshot - no performance analysis possible")
            
            # Capture stdout to get the backtest output
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            
            # Validate benchmark weights if benchmark is enabled
            benchmark_weights_final = None
            benchmark_rebalance_final = False  # Default value if benchmark is not used
            if use_benchmark and benchmark_weights:
                is_valid, error_msg = validate_benchmark_weights(benchmark_weights)
                if not is_valid:
                    st.error(f"❌ Benchmark configuration error: {error_msg}")
                    st.stop()
                else:
                    benchmark_weights_final = benchmark_weights
                    benchmark_rebalance_final = benchmark_rebalance
            
            # Run backtest
            st.info("Running backtest...")
            progress_bar.progress(50)
            
            perf, summary, detailed, benchmark_df, benchmark_comparison = backtest_rank_altbtc_short(
                df,
                btc_w=btc_weight,
                alt_w=alt_weight,
                top_n=top_n_alts,
                excluded=excluded_tokens,
                start_cap=initial_capital,
                detailed_output=True,
                benchmark_weights=benchmark_weights_final,
                benchmark_rebalance_weekly=benchmark_rebalance_final if use_benchmark else None,
            )
            
            # Restore stdout
            sys.stdout = old_stdout
            progress_bar.progress(80)
            
            # Check if backtest was successful
            if summary:
                # Create tabs for different views - add benchmark tab if benchmark is used
                if use_benchmark and benchmark_weights_final and not benchmark_df.empty:
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "Summary", "Performance Charts", "Benchmark Comparison", 
                        "Detailed Output", "Raw Data", "Benchmark Data"
                    ])
                else:
                    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Performance Charts", "Detailed Output", "Raw Data"])
                
                with tab1:
                    # Display summary in a nice format
                    st.header("Backtest Summary")
                    
                    # Display strategy parameters
                    st.subheader("Strategy Parameters")
                    params_cols = st.columns(3)
                    with params_cols[0]:
                        st.metric("Initial Capital", f"${initial_capital:,.2f}")
                    with params_cols[1]:
                        st.metric("BTC Weight", f"{btc_weight:.2%}")
                    with params_cols[2]:
                        st.metric("ALT Weight", f"{alt_weight:.2%}")
                    
                    st.metric("Number of ALTs in Basket", top_n_alts)
                    
                    # Display key metrics
                    st.subheader("Performance Metrics")
                    
                    # Create three columns
                    cols = st.columns(3)
                    
                    # First column - Returns
                    with cols[0]:
                        st.metric("Final Equity", f"${summary['final_equity']:,.2f}")
                        st.metric("Total Return", f"{summary['total_return_pct']:+.2f}%")
                        st.metric("Annualized Return", f"{summary['annualized_return']:+.2f}%")
                    
                    # Second column - Risk
                    with cols[1]:
                        st.metric("Maximum Drawdown", f"{summary['max_drawdown']:.2f}%")
                        st.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
                        st.metric("Sortino Ratio", f"{summary['sortino_ratio']:.2f}")
                    
                    # Third column - Other stats
                    with cols[2]:
                        st.metric("Win Rate", f"{summary['win_rate']:.1f}%")
                        st.metric("BTC P/L Contribution", f"${summary['cum_btc_pnl']:+,.2f}")
                        st.metric("ALT P/L Contribution", f"${summary['cum_alt_pnl']:+,.2f}")
                    
                    # Show benchmark comparison summary if available
                    if benchmark_comparison:
                        st.subheader("Benchmark Comparison")
                        bench_cols = st.columns(3)
                        
                        with bench_cols[0]:
                            st.metric(
                                "Alpha (Excess Return)",
                                f"{benchmark_comparison.get('alpha', 0):+.2f}%",
                                delta=f"{benchmark_comparison.get('strategy_vs_benchmark_total', 0):+.2f}% vs benchmark"
                            )
                        
                        with bench_cols[1]:
                            st.metric(
                                "Correlation",
                                f"{benchmark_comparison.get('correlation', 0):.3f}",
                                help="Correlation between strategy and benchmark returns (-1 to 1)"
                            )
                        
                        with bench_cols[2]:
                            benchmark_desc = " + ".join([f"{w*100:.0f}% {s}" for s, w in benchmark_weights_final.items()])
                            st.metric(
                                "Benchmark Return",
                                f"{benchmark_comparison.get('benchmark_total_return', 0):+.2f}%",
                                delta=benchmark_desc
                            )
                
                with tab2:
                    st.header("Performance Charts")
                    
                    # Equity curve
                    st.subheader("Equity Curve")
                    equity_fig = plot_equity_curve(perf, summary, start_dt, end_dt)
                    if equity_fig:
                        st.pyplot(equity_fig)
                    
                    # P/L contribution chart
                    st.subheader("BTC vs ALT Contribution")
                    contrib_fig = plot_btc_vs_alts(perf)
                    if contrib_fig:
                        st.pyplot(contrib_fig)
                
                # Add benchmark comparison tab if benchmark is enabled
                if use_benchmark and benchmark_weights_final and not benchmark_df.empty:
                    with tab3:
                        st.header("Benchmark Comparison")
                        
                        # Strategy vs Benchmark chart
                        st.subheader("Strategy vs Benchmark Performance")
                        comparison_fig = plot_strategy_vs_benchmark(
                            perf, benchmark_df, summary, benchmark_comparison, 
                            start_dt, end_dt, benchmark_weights_final
                        )
                        if comparison_fig:
                            st.pyplot(comparison_fig)
                        
                        # Rolling correlation chart
                        st.subheader("Rolling Correlation (12-Week Window)")
                        correlation_fig = plot_rolling_correlation(perf, benchmark_df, window=12)
                        if correlation_fig:
                            st.pyplot(correlation_fig)
                        
                        # Detailed comparison metrics
                        st.subheader("Detailed Comparison Metrics")
                        
                        # Determine if we have meaningful correlation and Sharpe data
                        num_weeks = len(perf) if not perf.empty else 0
                        has_correlation = num_weeks > 1
                        has_sharpe = num_weeks > 1
                        
                        # Format correlation display
                        if has_correlation:
                            correlation_str = f"{benchmark_comparison.get('correlation', 0):.3f}"
                        else:
                            correlation_str = "N/A (insufficient data)"
                        
                        # Format Sharpe ratio display  
                        strategy_sharpe = f"{summary['sharpe_ratio']:.2f}" if has_sharpe else "N/A"
                        benchmark_sharpe = f"{benchmark_comparison.get('benchmark_sharpe_ratio', 0):.2f}" if has_sharpe else "N/A"
                        sharpe_diff = f"{benchmark_comparison.get('strategy_vs_benchmark_sharpe', 0):+.2f}" if has_sharpe else "N/A"
                        
                        comparison_data = [
                            ["Metric", "Strategy", "Benchmark", "Difference"],
                            ["Total Return", f"{summary['total_return_pct']:+.2f}%", 
                             f"{benchmark_comparison.get('benchmark_total_return', 0):+.2f}%",
                             f"{benchmark_comparison.get('strategy_vs_benchmark_total', 0):+.2f}%"],
                            ["Annualized Return", f"{summary['annualized_return']:+.2f}%",
                             f"{benchmark_comparison.get('benchmark_annualized_return', 0):+.2f}%",
                             f"{benchmark_comparison.get('alpha', 0):+.2f}%"],
                            ["Max Drawdown", f"{summary['max_drawdown']:.2f}%",
                             f"{benchmark_comparison.get('benchmark_max_drawdown', 0):.2f}%",
                             f"{benchmark_comparison.get('strategy_vs_benchmark_drawdown', 0):+.2f}%"],
                            ["Sharpe Ratio", strategy_sharpe, benchmark_sharpe, sharpe_diff],
                            ["Correlation", "—", "—", correlation_str]
                        ]
                        
                        # Display as a clean table
                        comparison_df = pd.DataFrame(comparison_data[1:], columns=comparison_data[0])
                        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                        
                        # Add explanatory note for single-week data
                        if num_weeks == 1:
                            st.info("⚠️ **Note**: Single-week backtest data. Sharpe ratio and correlation require more data points for meaningful calculation.")
                        elif num_weeks < 4:
                            st.info("⚠️ **Note**: Limited data (<4 weeks). Statistical measures may not be reliable.")
                
                # Adjust tab numbers based on whether benchmark is enabled
                detailed_tab = tab4 if not (use_benchmark and benchmark_weights_final and not benchmark_df.empty) else tab4
                raw_data_tab = tab4 if not (use_benchmark and benchmark_weights_final and not benchmark_df.empty) else tab5
                
                with detailed_tab:
                    st.header("Detailed Backtest Output")
                    st.text(new_stdout.getvalue())
                
                with raw_data_tab:
                    st.header("Raw Data & Analytics")
                    
                    # Create sub-tabs for different types of raw data
                    raw_tab1, raw_tab2, raw_tab3, raw_tab4 = st.tabs([
                        "Strategy Performance", "Position Details", "Weekly Analysis", "Data Quality"
                    ])
                    
                    with raw_tab1:
                        st.subheader("Strategy Performance Data")
                        
                        # Enhanced performance metrics
                        if not perf.empty:
                            # Show key summary first
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Snapshots", len(perf) + 1)  # +1 for initial
                            with col2:
                                analysis_periods = len(perf) if not perf.empty else 0
                                st.metric("Analysis Periods", analysis_periods)
                            with col3:
                                if "Weekly_Return_Pct" in perf.columns:
                                    avg_weekly = perf["Weekly_Return_Pct"].mean()
                                    st.metric("Avg Weekly Return", f"{avg_weekly:.2f}%")
                            with col4:
                                if "Weekly_Return_Pct" in perf.columns:
                                    volatility = perf["Weekly_Return_Pct"].std()
                                    st.metric("Weekly Volatility", f"{volatility:.2f}%")
                            
                            # Performance data table with enhanced display options
                            st.write("**Performance Data Table:**")
                            
                            # Add data filtering options
                            show_all_cols = st.checkbox("Show all columns", value=False)
                            
                            if show_all_cols:
                                display_df = perf
                            else:
                                # Show key columns by default
                                key_cols = ["Date", "Equity_USD", "BTC_Price_USD", "Weekly_Return_Pct", 
                                           "Weekly_BTC_PNL_USD", "Weekly_ALT_PNL_USD"]
                                available_cols = [col for col in key_cols if col in perf.columns]
                                display_df = perf[available_cols]
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Download button for performance data
                            csv_perf = perf.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Performance CSV",
                                data=csv_perf,
                                file_name=f"strategy_performance_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    with raw_tab2:
                        st.subheader("Position Details")
                        
                        if not detailed.empty:
                            # Position analysis
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                total_positions = len(detailed)
                                st.metric("Total Position Records", total_positions)
                            with col2:
                                unique_assets = detailed["Symbol"].nunique() if "Symbol" in detailed.columns else 0
                                st.metric("Unique Assets", unique_assets)
                            with col3:
                                if "Type" in detailed.columns:
                                    position_types = detailed["Type"].value_counts().to_dict()
                                    types_str = ", ".join([f"{k}: {v}" for k, v in position_types.items()])
                                    st.metric("Position Types", len(position_types))
                                    st.caption(types_str)
                            
                            # Position filtering
                            if "Symbol" in detailed.columns:
                                selected_symbols = st.multiselect(
                                    "Filter by Asset",
                                    options=sorted(detailed["Symbol"].unique()),
                                    default=[]
                                )
                                
                                if selected_symbols:
                                    filtered_detailed = detailed[detailed["Symbol"].isin(selected_symbols)]
                                else:
                                    filtered_detailed = detailed
                            else:
                                filtered_detailed = detailed
                            
                            st.dataframe(filtered_detailed, use_container_width=True)
                            
                            # Position summary by asset
                            if "Symbol" in detailed.columns and "PnL_USD" in detailed.columns:
                                st.write("**Position Summary by Asset:**")
                                position_summary = detailed.groupby(["Symbol", "Type"]).agg({
                                    "PnL_USD": ["sum", "mean", "count"],
                                    "Value_USD": ["mean", "max"] if "Value_USD" in detailed.columns else ["count"]
                                }).round(2)
                                
                                # Flatten column names
                                position_summary.columns = ['_'.join(col).strip() for col in position_summary.columns.values]
                                position_summary = position_summary.reset_index()
                                
                                st.dataframe(position_summary, use_container_width=True)
                            
                            # Download button for position data
                            csv_detailed = detailed.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Position Details CSV",
                                data=csv_detailed,
                                file_name=f"position_details_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No detailed position data available.")
                    
                    with raw_tab3:
                        st.subheader("Weekly Analysis")
                        
                        if not perf.empty and "Weekly_Return_Pct" in perf.columns:
                            # Weekly return statistics
                            returns = perf["Weekly_Return_Pct"].dropna()
                            
                            if len(returns) > 0:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Return Statistics:**")
                                    stats_data = {
                                        "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Skewness", "Kurtosis"],
                                        "Value": [
                                            f"{returns.mean():.2f}%",
                                            f"{returns.median():.2f}%",
                                            f"{returns.std():.2f}%",
                                            f"{returns.min():.2f}%",
                                            f"{returns.max():.2f}%",
                                            f"{returns.skew():.2f}",
                                            f"{returns.kurtosis():.2f}"
                                        ]
                                    }
                                    st.dataframe(pd.DataFrame(stats_data), hide_index=True)
                                
                                with col2:
                                    st.write("**Performance Buckets:**")
                                    positive_weeks = (returns > 0).sum()
                                    negative_weeks = (returns < 0).sum()
                                    flat_weeks = (returns == 0).sum()
                                    
                                    performance_buckets = pd.DataFrame({
                                        "Outcome": ["Positive", "Negative", "Flat"],
                                        "Weeks": [positive_weeks, negative_weeks, flat_weeks],
                                        "Percentage": [
                                            f"{positive_weeks/len(returns)*100:.1f}%",
                                            f"{negative_weeks/len(returns)*100:.1f}%",
                                            f"{flat_weeks/len(returns)*100:.1f}%"
                                        ]
                                    })
                                    st.dataframe(performance_buckets, hide_index=True)
                                
                                # Return distribution
                                st.write("**Weekly Return Distribution:**")
                                
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(10, 4))
                                
                                # Histogram
                                ax.hist(returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
                                ax.axvline(returns.mean(), color='red', linestyle='--', 
                                          label=f'Mean: {returns.mean():.2f}%')
                                ax.axvline(returns.median(), color='green', linestyle='--', 
                                          label=f'Median: {returns.median():.2f}%')
                                
                                ax.set_xlabel('Weekly Return (%)')
                                ax.set_ylabel('Frequency')
                                ax.set_title('Distribution of Weekly Returns')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                        else:
                            st.info("No weekly return data available for analysis.")
                    
                    with raw_tab4:
                        st.subheader("Data Quality & Validation")
                        
                        # Data quality checks
                        st.write("**Data Quality Report:**")
                        
                        quality_checks = []
                        
                        # Performance data checks
                        if not perf.empty:
                            quality_checks.append(["Performance Data", "✅ Available", f"{len(perf)} records"])
                            
                            # Check for missing values
                            missing_counts = perf.isnull().sum()
                            critical_cols = ["Date", "Equity_USD", "BTC_Price_USD"]
                            missing_critical = missing_counts[missing_counts.index.intersection(critical_cols)]
                            
                            if missing_critical.sum() > 0:
                                quality_checks.append(["Missing Critical Data", "⚠️ Issues Found", f"{missing_critical.sum()} missing values"])
                            else:
                                quality_checks.append(["Missing Critical Data", "✅ Clean", "No missing critical values"])
                            
                            # Check for date continuity
                            if "Date" in perf.columns:
                                dates = pd.to_datetime(perf["Date"]).sort_values()
                                date_gaps = dates.diff().dt.days
                                large_gaps = (date_gaps > 14).sum()  # More than 2 weeks
                                
                                if large_gaps > 0:
                                    quality_checks.append(["Date Continuity", "⚠️ Gaps Found", f"{large_gaps} gaps > 2 weeks"])
                                else:
                                    quality_checks.append(["Date Continuity", "✅ Continuous", "No significant gaps"])
                        else:
                            quality_checks.append(["Performance Data", "❌ Missing", "No performance data"])
                        
                        # Position data checks
                        if not detailed.empty:
                            quality_checks.append(["Position Data", "✅ Available", f"{len(detailed)} records"])
                            
                            # Check for zero or negative values where they shouldn't be
                            if "Value_USD" in detailed.columns:
                                negative_values = (detailed["Value_USD"] < 0).sum()
                                zero_values = (detailed["Value_USD"] == 0).sum()
                                
                                if negative_values > 0:
                                    quality_checks.append(["Position Values", "⚠️ Issues", f"{negative_values} negative values"])
                                elif zero_values > 0:
                                    quality_checks.append(["Position Values", "⚠️ Zero Values", f"{zero_values} zero values"])
                                else:
                                    quality_checks.append(["Position Values", "✅ Valid", "All positive values"])
                        else:
                            quality_checks.append(["Position Data", "❌ Missing", "No position data"])
                        
                        # Period alignment check
                        if not perf.empty:
                            snapshots = len(perf) + 1  # +1 for initial snapshot
                            analysis_periods = len(perf)
                            expected_periods = snapshots - 1
                            
                            if analysis_periods == expected_periods:
                                quality_checks.append(["Period Alignment", "✅ Correct", f"{analysis_periods} analysis periods from {snapshots} snapshots"])
                            else:
                                quality_checks.append(["Period Alignment", "⚠️ Mismatch", f"{analysis_periods} periods vs {expected_periods} expected"])
                        
                        # Display quality report
                        quality_df = pd.DataFrame(quality_checks, columns=["Check", "Status", "Details"])
                        st.dataframe(quality_df, hide_index=True, use_container_width=True)
                        
                        # Raw data configuration summary
                        st.write("**Configuration Summary:**")
                        config_data = [
                            ["Initial Capital", f"${initial_capital:,.2f}"],
                            ["BTC Weight", f"{btc_weight:.1%}"],
                            ["ALT Weight", f"{alt_weight:.1%}"],
                            ["Total Leverage", f"{btc_weight + alt_weight:.2f}x"],
                            ["ALT Basket Size", f"{top_n_alts} assets"],
                            ["Date Range", f"{start_date} to {end_date}"],
                            ["Excluded Tokens", f"{len(excluded_tokens)} excluded"]
                        ]
                        config_df = pd.DataFrame(config_data, columns=["Parameter", "Value"])
                        st.dataframe(config_df, hide_index=True, use_container_width=True)
                
                # Add benchmark data tab if benchmark is enabled
                if use_benchmark and benchmark_weights_final and not benchmark_df.empty:
                    with tab6:
                        st.header("Benchmark Data & Analysis")
                        
                        # Create benchmark sub-tabs
                        bench_tab1, bench_tab2, bench_tab3 = st.tabs([
                            "Portfolio Composition", "Performance Data", "Asset Breakdown"
                        ])
                        
                        with bench_tab1:
                            st.subheader("Benchmark Portfolio Composition")
                            
                            # Show composition as metrics
                            st.write("**Asset Allocation:**")
                            benchmark_cols = st.columns(min(len(benchmark_weights_final), 4))
                            
                            for i, (asset, weight) in enumerate(sorted(benchmark_weights_final.items(), key=lambda x: x[1], reverse=True)):
                                col_idx = i % len(benchmark_cols)
                                with benchmark_cols[col_idx]:
                                    st.metric(asset, f"{weight*100:.1f}%")
                            
                            # Total allocation check
                            total_weight = sum(benchmark_weights_final.values())
                            if abs(total_weight - 1.0) > 0.001:
                                st.warning(f"⚠️ Total allocation: {total_weight*100:.1f}% (should be 100%)")
                            else:
                                st.success(f"✅ Total allocation: {total_weight*100:.1f}%")
                            
                            # Benchmark summary metrics
                            if benchmark_comparison:
                                st.write("**Benchmark Performance Summary:**")
                                bench_summary_cols = st.columns(3)
                                
                                with bench_summary_cols[0]:
                                    st.metric("Total Return", f"{benchmark_comparison.get('benchmark_total_return', 0):+.2f}%")
                                with bench_summary_cols[1]:
                                    st.metric("Annualized Return", f"{benchmark_comparison.get('benchmark_annualized_return', 0):+.2f}%")
                                with bench_summary_cols[2]:
                                    st.metric("Max Drawdown", f"{benchmark_comparison.get('benchmark_max_drawdown', 0):.2f}%")
                        
                        with bench_tab2:
                            st.subheader("Benchmark Performance Data")
                            
                            # Performance overview
                            if not benchmark_df.empty:
                                bench_periods = len(benchmark_df)
                                analysis_periods_bench = bench_periods - 1 if bench_periods > 1 else 0
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Snapshots", bench_periods)
                                with col2:
                                    st.metric("Analysis Periods", analysis_periods_bench)
                                with col3:
                                    if "Weekly_Return_Pct" in benchmark_df.columns:
                                        avg_weekly_bench = benchmark_df["Weekly_Return_Pct"].mean()
                                        st.metric("Avg Weekly Return", f"{avg_weekly_bench:.2f}%")
                                
                                # Show full benchmark data with filtering options
                                show_all_bench_cols = st.checkbox("Show all benchmark columns", value=False)
                                
                                if show_all_bench_cols:
                                    display_benchmark_df = benchmark_df
                                else:
                                    # Show key columns
                                    key_bench_cols = ["Date", "Portfolio_Value", "Weekly_Return_Pct", "Period_Number"]
                                    available_bench_cols = [col for col in key_bench_cols if col in benchmark_df.columns]
                                    
                                    # Add asset value columns
                                    asset_value_cols = [col for col in benchmark_df.columns if col.endswith('_Value')]
                                    available_bench_cols.extend(asset_value_cols[:5])  # Show up to 5 asset values
                                    
                                    # Remove duplicates while preserving order
                                    unique_bench_cols = []
                                    for col in available_bench_cols:
                                        if col not in unique_bench_cols:
                                            unique_bench_cols.append(col)
                                    
                                    display_benchmark_df = benchmark_df[unique_bench_cols]
                                
                                st.dataframe(display_benchmark_df, use_container_width=True)
                                
                                # Download benchmark data
                                csv_benchmark = benchmark_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Benchmark CSV",
                                    data=csv_benchmark,
                                    file_name=f"benchmark_data_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        
                        with bench_tab3:
                            st.subheader("Asset-Level Breakdown")
                            
                            # Generate detailed asset breakdown
                            try:
                                from benchmark_analyzer import get_benchmark_breakdown
                                
                                breakdown_df = get_benchmark_breakdown(benchmark_df, benchmark_weights_final)
                                
                                if not breakdown_df.empty:
                                    st.write("**Individual Asset Performance:**")
                                    
                                    # Display breakdown table
                                    st.dataframe(breakdown_df, use_container_width=True)
                                    
                                    # Asset performance chart
                                    if "Total_Return_Pct" in breakdown_df.columns:
                                        st.write("**Asset Return Comparison:**")
                                        
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        
                                        assets = breakdown_df["Asset"]
                                        returns = breakdown_df["Total_Return_Pct"]
                                        
                                        colors = ['green' if r >= 0 else 'red' for r in returns]
                                        bars = ax.bar(assets, returns, color=colors, alpha=0.7)
                                        
                                        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
                                        ax.set_ylabel('Total Return (%)')
                                        ax.set_title('Asset Performance in Benchmark Portfolio')
                                        ax.tick_params(axis='x', rotation=45)
                                        
                                        # Add value labels on bars
                                        for bar, return_val in zip(bars, returns):
                                            height = bar.get_height()
                                            ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                                                   f'{return_val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                    
                                    # Top/bottom performers
                                    if "Total_Return_Pct" in breakdown_df.columns and len(breakdown_df) > 1:
                                        sorted_breakdown = breakdown_df.sort_values("Total_Return_Pct", ascending=False)
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("**🏆 Best Performer:**")
                                            best = sorted_breakdown.iloc[0]
                                            st.write(f"• **{best['Asset']}**: {best['Total_Return_Pct']:+.2f}%")
                                            st.write(f"• Contribution: ${best['Contribution_To_Portfolio']:+,.2f}")
                                            st.write(f"• Weight: {best['Target_Weight_Pct']:.1f}%")
                                        
                                        with col2:
                                            st.write("**📉 Worst Performer:**")
                                            worst = sorted_breakdown.iloc[-1]
                                            st.write(f"• **{worst['Asset']}**: {worst['Total_Return_Pct']:+.2f}%")
                                            st.write(f"• Contribution: ${worst['Contribution_To_Portfolio']:+,.2f}")
                                            st.write(f"• Weight: {worst['Target_Weight_Pct']:.1f}%")
                                    
                                    # Download breakdown data
                                    csv_breakdown = breakdown_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Asset Breakdown CSV",
                                        data=csv_breakdown,
                                        file_name=f"benchmark_breakdown_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.info("No asset breakdown data available.")
                            
                            except ImportError:
                                st.error("Asset breakdown functionality not available.")
                            except Exception as e:
                                st.error(f"Error generating asset breakdown: {e}")
                
                progress_bar.progress(100)
                st.success("Backtest completed successfully!")
            else:
                progress_bar.progress(100)
                st.error("Backtest did not produce a summary. Check the detailed output for errors.")
                st.text(new_stdout.getvalue())
else:
    # When the app is first loaded, show instructions
    st.info("Configure your backtest parameters in the sidebar and click 'Run Backtest' to start.")
    
    # Display explanation of the strategy
    st.header("About the Strategy")
    st.markdown("""
    ### Bitcoin Dominance Index Strategy
    
    This strategy is based on the principle that during certain market phases,
    Bitcoin tends to outperform altcoins, and vice versa. By going long on Bitcoin 
    and short on a basket of top altcoins, the strategy aims to:
    
    1. **Capture Bitcoin's relative strength** against altcoins
    2. **Reduce overall volatility** through the hedged approach
    3. **Provide potential returns** in both bull and bear markets
    
    ### Key Parameters
    
    - **BTC Long Weight**: Percentage of portfolio allocated to Bitcoin long position (0-300%)
    - **ALT Short Weight**: Percentage of portfolio allocated to shorting altcoins (0-300%)
    - **Number of ALTs in Short Basket**: How many top altcoins to include in the short basket
    - **Excluded Tokens**: Tokens that should never be included in the short basket (e.g., stablecoins)
    
    ### How It Works
    
    1. The strategy simulates futures perpetual positions for both BTC long and ALT shorts
    2. You can configure leverage from 0% to 300% for each leg independently
    3. Every week, positions are rebalanced to maintain target allocations
    4. Performance is tracked in USD terms with mark-to-market valuations
    5. The backtest shows detailed P/L for both the BTC long and ALT short components
    
    ### Leverage Examples
    
    - **Classic 1:1 Hedge**: 50% BTC long + 50% ALT short (total leverage = 1.0x)
    - **High Conviction BTC**: 100% BTC long + 50% ALT short (total leverage = 1.5x)
    - **High Conviction ALT Short**: 50% BTC long + 100% ALT short (total leverage = 1.5x)
    - **Maximum Leverage**: 150% BTC long + 150% ALT short (total leverage = 3.0x)
    - **Cash Reserve**: 30% BTC long + 30% ALT short (total leverage = 0.6x, 40% cash)
    """)

# Footer with additional information
st.sidebar.markdown("---")
st.sidebar.info(
    "**Note**: This app uses historical data from CoinMarketCap "
    "to simulate the strategy's performance. Past performance is not indicative of future results."
)