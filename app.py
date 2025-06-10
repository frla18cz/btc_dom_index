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

# Import the analyzer functionality directly from analyzer_weekly
from analyzer_weekly import (
    load_and_prepare,
    backtest_rank_altbtc_short,
    plot_equity_curve,
    plot_btc_vs_alts
)

# Import default configuration
from config.config import (
    EXCLUDED_SYMBOLS,
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_BTC_WEIGHT,
    BACKTEST_ALT_WEIGHT,
    BACKTEST_TOP_N_ALTS
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
        # Scrape new data
        new_df = scrape_historical(start_date=start_date)
        
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
                    progress_bar.progress(25, text="Scraping CoinMarketCap...")
                    
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
with col1:
    # Default start date to first Monday of 2025 if available, otherwise config default
    default_start_2025 = dt.date(2025, 1, 6)  # First Monday of 2025
    if available_start_date <= default_start_2025 <= available_end_date:
        default_start = default_start_2025
    else:
        default_start = max(BACKTEST_START_DATE.date(), available_start_date)
    
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        min_value=available_start_date,
        max_value=available_end_date,
        help=f"Available data starts from {available_start_date}"
    )
with col2:
    # Default end date to the last available data
    end_date = st.date_input(
        "End Date",
        value=available_end_date,  # Always use the last available date
        min_value=available_start_date,
        max_value=available_end_date,
        help=f"Available data ends on {available_end_date}"
    )

# Check if the selected date range is valid
if start_date >= end_date:
    st.sidebar.error("❌ End date must be after start date")

# Show selected period length
if start_date < end_date:
    period_days = (end_date - start_date).days
    period_weeks = period_days // 7
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

# Excluded tokens (advanced option)
show_advanced = st.sidebar.checkbox("Show Advanced Options")
excluded_tokens = EXCLUDED_SYMBOLS.copy()

if show_advanced:
    st.sidebar.subheader("Advanced Options")
    
    # Excluded tokens input
    excluded_tokens_input = st.sidebar.text_area(
        "Excluded Tokens (comma-separated)",
        value=", ".join(EXCLUDED_SYMBOLS),
        height=100,
    )
    excluded_tokens = [token.strip() for token in excluded_tokens_input.split(",") if token.strip()]

# Add a Run Backtest button
run_backtest = st.sidebar.button("Run Backtest", type="primary")

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
            st.success(f"✅ Loaded {len(df)} rows over {weeks_count} weeks ({actual_start} to {actual_end})")
            
            # Capture stdout to get the backtest output
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            
            # Run backtest
            st.info("Running backtest...")
            progress_bar.progress(50)
            
            perf, summary, detailed = backtest_rank_altbtc_short(
                df,
                btc_w=btc_weight,
                alt_w=alt_weight,
                top_n=top_n_alts,
                excluded=excluded_tokens,
                start_cap=initial_capital,
                detailed_output=True,
            )
            
            # Restore stdout
            sys.stdout = old_stdout
            progress_bar.progress(80)
            
            # Check if backtest was successful
            if summary:
                # Create tabs for different views
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
                
                with tab3:
                    st.header("Detailed Backtest Output")
                    st.text(new_stdout.getvalue())
                
                with tab4:
                    st.header("Raw Data")
                    
                    # Show performance data
                    st.subheader("Performance Data")
                    st.dataframe(perf)
                    
                    # Show detailed position data
                    if not detailed.empty:
                        st.subheader("Detailed Position Data")
                        st.dataframe(detailed)
                
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