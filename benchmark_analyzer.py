#!/usr/bin/env python3
"""
Benchmark analyzer for comparing strategy performance against custom benchmark portfolios.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import datetime as dt
from matplotlib.ticker import ScalarFormatter, PercentFormatter

from config.config import BENCHMARK_AVAILABLE_ASSETS, DEFAULT_BENCHMARK_WEIGHTS


def validate_benchmark_weights(weights: Dict[str, float]) -> Tuple[bool, str]:
    """
    Validate that benchmark weights sum to 100% (1.0).
    
    Args:
        weights: Dictionary mapping asset symbols to their weights (0-1)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not weights:
        return False, "No benchmark weights provided"
    
    total_weight = sum(weights.values())
    
    # Check for negative weights
    if any(w < 0 for w in weights.values()):
        return False, "Weights cannot be negative"
    
    # Check if weights sum to approximately 1.0 (allow small floating point errors)
    if abs(total_weight - 1.0) > 0.001:
        return False, f"Weights must sum to 100%. Current total: {total_weight:.1%}"
    
    return True, ""


def calculate_benchmark_performance(df: pd.DataFrame, benchmark_weights: Dict[str, float], 
                                  start_cap: float = 100000.0, rebalance_weekly: bool = False) -> pd.DataFrame:
    """
    Calculate benchmark portfolio performance using buy-and-hold or rebalanced strategy.
    Aligns with strategy calculation method for consistent period counting.
    
    Args:
        df: DataFrame with prepared cryptocurrency data
        benchmark_weights: Dictionary mapping asset symbols to their weights
        start_cap: Initial capital in USD
        rebalance_weekly: If True, rebalance to target weights each week; if False, buy-and-hold
        
    Returns:
        DataFrame with benchmark performance data including detailed asset breakdowns
    """
    # Validate weights
    is_valid, error_msg = validate_benchmark_weights(benchmark_weights)
    if not is_valid:
        raise ValueError(f"Invalid benchmark weights: {error_msg}")
    
    # Get available weeks
    weeks = sorted(df["rebalance_ts"].unique())
    if len(weeks) < 1:
        raise ValueError("Not enough weeks for benchmark calculation")
    
    # Initialize tracking with detailed asset breakdown
    benchmark_rows = []
    portfolio_value = start_cap
    positions = {}  # symbol -> {qty, initial_price, current_price, value, weight}
    
    # Initialize positions at the first week - align with strategy methodology
    first_week = weeks[0]
    week_data = df[df["rebalance_ts"] == first_week].set_index("sym")
    
    benchmark_type = "Weekly Rebalanced" if rebalance_weekly else "Buy & Hold"
    print(f"\n=== BENCHMARK PORTFOLIO INITIALIZATION ({benchmark_type}) ===")
    print(f"Initial Capital: ${start_cap:,.2f}")
    print(f"Portfolio Composition:")
    
    total_allocation = 0.0
    initialization_errors = []
    
    for symbol, weight in benchmark_weights.items():
        if symbol in week_data.index:
            price = week_data.loc[symbol, "price_usd"]
            if pd.notna(price) and price > 0:
                allocation = start_cap * weight
                qty = allocation / price
                positions[symbol] = {
                    "qty": qty,
                    "initial_price": price,
                    "current_price": price,
                    "value": allocation,
                    "weight": weight,
                    "rank": week_data.loc[symbol, "rank"] if "rank" in week_data.columns else 0
                }
                total_allocation += allocation
                print(f"  • {symbol}: {weight*100:.1f}% = ${allocation:,.2f} ({qty:.6f} coins @ ${price:.2f})")
            else:
                error_msg = f"Invalid price for {symbol}: {price}"
                initialization_errors.append(error_msg)
                print(f"  ⚠️  {symbol}: {weight*100:.1f}% - {error_msg}")
        else:
            error_msg = f"{symbol} not found in data"
            initialization_errors.append(error_msg)
            print(f"  ⚠️  {symbol}: {weight*100:.1f}% - {error_msg}")
    
    print(f"\nTotal Allocated: ${total_allocation:,.2f} ({total_allocation/start_cap*100:.1f}% of capital)")
    if initialization_errors:
        print(f"Initialization Warnings: {len(initialization_errors)} issues")
    
    # Process week-to-week transitions for performance tracking (align with strategy methodology)
    # Add initial week data point (portfolio setup)
    initial_row_data = {
        "Date": pd.Timestamp(weeks[0]),
        "Portfolio_Value": total_allocation,
        "Weekly_Return_Pct": 0.0,
        "Period_Number": 0,
        "Analysis_Period": 0  # Initial setup, not analysis period
    }
    
    # Add initial asset-level data
    for symbol in benchmark_weights:
        if symbol in positions:
            pos = positions[symbol]
            initial_row_data.update({
                f"{symbol}_Weight": pos["weight"],
                f"{symbol}_Qty": pos["qty"],
                f"{symbol}_Price": pos["current_price"],
                f"{symbol}_Value": pos["value"],
                f"{symbol}_Allocation_Pct": (pos["value"] / total_allocation * 100) if total_allocation > 0 else 0,
                f"{symbol}_Return_Pct": 0.0  # Initial setup
            })
        else:
            initial_row_data.update({
                f"{symbol}_Weight": benchmark_weights[symbol],
                f"{symbol}_Qty": 0,
                f"{symbol}_Price": 0,
                f"{symbol}_Value": 0,
                f"{symbol}_Allocation_Pct": 0,
                f"{symbol}_Return_Pct": 0
            })
    
    benchmark_rows.append(initial_row_data)
    
    # Now process week-to-week transitions (same as strategy approach)
    for i in range(len(weeks) - 1):
        current_week = weeks[i]
        next_week = weeks[i + 1]
        week_data = df[df["rebalance_ts"] == next_week].set_index("sym")
        
        # Update positions with new prices and calculate portfolio value
        prev_portfolio_value = total_allocation if i == 0 else portfolio_value
        portfolio_value = 0.0
        
        for symbol in positions:
            if symbol in week_data.index:
                current_price = week_data.loc[symbol, "price_usd"]
                if pd.notna(current_price) and current_price > 0:
                    positions[symbol]["current_price"] = current_price
                    positions[symbol]["value"] = positions[symbol]["qty"] * current_price
                    portfolio_value += positions[symbol]["value"]
                else:
                    # Keep previous value if price is invalid
                    portfolio_value += positions[symbol]["value"]
            else:
                # Keep previous value if asset not found
                portfolio_value += positions[symbol]["value"]
        
        # Calculate weekly return
        weekly_return_pct = ((portfolio_value - prev_portfolio_value) / prev_portfolio_value) * 100 if prev_portfolio_value > 0 else 0.0
        
        # Rebalance to target weights if requested
        if rebalance_weekly:
            rebalanced_positions = {}
            rebalanced_value = 0.0
            
            for symbol, target_weight in benchmark_weights.items():
                if symbol in week_data.index:
                    current_price = week_data.loc[symbol, "price_usd"]
                    if pd.notna(current_price) and current_price > 0:
                        # Calculate target value for this asset
                        target_value = portfolio_value * target_weight
                        # Calculate new quantity needed
                        new_qty = target_value / current_price
                        
                        rebalanced_positions[symbol] = {
                            "qty": new_qty,
                            "initial_price": positions[symbol]["initial_price"] if symbol in positions else current_price,
                            "current_price": current_price,
                            "value": target_value,
                            "weight": target_weight,
                            "rank": week_data.loc[symbol, "rank"] if "rank" in week_data.columns else 0
                        }
                        rebalanced_value += target_value
                    else:
                        # Keep existing position if price is invalid
                        if symbol in positions:
                            rebalanced_positions[symbol] = positions[symbol].copy()
                            rebalanced_value += positions[symbol]["value"]
                else:
                    # Keep existing position if asset not found
                    if symbol in positions:
                        rebalanced_positions[symbol] = positions[symbol].copy()
                        rebalanced_value += positions[symbol]["value"]
            
            # Update positions with rebalanced quantities
            positions = rebalanced_positions
            portfolio_value = rebalanced_value
        
        # Create detailed row with asset breakdown
        row_data = {
            "Date": pd.Timestamp(next_week),
            "Portfolio_Value": portfolio_value,
            "Weekly_Return_Pct": weekly_return_pct,
            "Period_Number": i + 1,
            "Analysis_Period": i + 1  # Analysis period number
        }
        
        # Add asset-level data
        for symbol in benchmark_weights:
            if symbol in positions:
                pos = positions[symbol]
                row_data.update({
                    f"{symbol}_Weight": pos["weight"],
                    f"{symbol}_Qty": pos["qty"],
                    f"{symbol}_Price": pos["current_price"],
                    f"{symbol}_Value": pos["value"],
                    f"{symbol}_Allocation_Pct": (pos["value"] / portfolio_value * 100) if portfolio_value > 0 else 0,
                    f"{symbol}_Return_Pct": ((pos["current_price"] - pos["initial_price"]) / pos["initial_price"] * 100) if pos["initial_price"] > 0 else 0
                })
            else:
                # Asset not available - set to zero
                row_data.update({
                    f"{symbol}_Weight": benchmark_weights[symbol],
                    f"{symbol}_Qty": 0,
                    f"{symbol}_Price": 0,
                    f"{symbol}_Value": 0,
                    f"{symbol}_Allocation_Pct": 0,
                    f"{symbol}_Return_Pct": 0
                })
        
        benchmark_rows.append(row_data)
    
    benchmark_df = pd.DataFrame(benchmark_rows)
    
    # Add summary information
    if not benchmark_df.empty:
        print(f"\n=== BENCHMARK PERFORMANCE SUMMARY ===")
        final_value = benchmark_df["Portfolio_Value"].iloc[-1]
        total_return = ((final_value - start_cap) / start_cap) * 100
        num_periods = len(benchmark_df)
        analysis_periods = num_periods - 1 if num_periods > 1 else 0
        
        print(f"Initial Value: ${start_cap:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Snapshots: {num_periods} | Analysis Periods: {analysis_periods}")
        
        if analysis_periods > 0:
            weekly_returns = benchmark_df["Weekly_Return_Pct"].dropna()
            if len(weekly_returns) > 0:
                avg_weekly_return = weekly_returns.mean()
                annualized_return = total_return * (52 / analysis_periods)
                print(f"Average Weekly Return: {avg_weekly_return:.2f}%")
                print(f"Annualized Return: {annualized_return:+.2f}%")
    
    return benchmark_df


def compare_strategy_vs_benchmark(strategy_perf: pd.DataFrame, benchmark_perf: pd.DataFrame, 
                                strategy_summary: dict, start_cap: float) -> dict:
    """
    Compare strategy performance against benchmark with enhanced analysis.
    
    Args:
        strategy_perf: Strategy performance DataFrame
        benchmark_perf: Benchmark performance DataFrame with detailed asset breakdowns
        strategy_summary: Strategy summary metrics
        start_cap: Initial capital
        
    Returns:
        Dictionary with comprehensive comparison metrics
    """
    if strategy_perf.empty or benchmark_perf.empty:
        return {}
    
    # Align dates for comparison
    strategy_aligned = strategy_perf.set_index("Date")["Equity_USD"]
    benchmark_aligned = benchmark_perf.set_index("Date")["Portfolio_Value"]
    
    # Find common dates
    common_dates = strategy_aligned.index.intersection(benchmark_aligned.index)
    if len(common_dates) < 1:
        return {}
    
    strategy_values = strategy_aligned.loc[common_dates]
    benchmark_values = benchmark_aligned.loc[common_dates]
    
    # Calculate benchmark metrics with proper period counting
    benchmark_total_return = ((benchmark_values.iloc[-1] - start_cap) / start_cap) * 100
    benchmark_weekly_returns = benchmark_perf["Weekly_Return_Pct"].dropna().values
    
    # Use actual analysis periods for annualization (number of weeks analyzed)
    # For period counting: if we have N data points, we have N-1 analysis periods
    analysis_periods = len(benchmark_perf) - 1 if len(benchmark_perf) > 1 else 1
    benchmark_annualized_return = benchmark_total_return * (52 / analysis_periods) if analysis_periods > 0 else 0
    
    # Calculate benchmark max drawdown
    benchmark_equity = benchmark_values.values
    benchmark_peak = np.maximum.accumulate(benchmark_equity)
    benchmark_drawdown = (benchmark_equity - benchmark_peak) / benchmark_peak
    benchmark_max_drawdown = abs(min(benchmark_drawdown)) * 100 if len(benchmark_drawdown) > 0 else 0
    
    # Calculate benchmark Sharpe ratio
    benchmark_sharpe = 0.0
    if len(benchmark_weekly_returns) > 1:  # Need at least 2 points for std calculation
        benchmark_ann_return = np.mean(benchmark_weekly_returns) * 52
        benchmark_ann_volatility = np.std(benchmark_weekly_returns) * np.sqrt(52)
        if benchmark_ann_volatility > 0:
            benchmark_sharpe = (benchmark_ann_return - 2.0) / benchmark_ann_volatility  # 2% risk-free rate
    elif len(benchmark_weekly_returns) == 1:
        # For single week, Sharpe ratio is not meaningful, set to 0
        benchmark_sharpe = 0.0
    
    # Calculate correlation
    strategy_returns = strategy_perf["Weekly_Return_Pct"].dropna().values
    benchmark_returns_aligned = benchmark_perf["Weekly_Return_Pct"].dropna().values
    
    correlation = 0.0
    if len(strategy_returns) > 1 and len(benchmark_returns_aligned) > 1:
        min_len = min(len(strategy_returns), len(benchmark_returns_aligned))
        if min_len > 1:
            correlation = np.corrcoef(strategy_returns[-min_len:], benchmark_returns_aligned[-min_len:])[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
    elif len(strategy_returns) == 1 and len(benchmark_returns_aligned) == 1:
        # For single week, correlation is not meaningful, set to N/A or 0
        correlation = 0.0
    
    # Calculate alpha (excess return vs benchmark)
    alpha = strategy_summary.get("annualized_return", 0) - benchmark_annualized_return
    
    comparison_result = {
        "benchmark_total_return": benchmark_total_return,
        "benchmark_annualized_return": benchmark_annualized_return,
        "benchmark_max_drawdown": benchmark_max_drawdown,
        "benchmark_sharpe_ratio": benchmark_sharpe,
        "correlation": correlation,
        "alpha": alpha,
        "strategy_vs_benchmark_total": strategy_summary.get("total_return_pct", 0) - benchmark_total_return,
        "strategy_vs_benchmark_sharpe": strategy_summary.get("sharpe_ratio", 0) - benchmark_sharpe,
        "strategy_vs_benchmark_drawdown": strategy_summary.get("max_drawdown", 0) - benchmark_max_drawdown,
        "analysis_periods_benchmark": analysis_periods,
        "num_common_dates": len(common_dates)
    }
    
    print(f"\n=== BENCHMARK vs STRATEGY COMPARISON ===")
    print(f"Analysis Periods: {analysis_periods}")
    print(f"Benchmark Total Return: {benchmark_total_return:+.2f}%")
    print(f"Strategy Total Return: {strategy_summary.get('total_return_pct', 0):+.2f}%")
    print(f"Alpha (Excess Return): {alpha:+.2f}%")
    print(f"Correlation: {correlation:.3f}")
    
    return comparison_result


def plot_strategy_vs_benchmark(strategy_perf: pd.DataFrame, benchmark_perf: pd.DataFrame,
                             strategy_summary: dict, comparison: dict, 
                             start_date: dt.datetime, end_date: dt.datetime,
                             benchmark_weights: Dict[str, float]) -> plt.Figure:
    """
    Plot strategy vs benchmark equity curves.
    """
    if strategy_perf.empty or benchmark_perf.empty:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Top plot: Normalized equity curves ---
    # Normalize both to start at 100% for easy comparison
    strategy_normalized = (strategy_perf["Equity_USD"] / strategy_perf["Equity_USD"].iloc[0]) * 100
    benchmark_normalized = (benchmark_perf["Portfolio_Value"] / benchmark_perf["Portfolio_Value"].iloc[0]) * 100
    
    ax1.plot(strategy_perf["Date"], strategy_normalized, 
             linewidth=2, color='blue', label='Strategy', alpha=0.9)
    ax1.plot(benchmark_perf["Date"], benchmark_normalized, 
             linewidth=2, color='orange', label='Benchmark', alpha=0.9)
    
    # Create benchmark description
    benchmark_desc = " + ".join([f"{w*100:.0f}% {s}" for s, w in benchmark_weights.items() if w > 0])
    
    ax1.set_title(f"Strategy vs Benchmark Performance\n"
                  f"Benchmark: {benchmark_desc} | Period: {start_date.date()} to {end_date.date()}", 
                  fontsize=14)
    ax1.set_ylabel("Normalized Value (%)", fontsize=12)
    ax1.axhline(100, color='gray', linestyle='--', alpha=0.5, label='Starting Value (100%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Add performance metrics
    metrics_text = (
        f"Strategy: {strategy_summary.get('total_return_pct', 0):+.1f}% | "
        f"Benchmark: {comparison.get('benchmark_total_return', 0):+.1f}% | "
        f"Alpha: {comparison.get('alpha', 0):+.1f}%\n"
        f"Strategy Sharpe: {strategy_summary.get('sharpe_ratio', 0):.2f} | "
        f"Benchmark Sharpe: {comparison.get('benchmark_sharpe_ratio', 0):.2f} | "
        f"Correlation: {comparison.get('correlation', 0):.2f}"
    )
    ax1.text(0.02, 0.02, metrics_text, transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))
    
    # --- Bottom plot: Relative performance (Strategy / Benchmark) ---
    # Calculate relative performance
    common_dates = strategy_perf["Date"].isin(benchmark_perf["Date"])
    if common_dates.any():
        strategy_subset = strategy_perf[common_dates].reset_index(drop=True)
        
        # Align by date
        merged = pd.merge(strategy_subset[["Date", "Equity_USD"]], 
                         benchmark_perf[["Date", "Portfolio_Value"]], 
                         on="Date", how="inner")
        
        if not merged.empty:
            relative_perf = (merged["Equity_USD"] / merged["Portfolio_Value"]) * 100
            
            # Color code the bars
            colors = ['green' if x >= 100 else 'red' for x in relative_perf]
            
            ax2.bar(merged["Date"], relative_perf, color=colors, alpha=0.7, width=6)
            ax2.axhline(100, color='black', linestyle='-', linewidth=0.8)
            ax2.set_ylabel("Relative Performance\n(Strategy/Benchmark %)", fontsize=11)
            ax2.set_xlabel("Date", fontsize=12)
            ax2.grid(True, axis='y', alpha=0.3)
            
            # Set y-axis to show percentage
            ax2.yaxis.set_major_formatter(PercentFormatter())
    
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_rolling_correlation(strategy_perf: pd.DataFrame, benchmark_perf: pd.DataFrame,
                           window: int = 12) -> Optional[plt.Figure]:
    """
    Plot rolling correlation between strategy and benchmark.
    """
    if strategy_perf.empty or benchmark_perf.empty or len(strategy_perf) < window:
        return None
    
    # Get weekly returns
    strategy_returns = strategy_perf["Weekly_Return_Pct"].dropna()
    benchmark_returns = benchmark_perf["Weekly_Return_Pct"].dropna()
    
    # Align the returns
    min_len = min(len(strategy_returns), len(benchmark_returns))
    if min_len < window:
        return None
    
    strategy_returns = strategy_returns.iloc[-min_len:].reset_index(drop=True)
    benchmark_returns = benchmark_returns.iloc[-min_len:].reset_index(drop=True)
    dates = strategy_perf["Date"].iloc[-min_len:].reset_index(drop=True)
    
    # Calculate rolling correlation
    rolling_corr = []
    rolling_dates = []
    
    for i in range(window - 1, len(strategy_returns)):
        start_idx = i - window + 1
        end_idx = i + 1
        
        corr = np.corrcoef(strategy_returns.iloc[start_idx:end_idx], 
                          benchmark_returns.iloc[start_idx:end_idx])[0, 1]
        
        if not np.isnan(corr):
            rolling_corr.append(corr)
            rolling_dates.append(dates.iloc[i])
    
    if not rolling_corr:
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(rolling_dates, rolling_corr, linewidth=2, color='purple')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Moderate Correlation (0.5)')
    ax.axhline(-0.5, color='gray', linestyle='--', alpha=0.7, label='Moderate Negative Correlation (-0.5)')
    
    ax.set_title(f"Rolling Correlation ({window}-Week Window)\nStrategy vs Benchmark", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def get_benchmark_breakdown(benchmark_df: pd.DataFrame, benchmark_weights: Dict[str, float]) -> pd.DataFrame:
    """
    Create a detailed breakdown of benchmark performance by asset.
    
    Args:
        benchmark_df: Benchmark performance DataFrame
        benchmark_weights: Dictionary of asset weights
        
    Returns:
        DataFrame with asset-level breakdown
    """
    if benchmark_df.empty or not benchmark_weights:
        return pd.DataFrame()
    
    breakdown_rows = []
    
    for symbol in benchmark_weights:
        weight_col = f"{symbol}_Weight"
        value_col = f"{symbol}_Value"
        return_col = f"{symbol}_Return_Pct"
        
        if weight_col in benchmark_df.columns and value_col in benchmark_df.columns:
            initial_value = benchmark_df[value_col].iloc[0] if len(benchmark_df) > 0 else 0
            final_value = benchmark_df[value_col].iloc[-1] if len(benchmark_df) > 0 else 0
            
            total_return = ((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0
            target_weight = benchmark_weights[symbol] * 100
            
            # Calculate actual allocation over time
            if len(benchmark_df) > 0:
                portfolio_values = benchmark_df["Portfolio_Value"]
                asset_values = benchmark_df[value_col]
                actual_weights = (asset_values / portfolio_values * 100).mean() if not portfolio_values.empty else 0
            else:
                actual_weights = 0
            
            breakdown_rows.append({
                "Asset": symbol,
                "Target_Weight_Pct": target_weight,
                "Actual_Weight_Pct": actual_weights,
                "Initial_Value": initial_value,
                "Final_Value": final_value,
                "Total_Return_Pct": total_return,
                "Contribution_To_Portfolio": (final_value - initial_value) if initial_value > 0 else 0
            })
    
    return pd.DataFrame(breakdown_rows)


def print_benchmark_breakdown(benchmark_df: pd.DataFrame, benchmark_weights: Dict[str, float]):
    """
    Print a detailed breakdown of benchmark performance.
    
    Args:
        benchmark_df: Benchmark performance DataFrame
        benchmark_weights: Dictionary of asset weights
    """
    if benchmark_df.empty:
        print("No benchmark data available for breakdown.")
        return
    
    breakdown_df = get_benchmark_breakdown(benchmark_df, benchmark_weights)
    
    if breakdown_df.empty:
        print("No benchmark breakdown data available.")
        return
    
    print("\n" + "="*80)
    print("BENCHMARK PORTFOLIO - DETAILED ASSET BREAKDOWN")
    print("="*80)
    
    # Summary information
    initial_portfolio = benchmark_df["Portfolio_Value"].iloc[0]
    final_portfolio = benchmark_df["Portfolio_Value"].iloc[-1]
    total_return = ((final_portfolio - initial_portfolio) / initial_portfolio * 100)
    
    print(f"\nPORTFOLIO SUMMARY:")
    print(f"Initial Value: ${initial_portfolio:,.2f}")
    print(f"Final Value: ${final_portfolio:,.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"Number of Assets: {len(breakdown_df)}")
    
    # Asset breakdown table
    print(f"\nASSET BREAKDOWN:")
    
    table_data = []
    for _, row in breakdown_df.iterrows():
        table_data.append([
            row["Asset"],
            f"{row['Target_Weight_Pct']:.1f}%",
            f"{row['Actual_Weight_Pct']:.1f}%",
            f"${row['Initial_Value']:,.2f}",
            f"${row['Final_Value']:,.2f}",
            f"{row['Total_Return_Pct']:+.2f}%",
            f"${row['Contribution_To_Portfolio']:+,.2f}"
        ])
    
    headers = [
        "Asset", "Target %", "Actual %", "Initial Value", 
        "Final Value", "Return %", "P&L Contribution"
    ]
    
    from tabulate import tabulate
    print(tabulate(table_data, headers=headers, tablefmt="grid", numalign="right"))
    
    # Performance attribution
    total_contribution = breakdown_df["Contribution_To_Portfolio"].sum()
    print(f"\nPERFORMANCE ATTRIBUTION:")
    print(f"Total P&L: ${total_contribution:+,.2f}")
    
    # Top contributors
    top_contributors = breakdown_df.nlargest(3, "Contribution_To_Portfolio")
    print(f"\nTOP CONTRIBUTORS:")
    for _, row in top_contributors.iterrows():
        contribution_pct = (row["Contribution_To_Portfolio"] / abs(total_contribution) * 100) if total_contribution != 0 else 0
        print(f"  • {row['Asset']}: ${row['Contribution_To_Portfolio']:+,.2f} ({contribution_pct:+.1f}% of total P&L)")


def get_available_assets_from_data(df: pd.DataFrame) -> list:
    """
    Get list of available assets from the dataset that can be used for benchmarks.
    
    Args:
        df: DataFrame with cryptocurrency data
        
    Returns:
        List of available asset symbols sorted by market cap
    """
    if df.empty:
        return BENCHMARK_AVAILABLE_ASSETS
    
    # Get symbols that appear in the most recent week
    latest_week = df["rebalance_ts"].max()
    latest_data = df[df["rebalance_ts"] == latest_week]
    
    # Filter for assets that have valid price and market cap data
    valid_assets = latest_data[
        (latest_data["price_usd"].notna()) & 
        (latest_data["price_usd"] > 0) &
        (latest_data["market_cap_usd"].notna()) &
        (latest_data["market_cap_usd"] > 0)
    ]
    
    # Sort by market cap (descending) and return symbols
    sorted_assets = valid_assets.sort_values("market_cap_usd", ascending=False)["sym"].tolist()
    
    # Filter to only include predefined benchmark assets that are available
    available_benchmark_assets = [asset for asset in BENCHMARK_AVAILABLE_ASSETS if asset in sorted_assets]
    
    return available_benchmark_assets[:20]  # Limit to top 20 for UI purposes


def get_benchmark_breakdown(benchmark_df: pd.DataFrame, benchmark_weights: Dict[str, float]) -> pd.DataFrame:
    """
    Get detailed breakdown of benchmark performance by asset.
    
    Args:
        benchmark_df: Benchmark performance DataFrame
        benchmark_weights: Dictionary mapping asset symbols to their weights
        
    Returns:
        DataFrame with asset-level performance breakdown
    """
    if benchmark_df.empty:
        return pd.DataFrame()
    
    breakdown_data = []
    
    for symbol, weight in benchmark_weights.items():
        if f"{symbol}_Value" in benchmark_df.columns and f"{symbol}_Price" in benchmark_df.columns:
            # Get start and end values
            start_value = benchmark_df[f"{symbol}_Value"].iloc[0]
            end_value = benchmark_df[f"{symbol}_Value"].iloc[-1]
            
            # Get start and end prices if available
            price_cols = [col for col in benchmark_df.columns if col.startswith(f"{symbol}_Price")]
            if price_cols:
                start_price = benchmark_df[price_cols[0]].iloc[0]
                end_price = benchmark_df[price_cols[0]].iloc[-1]
                price_return = ((end_price - start_price) / start_price * 100) if start_price > 0 else 0
            else:
                start_price = end_price = price_return = 0
            
            # Get quantity if available
            qty_cols = [col for col in benchmark_df.columns if col.startswith(f"{symbol}_Qty")]
            quantity = benchmark_df[qty_cols[0]].iloc[0] if qty_cols else 0
            
            # Calculate returns
            value_return = ((end_value - start_value) / start_value * 100) if start_value > 0 else 0
            
            breakdown_data.append({
                "Asset": symbol,
                "Weight_Pct": weight * 100,
                "Start_Price": start_price,
                "End_Price": end_price,
                "Price_Return_Pct": price_return,
                "Total_Return_Pct": price_return,  # Alias for compatibility
                "Start_Value": start_value,
                "End_Value": end_value,
                "Value_Return_Pct": value_return,
                "Quantity": quantity,
                "PnL_USD": end_value - start_value
            })
    
    return pd.DataFrame(breakdown_data)


def print_benchmark_breakdown(benchmark_df: pd.DataFrame, benchmark_weights: Dict[str, float],
                             benchmark_comparison: dict, period_description: str = ""):
    """
    Print detailed benchmark construction and performance breakdown.
    
    Args:
        benchmark_df: Benchmark performance DataFrame
        benchmark_weights: Dictionary mapping asset symbols to their weights  
        benchmark_comparison: Benchmark comparison metrics
        period_description: Description of the analysis period
    """
    if benchmark_df.empty:
        print("No benchmark data available for breakdown.")
        return
    
    print(f"\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print(f"┃                          BENCHMARK RAW DATA BREAKDOWN                        ┃")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    
    if period_description:
        print(f"\nPeriod: {period_description}")
    
    print(f"Analysis Periods: {len(benchmark_df) - 1 if len(benchmark_df) > 1 else 1}")
    print(f"Data Points: {len(benchmark_df)}")
    
    # Get asset breakdown
    breakdown_df = get_benchmark_breakdown(benchmark_df, benchmark_weights)
    
    if not breakdown_df.empty:
        print(f"\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print(f"│                           ASSET-LEVEL PERFORMANCE                           │")
        print(f"├─────────────────────────────────────────────────────────────────────────────┤")
        
        for _, row in breakdown_df.iterrows():
            print(f"│ {row['Asset']:>6} ({row['Weight_Pct']:5.1f}%):  ${row['Start_Price']:8.2f} → ${row['End_Price']:8.2f}  ({row['Price_Return_Pct']:+6.2f}%)  P&L: ${row['PnL_USD']:+8.2f} │")
        
        print(f"└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Show portfolio summary
    if len(benchmark_df) > 0:
        start_value = benchmark_df["Portfolio_Value"].iloc[0]
        end_value = benchmark_df["Portfolio_Value"].iloc[-1]
        total_return = ((end_value - start_value) / start_value * 100) if start_value > 0 else 0
        
        print(f"\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print(f"│                           PORTFOLIO SUMMARY                                  │")
        print(f"├─────────────────────────────────────────────────────────────────────────────┤")
        print(f"│ Start Value:               ${start_value:12,.2f}                              │")
        print(f"│ End Value:                 ${end_value:12,.2f}                              │")
        print(f"│ Total Return:              {total_return:+12.2f}%                             │")
        print(f"│ Annualized Return:         {benchmark_comparison.get('benchmark_annualized_return', 0):+12.2f}%                             │")
        print(f"│ Total P&L:                 ${end_value - start_value:+12,.2f}                              │")
        print(f"└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Show period-by-period if multiple periods
    if len(benchmark_df) > 1:
        print(f"\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print(f"│                           PERIOD-BY-PERIOD BREAKDOWN                         │")
        print(f"├─────────────────────────────────────────────────────────────────────────────┤")
        print(f"│     Date        Portfolio Value    Weekly Return                            │")
        print(f"├─────────────────────────────────────────────────────────────────────────────┤")
        
        for _, row in benchmark_df.iterrows():
            date_str = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
            print(f"│ {date_str}      ${row['Portfolio_Value']:12,.2f}        {row['Weekly_Return_Pct']:+6.2f}%                   │")
        
        print(f"└─────────────────────────────────────────────────────────────────────────────┘")