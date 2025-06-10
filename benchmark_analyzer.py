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
                                  start_cap: float = 100000.0) -> pd.DataFrame:
    """
    Calculate benchmark portfolio performance using buy-and-hold strategy.
    
    Args:
        df: DataFrame with prepared cryptocurrency data
        benchmark_weights: Dictionary mapping asset symbols to their weights
        start_cap: Initial capital in USD
        
    Returns:
        DataFrame with benchmark performance data
    """
    # Validate weights
    is_valid, error_msg = validate_benchmark_weights(benchmark_weights)
    if not is_valid:
        raise ValueError(f"Invalid benchmark weights: {error_msg}")
    
    # Get available weeks
    weeks = sorted(df["rebalance_ts"].unique())
    if len(weeks) < 2:
        raise ValueError("Not enough weeks for benchmark calculation")
    
    # Initialize tracking
    benchmark_rows = []
    
    for i, week in enumerate(weeks):
        week_data = df[df["rebalance_ts"] == week].set_index("sym")
        
        if i == 0:
            # Initialize positions at the first week
            portfolio_value = start_cap
            positions = {}  # symbol -> {qty, value}
            
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
                            "value": allocation
                        }
        else:
            # Update positions with new prices
            portfolio_value = 0.0
            
            for symbol in positions:
                if symbol in week_data.index:
                    current_price = week_data.loc[symbol, "price_usd"]
                    if pd.notna(current_price) and current_price > 0:
                        positions[symbol]["current_price"] = current_price
                        positions[symbol]["value"] = positions[symbol]["qty"] * current_price
                        portfolio_value += positions[symbol]["value"]
        
        # Calculate weekly return if not first week
        weekly_return_pct = 0.0
        if i > 0 and len(benchmark_rows) > 0:
            prev_value = benchmark_rows[-1]["Portfolio_Value"]
            if prev_value > 0:
                weekly_return_pct = ((portfolio_value - prev_value) / prev_value) * 100
        
        # Store benchmark performance data
        benchmark_rows.append({
            "Date": pd.Timestamp(week),
            "Portfolio_Value": portfolio_value,
            "Weekly_Return_Pct": weekly_return_pct,
            **{f"{symbol}_Weight": benchmark_weights.get(symbol, 0) for symbol in benchmark_weights},
            **{f"{symbol}_Value": positions.get(symbol, {}).get("value", 0) for symbol in benchmark_weights}
        })
    
    return pd.DataFrame(benchmark_rows)


def compare_strategy_vs_benchmark(strategy_perf: pd.DataFrame, benchmark_perf: pd.DataFrame, 
                                strategy_summary: dict, start_cap: float) -> dict:
    """
    Compare strategy performance against benchmark.
    
    Args:
        strategy_perf: Strategy performance DataFrame
        benchmark_perf: Benchmark performance DataFrame  
        strategy_summary: Strategy summary metrics
        start_cap: Initial capital
        
    Returns:
        Dictionary with comparison metrics
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
    
    # Calculate benchmark metrics
    benchmark_total_return = ((benchmark_values.iloc[-1] - start_cap) / start_cap) * 100
    benchmark_weekly_returns = benchmark_perf["Weekly_Return_Pct"].dropna().values
    benchmark_annualized_return = benchmark_total_return * (52 / len(benchmark_weekly_returns)) if len(benchmark_weekly_returns) > 0 else 0
    
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
    
    return {
        "benchmark_total_return": benchmark_total_return,
        "benchmark_annualized_return": benchmark_annualized_return,
        "benchmark_max_drawdown": benchmark_max_drawdown,
        "benchmark_sharpe_ratio": benchmark_sharpe,
        "correlation": correlation,
        "alpha": alpha,
        "strategy_vs_benchmark_total": strategy_summary.get("total_return_pct", 0) - benchmark_total_return,
        "strategy_vs_benchmark_sharpe": strategy_summary.get("sharpe_ratio", 0) - benchmark_sharpe,
        "strategy_vs_benchmark_drawdown": strategy_summary.get("max_drawdown", 0) - benchmark_max_drawdown
    }


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