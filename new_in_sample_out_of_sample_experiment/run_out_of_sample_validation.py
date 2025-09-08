#!/usr/bin/env python3
"""
Out-of-Sample Validation Script for New Experiment

Tests selected pairs from in-sample filtering on 2024 out-of-sample data.
Creates equity curves, drawdown curves, and metrics for each pair.

Period: 2024-01-01 to 2025-01-01
Algorithm: Same CMMA-based intermarket difference strategy
Output: Individual pair visualizations and performance metrics
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Import core trading functions
from core import (
    cmma, threshold_revert_signal, load_csv_with_cache, align_frames,
    calculate_metrics_from_trades, calculate_max_drawdown, run_single_trading_type_backtest,
    generate_equity_curve_from_trades, get_available_coins, create_trade_record,
    LOOKBACK, ATR_LOOKBACK, THRESHOLD
)

# Add parent directory to path to import trading functions
sys.path.append('..')
from parameter_optimization_experiment.trades_from_signal import get_trades_from_signal

# -----------------------
# Configuration
# -----------------------
IN_SAMPLE_DATA_DIR = "data/in_sample"
OUT_OF_SAMPLE_DATA_DIR = "data/out_of_sample"
FILTERED_PAIRS_FILE = "results/filtered_pairs/new_experiment_filtered_pairs.csv"
RESULTS_DIR = "results/out_of_sample"
TRADES_DIR = os.path.join(RESULTS_DIR, "trades")
FIGURES_DIR = "results/pair_figures"

# Algorithm parameters imported from core
# LOOKBACK, ATR_LOOKBACK, THRESHOLD are now imported from core.py
MIN_OVERLAP = 100  # Lower threshold for 1-year data

# Output file
OUT_CSV = os.path.join(RESULTS_DIR, "oos_validation_results.csv")


# -----------------------
# Trading Algorithm (imported from core.py)
# -----------------------
# cmma() and threshold_revert_signal() functions are now imported from core.py


def load_csv_data(coin: str, data_type: str) -> pd.DataFrame:
    """Load CSV file for a coin (in_sample or out_of_sample)."""
    
    if data_type == "in_sample":
        return load_csv_with_cache(coin, IN_SAMPLE_DATA_DIR, "is", {})
    elif data_type == "out_of_sample":
        return load_csv_with_cache(coin, OUT_OF_SAMPLE_DATA_DIR, "oos", {})
    else:
        raise ValueError("data_type must be 'in_sample' or 'out_of_sample'")


# calculate_metrics_from_trades() and calculate_max_drawdown() functions are now imported from core.py


def save_oos_detailed_trades(ref_coin: str, trading_coin: str, trading_type: str):
    """Save detailed OOS trade data for a pair and trading type."""
    
    ref_df = load_csv_data(ref_coin, "out_of_sample")
    traded_df = load_csv_data(trading_coin, "out_of_sample")
    
    if ref_df is None or traded_df is None:
        return
    
    ref_df, traded_df = align_frames(ref_df, traded_df)
    
    if len(ref_df) < MIN_OVERLAP:
        return
    
    try:
        # Generate signals and trades
        traded_df = traded_df.copy()
        traded_df["diff"] = np.log(traded_df["close"]).diff()
        traded_df["next_return"] = traded_df["diff"].shift(-1)
        
        ref_cmma = cmma(ref_df, LOOKBACK, ATR_LOOKBACK)
        trd_cmma = cmma(traded_df, LOOKBACK, ATR_LOOKBACK)
        intermarket_diff = trd_cmma - ref_cmma
        traded_df["sig"] = threshold_revert_signal(intermarket_diff, THRESHOLD)
        
        # Get trades
        long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, traded_df["sig"].values)
        
        # Select trades based on trading type
        if trading_type == 'longs':
            selected_trades = long_trades
        elif trading_type == 'shorts':
            selected_trades = short_trades
        else:  # 'both'
            selected_trades = all_trades
        
        # Save trades JSON
        trade_dir = os.path.join(TRADES_DIR, trading_coin)
        os.makedirs(trade_dir, exist_ok=True)
        
        trades_data = []
        for entry_time, trade in selected_trades.iterrows():
            trade_record = create_trade_record(trade, entry_time)
            trades_data.append(trade_record)
        
        # Save to JSON
        trades_file = os.path.join(trade_dir, f"{trading_coin}_{ref_coin}_{trading_type}_trades.json")
        with open(trades_file, 'w') as f:
            json.dump(trades_data, f, indent=2)
            
    except Exception:
        pass


def run_oos_validation(ref_coin: str, trading_coin: str, in_sample_metrics: dict) -> dict:
    """
    Run out-of-sample validation for a single pair.
    Now uses core functions for consistency and saves trades.
    
    Args:
        ref_coin: Reference coin symbol
        trading_coin: Trading coin symbol  
        in_sample_metrics: Original in-sample performance metrics
        
    Returns:
        Combined metrics dictionary or None if failed
    """
    
    # Load out-of-sample data
    ref_df = load_csv_data(ref_coin, "out_of_sample")
    traded_df = load_csv_data(trading_coin, "out_of_sample")
    
    # Use trading type from in-sample metrics if available
    trading_type = in_sample_metrics['trading_type']
    
    # Use core function for backtesting
    oos_metrics = run_single_trading_type_backtest(ref_df, traded_df, ref_coin, trading_coin, trading_type, MIN_OVERLAP)
    
    if oos_metrics is None:
        return None
    
    # Save detailed trades
    save_oos_detailed_trades(ref_coin, trading_coin, trading_type)
    
    # Combine with in-sample metrics
    result = {
        'reference_coin': ref_coin,
        'trading_coin': trading_coin,
        'trading_type': trading_type,
        
        # In-sample metrics (prefixed with is_)
        'is_total_cumulative_return': in_sample_metrics.get('total_cumulative_return', 0.0),
        'is_profit_factor': in_sample_metrics.get('profit_factor', 0.0),
        'is_max_drawdown': in_sample_metrics.get('max_drawdown', 0.0),
        'is_last_year_drawdown': in_sample_metrics.get('last_year_drawdown', 0.0),
        'is_sharpe_ratio': in_sample_metrics.get('sharpe_ratio', 0.0),
        'is_volatility': in_sample_metrics.get('volatility', 0.0),
        'is_num_trades': in_sample_metrics.get('num_trades', 0),
        
        # Out-of-sample metrics (prefixed with oos_)
        'oos_total_cumulative_return': oos_metrics['total_cumulative_return'],
        'oos_profit_factor': oos_metrics['profit_factor'],
        'oos_max_drawdown': oos_metrics['max_drawdown'],
        'oos_last_year_drawdown': oos_metrics['last_year_drawdown'],
        'oos_sharpe_ratio': oos_metrics['sharpe_ratio'],
        'oos_volatility': oos_metrics['volatility'],
        'oos_num_trades': oos_metrics['num_trades'],
        'oos_data_points': oos_metrics['data_points']
    }
    
    return result


def create_pair_visualizations(ref_coin: str, trading_coin: str, in_sample_metrics: dict):
    """Create equity curve and drawdown visualizations for a pair."""
    
    # Load out-of-sample data
    ref_df = load_csv_data(ref_coin, "out_of_sample")
    traded_df = load_csv_data(trading_coin, "out_of_sample")
    
    if ref_df is None or traded_df is None:
        return
    
    # Use trading type from in-sample metrics
    trading_type = in_sample_metrics.get('trading_type', 'both')
    
    # Use core function to get trades and metrics
    oos_metrics = run_single_trading_type_backtest(ref_df, traded_df, ref_coin, trading_coin, trading_type, MIN_OVERLAP)
    
    if oos_metrics is None:
        return
    
    try:
        # Re-run the core logic to get trade data for visualization
        ref_df, traded_df = align_frames(ref_df, traded_df)
        
        if len(ref_df) < MIN_OVERLAP:
            return
        
        # Calculate signals
        traded_df = traded_df.copy()
        traded_df["diff"] = np.log(traded_df["close"]).diff()
        traded_df["next_return"] = traded_df["diff"].shift(-1)
        
        ref_cmma = cmma(ref_df, LOOKBACK, ATR_LOOKBACK)
        trd_cmma = cmma(traded_df, LOOKBACK, ATR_LOOKBACK)
        intermarket_diff = trd_cmma - ref_cmma
        traded_df["sig"] = threshold_revert_signal(intermarket_diff, THRESHOLD)
        
        # Get detailed trades
        long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, traded_df["sig"].values)
        
        # Select trades based on trading type
        if trading_type == 'longs':
            selected_trades = long_trades
        elif trading_type == 'shorts':
            selected_trades = short_trades
        else:  # 'both'
            selected_trades = all_trades
        
        if len(selected_trades) == 0:
            return
        
        # Generate equity curve using core function
        portfolio_values = generate_equity_curve_from_trades(selected_trades, 1000.0)
        
        if portfolio_values is None or len(portfolio_values) == 0:
            return
        
        # Calculate drawdown
        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values - running_max) / running_max
        
        # Create pair directory
        pair_name = f"{trading_coin}_{ref_coin}"
        pair_dir = os.path.join(FIGURES_DIR, pair_name)
        os.makedirs(pair_dir, exist_ok=True)
        
        # 1. Equity Curve
        plt.figure(figsize=(15, 8))
        plt.plot(portfolio_values.index, portfolio_values.values, linewidth=1.5, color='blue')
        plt.axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        plt.title(f'Out-of-Sample Equity Curve: {trading_coin} (ref: {ref_coin})', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        equity_file = os.path.join(pair_dir, 'oos_equity_curve.png')
        plt.savefig(equity_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Drawdown Curve
        plt.figure(figsize=(15, 8))
        plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
        plt.title(f'Out-of-Sample Drawdown: {trading_coin} (ref: {ref_coin})', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        drawdown_file = os.path.join(pair_dir, 'oos_drawdown_curve.png')
        plt.savefig(drawdown_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Save metrics JSON
        oos_metrics = {
            'pair_info': {
                'reference_coin': ref_coin,
                'trading_coin': trading_coin,
                'period': '2024-01-01 to 2025-01-01'
            },
            'in_sample_metrics': in_sample_metrics,
            'oos_metrics': {
                'total_cumulative_return': float(portfolio_values.iloc[-1] / 1000),  # Portfolio return ratio
                'profit_factor': oos_metrics['profit_factor'],
                'max_drawdown': float(drawdown.min()),
                'sharpe_ratio': oos_metrics['sharpe_ratio'],
                'num_trades': oos_metrics['num_trades'],
                'final_portfolio_value': float(portfolio_values.iloc[-1]),
                'trading_type': trading_type
            }
        }
        
        metrics_file = os.path.join(pair_dir, 'oos_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(oos_metrics, f, indent=2)
        
    except Exception as e:
        print(f"⚠️  Could not create visualizations for {ref_coin}-{trading_coin}: {e}")


def main():
    """Main execution function."""
    
    print("🔍 Out-of-Sample Validation (2024 Data)")
    print("=" * 45)
    
    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TRADES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load filtered pairs
    if not os.path.exists(FILTERED_PAIRS_FILE):
        print(f"❌ Filtered pairs file not found: {FILTERED_PAIRS_FILE}")
        print("Run apply_custom_filters.py first to generate filtered pairs.")
        return
    
    filtered_pairs = pd.read_csv(FILTERED_PAIRS_FILE)
    print(f"📊 Loaded {len(filtered_pairs)} filtered pairs for validation")
    
    # Run OOS validation
    results = []
    successful_validations = 0
    failed_validations = 0
    
    print(f"\n🔄 Running out-of-sample validation...")
    
    for _, pair in tqdm(filtered_pairs.iterrows(), total=len(filtered_pairs), desc="Validating pairs"):
        ref_coin = pair['reference_coin']
        trading_coin = pair['trading_coin']
        
        # Prepare in-sample metrics (include all available fields)
        in_sample_metrics = {
            'total_cumulative_return': pair.get('total_cumulative_return', 0.0),
            'profit_factor': pair.get('profit_factor', 0.0),
            'max_drawdown': pair.get('max_drawdown', 0.0),
            'last_year_drawdown': pair.get('last_year_drawdown', 0.0),
            'sharpe_ratio': pair.get('sharpe_ratio', 0.0),
            'volatility': pair.get('volatility', 0.0),
            'num_trades': pair.get('num_trades', 0),
            'trading_type': pair.get('trading_type', 'both')
        }
        
        # Run OOS validation
        result = run_oos_validation(ref_coin, trading_coin, in_sample_metrics)
        
        if result is not None:
            results.append(result)
            successful_validations += 1
            
            # Create visualizations
            create_pair_visualizations(ref_coin, trading_coin, in_sample_metrics)
        else:
            failed_validations += 1
    
    if not results:
        print("❌ No successful validations!")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(OUT_CSV, index=False, float_format='%.6f')
    
    # Analysis
    print(f"\n📊 OUT-OF-SAMPLE VALIDATION RESULTS:")
    print(f"   ✅ Successful validations: {successful_validations}")
    print(f"   ❌ Failed validations: {failed_validations}")
    print(f"   📈 Average OOS Profit Factor: {results_df['oos_profit_factor'].mean():.3f}")
    print(f"   📉 Average OOS Max Drawdown: {results_df['oos_max_drawdown'].mean():.2%}")
    print(f"   ⚡ Average OOS Sharpe Ratio: {results_df['oos_sharpe_ratio'].mean():.3f}")
    print(f"   📊 Average OOS Return: {results_df['oos_total_cumulative_return'].mean():.3f}")
    
    # Performance comparison
    print(f"\n📊 IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON:")
    is_avg_pf = results_df['is_profit_factor'].mean()
    oos_avg_pf = results_df['oos_profit_factor'].mean()
    is_avg_return = results_df['is_total_cumulative_return'].mean()
    oos_avg_return = results_df['oos_total_cumulative_return'].mean()
    is_avg_dd = results_df['is_max_drawdown'].mean()
    oos_avg_dd = results_df['oos_max_drawdown'].mean()
    
    print(f"   Profit Factor: IS={is_avg_pf:.2f} → OOS={oos_avg_pf:.2f} (Δ: {oos_avg_pf-is_avg_pf:+.2f})")
    print(f"   Total Return: IS={is_avg_return:.2f} → OOS={oos_avg_return:.2f} (Δ: {oos_avg_return-is_avg_return:+.2f})")
    print(f"   Max Drawdown: IS={is_avg_dd:.2%} → OOS={oos_avg_dd:.2%} (Δ: {oos_avg_dd-is_avg_dd:+.2%})")
    
    # Top OOS performers
    top_oos = results_df.nlargest(10, 'oos_profit_factor')
    print(f"\n🏆 TOP 10 OUT-OF-SAMPLE PERFORMERS:")
    print(f"   {'Rank':<4} {'Ref Coin':<8} {'Trading Coin':<12} {'OOS_PF':<8} {'OOS_Ret':<9} {'OOS_DD':<9} {'IS_PF':<8}")
    print(f"   {'-'*4} {'-'*8} {'-'*12} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")
    
    for i, (_, row) in enumerate(top_oos.iterrows(), 1):
        print(f"   {i:<4} {row['reference_coin']:<8} {row['trading_coin']:<12} "
              f"{row['oos_profit_factor']:<8.2f} {row['oos_total_cumulative_return']:<9.2f} "
              f"{row['oos_max_drawdown']:<9.2%} {row['is_profit_factor']:<8.2f}")
    
    print(f"\n💾 Results saved to: {OUT_CSV}")
    print(f"📁 Detailed trades saved to: {TRADES_DIR}")
    print(f"📊 Individual pair figures saved to: {FIGURES_DIR}")
    
    print(f"\n✅ Out-of-sample validation completed!")
    print(f"Ready for portfolio simulation: python simulate_portfolio_2024.py")


if __name__ == "__main__":
    main()