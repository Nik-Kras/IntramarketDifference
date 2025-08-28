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
import pandas_ta as ta
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add parent directory to path to import trading functions
sys.path.append('..')
from trades_from_signal import get_trades_from_signal

# -----------------------
# Configuration
# -----------------------
IN_SAMPLE_DATA_DIR = "data/in_sample"
OUT_OF_SAMPLE_DATA_DIR = "data/out_of_sample"
FILTERED_PAIRS_FILE = "results/filtered_pairs/new_experiment_filtered_pairs.csv"
RESULTS_DIR = "results/out_of_sample"
FIGURES_DIR = "results/pair_figures"

# Algorithm parameters (same as in-sample)
LOOKBACK = 24
ATR_LOOKBACK = 168
THRESHOLD = 0.25
MIN_OVERLAP = 100  # Lower threshold for 1-year data

# Output file
OUT_CSV = os.path.join(RESULTS_DIR, "oos_validation_results.csv")


# -----------------------
# Trading Algorithm (Identical)
# -----------------------
def cmma(ohlc: pd.DataFrame, lookback: int, atr_lookback: int = 168) -> pd.Series:
    """Close-minus-MA normalized by ATR * sqrt(L)."""
    atr = ta.atr(ohlc["high"], ohlc["low"], ohlc["close"], atr_lookback)
    ma = ohlc["close"].rolling(lookback).mean()
    ind = (ohlc["close"] - ma) / (atr * lookback ** 0.5)
    return ind


def threshold_revert_signal(ind: pd.Series, threshold: float) -> np.ndarray:
    """Generate mean reversion signals based on threshold crossings."""
    signal = np.zeros(len(ind))
    position = 0
    values = ind.values
    for i in range(len(values)):
        v = values[i]
        if not np.isnan(v):
            if v > threshold:
                position = 1
            if v < -threshold:
                position = -1
            if position == 1 and v <= 0:
                position = 0
            if position == -1 and v >= 0:
                position = 0
        signal[i] = position
    return signal


def load_csv_data(coin: str, data_type: str) -> pd.DataFrame:
    """Load CSV file for a coin (in_sample or out_of_sample)."""
    
    if data_type == "in_sample":
        filepath = os.path.join(IN_SAMPLE_DATA_DIR, f"{coin}_is.csv")
    elif data_type == "out_of_sample":
        filepath = os.path.join(OUT_OF_SAMPLE_DATA_DIR, f"{coin}_oos.csv")
    else:
        raise ValueError("data_type must be 'in_sample' or 'out_of_sample'")
    
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        
        # Ensure required columns
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                return None
        
        return df.sort_index().dropna()
        
    except Exception:
        return None


def align_frames(a: pd.DataFrame, b: pd.DataFrame) -> tuple:
    """Align two dataframes by common index."""
    ix = a.index.intersection(b.index)
    return a.loc[ix].copy(), b.loc[ix].copy()


def run_oos_validation(ref_coin: str, trading_coin: str, in_sample_metrics: dict) -> dict:
    """
    Run out-of-sample validation for a single pair.
    
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
    
    if ref_df is None or traded_df is None:
        return None
    
    # Align data
    ref_df, traded_df = align_frames(ref_df, traded_df)
    
    if len(ref_df) < MIN_OVERLAP:
        return None
    
    try:
        # Calculate returns
        traded_df = traded_df.copy()
        traded_df["diff"] = np.log(traded_df["close"]).diff()
        traded_df["next_return"] = traded_df["diff"].shift(-1)
        
        # Calculate CMMA indicators
        ref_cmma = cmma(ref_df, LOOKBACK, ATR_LOOKBACK)
        trd_cmma = cmma(traded_df, LOOKBACK, ATR_LOOKBACK)
        intermarket_diff = trd_cmma - ref_cmma
        
        # Generate signal
        traded_df["sig"] = threshold_revert_signal(intermarket_diff, THRESHOLD)
        
        # Calculate returns
        rets = traded_df["sig"] * traded_df["next_return"]
        rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(rets) == 0:
            return None
        
        # Get detailed trades
        long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, traded_df["sig"].values)
        
        # Calculate OOS metrics
        total_return = rets.sum()
        
        # Profit factor
        gains = rets[rets > 0].sum()
        losses = rets[rets < 0].sum()
        profit_factor = gains / abs(losses) if losses < 0 else np.inf if gains > 0 else 0
        
        # Max drawdown
        cumulative = rets.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio
        sharpe_ratio = rets.mean() / rets.std() * np.sqrt(8760) if rets.std() > 0 else 0  # Hourly data
        
        # Volatility
        volatility = rets.std() * np.sqrt(8760)  # Annualized
        
        # Number of trades
        num_trades = len(all_trades)
        
        # Combine with in-sample metrics
        result = {
            'reference_coin': ref_coin,
            'trading_coin': trading_coin,
            
            # In-sample metrics (prefixed with is_)
            'is_total_cumulative_return': in_sample_metrics['total_cumulative_return'],
            'is_profit_factor': in_sample_metrics['profit_factor'],
            'is_max_drawdown': in_sample_metrics['max_drawdown'],
            'is_sharpe_ratio': in_sample_metrics['sharpe_ratio'],
            'is_num_trades': in_sample_metrics['num_trades'],
            
            # Out-of-sample metrics (prefixed with oos_)
            'oos_total_cumulative_return': total_return,
            'oos_profit_factor': profit_factor,
            'oos_max_drawdown': max_drawdown,
            'oos_sharpe_ratio': sharpe_ratio,
            'oos_volatility': volatility,
            'oos_num_trades': num_trades,
            'oos_data_points': len(rets)
        }
        
        return result
        
    except Exception as e:
        return None


def create_pair_visualizations(ref_coin: str, trading_coin: str, in_sample_metrics: dict):
    """Create equity curve and drawdown visualizations for a pair."""
    
    # Load out-of-sample data
    ref_df = load_csv_data(ref_coin, "out_of_sample")
    traded_df = load_csv_data(trading_coin, "out_of_sample")
    
    if ref_df is None or traded_df is None:
        return
    
    # Align data
    ref_df, traded_df = align_frames(ref_df, traded_df)
    
    if len(ref_df) < MIN_OVERLAP:
        return
    
    try:
        # Generate signals and returns
        traded_df = traded_df.copy()
        traded_df["diff"] = np.log(traded_df["close"]).diff()
        traded_df["next_return"] = traded_df["diff"].shift(-1)
        
        ref_cmma = cmma(ref_df, LOOKBACK, ATR_LOOKBACK)
        trd_cmma = cmma(traded_df, LOOKBACK, ATR_LOOKBACK)
        intermarket_diff = trd_cmma - ref_cmma
        traded_df["sig"] = threshold_revert_signal(intermarket_diff, THRESHOLD)
        
        rets = traded_df["sig"] * traded_df["next_return"]
        rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(rets) == 0:
            return
        
        # Calculate equity curve
        cumulative_returns = rets.cumsum()
        equity_curve = np.exp(cumulative_returns) * 1000  # Start with $1000
        
        # Calculate drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        
        # Create pair directory
        pair_name = f"{ref_coin}_{trading_coin}"
        pair_dir = os.path.join(FIGURES_DIR, pair_name)
        os.makedirs(pair_dir, exist_ok=True)
        
        # 1. Equity Curve
        plt.figure(figsize=(15, 8))
        plt.plot(equity_curve.index, equity_curve.values, linewidth=1.5, color='blue')
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
                'total_cumulative_return': float(cumulative_returns.iloc[-1]),
                'profit_factor': float(rets[rets > 0].sum() / abs(rets[rets < 0].sum())) if rets[rets < 0].sum() < 0 else float('inf'),
                'max_drawdown': float(drawdown.min()),
                'sharpe_ratio': float(rets.mean() / rets.std() * np.sqrt(8760)) if rets.std() > 0 else 0,
                'num_trades': len(rets[rets != 0]),
                'final_portfolio_value': float(equity_curve.iloc[-1])
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
        
        # Prepare in-sample metrics
        in_sample_metrics = {
            'total_cumulative_return': pair['total_cumulative_return'],
            'profit_factor': pair['profit_factor'],
            'max_drawdown': pair['max_drawdown'],
            'sharpe_ratio': pair['sharpe_ratio'],
            'num_trades': pair['num_trades']
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
    print(f"📊 Individual pair figures saved to: {FIGURES_DIR}")
    
    print(f"\n✅ Out-of-sample validation completed!")
    print(f"Ready for portfolio simulation: python simulate_portfolio_2024.py")


if __name__ == "__main__":
    main()