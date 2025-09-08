#!/usr/bin/env python3
"""
In-Sample Backtesting Experiment (2022-2024)

Tests all possible coin pair combinations using the intermarket difference strategy.
Uses the exact same algorithm as the original run_all.py script.

Period: 2022-01-01 to 2024-01-01
Algorithm: CMMA-based intermarket difference mean reversion
Output: Complete results with permutation-style distributions
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Import core trading functions
from core import (
    cmma, threshold_revert_signal, load_csv_with_cache, align_frames,
    calculate_metrics_from_trades, calculate_max_drawdown, run_pair_backtest_core,
    generate_equity_curve_from_trades as core_generate_equity_curve, get_available_coins, create_trade_record,
    LOOKBACK, ATR_LOOKBACK, THRESHOLD, MIN_OVERLAP
)

# Add parent directory to path to import trading functions
sys.path.append('..')
from parameter_optimization_experiment.trades_from_signal import get_trades_from_signal

# -----------------------
# Configuration
# -----------------------
DATA_DIR = "data/in_sample"
RESULTS_DIR = "results/in_sample"
TRADES_DIR = os.path.join(RESULTS_DIR, "trades")
DISTRIBUTIONS_DIR = os.path.join(RESULTS_DIR, "distributions")

# Algorithm parameters imported from core
# LOOKBACK, ATR_LOOKBACK, THRESHOLD, MIN_OVERLAP are now imported from core.py

# Output files
OUT_CSV = os.path.join(RESULTS_DIR, "in_sample_results.csv")


# -----------------------
# Trading Algorithm (imported from core.py)
# -----------------------
# cmma() and threshold_revert_signal() functions are now imported from core.py


def load_in_sample_csv(coin: str) -> pd.DataFrame:
    """Load in-sample CSV file for a coin with caching."""
    return load_csv_with_cache(coin, DATA_DIR, "is", _data_cache)


# calculate_metrics_from_trades() function is now imported from core.py


def run_pair_backtest(ref_coin: str, trading_coin: str) -> list:
    """
    Run backtest for a single pair with three trading types: longs, shorts, both.
    Returns list of metrics dictionaries (one for each trading type).
    """
    
    # Load data
    ref_df = load_in_sample_csv(ref_coin)
    traded_df = load_in_sample_csv(trading_coin)
    
    # Use core backtesting function
    return run_pair_backtest_core(ref_df, traded_df, ref_coin, trading_coin, MIN_OVERLAP)


def save_detailed_trades(ref_coin: str, trading_coin: str, trading_type: str):
    """Save detailed trade data for a pair and trading type."""
    
    ref_df = load_in_sample_csv(ref_coin)
    traded_df = load_in_sample_csv(trading_coin)
    
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
            # Calculate log return
            if trade['type'] == 1:  # Long trade
                log_return = np.log(trade['exit_price'] / trade['entry_price'])
            else:  # Short trade  
                log_return = np.log(trade['entry_price'] / trade['exit_price'])
            
            trade_record = create_trade_record(trade, entry_time)
            trades_data.append(trade_record)
        
        # Save to JSON
        trades_file = os.path.join(trade_dir, f"{trading_coin}_{ref_coin}_{trading_type}_trades.json")
        with open(trades_file, 'w') as f:
            json.dump(trades_data, f, indent=2)
            
    except Exception:
        pass


def load_trades_and_generate_equity_curve(ref_coin: str, trading_coin: str, trading_type: str = 'both') -> pd.Series:
    """Load trade data from JSON and generate equity curve."""
    
    trades_file = os.path.join(TRADES_DIR, trading_coin, f"{trading_coin}_{ref_coin}_{trading_type}_trades.json")
    
    if not os.path.exists(trades_file):
        # Try to generate trades if not saved yet
        save_detailed_trades(ref_coin, trading_coin, trading_type)
        if not os.path.exists(trades_file):
            return None
    
    try:
        # Load trades
        with open(trades_file, 'r') as f:
            trades_data = json.load(f)
        
        if not trades_data:
            return None
        
        # Convert to DataFrame for core function
        trades_list = []
        for trade in trades_data:
            if trade['time_exited'] and trade['log_return'] is not None:
                # Create proper format for core function
                trades_list.append({
                    'exit_time': pd.to_datetime(trade['time_exited']),
                    'entry_price': 100.0,  # Dummy value - log_return is already calculated
                    'exit_price': 100.0 * np.exp(trade['log_return']),  # Reconstruct from log return
                    'type': 1 if trade['trade_type'] == 'long' else -1
                })
        
        if not trades_list:
            return None
        
        # Convert to DataFrame and call core function
        trades_df = pd.DataFrame(trades_list).set_index('exit_time')
        equity_curve = core_generate_equity_curve(trades_df)
        
        return equity_curve
        
    except Exception:
        return None


# calculate_max_drawdown() function is now imported from core.py


def get_available_in_sample_coins() -> list:
    """Get list of available coins from in-sample data."""
    return get_available_coins(DATA_DIR, "is")


# Global cache for loaded data to avoid repeated file I/O
_data_cache = {}


def create_distribution_plots(results_df: pd.DataFrame):
    """Create distribution plots similar to permutations directory."""
    
    print(f"\n📊 Creating distribution visualizations...")
    
    os.makedirs(DISTRIBUTIONS_DIR, exist_ok=True)
    
    # Group by trading coin and trading type
    trading_coins = results_df['trading_coin'].unique()
    trading_types = results_df['trading_type'].unique()
    
    for trading_coin in tqdm(trading_coins, desc="Creating distributions"):
        
        for trading_type in trading_types:
            coin_results = results_df[
                (results_df['trading_coin'] == trading_coin) & 
                (results_df['trading_type'] == trading_type)
            ]
            
            if len(coin_results) < 5:  # Skip coins with too few pairs
                continue
            
            coin_dir = os.path.join(DISTRIBUTIONS_DIR, f"{trading_coin}_{trading_type}")
            os.makedirs(coin_dir, exist_ok=True)
            
            # Profit Factor Distribution
            plt.figure(figsize=(12, 8))
            plt.hist(coin_results['profit_factor'], bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(coin_results['profit_factor'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {coin_results["profit_factor"].mean():.2f}')
            plt.title(f'Profit Factor Distribution - {trading_coin} ({trading_type})')
            plt.xlabel('Profit Factor')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(coin_dir, 'profit_factor_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Drawdown Distribution
            plt.figure(figsize=(12, 8))
            plt.hist(coin_results['max_drawdown'], bins=50, alpha=0.7, color='red', edgecolor='black')
            plt.axvline(coin_results['max_drawdown'].mean(), color='blue', linestyle='--',
                       label=f'Mean: {coin_results["max_drawdown"].mean():.2%}')
            plt.title(f'Max Drawdown Distribution - {trading_coin} ({trading_type})')
            plt.xlabel('Max Drawdown')
            plt.ylabel('Frequency')
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(coin_dir, 'drawdown_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Sharpe Ratio Distribution
            plt.figure(figsize=(12, 8))
            plt.hist(coin_results['sharpe_ratio'], bins=50, alpha=0.7, color='green', edgecolor='black')
            plt.axvline(coin_results['sharpe_ratio'].mean(), color='red', linestyle='--',
                       label=f'Mean: {coin_results["sharpe_ratio"].mean():.2f}')
            plt.title(f'Sharpe Ratio Distribution - {trading_coin} ({trading_type})')
            plt.xlabel('Sharpe Ratio')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(coin_dir, 'sharpe_ratio_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Top performers equity curves (top 10 by profit factor)
            top_performers = coin_results.nlargest(10, 'profit_factor')
            
            # Generate real equity curves from trades
            plt.figure(figsize=(15, 10))
            curves_plotted = 0
            for _, row in top_performers.iterrows():
                ref_coin = row['reference_coin']
                
                # Use the corrected function to load trades and generate equity curve
                try:
                    equity_curve = load_trades_and_generate_equity_curve(ref_coin, trading_coin, trading_type)
                    if equity_curve is not None and len(equity_curve) > 0:
                        plt.plot(equity_curve.index, equity_curve, 
                                label=f"{ref_coin} (PF: {row['profit_factor']:.2f})", 
                                linewidth=2, alpha=0.8)
                        curves_plotted += 1
                except Exception:
                    continue
            
            # Only save if we have at least one curve to plot
            if curves_plotted == 0:
                plt.close()
                continue
            
            plt.title(f'Top 10 Performers Equity Curves - {trading_coin} ({trading_type})')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(coin_dir, 'top_performers_equity_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Reference coin metrics
            ref_coin_metrics = {}
            for ref_coin in coin_results['reference_coin'].unique():
                ref_data = coin_results[coin_results['reference_coin'] == ref_coin]
                ref_coin_metrics[ref_coin] = {
                    'num_pairs': len(ref_data),
                    'avg_profit_factor': float(ref_data['profit_factor'].mean()),
                    'avg_total_return': float(ref_data['total_cumulative_return'].mean()),
                    'avg_max_drawdown': float(ref_data['max_drawdown'].mean()),
                    'avg_sharpe_ratio': float(ref_data['sharpe_ratio'].mean()),
                    'avg_volatility': float(ref_data['volatility'].mean()),
                    'avg_num_trades': float(ref_data['num_trades'].mean())
                }
            
            # Save summary statistics
            summary_stats = {
                'trading_coin': trading_coin,
                'trading_type': trading_type,
                'num_pairs': len(coin_results),
                'avg_profit_factor': float(coin_results['profit_factor'].mean()),
                'avg_total_return': float(coin_results['total_cumulative_return'].mean()),
                'avg_max_drawdown': float(coin_results['max_drawdown'].mean()),
                'avg_sharpe_ratio': float(coin_results['sharpe_ratio'].mean()),
                'best_profit_factor': float(coin_results['profit_factor'].max()),
                'best_reference_coin': coin_results.loc[coin_results['profit_factor'].idxmax(), 'reference_coin'],
                'reference_coin_metrics': ref_coin_metrics
            }
            
            with open(os.path.join(coin_dir, 'summary_stats.json'), 'w') as f:
                json.dump(summary_stats, f, indent=2)


def main():
    """Main execution function."""
    
    print("🚀 In-Sample Backtesting Experiment (2022-2024)")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TRADES_DIR, exist_ok=True)
    
    # Get available coins
    coins = get_available_in_sample_coins()
    
    if not coins:
        print("❌ No coin data found in data/in_sample directory!")
        print("Run prepare_data.py first to create the data files.")
        return
    
    print(f"📊 Found {len(coins)} coins")
    print(f"💫 Testing ~{len(coins) * (len(coins) - 1)} pair combinations")
    print(f"⏱️  Estimated time: ~{len(coins) * (len(coins) - 1) * 0.5 / 60:.0f} minutes")
    
    # Run backtests
    results = []
    successful_pairs = 0
    failed_pairs = 0
    
    print(f"\n🔄 Running backtests...")
    
    # Pre-load all data to cache
    print("📂 Pre-loading data to cache...")
    for coin in tqdm(coins, desc="Loading data"):
        load_in_sample_csv(coin)
    
    print("🔄 Starting pair backtests...")
    
    for i, ref_coin in enumerate(coins):
        print(f"\n📊 Processing reference coin {i+1}/{len(coins)}: {ref_coin}")
        
        for trading_coin in tqdm(coins, desc=f"Trading vs {ref_coin}", leave=False):
            if ref_coin == trading_coin:
                continue
            
            # Run backtest (returns list of results for 3 trading types)
            pair_results = run_pair_backtest(ref_coin, trading_coin)
            
            if pair_results:  # List of 3 results (both, longs, shorts)
                results.extend(pair_results)
                successful_pairs += len(pair_results)
                
                # Save detailed trades for each trading type
                for result in pair_results:
                    save_detailed_trades(ref_coin, trading_coin, result['trading_type'])
            else:
                failed_pairs += 1
        
        # Save intermediate results every 10 reference coins
        if (i + 1) % 10 == 0 and results:
            temp_df = pd.DataFrame(results)
            temp_file = os.path.join(RESULTS_DIR, f"partial_results_{i+1}_coins.csv")
            temp_df.to_csv(temp_file, index=False, float_format='%.6f')
            print(f"💾 Intermediate results saved: {temp_file}")
    
    print(f"\n🔄 Processed all {len(coins)} reference coins")
    
    if not results:
        print("❌ No successful backtests!")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(OUT_CSV, index=False, float_format='%.6f')
    
    # Create distribution plots
    create_distribution_plots(results_df)
    
    # Summary statistics
    print(f"\n📊 BACKTESTING RESULTS:")
    print(f"   ✅ Successful pairs: {successful_pairs:,}")
    print(f"   ❌ Failed pairs: {failed_pairs:,}")
    print(f"   📈 Average Profit Factor: {results_df['profit_factor'].mean():.3f}")
    print(f"   📉 Average Max Drawdown: {results_df['max_drawdown'].mean():.2%}")
    print(f"   ⚡ Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.3f}")
    print(f"   📊 Average Return: {results_df['total_cumulative_return'].mean():.3f}")
    
    # Top performers
    top_pairs = results_df.nlargest(10, 'profit_factor')
    print(f"\n🏆 TOP 10 PAIRS BY PROFIT FACTOR:")
    print(f"   {'Rank':<4} {'Ref Coin':<8} {'Trading Coin':<12} {'PF':<8} {'Return':<9} {'Drawdown':<10} {'Sharpe':<8}")
    print(f"   {'-'*4} {'-'*8} {'-'*12} {'-'*8} {'-'*9} {'-'*10} {'-'*8}")
    
    for i, (_, row) in enumerate(top_pairs.iterrows(), 1):
        print(f"   {i:<4} {row['reference_coin']:<8} {row['trading_coin']:<12} "
              f"{row['profit_factor']:<8.2f} {row['total_cumulative_return']:<9.2f} "
              f"{row['max_drawdown']:<10.2%} {row['sharpe_ratio']:<8.2f}")
    
    print(f"\n💾 Results saved to: {OUT_CSV}")
    print(f"📊 Distributions saved to: {DISTRIBUTIONS_DIR}")
    print(f"📁 Detailed trades saved to: {TRADES_DIR}")
    
    # Clean up temporary CSV files
    print(f"\n🧹 Cleaning up temporary files...")
    temp_files = glob.glob(os.path.join(RESULTS_DIR, "partial_results_*_coins.csv"))
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"   🗑️  Deleted: {os.path.basename(temp_file)}")
        except Exception:
            pass
    
    print(f"\n✅ In-sample experiment completed!")
    print(f"Ready for filtering step: python apply_custom_filters.py")


if __name__ == "__main__":
    main()