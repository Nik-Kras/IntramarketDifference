#!/usr/bin/env python3
"""
All Pairs Backtesting for Parameter Optimization Experiment

Runs comprehensive backtesting on all coin pairs using In-Sample data.
Generates both A->B and B->A pairs and saves detailed trade data.

Key differences from previous scripts:
- Uses only In-Sample data for backtesting
- Generates all unique directional pairs (A,B) and (B,A) 
- Saves detailed trade records for each pair
- Focuses on trade-level data collection for parameter optimization
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import core trading functions
from core import (
    cmma, threshold_revert_signal, load_csv_with_cache, align_frames,
    calculate_metrics_from_trades, run_pair_backtest_core, create_trade_record,
    LOOKBACK, ATR_LOOKBACK, THRESHOLD, MIN_OVERLAP
)

from trades_from_signal import get_trades_from_signal

# Configuration
DATA_DIR = "data/in_sample"
TRADES_DIR = "in_sample/trades"
MIN_OVERLAP = 500  # Minimum data points for pair analysis

# Global cache for loaded data
_data_cache = {}

def load_in_sample_csv(coin: str) -> pd.DataFrame:
    """Load in-sample CSV file for a coin with caching."""
    return load_csv_with_cache(coin, DATA_DIR, "is", _data_cache)

def get_available_coins() -> list:
    """Get list of available coins from in-sample data."""
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_is.csv"))
    coins = [os.path.basename(f).replace("_is.csv", "") for f in csv_files]
    return sorted(coins)

def generate_all_pairs(coins: list) -> list:
    """Generate all unique directional pairs (including A,B and B,A)."""
    pairs = []
    for ref_coin in coins:
        for trading_coin in coins:
            if ref_coin != trading_coin:
                pairs.append((ref_coin, trading_coin))
    return pairs

def run_pair_backtest_and_save_trades(ref_coin: str, trading_coin: str) -> dict:
    """
    Run backtest for a single pair and save detailed trade data.
    
    Args:
        ref_coin: Reference coin (predictor)
        trading_coin: Coin being traded
        
    Returns:
        Dictionary with pair results or None if failed
    """
    
    # Load data
    ref_df = load_in_sample_csv(ref_coin)
    traded_df = load_in_sample_csv(trading_coin)
    
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
        
        # Get detailed trades
        long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, traded_df["sig"].values)
        
        # Save detailed trade data for each trading type
        pair_results = {}
        
        for trading_type, trades_df in [('longs', long_trades), ('shorts', short_trades), ('both', all_trades)]:
            
            # Create trades directory
            trade_dir = os.path.join(TRADES_DIR, f"{ref_coin}_{trading_coin}")
            os.makedirs(trade_dir, exist_ok=True)
            
            # Convert trades to JSON format
            trades_data = []
            for entry_time, trade in trades_df.iterrows():
                # Calculate log return
                if trade['type'] == 1:  # Long trade
                    log_return = np.log(trade['exit_price'] / trade['entry_price'])
                else:  # Short trade  
                    log_return = np.log(trade['entry_price'] / trade['exit_price'])
                
                trade_record = {
                    'time_entered': entry_time.isoformat() if pd.notna(entry_time) else None,
                    'time_exited': trade['exit_time'].isoformat() if pd.notna(trade['exit_time']) else None,
                    'log_return': float(log_return) if pd.notna(log_return) else None,
                    'trade_type': 'long' if trade['type'] == 1 else 'short'
                }
                trades_data.append(trade_record)
            
            # Save trades JSON
            trades_file = os.path.join(trade_dir, f"{trading_type}_trades.json")
            with open(trades_file, 'w') as f:
                json.dump(trades_data, f, indent=2)
            
            # Calculate metrics for this trading type
            data_points = len(traded_df)
            metrics = calculate_metrics_from_trades(trades_df, data_points)
            metrics.update({
                'reference_coin': ref_coin,
                'trading_coin': trading_coin,
                'trading_type': trading_type,
                'data_points': data_points,
                'trades_saved': len(trades_data)
            })
            
            pair_results[trading_type] = metrics
        
        return pair_results
        
    except Exception as e:
        print(f"Error processing {ref_coin}->{trading_coin}: {str(e)}")
        return None

def main():
    """Main execution function."""
    
    print("🚀 All Pairs Backtesting - Parameter Optimization Experiment")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(TRADES_DIR, exist_ok=True)
    
    # Get available coins
    coins = get_available_coins()
    
    if not coins:
        print("❌ No coin data found in data/in_sample directory!")
        print("Run split_data.py first to create the data files.")
        return
    
    print(f"📊 Found {len(coins)} coins")
    
    # Generate all pairs (including directional: A,B and B,A)
    all_pairs = generate_all_pairs(coins)
    print(f"💫 Testing {len(all_pairs)} directional pair combinations")
    print(f"⏱️  Estimated time: ~{len(all_pairs) * 0.5 / 60:.0f} minutes")
    
    # Pre-load all data to cache
    print(f"\n📂 Pre-loading {len(coins)} coins to cache...")
    for coin in tqdm(coins, desc="Loading data"):
        load_in_sample_csv(coin)
    
    print("🔄 Starting pair backtests...")
    
    # Track results
    all_results = []
    successful_pairs = 0
    failed_pairs = 0
    
    # Process all pairs
    for i, (ref_coin, trading_coin) in enumerate(tqdm(all_pairs, desc="Processing pairs")):
        
        # Run backtest and save trades
        pair_results = run_pair_backtest_and_save_trades(ref_coin, trading_coin)
        
        if pair_results:
            # Extract results for each trading type
            for trading_type, metrics in pair_results.items():
                all_results.append(metrics)
            
            successful_pairs += 1
        else:
            failed_pairs += 1
        
        # Progress update every 1000 pairs
        if (i + 1) % 1000 == 0:
            print(f"\n📊 Progress: {i+1}/{len(all_pairs)} pairs processed")
            print(f"   ✅ Successful: {successful_pairs}")
            print(f"   ❌ Failed: {failed_pairs}")
            
            # Save intermediate results
            if all_results:
                temp_df = pd.DataFrame(all_results)
                temp_file = f"partial_results_{i+1}_pairs.csv"
                temp_df.to_csv(temp_file, index=False, float_format='%.6f')
                print(f"   💾 Intermediate results saved: {temp_file}")
    
    print(f"\n🔄 Processed all {len(all_pairs)} pairs")
    
    if not all_results:
        print("❌ No successful backtests!")
        return
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_file = "all_pairs_backtest_results.csv"
    results_df.to_csv(results_file, index=False, float_format='%.6f')
    
    # Summary statistics
    print(f"\n📊 BACKTESTING RESULTS:")
    print(f"   ✅ Successful pairs: {successful_pairs:,}")
    print(f"   ❌ Failed pairs: {failed_pairs:,}")
    print(f"   📊 Total result records: {len(all_results):,}")
    print(f"   📈 Average Profit Factor: {results_df['profit_factor'].mean():.3f}")
    print(f"   📉 Average Max Drawdown: {results_df['max_drawdown'].mean():.2%}")
    print(f"   ⚡ Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.3f}")
    print(f"   🔄 Average Trades per Pair: {results_df['num_trades'].mean():.1f}")
    
    # Top performers by trading type
    for trading_type in ['both', 'longs', 'shorts']:
        type_results = results_df[results_df['trading_type'] == trading_type]
        if len(type_results) > 0:
            top_pairs = type_results.nlargest(5, 'profit_factor')
            print(f"\n🏆 TOP 5 PAIRS - {trading_type.upper()}:")
            print(f"   {'Ref Coin':<8} {'Trading Coin':<12} {'PF':<8} {'Return':<9} {'Trades':<7} {'Sharpe':<8}")
            print(f"   {'-'*8} {'-'*12} {'-'*8} {'-'*9} {'-'*7} {'-'*8}")
            
            for _, row in top_pairs.iterrows():
                print(f"   {row['reference_coin']:<8} {row['trading_coin']:<12} "
                      f"{row['profit_factor']:<8.2f} {row['total_cumulative_return']:<9.2f} "
                      f"{row['num_trades']:<7.0f} {row['sharpe_ratio']:<8.2f}")
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"📁 Detailed trades saved to: {TRADES_DIR}/")
    
    # Trade data statistics
    total_trades_saved = results_df['trades_saved'].sum()
    avg_trades_per_pair = results_df['trades_saved'].mean()
    
    print(f"\n📈 TRADE DATA STATISTICS:")
    print(f"   📊 Total trades saved: {total_trades_saved:,}")
    print(f"   📊 Average trades per pair: {avg_trades_per_pair:.1f}")
    print(f"   📁 Trade files organized by pair in: {TRADES_DIR}/")
    
    # Clean up temporary files
    print(f"\n🧹 Cleaning up temporary files...")
    temp_files = glob.glob("partial_results_*_pairs.csv")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"   🗑️  Deleted: {temp_file}")
        except Exception:
            pass
    
    print(f"\n✅ All pairs backtesting completed!")
    print(f"📊 Ready for parameter optimization analysis")

if __name__ == "__main__":
    main()