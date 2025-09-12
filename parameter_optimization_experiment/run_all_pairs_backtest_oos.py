#!/usr/bin/env python3
"""
Out-of-Sample All Pairs Backtesting for Parameter Optimization Experiment

Runs comprehensive backtesting on all coin pairs using Out-of-Sample data (2024+).
Generates both A->B and B->A pairs and saves detailed trade data for portfolio simulation.

This script creates the trade data that window_portfolio_simulator.py will load.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Import core trading functions
from core import (
    cmma, threshold_revert_signal, load_csv_with_cache, align_frames,
    LOOKBACK, ATR_LOOKBACK, THRESHOLD, MIN_OVERLAP
)

# Add parent directory to path to import trading functions
sys.path.append('..')
from parameter_optimization_experiment.trades_from_signal import get_trades_from_signal

# Configuration
DATA_DIR = "data/out_of_sample"
TRADES_DIR = "out_of_sample/trades"
MIN_OVERLAP = 100  # Lower minimum for OOS data (shorter time period)

# Global cache for loaded data
_data_cache = {}

def load_oos_csv(coin: str) -> pd.DataFrame:
    """Load out-of-sample CSV file for a coin with caching."""
    return load_csv_with_cache(coin, DATA_DIR, "oos", _data_cache)

def get_available_oos_coins() -> list:
    """Get list of available coins from out-of-sample data."""
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_oos.csv"))
    coins = [os.path.basename(f).replace("_oos.csv", "") for f in csv_files]
    return sorted(coins)

def generate_all_pairs(coins: list) -> list:
    """Generate all unique directional pairs (including A,B and B,A)."""
    pairs = []
    for ref_coin in coins:
        for trading_coin in coins:
            if ref_coin != trading_coin:
                pairs.append((ref_coin, trading_coin))
    return pairs

def run_oos_pair_backtest_and_save_trades(ref_coin: str, trading_coin: str) -> dict:
    """
    Run OOS backtest for a single pair and save detailed trade data.
    
    Args:
        ref_coin: Reference coin (predictor)
        trading_coin: Coin being traded
        
    Returns:
        Dictionary with pair results or None if failed
    """
    
    # Load OOS data
    ref_df = load_oos_csv(ref_coin)
    traded_df = load_oos_csv(trading_coin)
    
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
        
        # Create output directory
        pair_dir = os.path.join(TRADES_DIR, f"{ref_coin}_{trading_coin}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Convert and save each trading type
        trade_types = {
            'longs': long_trades,
            'shorts': short_trades, 
            'both': all_trades
        }
        
        saved_counts = {}
        
        for trading_type, trades_df in trade_types.items():
            if len(trades_df) > 0:
                # Convert to JSON format
                trades_list = []
                for entry_time, trade in trades_df.iterrows():
                    # Calculate log return for consistency
                    if trade['type'] == 1:  # Long trade
                        log_return = np.log(trade['exit_price'] / trade['entry_price'])
                    else:  # Short trade
                        log_return = np.log(trade['entry_price'] / trade['exit_price'])
                    
                    trades_list.append({
                        'time_entered': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'time_exited': trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        'log_return': float(log_return),
                        'trade_type': 'long' if trade['type'] == 1 else 'short'
                    })
                
                # Save to JSON file
                output_file = os.path.join(pair_dir, f"{trading_type}_trades.json")
                with open(output_file, 'w') as f:
                    json.dump(trades_list, f, indent=2)
                
                saved_counts[trading_type] = len(trades_list)
            else:
                saved_counts[trading_type] = 0
        
        return {
            'ref_coin': ref_coin,
            'trading_coin': trading_coin,
            'longs_trades': saved_counts.get('longs', 0),
            'shorts_trades': saved_counts.get('shorts', 0),
            'total_trades': saved_counts.get('both', 0),
            'data_overlap': len(ref_df)
        }
        
    except Exception as e:
        print(f"Error processing {ref_coin}->{trading_coin}: {str(e)}")
        return None

def main():
    """Main execution function."""
    
    print("🚀 Out-of-Sample All Pairs Backtesting")
    print("=" * 50)
    print("Generating trade data for portfolio simulation...")
    
    # Create output directory
    os.makedirs(TRADES_DIR, exist_ok=True)
    
    # Get available coins
    coins = get_available_oos_coins()
    print(f"📊 Found {len(coins)} coins with OOS data")
    
    # Generate all pairs
    all_pairs = generate_all_pairs(coins)
    print(f"🔄 Processing {len(all_pairs):,} directional pairs...")
    
    # Process all pairs
    results = []
    successful_pairs = 0
    failed_pairs = 0
    
    for ref_coin, trading_coin in tqdm(all_pairs, desc="Processing pairs"):
        result = run_oos_pair_backtest_and_save_trades(ref_coin, trading_coin)
        
        if result:
            results.append(result)
            successful_pairs += 1
        else:
            failed_pairs += 1
    
    print(f"\n📊 BACKTESTING COMPLETE:")
    print(f"   Successful pairs: {successful_pairs:,}")
    print(f"   Failed pairs: {failed_pairs:,}")
    print(f"   Success rate: {successful_pairs/(successful_pairs+failed_pairs)*100:.1f}%")
    
    # Save summary
    if results:
        results_df = pd.DataFrame(results)
        summary_file = os.path.join(TRADES_DIR, "oos_backtest_summary.csv")
        results_df.to_csv(summary_file, index=False)
        
        print(f"   Total trades generated: {results_df['total_trades'].sum():,}")
        print(f"   Long trades: {results_df['longs_trades'].sum():,}")
        print(f"   Short trades: {results_df['shorts_trades'].sum():,}")
        print(f"📁 Trade data saved to: {TRADES_DIR}/")
        print(f"📋 Summary saved to: {summary_file}")
    
    print(f"\n✅ Ready for portfolio simulation!")
    print(f"📂 Trade files: {TRADES_DIR}/{{RefCoin}}_{{TradingCoin}}/{{type}}_trades.json")

if __name__ == "__main__":
    main()