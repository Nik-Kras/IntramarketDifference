#!/usr/bin/env python3
"""
Test to understand how the original system handles trade types.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('/Users/nikitakrasnytskyi/Desktop/IntramarketDifference')

from run_all import run_pair, load_one_csv, extract_coin_name

def test_original_algorithm():
    """Test the original algorithm to understand its trade type behavior."""
    
    print("🔍 Testing original algorithm behavior...")
    
    # Load some test data
    ref_path = "/Users/nikitakrasnytskyi/Desktop/IntramarketDifference/data/BTCUSDT_IS.csv"
    traded_path = "/Users/nikitakrasnytskyi/Desktop/IntramarketDifference/data/ETHUSDT_IS.csv"
    
    ref_df = load_one_csv(ref_path)
    traded_df = load_one_csv(traded_path)
    
    print(f"📊 Loaded data: BTC (ref) {ref_df.shape}, ETH (traded) {traded_df.shape}")
    
    # Run original algorithm
    result = run_pair("BTC", "ETH", ref_df, traded_df)
    
    print(f"\n📈 Original algorithm results:")
    print(f"   Total Return: {result['Total Cummulative Return']:.4f}")
    print(f"   Profit Factor: {result['Profit Factor']:.4f}")
    print(f"   Number of Trades: {result['Number of Trades']}")
    print(f"   Max Drawdown: {result['Max DrawDown']:.4f}")
    print(f"   Sharpe Ratio: {result['Sharpe Ratio']:.4f}")
    
    # Now check what happens if we manually extract signals and check long vs short
    from run_all import cmma, threshold_revert_signal, align_frames
    from parameter_optimization_experiment.trades_from_signal import get_trades_from_signal
    
    ref_df, traded_df = align_frames(ref_df, traded_df)
    traded_df = traded_df.copy()
    traded_df["diff"] = np.log(traded_df["close"]).diff()
    traded_df["next_return"] = traded_df["diff"].shift(-1)
    
    # Calculate indicators
    ref_cmma = cmma(ref_df, 24, 168)
    trd_cmma = cmma(traded_df, 24, 168)
    intermarket_diff = trd_cmma - ref_cmma
    traded_df["sig"] = threshold_revert_signal(intermarket_diff, 0.25)
    
    # Get trades
    long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, traded_df["sig"].values)
    
    print(f"\n🔍 Signal analysis:")
    print(f"   Total signals: {len(traded_df['sig'])}")
    print(f"   Long positions: {(traded_df['sig'] == 1).sum()}")
    print(f"   Short positions: {(traded_df['sig'] == -1).sum()}")
    print(f"   Flat positions: {(traded_df['sig'] == 0).sum()}")
    
    print(f"\n📋 Trade analysis:")
    print(f"   All trades: {len(all_trades)}")
    print(f"   Long trades: {len(long_trades)}")
    print(f"   Short trades: {len(short_trades)}")
    
    # Calculate returns for each type
    rets_all = traded_df["sig"] * traded_df["next_return"]
    rets_all = rets_all.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Long-only returns
    long_mask = traded_df["sig"] == 1
    rets_long = rets_all[long_mask].dropna()
    
    # Short-only returns  
    short_mask = traded_df["sig"] == -1
    rets_short = rets_all[short_mask].dropna()
    
    print(f"\n📊 Return analysis:")
    print(f"   All returns: {len(rets_all)} periods")
    print(f"   Long returns: {len(rets_long)} periods")
    print(f"   Short returns: {len(rets_short)} periods")
    
    print(f"\n💰 Performance comparison:")
    
    # Calculate metrics for each
    def calc_metrics(rets):
        if len(rets) == 0:
            return {"total_return": np.nan, "profit_factor": np.nan}
        
        total_return = rets.cumsum().iloc[-1]
        simple_rets = np.exp(rets) - 1
        gains = simple_rets[simple_rets > 0].sum()
        losses = simple_rets[simple_rets < 0].abs().sum()
        profit_factor = gains / losses if losses > 0 else np.inf
        
        return {"total_return": total_return, "profit_factor": profit_factor}
    
    all_metrics = calc_metrics(rets_all)
    long_metrics = calc_metrics(rets_long)  
    short_metrics = calc_metrics(rets_short)
    
    print(f"   ALL:   Return={all_metrics['total_return']:.4f}, PF={all_metrics['profit_factor']:.4f}")
    print(f"   LONG:  Return={long_metrics['total_return']:.4f}, PF={long_metrics['profit_factor']:.4f}")
    print(f"   SHORT: Return={short_metrics['total_return']:.4f}, PF={short_metrics['profit_factor']:.4f}")
    
    # Compare with original result
    original_return = result['Total Cummulative Return']
    original_pf = result['Profit Factor']
    
    print(f"\n🔍 VERIFICATION:")
    print(f"   Original total return: {original_return:.4f}")
    print(f"   Calculated ALL return: {all_metrics['total_return']:.4f}")
    print(f"   Match: {abs(original_return - all_metrics['total_return']) < 0.001}")
    
    print(f"   Original PF: {original_pf:.4f}")
    print(f"   Calculated ALL PF: {all_metrics['profit_factor']:.4f}")
    print(f"   Match: {abs(original_pf - all_metrics['profit_factor']) < 0.001}")

if __name__ == "__main__":
    test_original_algorithm()