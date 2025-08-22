#!/usr/bin/env python3
"""
Test script to verify the equity curve bug fix.
"""

import sys
import os
sys.path.append('/Users/nikitakrasnytskyi/Desktop/IntramarketDifference')

from oos_backtest import run_oos_backtest_single_year, save_detailed_results_both_years
import json

def test_equity_fix():
    """Test the ZRX-FIL long pair that showed the discrepancy."""
    
    print("🔍 Testing equity curve fix for ZRX-FIL long pair...")
    
    # Test the pair that showed the bug
    trading_coin = "FIL"
    reference_coin = "ZRX" 
    trade_type = "long"
    
    # Test 2023 data
    print(f"\n📊 Testing {reference_coin}-{trading_coin} ({trade_type}) for 2023...")
    
    result_2023 = run_oos_backtest_single_year(trading_coin, reference_coin, trade_type, 2023)
    print(f"OOS Total Return (from backtest): {result_2023['oos_total_return']:.4f}")
    
    # Save detailed results (this will regenerate the equity curve)
    print(f"🔄 Regenerating equity curve with fix...")
    save_detailed_results_both_years(trading_coin, reference_coin, trade_type)
    
    # Read the JSON file to see the values
    json_file = f"oos_experiments/{reference_coin}_{trading_coin}_{trade_type}/oos_metrics_combined.json"
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        json_return_2023 = data['oos_metrics_2023']['total_cumulative_return']
        print(f"OOS Total Return (from JSON): {json_return_2023:.4f}")
        
        # Check if they match
        if abs(result_2023['oos_total_return'] - json_return_2023) < 0.0001:
            print("✅ JSON values match backtest calculation!")
        else:
            print("❌ JSON values still don't match!")
            print(f"   Difference: {abs(result_2023['oos_total_return'] - json_return_2023):.6f}")
    
    print(f"\n📈 Equity curve regenerated at: oos_experiments/{reference_coin}_{trading_coin}_{trade_type}/")
    print(f"🔍 Check the equity curve figure - it should now show returns consistent with the JSON total_cumulative_return")
    
    # Calculate what the final equity value should be
    final_equity_should_be = 1000 * (1 + result_2023['oos_total_return'])  # For small returns, approx
    final_equity_exact = 1000 * exp(result_2023['oos_total_return'])  # Exact calculation
    
    print(f"\n📊 EXPECTED EQUITY CURVE VALUES:")
    print(f"   Starting value: $1,000")
    print(f"   Final value (exact): ${final_equity_exact:.2f}")
    print(f"   Total return: {result_2023['oos_total_return']:.4f} ({result_2023['oos_total_return']*100:.2f}%)")

if __name__ == "__main__":
    from math import exp
    test_equity_fix()