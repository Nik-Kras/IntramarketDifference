#!/usr/bin/env python3
"""
Test the selection filters functionality (without minimum trades filter)
"""

import pandas as pd
from window_analyzer_fast import apply_selection_filters

def test_selection_filters():
    """Test the selection filters with sample data."""
    
    # Create sample data
    sample_data = pd.DataFrame([
        {'reference_coin': 'BTC', 'trading_coin': 'ETH', 'trading_type': 'both', 
         'sharpe_ratio': 2.5, 'max_drawdown': 0.3, 'num_trades': 50},  # Should pass all filters
        {'reference_coin': 'BTC', 'trading_coin': 'ADA', 'trading_type': 'both',
         'sharpe_ratio': 2.2, 'max_drawdown': 0.25, 'num_trades': 15},  # Should pass all filters
        {'reference_coin': 'ETH', 'trading_coin': 'ADA', 'trading_type': 'both',
         'sharpe_ratio': 1.8, 'max_drawdown': 0.2, 'num_trades': 100}, # Should fail Sharpe filter
        {'reference_coin': 'BTC', 'trading_coin': 'DOT', 'trading_type': 'both',
         'sharpe_ratio': 2.3, 'max_drawdown': 0.6, 'num_trades': 80},  # Should fail drawdown filter
        {'reference_coin': 'ETH', 'trading_coin': 'DOT', 'trading_type': 'both',
         'sharpe_ratio': 3.0, 'max_drawdown': 0.15, 'num_trades': 5},  # Should pass all filters
    ])
    
    print("🔬 Testing Selection Filters")
    print("=" * 50)
    print("Filter criteria:")
    print("  - Sharpe Ratio > 2.0")
    print("  - Max Drawdown < 50%")
    print("")
    
    selected, stats = apply_selection_filters(sample_data)
    
    print(f"Results: {len(selected)} pairs selected out of {len(sample_data)}")
    print(f"  After Sharpe filter: {stats['after_sharpe_filter']} pairs")
    print(f"  After Drawdown filter: {stats['after_drawdown_filter']} pairs")
    
    if len(selected) > 0:
        print("\nSelected pairs:")
        for _, row in selected.iterrows():
            print(f"  {row['reference_coin']}-{row['trading_coin']}: "
                  f"Sharpe={row['sharpe_ratio']:.1f}, DD={row['max_drawdown']:.1%}, "
                  f"Trades={row['num_trades']}")
    else:
        print("\nNo pairs selected!")
    
    print("\nExpected results:")
    print("  BTC-ETH: PASS (Sharpe=2.5>2.0, DD=30%<50%)")
    print("  BTC-ADA: PASS (Sharpe=2.2>2.0, DD=25%<50%)")
    print("  ETH-ADA: FAIL (Sharpe=1.8<2.0)")
    print("  BTC-DOT: FAIL (DD=60%>50%)")
    print("  ETH-DOT: PASS (Sharpe=3.0>2.0, DD=15%<50%)")

if __name__ == "__main__":
    test_selection_filters()