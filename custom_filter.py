#!/usr/bin/env python3
"""
Custom Filter Script with User-Specified Parameters

Applies the following filters:
- Sharpe Ratio > 1.0
- max_drawdown better than -75% (> -0.75)
- last_year_drawdown better than -40% (> -0.40)
- PF_quantile is in top 5% (<= 5.0)
- DD_quantile is in top 5% (<= 5.0)
"""

import sys
import os
sys.path.append('.')
from filter_pairs import PairFilter

def main():
    """Run custom filter with user-specified parameters."""
    
    print("🎯 Running Custom Filter with User Parameters")
    print("=" * 60)
    
    # Initialize filter
    filter_engine = PairFilter("all_pairs_metrics.csv")
    
    # Define custom filters based on user requirements
    custom_filters = [
        {
            'column': 'sharpe_ratio',
            'operator': '>',
            'value': 1.0,
            'name': 'Sharpe Ratio > 1.0'
        },
        {
            'column': 'max_drawdown',
            'operator': '>',
            'value': -0.75,  # Better than -75%
            'name': 'Max Drawdown better than -75%'
        },
        {
            'column': 'last_year_drawdown',
            'operator': '>',
            'value': -0.40,  # Better than -40%
            'name': 'Last Year Drawdown better than -40%'
        },
        {
            'column': 'profit_factor_quantile',
            'operator': '<=',
            'value': 5.0,  # Top 5%
            'name': 'PF Quantile in top 5%'
        },
        {
            'column': 'drawdown_quantile',
            'operator': '<=',
            'value': 5.0,  # Top 5%
            'name': 'DD Quantile in top 5%'
        }
    ]
    
    print("\n📋 Filter Parameters:")
    print("-" * 30)
    for i, f in enumerate(custom_filters, 1):
        print(f"{i}. {f['name']}: {f['column']} {f['operator']} {f['value']}")
    
    # Apply the filters
    result = filter_engine.apply_filters(custom_filters, "user_custom_filter")
    
    print(f"\n📊 Custom Filter Summary:")
    print(f"   Total pairs found: {len(result)}")
    if len(result) > 0:
        print(f"   Best profit factor: {result['profit_factor'].max():.2f}")
        print(f"   Best return: {result['total_cumulative_return'].max():.1f}x")
        print(f"   Best drawdown: {result['max_drawdown'].max():.1%}")
        print(f"   Average Sharpe: {result['sharpe_ratio'].mean():.2f}")
    
    return result

if __name__ == "__main__":
    main()