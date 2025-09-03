#!/usr/bin/env python3
"""
Debug Rolling Results - Compare pair metrics across windows

Analyzes pair-level metrics to identify if different windows are producing identical results.
This helps debug issues where portfolio performance doesn't vary between window lengths.
"""

import os
import pandas as pd
from pathlib import Path

def compare_period_windows(period_dir: str) -> None:
    """Compare pair metrics across different windows within a period."""
    
    period_path = Path(period_dir)
    period_name = period_path.name
    
    print(f"\n🔍 Analyzing period: {period_name}")
    print("=" * 60)
    
    window_dirs = [d for d in period_path.iterdir() if d.is_dir() and d.name.endswith('mo')]
    window_dirs.sort()
    
    if len(window_dirs) < 2:
        print("⚠️  Not enough windows to compare")
        return
    
    # Load metrics for each window
    window_metrics = {}
    
    for window_dir in window_dirs:
        window_name = window_dir.name
        
        # Check for pair summary files
        summary_file = window_dir / 'selected_pairs_summary.csv'
        oos_metrics_file = window_dir / 'pair_oos_metrics.csv'
        
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            window_metrics[window_name] = {
                'summary_file': summary_file,
                'pairs': len(df),
                'summary_hash': pd.util.hash_pandas_object(df.sort_values(['reference_coin', 'trading_coin'])).sum()
            }
            
            print(f"  {window_name}: {len(df)} pairs")
            
            # Show top 5 pairs for visual comparison
            top_pairs = df.head(5)[['reference_coin', 'trading_coin', 'sharpe_ratio']].to_string(index=False)
            print(f"    Top 5 pairs:\n{top_pairs}")
            
        else:
            print(f"  {window_name}: ❌ No summary file found")
    
    # Compare hashes to detect identical results
    print(f"\n📊 Hash Comparison:")
    hashes = {w: data['summary_hash'] for w, data in window_metrics.items()}
    
    # Group windows by identical hashes
    hash_groups = {}
    for window, hash_val in hashes.items():
        if hash_val not in hash_groups:
            hash_groups[hash_val] = []
        hash_groups[hash_val].append(window)
    
    for hash_val, windows in hash_groups.items():
        if len(windows) > 1:
            print(f"  ❌ IDENTICAL: {', '.join(windows)} (hash: {hash_val})")
        else:
            print(f"  ✅ UNIQUE: {windows[0]} (hash: {hash_val})")
    
    # Detailed comparison of first two windows
    if len(window_dirs) >= 2:
        print(f"\n🔬 Detailed Comparison: {window_dirs[0].name} vs {window_dirs[1].name}")
        
        try:
            df1 = pd.read_csv(window_dirs[0] / 'selected_pairs_summary.csv')
            df2 = pd.read_csv(window_dirs[1] / 'selected_pairs_summary.csv')
            
            # Sort for proper comparison
            df1_sorted = df1.sort_values(['reference_coin', 'trading_coin']).reset_index(drop=True)
            df2_sorted = df2.sort_values(['reference_coin', 'trading_coin']).reset_index(drop=True)
            
            # Check if pairs are identical
            pairs1 = set(zip(df1_sorted['reference_coin'], df1_sorted['trading_coin']))
            pairs2 = set(zip(df2_sorted['reference_coin'], df2_sorted['trading_coin']))
            
            common_pairs = pairs1 & pairs2
            unique_to_1 = pairs1 - pairs2
            unique_to_2 = pairs2 - pairs1
            
            print(f"  Common pairs: {len(common_pairs)}")
            print(f"  Unique to {window_dirs[0].name}: {len(unique_to_1)}")
            print(f"  Unique to {window_dirs[1].name}: {len(unique_to_2)}")
            
            if unique_to_1:
                print(f"    Examples unique to {window_dirs[0].name}: {list(unique_to_1)[:3]}")
            if unique_to_2:
                print(f"    Examples unique to {window_dirs[1].name}: {list(unique_to_2)[:3]}")
                
        except Exception as e:
            print(f"  Error in detailed comparison: {e}")

def main():
    """Main analysis function."""
    
    results_dir = "results/rolling_experiments"
    
    print("🐛 Rolling Results Debug Analysis")
    print("=" * 50)
    print("Comparing pair metrics across windows to detect duplicate results")
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Find all period directories
    results_path = Path(results_dir)
    period_dirs = [d for d in results_path.iterdir() 
                   if d.is_dir() and 'IS' in d.name and 'OOS' in d.name]
    
    if not period_dirs:
        print("❌ No period directories found")
        return
    
    print(f"📁 Found {len(period_dirs)} periods to analyze")
    
    # Analyze each period
    for period_dir in sorted(period_dirs):
        compare_period_windows(str(period_dir))
    
    print(f"\n✅ Analysis complete!")
    print(f"📋 If you see 'IDENTICAL' entries, those windows have bugs")
    print(f"📋 Each window should produce unique pair selections")

if __name__ == "__main__":
    main()