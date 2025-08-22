#!/usr/bin/env python3
"""
Custom Pair Filtering Script - filter_pairs.py

Applies custom filters to the all_pairs_metrics.csv table and generates
a filtered result without modifying the original file.

Usage:
    python filter_pairs.py

Available columns for filtering:
- trading_coin, reference_coin, trade_type (string filters)
- profit_factor, profit_factor_quantile, total_cumulative_return
- max_drawdown, last_year_drawdown, num_trades, sharpe_ratio, drawdown_quantile
- passes_filter_1_drawdown, passes_filter_2_return, passes_filter_3_trades,
  passes_filter_4_statistical, passes_filter_5_2022, passes_all_filters (boolean)

Author: IntramarketDifference Analysis
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


# Configuration
INPUT_FILE = "all_pairs_metrics.csv"
OUTPUT_DIR = "filtered_results"


class PairFilter:
    """Class for applying custom filters to trading pairs data."""
    
    def __init__(self, input_file: str):
        """Initialize with input CSV file."""
        self.input_file = input_file
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load the CSV data."""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        print(f"📂 Loading data from {self.input_file}...")
        self.df = pd.read_csv(self.input_file)
        print(f"✅ Loaded {len(self.df)} pairs")
        
        # Display available columns
        print(f"\n📊 Available columns:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Display data types
        print(f"\n🔍 Data types:")
        numeric_cols = []
        string_cols = []
        boolean_cols = []
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            if dtype in ['bool']:
                boolean_cols.append(col)
            elif dtype in ['object']:
                string_cols.append(col)
            else:
                numeric_cols.append(col)
        
        if numeric_cols:
            print(f"  📈 Numeric: {', '.join(numeric_cols)}")
        if string_cols:
            print(f"  📝 String: {', '.join(string_cols)}")
        if boolean_cols:
            print(f"  ✅ Boolean: {', '.join(boolean_cols)}")
    
    def apply_filters(self, filters: list, output_filename: str = None) -> pd.DataFrame:
        """
        Apply custom filters to the data.
        
        Args:
            filters: List of filter dictionaries. Each filter should have:
                - 'column': Column name to filter
                - 'operator': Comparison operator ('>', '<', '>=', '<=', '==', '!=', 'in', 'not_in')
                - 'value': Value to compare against
                - 'name': Optional human-readable name for the filter
            output_filename: Optional custom output filename
        
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"\n🔍 Applying {len(filters)} filters...")
        print("=" * 50)
        
        filtered_df = self.df.copy()
        filter_results = []
        
        for i, filter_config in enumerate(filters, 1):
            column = filter_config['column']
            operator = filter_config['operator']
            value = filter_config['value']
            filter_name = filter_config.get('name', f"Filter {i}")
            
            if column not in self.df.columns:
                print(f"❌ Warning: Column '{column}' not found. Skipping filter.")
                continue
            
            initial_count = len(filtered_df)
            
            # Apply the filter based on operator
            if operator == '>':
                mask = filtered_df[column] > value
            elif operator == '<':
                mask = filtered_df[column] < value
            elif operator == '>=':
                mask = filtered_df[column] >= value
            elif operator == '<=':
                mask = filtered_df[column] <= value
            elif operator == '==':
                mask = filtered_df[column] == value
            elif operator == '!=':
                mask = filtered_df[column] != value
            elif operator == 'in':
                if isinstance(value, (list, tuple)):
                    mask = filtered_df[column].isin(value)
                else:
                    mask = filtered_df[column].isin([value])
            elif operator == 'not_in':
                if isinstance(value, (list, tuple)):
                    mask = ~filtered_df[column].isin(value)
                else:
                    mask = ~filtered_df[column].isin([value])
            else:
                print(f"❌ Unknown operator '{operator}'. Skipping filter.")
                continue
            
            filtered_df = filtered_df[mask]
            final_count = len(filtered_df)
            
            print(f"Filter {i}: {filter_name}")
            print(f"  📋 {column} {operator} {value}")
            print(f"  📊 {initial_count} → {final_count} pairs ({final_count - initial_count:+d})")
            
            filter_results.append({
                'filter_name': filter_name,
                'column': column,
                'operator': operator,
                'value': value,
                'before_count': initial_count,
                'after_count': final_count,
                'removed_count': initial_count - final_count
            })
        
        print("=" * 50)
        print(f"🎯 Final result: {len(filtered_df)} pairs (from {len(self.df)} original)")
        
        # Save results
        if len(filtered_df) > 0:
            self.save_filtered_results(filtered_df, filters, filter_results, output_filename)
        else:
            print("⚠️  No pairs passed all filters!")
        
        return filtered_df
    
    def save_filtered_results(self, filtered_df: pd.DataFrame, filters: list, 
                            filter_results: list, output_filename: str = None):
        """Save filtered results with metadata."""
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"filtered_pairs_{timestamp}.csv"
        
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
        
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Save filtered data
        filtered_df_sorted = filtered_df.sort_values('profit_factor', ascending=False, na_position='last')
        filtered_df_sorted.to_csv(output_path, index=False, float_format='%.4f')
        
        # Create filter summary file
        summary_filename = output_filename.replace('.csv', '_summary.txt')
        summary_path = os.path.join(OUTPUT_DIR, summary_filename)
        
        with open(summary_path, 'w') as f:
            f.write("CUSTOM FILTER RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Output file: {output_filename}\n\n")
            
            f.write(f"FILTER CHAIN:\n")
            f.write("-" * 20 + "\n")
            for result in filter_results:
                f.write(f"{result['filter_name']}: {result['column']} {result['operator']} {result['value']}\n")
                f.write(f"  Result: {result['before_count']} → {result['after_count']} pairs\n\n")
            
            f.write(f"FINAL RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total pairs remaining: {len(filtered_df_sorted)}\n")
            f.write(f"Reduction: {len(self.df) - len(filtered_df_sorted)} pairs removed\n")
            f.write(f"Retention rate: {100 * len(filtered_df_sorted) / len(self.df):.1f}%\n\n")
            
            if len(filtered_df_sorted) > 0:
                f.write(f"TOP 10 PAIRS BY PROFIT FACTOR:\n")
                f.write("-" * 30 + "\n")
                top_pairs = filtered_df_sorted.head(10)
                for _, row in top_pairs.iterrows():
                    f.write(f"{row['trading_coin']}-{row['reference_coin']} ({row['trade_type']}): ")
                    f.write(f"PF={row['profit_factor']:.2f}, Return={row['total_cumulative_return']:.1f}x\n")
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Filtered data: {output_path}")
        print(f"   📋 Filter summary: {summary_path}")
    
    def get_column_stats(self, column: str):
        """Get basic statistics for a column to help with filter design."""
        if column not in self.df.columns:
            print(f"❌ Column '{column}' not found.")
            return
        
        col_data = self.df[column]
        print(f"\n📊 Statistics for '{column}':")
        print("-" * 40)
        
        if col_data.dtype == 'bool':
            value_counts = col_data.value_counts()
            print(f"True: {value_counts.get(True, 0)}")
            print(f"False: {value_counts.get(False, 0)}")
        elif col_data.dtype == 'object':
            print(f"Unique values: {col_data.nunique()}")
            print(f"Most common:")
            for val, count in col_data.value_counts().head(10).items():
                print(f"  {val}: {count}")
        else:
            print(f"Count: {col_data.count()}")
            print(f"Mean: {col_data.mean():.4f}")
            print(f"Median: {col_data.median():.4f}")
            print(f"Std: {col_data.std():.4f}")
            print(f"Min: {col_data.min():.4f}")
            print(f"Max: {col_data.max():.4f}")
            print(f"25th percentile: {col_data.quantile(0.25):.4f}")
            print(f"75th percentile: {col_data.quantile(0.75):.4f}")


def main():
    """Main execution with example filters."""
    
    # Initialize filter
    filter_engine = PairFilter(INPUT_FILE)
    
    print("\n" + "=" * 60)
    print("CUSTOM PAIR FILTERING EXAMPLES")
    print("=" * 60)
    
    # Example 1: High performance pairs
    print("\n🚀 Example 1: High Performance Pairs")
    print("-" * 40)
    
    high_perf_filters = [
        {
            'column': 'max_drawdown',
            'operator': '>',
            'value': -0.5,  # Drawdown better than -50%
            'name': 'Max Drawdown > -50%'
        },
        {
            'column': 'profit_factor',
            'operator': '>',
            'value': 2.0,  # Profit factor > 2.0
            'name': 'Profit Factor > 2.0'
        },
        {
            'column': 'total_cumulative_return',
            'operator': '>',
            'value': 5.0,  # Return > 5x (500%)
            'name': 'Total Return > 5x'
        },
        {
            'column': 'num_trades',
            'operator': '>=',
            'value': 100,  # At least 100 trades
            'name': 'Minimum 100 trades'
        }
    ]
    
    high_perf_result = filter_engine.apply_filters(high_perf_filters, "high_performance_pairs")
    
    # Example 2: Conservative pairs
    print("\n🛡️  Example 2: Conservative Pairs")
    print("-" * 40)
    
    conservative_filters = [
        {
            'column': 'max_drawdown',
            'operator': '>',
            'value': -0.3,  # Max 30% drawdown
            'name': 'Max Drawdown > -30%'
        },
        {
            'column': 'last_year_drawdown',
            'operator': '>',
            'value': -0.2,  # Last year drawdown > -20%
            'name': '2022 Drawdown > -20%'
        },
        {
            'column': 'sharpe_ratio',
            'operator': '>',
            'value': 0.5,  # Positive risk-adjusted returns
            'name': 'Sharpe Ratio > 0.5'
        },
        {
            'column': 'profit_factor',
            'operator': '>',
            'value': 1.5,  # Profitable
            'name': 'Profit Factor > 1.5'
        }
    ]
    
    conservative_result = filter_engine.apply_filters(conservative_filters, "conservative_pairs")
    
    # Example 3: Short-only trades
    print("\n📉 Example 3: Short-Only High Performers")
    print("-" * 40)
    
    short_filters = [
        {
            'column': 'trade_type',
            'operator': '==',
            'value': 'short',
            'name': 'Short trades only'
        },
        {
            'column': 'profit_factor',
            'operator': '>',
            'value': 3.0,
            'name': 'Profit Factor > 3.0'
        },
        {
            'column': 'profit_factor_quantile',
            'operator': '<=',
            'value': 5.0,  # Top 5% vs random
            'name': 'Top 5% statistical significance'
        }
    ]
    
    short_result = filter_engine.apply_filters(short_filters, "short_only_high_perf")
    
    # Example 4: Specific coins
    print("\n🪙 Example 4: Specific Trading Coins")
    print("-" * 40)
    
    specific_coins_filters = [
        {
            'column': 'trading_coin',
            'operator': 'in',
            'value': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL'],
            'name': 'Major coins only'
        },
        {
            'column': 'profit_factor',
            'operator': '>',
            'value': 1.2,
            'name': 'Profit Factor > 1.2'
        },
        {
            'column': 'passes_all_filters',
            'operator': '==',
            'value': True,
            'name': 'Passes all original filters'
        }
    ]
    
    specific_result = filter_engine.apply_filters(specific_coins_filters, "major_coins_filtered")
    
    print(f"\n🎉 Filtering completed! Check the '{OUTPUT_DIR}' directory for results.")
    print("\n💡 To create your own filters, modify the filter dictionaries in this script")
    print("   or create a new script using the PairFilter class.")


if __name__ == "__main__":
    main()