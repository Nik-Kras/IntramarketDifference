#!/usr/bin/env python3
"""
Custom Filters Script for New In-Sample/Out-of-Sample Experiment

Applies filtering criteria to in-sample results:
- Sharpe Ratio > 1.0
- Max Drawdown better than -50%

Saves filtered pairs for out-of-sample validation.
"""

import os
import pandas as pd
from datetime import datetime


class PairFilter:
    """Filter trading pairs based on performance criteria."""
    
    def __init__(self, results_file: str):
        """Initialize with results CSV file."""
        
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        self.df = pd.read_csv(results_file)
        self.original_count = len(self.df)
        
        print(f"📊 Loaded {self.original_count} pairs from results")
        
    def apply_filters(self, filters: list, output_name: str) -> pd.DataFrame:
        """
        Apply a series of filters to the data.
        
        Args:
            filters: List of filter dictionaries with 'column', 'operator', 'value', 'name'
            output_name: Base name for output files
            
        Returns:
            Filtered DataFrame
        """
        
        filtered_df = self.df.copy()
        filter_summary = []
        
        print(f"\n🔍 Applying {len(filters)} filters...")
        print("-" * 50)
        
        for i, filter_config in enumerate(filters, 1):
            column = filter_config['column']
            operator = filter_config['operator']
            value = filter_config['value']
            name = filter_config['name']
            
            initial_count = len(filtered_df)
            
            # Apply filter
            if operator == '>':
                filtered_df = filtered_df[filtered_df[column] > value]
            elif operator == '>=':
                filtered_df = filtered_df[filtered_df[column] >= value]
            elif operator == '<':
                filtered_df = filtered_df[filtered_df[column] < value]
            elif operator == '<=':
                filtered_df = filtered_df[filtered_df[column] <= value]
            elif operator == '==':
                filtered_df = filtered_df[filtered_df[column] == value]
            elif operator == '!=':
                filtered_df = filtered_df[filtered_df[column] != value]
            else:
                raise ValueError(f"Unsupported operator: {operator}")
            
            final_count = len(filtered_df)
            passed_count = final_count
            failed_count = initial_count - final_count
            
            filter_summary.append({
                'filter_name': name,
                'initial_pairs': initial_count,
                'passed': passed_count,
                'failed': failed_count,
                'pass_rate': (passed_count / initial_count * 100) if initial_count > 0 else 0
            })
            
            print(f"   {i}. {name}")
            print(f"      Before: {initial_count:,} pairs")
            print(f"      After:  {passed_count:,} pairs")
            print(f"      Failed: {failed_count:,} pairs ({failed_count/initial_count*100:.1f}%)")
        
        # Save filtered results
        results_dir = "results/filtered_pairs"
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, f"{output_name}.csv")
        filtered_df.to_csv(output_file, index=False, float_format='%.6f')
        
        # Create summary
        summary_text = self._create_summary(filter_summary, filtered_df, output_name)
        summary_file = os.path.join(results_dir, f"{output_name}_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        print(f"\n📊 FILTERING COMPLETE:")
        print(f"   Original pairs: {self.original_count:,}")
        print(f"   Filtered pairs: {len(filtered_df):,}")
        print(f"   Survival rate: {len(filtered_df)/self.original_count*100:.1f}%")
        print(f"   Output file: {output_file}")
        print(f"   Summary file: {summary_file}")
        
        return filtered_df
    
    def _create_summary(self, filter_summary: list, final_df: pd.DataFrame, output_name: str) -> str:
        """Create summary text for filtering results."""
        
        summary = f"Custom Filter Results - {output_name}\n"
        summary += "=" * 50 + "\n\n"
        summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"Original pairs: {self.original_count:,}\n"
        summary += f"Final pairs: {len(final_df):,}\n"
        summary += f"Overall survival rate: {len(final_df)/self.original_count*100:.2f}%\n\n"
        
        summary += "Filter Application Results:\n"
        summary += "-" * 30 + "\n"
        
        for i, filter_info in enumerate(filter_summary, 1):
            summary += f"{i}. {filter_info['filter_name']}\n"
            summary += f"   Initial: {filter_info['initial_pairs']:,} pairs\n"
            summary += f"   Passed:  {filter_info['passed']:,} pairs ({filter_info['pass_rate']:.1f}%)\n"
            summary += f"   Failed:  {filter_info['failed']:,} pairs\n\n"
        
        if len(final_df) > 0:
            summary += "Final Dataset Statistics:\n"
            summary += "-" * 25 + "\n"
            summary += f"Average Profit Factor: {final_df['profit_factor'].mean():.3f}\n"
            summary += f"Average Total Return: {final_df['total_cumulative_return'].mean():.3f}\n"
            summary += f"Average Max Drawdown: {final_df['max_drawdown'].mean():.2%}\n"
            summary += f"Average Sharpe Ratio: {final_df['sharpe_ratio'].mean():.3f}\n"
            summary += f"Average Number of Trades: {final_df['num_trades'].mean():.0f}\n\n"
            
            summary += "Best Performers:\n"
            summary += "-" * 15 + "\n"
            top_performers = final_df.nlargest(10, 'profit_factor')
            summary += f"{'Rank':<4} {'Ref Coin':<8} {'Trading Coin':<12} {'PF':<8} {'Return':<9} {'Drawdown':<10} {'Sharpe':<8}\n"
            summary += f"{'-'*4} {'-'*8} {'-'*12} {'-'*8} {'-'*9} {'-'*10} {'-'*8}\n"
            
            for i, (_, row) in enumerate(top_performers.iterrows(), 1):
                summary += f"{i:<4} {row['reference_coin']:<8} {row['trading_coin']:<12} "
                summary += f"{row['profit_factor']:<8.2f} {row['total_cumulative_return']:<9.2f} "
                summary += f"{row['max_drawdown']:<10.2%} {row['sharpe_ratio']:<8.2f}\n"
        
        return summary


def main():
    """Main execution function."""
    
    print("🎯 Applying Custom Filters to In-Sample Results")
    print("=" * 55)
    
    # Load results
    results_file = "results/in_sample/in_sample_results.csv"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("Run run_in_sample_experiment.py first to generate results.")
        return
    
    # Initialize filter engine
    filter_engine = PairFilter(results_file)
    
    # Define custom filters
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
            'value': -0.50,  # Better than -50%
            'name': 'Max Drawdown better than -50%'
        }
    ]
    
    print("\n📋 Filter Parameters:")
    print("-" * 30)
    for i, f in enumerate(custom_filters, 1):
        print(f"{i}. {f['name']}: {f['column']} {f['operator']} {f['value']}")
    
    # Apply the filters
    result = filter_engine.apply_filters(custom_filters, "new_experiment_filtered_pairs")
    
    if len(result) == 0:
        print("\n❌ No pairs passed the filters!")
        print("Consider relaxing the filter criteria.")
        return
    
    print(f"\n📊 Filter Results Summary:")
    print(f"   Total pairs selected: {len(result)}")
    print(f"   Best profit factor: {result['profit_factor'].max():.2f}")
    print(f"   Best return: {result['total_cumulative_return'].max():.1f}")
    print(f"   Best drawdown: {result['max_drawdown'].max():.1%}")
    print(f"   Average Sharpe: {result['sharpe_ratio'].mean():.2f}")
    
    # Trading coin analysis
    trading_coin_counts = result['trading_coin'].value_counts()
    print(f"\n📈 Top Trading Coins (by number of selected pairs):")
    for coin, count in trading_coin_counts.head(10).items():
        avg_pf = result[result['trading_coin'] == coin]['profit_factor'].mean()
        print(f"   {coin}: {count} pairs (avg PF: {avg_pf:.2f})")
    
    # Reference coin analysis  
    ref_coin_counts = result['reference_coin'].value_counts()
    print(f"\n📊 Top Reference Coins (by number of selected pairs):")
    for coin, count in ref_coin_counts.head(10).items():
        avg_pf = result[result['reference_coin'] == coin]['profit_factor'].mean()
        print(f"   {coin}: {count} pairs (avg PF: {avg_pf:.2f})")
    
    print(f"\n✅ Filtering completed!")
    print(f"Ready for out-of-sample validation: python run_out_of_sample_validation.py")
    
    return result


if __name__ == "__main__":
    main()