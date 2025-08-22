#!/usr/bin/env python3
"""
Pair Selection Script - selecting_pairs.py

Selects the best trading pairs based on clear, interpretable criteria:
1. Max drawdown < -50% (both overall and last year 2022)
2. Total cumulative return >= 600% (6x initial capital)
3. Number of trades >= 800
4. Statistical significance: Top 5% vs random sequences
5. Strong performance in 2022 specifically

Author: IntramarketDifference Analysis
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
PERMUTATIONS_DIR = "permutations"
TRADES_DIR = "trades"
OUTPUT_FILE_FULL = "all_pairs_metrics.csv"
OUTPUT_FILE_FILTERED = "selected_pairs.csv"
OUTPUT_FILE_TEMP = "all_pairs_metrics_temp.csv"  # Temporary file for incremental saves
INITIAL_CAPITAL = 1000.0

# Progress tracking
SAVE_INTERVAL = 1000  # Save every 1000 processed pairs

# Selection Thresholds
MAX_OVERALL_DRAWDOWN_THRESHOLD = -0.70    # -70% maximum overall drawdown
MAX_LAST_YEAR_DRAWDOWN_THRESHOLD = -0.30  # -30% maximum last year drawdown
MIN_TOTAL_RETURN = 4.0                    # 4x initial capital
MIN_TRADES = 200                          # Minimum number of trades
MAX_PROFIT_FACTOR_QUANTILE = 5.0          # Top 5% performers vs random
ANALYSIS_YEAR = 2022                      # Year for last-year analysis


def load_permutation_results() -> List[Dict]:
    """Load all permutation test results from JSON files."""
    print("📂 Loading permutation test results...")
    
    all_results = []
    
    # Get all trading coin directories
    if not os.path.exists(PERMUTATIONS_DIR):
        raise FileNotFoundError(f"Permutations directory not found: {PERMUTATIONS_DIR}")
    
    trading_coins = [d for d in os.listdir(PERMUTATIONS_DIR) 
                    if os.path.isdir(os.path.join(PERMUTATIONS_DIR, d))]
    
    print(f"Found {len(trading_coins)} trading coins: {', '.join(trading_coins)}")
    
    for trading_coin in trading_coins:
        coin_dir = os.path.join(PERMUTATIONS_DIR, trading_coin)
        
        # Load all three trade types
        for trade_type in ['combined', 'longs', 'shorts']:
            results_file = os.path.join(coin_dir, f'results_{trade_type}.json')
            
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    
                    # Add trading coin info to each result
                    for result in data.get('results', []):
                        result['trading_coin'] = trading_coin
                        result['trade_type_file'] = trade_type
                        all_results.append(result)
                        
                except Exception as e:
                    print(f"Warning: Could not load {results_file}: {e}")
            else:
                print(f"Warning: Missing results file: {results_file}")
    
    print(f"✅ Loaded {len(all_results)} pair configurations")
    return all_results


def load_trade_data(trading_coin: str, reference_coin: str) -> Optional[List[Dict]]:
    """Load individual trade data from JSON file."""
    filename = f"{reference_coin}_{trading_coin}_trades.json"
    filepath = os.path.join(TRADES_DIR, trading_coin, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            trades = json.load(f)
        return trades
    except Exception as e:
        print(f"Warning: Could not load trade data {filepath}: {e}")
        return None


def calculate_total_cumulative_return(trades: List[Dict], trade_type: str = 'both') -> float:
    """
    Calculate total cumulative return using log returns.
    
    Args:
        trades: List of trade dictionaries
        trade_type: 'both', 'long', or 'short'
    
    Returns:
        Total cumulative return as ratio (5.0 = 600% return)
    """
    if not trades:
        return 0.0
    
    # Filter trades by type
    if trade_type == 'both':
        filtered_trades = trades
    elif trade_type == 'long':
        filtered_trades = [t for t in trades if t['trade_type'] == 'long']
    elif trade_type == 'short':
        filtered_trades = [t for t in trades if t['trade_type'] == 'short']
    else:
        return 0.0
    
    # Get log returns
    log_returns = [t['log_return'] for t in filtered_trades if t['log_return'] is not None]
    
    if not log_returns:
        return 0.0
    
    # Calculate cumulative return: (final_value / initial_value) - 1
    total_log_return = sum(log_returns)
    cumulative_return = np.exp(total_log_return) - 1
    
    return float(cumulative_return)


def calculate_max_drawdown_from_trades(trades: List[Dict], trade_type: str = 'both') -> float:
    """Calculate maximum drawdown from individual trades."""
    if not trades:
        return 0.0
    
    # Filter trades by type
    if trade_type == 'both':
        filtered_trades = trades
    elif trade_type == 'long':
        filtered_trades = [t for t in trades if t['trade_type'] == 'long']
    elif trade_type == 'short':
        filtered_trades = [t for t in trades if t['trade_type'] == 'short']
    else:
        return 0.0
    
    # Get log returns and sort by exit time
    trade_data = []
    for t in filtered_trades:
        if t['log_return'] is not None and t['time_exited'] is not None:
            exit_time = pd.to_datetime(t['time_exited'])
            trade_data.append((exit_time, t['log_return']))
    
    if not trade_data:
        return 0.0
    
    # Sort by time
    trade_data.sort(key=lambda x: x[0])
    log_returns = [x[1] for x in trade_data]
    
    # Calculate cumulative equity and drawdown
    cumsum_returns = np.cumsum(log_returns)
    equity_curve = INITIAL_CAPITAL * np.exp(cumsum_returns)
    
    # Calculate drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    
    return float(drawdown.min())


def calculate_2022_drawdown(trades: List[Dict], trade_type: str = 'both') -> float:
    """Calculate maximum drawdown for trades in 2022 only."""
    if not trades:
        return 0.0
    
    # Filter trades by type and year 2022
    filtered_trades = []
    for t in trades:
        if (t['log_return'] is not None and 
            t['time_exited'] is not None):
            
            exit_time = pd.to_datetime(t['time_exited'])
            if exit_time.year == ANALYSIS_YEAR:
                if (trade_type == 'both' or 
                    (trade_type == 'long' and t['trade_type'] == 'long') or 
                    (trade_type == 'short' and t['trade_type'] == 'short')):
                    filtered_trades.append((exit_time, t['log_return']))
    
    if not filtered_trades:
        return 0.0  # No trades in 2022
    
    # Sort by time and calculate drawdown
    filtered_trades.sort(key=lambda x: x[0])
    log_returns = [x[1] for x in filtered_trades]
    
    # Calculate equity curve for 2022
    cumsum_returns = np.cumsum(log_returns)
    equity_curve = INITIAL_CAPITAL * np.exp(cumsum_returns)
    
    # Calculate drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    
    return float(drawdown.min())


def calculate_sharpe_ratio_from_trades(trades: List[Dict], trade_type: str = 'both', risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from individual trades."""
    if not trades:
        return np.nan
    
    # Filter trades by type
    if trade_type == 'both':
        filtered_trades = trades
    elif trade_type == 'long':
        filtered_trades = [t for t in trades if t['trade_type'] == 'long']
    elif trade_type == 'short':
        filtered_trades = [t for t in trades if t['trade_type'] == 'short']
    else:
        return np.nan
    
    # Get log returns
    log_returns = [t['log_return'] for t in filtered_trades if t['log_return'] is not None]
    
    if len(log_returns) == 0:
        return np.nan
    
    # Convert to pandas Series for calculation
    returns_series = pd.Series(log_returns)
    
    if len(returns_series) == 0 or returns_series.std() == 0:
        return np.nan
    
    # Calculate Sharpe ratio (assuming returns are already log returns)
    excess_returns = returns_series - risk_free_rate / 252  # Assuming daily returns
    sharpe_ratio = excess_returns.mean() / returns_series.std() * np.sqrt(252)  # Annualized
    
    return float(sharpe_ratio)


def save_incremental_results(temp_results: List[Dict], current_count: int, total_count: int, final: bool = False):
    """Save incremental results to temporary CSV file."""
    if not temp_results:
        return
    
    # Convert to DataFrame with same structure as final output
    temp_data = []
    for result in temp_results:
        temp_data.append({
            'trading_coin': result['trading_coin'],
            'reference_coin': result['reference_coin'],
            'trade_type': result['trade_type'],
            'profit_factor': result.get('algo_profit_factor', np.nan),
            'profit_factor_quantile': result.get('profit_factor_quantile', np.nan),
            'total_cumulative_return': result['total_cumulative_return'],
            'max_drawdown': result['overall_max_drawdown'],
            'last_year_drawdown': result['last_year_drawdown'],
            'num_trades': result.get('num_trades', 0),
            'sharpe_ratio': result.get('sharpe_ratio', np.nan),
            'drawdown_quantile': result.get('drawdown_quantile', np.nan),
            'passes_filter_1_drawdown': result['overall_max_drawdown'] >= MAX_OVERALL_DRAWDOWN_THRESHOLD,
            'passes_filter_2_return': result['total_cumulative_return'] >= MIN_TOTAL_RETURN,
            'passes_filter_3_trades': result.get('num_trades', 0) >= MIN_TRADES,
            'passes_filter_4_statistical': result.get('profit_factor_quantile', 100) <= MAX_PROFIT_FACTOR_QUANTILE,
            'passes_filter_5_2022': result['last_year_drawdown'] >= MAX_LAST_YEAR_DRAWDOWN_THRESHOLD,
            'passes_all_filters': (
                result['overall_max_drawdown'] >= MAX_OVERALL_DRAWDOWN_THRESHOLD and
                result['total_cumulative_return'] >= MIN_TOTAL_RETURN and
                result.get('num_trades', 0) >= MIN_TRADES and
                result.get('profit_factor_quantile', 100) <= MAX_PROFIT_FACTOR_QUANTILE and
                result['last_year_drawdown'] >= MAX_LAST_YEAR_DRAWDOWN_THRESHOLD
            )
        })
    
    temp_df = pd.DataFrame(temp_data)
    temp_df = temp_df.sort_values('profit_factor', ascending=False, na_position='last')
    
    # Append to existing file or create new one
    mode = 'a' if os.path.exists(OUTPUT_FILE_TEMP) and not final else 'w'
    header = not os.path.exists(OUTPUT_FILE_TEMP) or final or mode == 'w'
    
    temp_df.to_csv(OUTPUT_FILE_TEMP, mode=mode, index=False, float_format='%.4f', header=header)
    
    if final:
        print(f"\n📊 Final incremental save: {current_count}/{total_count} pairs → {OUTPUT_FILE_TEMP}")
        # Also copy to final filename for immediate inspection
        temp_df_full = pd.read_csv(OUTPUT_FILE_TEMP)
        temp_df_full.to_csv(OUTPUT_FILE_FULL.replace('.csv', '_preview.csv'), index=False, float_format='%.4f')
        print(f"📋 Preview available at: {OUTPUT_FILE_FULL.replace('.csv', '_preview.csv')}")
    else:
        print(f"\n💾 Incremental save: {current_count}/{total_count} pairs → {OUTPUT_FILE_TEMP}")


def main():
    """Main execution function."""
    print("🚀 Starting Pair Selection Analysis")
    print("=" * 60)
    
    # Load all permutation results
    results = load_permutation_results()
    
    if not results:
        print("❌ No permutation results found!")
        return
    
    print(f"\n📊 Starting with {len(results)} pair configurations")
    
    # Process each result
    enhanced_results = []
    failed_loads = 0
    
    print("\n🔄 Calculating additional metrics...")
    
    # Initialize temporary results list
    temp_results = []
    
    # Use tqdm for progress bar with estimated time
    for i, result in enumerate(tqdm(results, desc="Processing pairs", unit="pair")):
        trading_coin = result['trading_coin']
        reference_coin = result['reference_coin']
        trade_type = result['trade_type']
        
        # Load individual trade data
        trades = load_trade_data(trading_coin, reference_coin)
        
        if trades is None:
            failed_loads += 1
            continue
        
        # Map trade type names
        trade_type_map = {'both': 'both', 'long': 'long', 'short': 'short'}
        actual_trade_type = trade_type_map.get(trade_type, 'both')
        
        # Calculate additional metrics
        try:
            total_return = calculate_total_cumulative_return(trades, actual_trade_type)
            overall_drawdown = calculate_max_drawdown_from_trades(trades, actual_trade_type)
            last_year_drawdown = calculate_2022_drawdown(trades, actual_trade_type)
            sharpe_ratio = calculate_sharpe_ratio_from_trades(trades, actual_trade_type)
            
            # Create enhanced result
            enhanced_result = result.copy()
            enhanced_result.update({
                'total_cumulative_return': total_return,
                'overall_max_drawdown': overall_drawdown,
                'last_year_drawdown': last_year_drawdown,
                'sharpe_ratio': sharpe_ratio
            })
            
            enhanced_results.append(enhanced_result)
            temp_results.append(enhanced_result)
            
            # Incremental save every SAVE_INTERVAL pairs
            if (i + 1) % SAVE_INTERVAL == 0:
                save_incremental_results(temp_results, i + 1, len(results))
                temp_results = []  # Clear temporary results
                
        except Exception as e:
            print(f"\nWarning: Failed to process {trading_coin}-{reference_coin}: {e}")
            failed_loads += 1
    
    # Save any remaining results
    if temp_results:
        save_incremental_results(temp_results, len(results), len(results), final=True)
    
    print(f"\n✅ Enhanced {len(enhanced_results)} configurations")
    if failed_loads > 0:
        print(f"⚠️  Failed to load {failed_loads} configurations")
    
    # Apply filters
    print(f"\n🔍 Applying Selection Filters")
    print("-" * 40)
    
    current_results = enhanced_results
    
    # Filter 1: Overall Max Drawdown Threshold
    filter1_passed = [r for r in current_results if r['overall_max_drawdown'] >= MAX_OVERALL_DRAWDOWN_THRESHOLD]
    print(f"Filter 1 - Overall Max Drawdown >= {MAX_OVERALL_DRAWDOWN_THRESHOLD*100:.0f}%: {len(filter1_passed)}/{len(current_results)} passed")
    current_results = filter1_passed
    
    # Filter 2: Total Cumulative Return
    filter2_passed = [r for r in current_results if r['total_cumulative_return'] >= MIN_TOTAL_RETURN]
    print(f"Filter 2 - Total Return >= {MIN_TOTAL_RETURN:.1f}x ({MIN_TOTAL_RETURN*100:.0f}%): {len(filter2_passed)}/{len(current_results)} passed")
    current_results = filter2_passed
    
    # Filter 3: Minimum Trade Count
    filter3_passed = [r for r in current_results if r.get('num_trades', 0) >= MIN_TRADES]
    print(f"Filter 3 - Number of Trades >= {MIN_TRADES}: {len(filter3_passed)}/{len(current_results)} passed")
    current_results = filter3_passed
    
    # Filter 4: Statistical Significance
    filter4_passed = [r for r in current_results if r.get('profit_factor_quantile', 100) <= MAX_PROFIT_FACTOR_QUANTILE]
    print(f"Filter 4 - Top {MAX_PROFIT_FACTOR_QUANTILE:.0f}% vs Random: {len(filter4_passed)}/{len(current_results)} passed")
    current_results = filter4_passed
    
    # Filter 5: Last Year (2022) Drawdown
    filter5_passed = [r for r in current_results if r['last_year_drawdown'] >= MAX_LAST_YEAR_DRAWDOWN_THRESHOLD]
    print(f"Filter 5 - 2022 Max Drawdown >= {MAX_LAST_YEAR_DRAWDOWN_THRESHOLD*100:.0f}%: {len(filter5_passed)}/{len(current_results)} passed")
    current_results = filter5_passed
    
    print(f"\n🎯 Final Selection: {len(current_results)} pairs passed all filters")
    
    # Create comprehensive output DataFrame with ALL enhanced results
    print(f"\n💾 Preparing comprehensive metrics table...")
    all_output_data = []
    for result in enhanced_results:
        all_output_data.append({
            'trading_coin': result['trading_coin'],
            'reference_coin': result['reference_coin'],
            'trade_type': result['trade_type'],
            'profit_factor': result.get('algo_profit_factor', np.nan),
            'profit_factor_quantile': result.get('profit_factor_quantile', np.nan),
            'total_cumulative_return': result['total_cumulative_return'],
            'max_drawdown': result['overall_max_drawdown'],
            'last_year_drawdown': result['last_year_drawdown'],
            'num_trades': result.get('num_trades', 0),
            'sharpe_ratio': result.get('sharpe_ratio', np.nan),
            'drawdown_quantile': result.get('drawdown_quantile', np.nan),
            'passes_filter_1_drawdown': result['overall_max_drawdown'] >= MAX_OVERALL_DRAWDOWN_THRESHOLD,
            'passes_filter_2_return': result['total_cumulative_return'] >= MIN_TOTAL_RETURN,
            'passes_filter_3_trades': result.get('num_trades', 0) >= MIN_TRADES,
            'passes_filter_4_statistical': result.get('profit_factor_quantile', 100) <= MAX_PROFIT_FACTOR_QUANTILE,
            'passes_filter_5_2022': result['last_year_drawdown'] >= MAX_LAST_YEAR_DRAWDOWN_THRESHOLD,
            'passes_all_filters': (
                result['overall_max_drawdown'] >= MAX_OVERALL_DRAWDOWN_THRESHOLD and
                result['total_cumulative_return'] >= MIN_TOTAL_RETURN and
                result.get('num_trades', 0) >= MIN_TRADES and
                result.get('profit_factor_quantile', 100) <= MAX_PROFIT_FACTOR_QUANTILE and
                result['last_year_drawdown'] >= MAX_LAST_YEAR_DRAWDOWN_THRESHOLD
            )
        })
    
    # Create and save comprehensive DataFrame
    df_all = pd.DataFrame(all_output_data)
    df_all = df_all.sort_values('profit_factor', ascending=False, na_position='last')
    df_all.to_csv(OUTPUT_FILE_FULL, index=False, float_format='%.4f')
    print(f"📊 Full metrics saved to: {OUTPUT_FILE_FULL} ({len(df_all)} pairs)")
    
    if len(current_results) == 0:
        print("❌ No pairs passed all selection criteria!")
        print(f"📈 However, you can analyze all pairs in {OUTPUT_FILE_FULL}")
        return
    
    # Create filtered output DataFrame for pairs that passed all filters
    filtered_output_data = []
    for result in current_results:
        filtered_output_data.append({
            'trading_coin': result['trading_coin'],
            'reference_coin': result['reference_coin'],
            'trade_type': result['trade_type'],
            'profit_factor': result.get('algo_profit_factor', np.nan),
            'profit_factor_quantile': result.get('profit_factor_quantile', np.nan),
            'total_cumulative_return': result['total_cumulative_return'],
            'max_drawdown': result['overall_max_drawdown'],
            'last_year_drawdown': result['last_year_drawdown'],
            'num_trades': result.get('num_trades', 0),
            'sharpe_ratio': result.get('sharpe_ratio', np.nan),
            'drawdown_quantile': result.get('drawdown_quantile', np.nan)
        })
    
    # Create and save filtered DataFrame
    df_filtered = pd.DataFrame(filtered_output_data)
    df_filtered = df_filtered.sort_values('profit_factor', ascending=False)
    df_filtered.to_csv(OUTPUT_FILE_FILTERED, index=False, float_format='%.4f')
    print(f"🎯 Filtered pairs saved to: {OUTPUT_FILE_FILTERED} ({len(df_filtered)} pairs)")
    
    # Display summary statistics for both datasets
    print("\n📈 Summary Statistics:")
    print("=" * 60)
    
    print(f"\n📊 ALL PAIRS ({len(df_all)} total):")
    print("-" * 40)
    print(f"Trading coins represented: {df_all['trading_coin'].nunique()}")
    print(f"Reference coins represented: {df_all['reference_coin'].nunique()}")
    print(f"Trade types: {df_all['trade_type'].value_counts().to_dict()}")
    print(f"Pairs passing all filters: {df_all['passes_all_filters'].sum()} ({100*df_all['passes_all_filters'].mean():.1f}%)")
    
    # Filter breakdown
    print(f"\nFilter Pass Rates:")
    print(f"  Filter 1 (Drawdown): {df_all['passes_filter_1_drawdown'].sum()} ({100*df_all['passes_filter_1_drawdown'].mean():.1f}%)")
    print(f"  Filter 2 (Return): {df_all['passes_filter_2_return'].sum()} ({100*df_all['passes_filter_2_return'].mean():.1f}%)")
    print(f"  Filter 3 (Trades): {df_all['passes_filter_3_trades'].sum()} ({100*df_all['passes_filter_3_trades'].mean():.1f}%)")
    print(f"  Filter 4 (Statistical): {df_all['passes_filter_4_statistical'].sum()} ({100*df_all['passes_filter_4_statistical'].mean():.1f}%)")
    print(f"  Filter 5 (2022): {df_all['passes_filter_5_2022'].sum()} ({100*df_all['passes_filter_5_2022'].mean():.1f}%)")
    
    valid_pf = df_all['profit_factor'].dropna()
    valid_ret = df_all['total_cumulative_return'].dropna()
    if len(valid_pf) > 0:
        print(f"\nProfit Factor (all pairs):")
        print(f"  Mean: {valid_pf.mean():.2f}, Median: {valid_pf.median():.2f}")
        print(f"  Range: {valid_pf.min():.2f} - {valid_pf.max():.2f}")
    if len(valid_ret) > 0:
        print(f"Total Return (all pairs):")
        print(f"  Mean: {valid_ret.mean():.1f}x, Median: {valid_ret.median():.1f}x")
        print(f"  Range: {valid_ret.min():.1f}x - {valid_ret.max():.1f}x")
    
    if len(df_filtered) > 0:
        print(f"\n🎯 FILTERED PAIRS ({len(df_filtered)} selected):")
        print("-" * 40)
        print(f"Trading coins represented: {df_filtered['trading_coin'].nunique()}")
        print(f"Reference coins represented: {df_filtered['reference_coin'].nunique()}")
        print(f"Trade types: {df_filtered['trade_type'].value_counts().to_dict()}")
        print()
        print("Profit Factor Statistics (filtered):")
        print(f"  Mean: {df_filtered['profit_factor'].mean():.2f}")
        print(f"  Median: {df_filtered['profit_factor'].median():.2f}")
        print(f"  Range: {df_filtered['profit_factor'].min():.2f} - {df_filtered['profit_factor'].max():.2f}")
        print()
        print("Total Return Statistics (filtered):")
        print(f"  Mean: {df_filtered['total_cumulative_return'].mean():.1f}x ({df_filtered['total_cumulative_return'].mean()*100:.0f}%)")
        print(f"  Median: {df_filtered['total_cumulative_return'].median():.1f}x ({df_filtered['total_cumulative_return'].median()*100:.0f}%)")
        print(f"  Range: {df_filtered['total_cumulative_return'].min():.1f}x - {df_filtered['total_cumulative_return'].max():.1f}x")
    
    print(f"\n🎉 Analysis completed!")
    print(f"📁 Files generated:")
    print(f"   📊 {OUTPUT_FILE_FULL} - All pairs with comprehensive metrics")
    print(f"   🎯 {OUTPUT_FILE_FILTERED} - Filtered pairs meeting all criteria")
    
    # Clean up temporary files
    if os.path.exists(OUTPUT_FILE_TEMP):
        os.remove(OUTPUT_FILE_TEMP)
        print(f"   🧹 Cleaned up temporary file: {OUTPUT_FILE_TEMP}")
    
    # Note about preview file
    preview_file = OUTPUT_FILE_FULL.replace('.csv', '_preview.csv')
    if os.path.exists(preview_file):
        print(f"   📋 Preview file available: {preview_file}")
        print(f"       (Contains data processed during execution - you can delete this)")


if __name__ == "__main__":
    main()