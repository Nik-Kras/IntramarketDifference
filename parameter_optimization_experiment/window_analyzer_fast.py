#!/usr/bin/env python3
"""
Optimized Window Analyzer for Parameter Optimization Experiment

Uses pre-converted CSV files for 10x faster trade loading.
Filters trades by date windows and calculates metrics for specific time periods.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

def load_all_trades_from_csv_directory(trades_dir: str = "in_sample/trades", 
                                     csv_suffix: str = "_fast") -> pd.DataFrame:
    """Load all trade data from CSV files (much faster than JSON)."""
    
    if not os.path.exists(trades_dir):
        return pd.DataFrame()
    
    # Find all pair directories with CSV files
    pair_dirs = [d for d in os.listdir(trades_dir) 
                if os.path.isdir(os.path.join(trades_dir, d))]
    
    print(f"   Found {len(pair_dirs)} pair directories to process...")
    
    all_trades_dfs = []
    
    for pair_dir in tqdm(pair_dirs, desc="Loading CSV trades"):
        pair_path = os.path.join(trades_dir, pair_dir)
        csv_file = os.path.join(pair_path, f"both_trades{csv_suffix}.csv")
        
        if os.path.exists(csv_file):
            try:
                # Fast CSV loading
                df = pd.read_csv(csv_file)
                
                # Vectorized datetime conversion
                df['entry_time'] = pd.to_datetime(df['time_entered'])
                df['exit_time'] = pd.to_datetime(df['time_exited'])
                
                # Filter valid trades
                df = df.dropna(subset=['entry_time', 'exit_time', 'log_return'])
                
                if len(df) > 0:
                    all_trades_dfs.append(df)
                    
            except Exception:
                continue
    
    # Concatenate all DataFrames at once (much faster than list appending)
    if all_trades_dfs:
        all_trades_df = pd.concat(all_trades_dfs, ignore_index=True)
        print(f"   Loaded {len(all_trades_df):,} total trades")
        return all_trades_df
    else:
        return pd.DataFrame()

def filter_trades_by_window_fast(all_trades_df: pd.DataFrame, 
                                window_start: str, window_end: str) -> pd.DataFrame:
    """Filter trades by window using vectorized operations."""
    
    if all_trades_df.empty:
        return pd.DataFrame()
    
    start_date = pd.to_datetime(window_start)
    end_date = pd.to_datetime(window_end)
    
    # Vectorized filtering (much faster than loops)
    mask = (all_trades_df['entry_time'] >= start_date) & (all_trades_df['entry_time'] < end_date)
    windowed_trades = all_trades_df[mask].copy()
    
    return windowed_trades

def calculate_pair_metrics_from_df(trades_df: pd.DataFrame) -> Dict:
    """Calculate performance metrics from DataFrame of trades."""
    
    if trades_df.empty:
        return {
            'total_cumulative_return': 0.0,
            'growth_factor': 1.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'num_trades': 0
        }

    # Vectorized calculations
    log_returns = trades_df['log_return'].astype(float)
    ar_returns = np.exp(log_returns) - 1.0

    # Growth metrics
    growth_factor = float(np.exp(log_returns.sum()))
    total_cumulative_return = growth_factor - 1.0

    # Profit factor
    gross_profit = float(ar_returns[ar_returns > 0].sum())
    gross_loss = float(-ar_returns[ar_returns < 0].sum())
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float('inf') if gross_profit > 0 else 0.0

    # Max drawdown
    equity = np.exp(log_returns.cumsum())
    peak = equity.cummax()
    drawdown = 1.0 - (equity / peak)
    max_drawdown = float(drawdown.max())

    # Annualization
    times = pd.to_datetime(trades_df['time_entered'])
    days_span = max((times.max() - times.min()).days, 1)
    trades_per_year = len(trades_df) * 365.0 / days_span

    # Sharpe & Volatility
    ar_std = float(ar_returns.std(ddof=1)) if len(ar_returns) > 1 else 0.0
    if ar_std > 0:
        sharpe_per_trade = float((ar_returns.mean()) / ar_std)
        sharpe_ratio = sharpe_per_trade * np.sqrt(trades_per_year)
    else:
        sharpe_ratio = 0.0

    volatility = ar_std * np.sqrt(trades_per_year)

    return {
        'total_cumulative_return': total_cumulative_return,
        'growth_factor': growth_factor,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'volatility': volatility,
        'num_trades': len(trades_df)
    }

def group_and_analyze_trades_fast(windowed_trades_df: pd.DataFrame) -> pd.DataFrame:
    """Group trades and calculate metrics using vectorized operations."""
    
    if windowed_trades_df.empty:
        return pd.DataFrame()
    
    # Group by unique combination
    grouped = windowed_trades_df.groupby(['trading_coin', 'reference_coin', 'strategy_type'])
    print(f"   Processing {len(grouped)} unique pairs...")
    
    results = []
    for (trading_coin, ref_coin, strategy_type), group_df in tqdm(grouped, desc="   Calculating metrics"):
        metrics = calculate_pair_metrics_from_df(group_df)
        metrics.update({
            'trading_coin': trading_coin,
            'reference_coin': ref_coin,
            'trading_type': strategy_type
        })
        results.append(metrics)
    
    return pd.DataFrame(results)

def apply_selection_filters(results_df: pd.DataFrame,
                          sharpe_threshold: float = 2.0, 
                          drawdown_threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
    """Apply selection criteria and generate detailed filtering report.
    
    Args:
        results_df: DataFrame with pair metrics
        sharpe_threshold: Minimum Sharpe ratio (default: 2.0)
        drawdown_threshold: Maximum drawdown as positive value (default: 0.5 = 50%)
    """
    
    initial_count = len(results_df)
    
    # Track filtering steps
    filter_stats = {
        'initial_pairs': initial_count,
        'after_sharpe_filter': 0,
        'after_drawdown_filter': 0,
        'sharpe_rejected': [],
        'drawdown_rejected': [],
        'final_selected': []
    }
    
    # Apply Sharpe ratio filter
    sharpe_passed = results_df[results_df['sharpe_ratio'] > sharpe_threshold].copy()
    sharpe_rejected = results_df[results_df['sharpe_ratio'] <= sharpe_threshold]
    
    filter_stats['after_sharpe_filter'] = len(sharpe_passed)
    filter_stats['sharpe_rejected'] = [
        f"{row['reference_coin']}_{row['trading_coin']}_{row['trading_type']} (Sharpe: {row['sharpe_ratio']:.2f})"
        for _, row in sharpe_rejected.iterrows()
    ]
    
    sharpe_reject_pct = len(sharpe_rejected) / initial_count * 100 if initial_count > 0 else 0
    print(f"   Sharpe filter: rejected {len(sharpe_rejected)} pairs ({sharpe_reject_pct:.1f}%)")
    
    # Apply drawdown filter
    final_selected = sharpe_passed[sharpe_passed['max_drawdown'] < drawdown_threshold].copy()
    drawdown_rejected = sharpe_passed[sharpe_passed['max_drawdown'] >= drawdown_threshold]
    
    filter_stats['after_drawdown_filter'] = len(final_selected)
    filter_stats['drawdown_rejected'] = [
        f"{row['reference_coin']}_{row['trading_coin']}_{row['trading_type']} (DD: {row['max_drawdown']:.2%})"
        for _, row in drawdown_rejected.iterrows()
    ]
    
    drawdown_reject_pct = len(drawdown_rejected) / len(sharpe_passed) * 100 if len(sharpe_passed) > 0 else 0
    print(f"   Drawdown filter: rejected {len(drawdown_rejected)} pairs ({drawdown_reject_pct:.1f}%)")
    
    filter_stats['final_selected'] = [
        f"{row['reference_coin']}_{row['trading_coin']}_{row['trading_type']}"
        for _, row in final_selected.iterrows()
    ]
    
    return final_selected, filter_stats

def generate_selection_report(filter_stats: Dict, window_start: str, window_end: str, 
                            output_file: str):
    """Generate detailed text report of the filtering process."""
    
    report_lines = [
        "PAIR SELECTION REPORT",
        "=" * 50,
        f"Window: {window_start} to {window_end}",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "FILTERING CRITERIA",
        "-" * 20,
        "Sharpe Ratio > 2.0",
        "Max Drawdown < 50%",
        "",
        "FILTERING RESULTS", 
        "-" * 20,
        f"Initial pairs: {filter_stats['initial_pairs']:,}",
        f"After Sharpe filter: {filter_stats['after_sharpe_filter']:,} ({filter_stats['after_sharpe_filter']/filter_stats['initial_pairs']*100:.1f}%)",
        f"After Drawdown filter: {filter_stats['after_drawdown_filter']:,} ({filter_stats['after_drawdown_filter']/filter_stats['initial_pairs']*100:.1f}%)",
        f"After Trades filter: {filter_stats['after_trades_filter']:,} ({filter_stats['after_trades_filter']/filter_stats['initial_pairs']*100:.1f}%)",
        "",
        f"REJECTED BY SHARPE RATIO ({len(filter_stats['sharpe_rejected'])} pairs):",
        "-" * 30
    ]
    
    # Add rejected pairs (limit to first 50)
    for rejected in filter_stats['sharpe_rejected'][:50]:
        report_lines.append(f"  {rejected}")
    if len(filter_stats['sharpe_rejected']) > 50:
        report_lines.append(f"  ... and {len(filter_stats['sharpe_rejected']) - 50} more")
    
    report_lines.extend([
        "",
        f"REJECTED BY DRAWDOWN ({len(filter_stats['drawdown_rejected'])} pairs):",
        "-" * 30
    ])
    
    for rejected in filter_stats['drawdown_rejected'][:50]:
        report_lines.append(f"  {rejected}")
    if len(filter_stats['drawdown_rejected']) > 50:
        report_lines.append(f"  ... and {len(filter_stats['drawdown_rejected']) - 50} more")
    
    report_lines.extend([
        "",
        f"REJECTED BY TRADES FILTER ({len(filter_stats['trades_rejected'])} pairs):",
        "-" * 30
    ])
    
    for rejected in filter_stats['trades_rejected'][:50]:
        report_lines.append(f"  {rejected}")
    if len(filter_stats['trades_rejected']) > 50:
        report_lines.append(f"  ... and {len(filter_stats['trades_rejected']) - 50} more")
    
    report_lines.extend([
        "",
        f"FINAL SELECTED PAIRS ({len(filter_stats['final_selected'])} pairs):",
        "-" * 30
    ])
    
    for selected in filter_stats['final_selected']:
        report_lines.append(f"  {selected}")
    
    # Save report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

def analyze_window_fast(window_start: str, window_end: str, 
                       trades_dir: str = "in_sample/trades", 
                       csv_suffix: str = "_fast") -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Fast window analysis using pre-converted CSV files.
    
    Returns:
        (all_window_results, selected_pairs, filter_stats)
    """
    
    print(f"🔍 Analyzing window: {window_start} to {window_end}")
    
    # Load all trades from CSV files (much faster)
    print("📂 Loading all trade data from CSV...")
    all_trades_df = load_all_trades_from_csv_directory(trades_dir, csv_suffix)
    
    if all_trades_df.empty:
        print("❌ No trades loaded!")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Filter by window using vectorized operations
    print(f"⏰ Filtering trades for window...")
    windowed_trades_df = filter_trades_by_window_fast(all_trades_df, window_start, window_end)
    print(f"   {len(windowed_trades_df):,} trades in window")
    
    if len(all_trades_df) > 0:
        filter_pct = (len(all_trades_df) - len(windowed_trades_df)) / len(all_trades_df) * 100
        print(f"   Filtered out {len(all_trades_df) - len(windowed_trades_df):,} trades ({filter_pct:.1f}%)")
    
    if windowed_trades_df.empty:
        print("❌ No trades found in window!")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Group and calculate metrics
    print("📊 Calculating pair metrics...")
    results_df = group_and_analyze_trades_fast(windowed_trades_df)
    print(f"   Analyzed {len(results_df)} unique pairs")
    
    # Apply filters
    print("🔍 Applying selection filters...")
    selected_pairs, filter_stats = apply_selection_filters(results_df)
    print(f"   Selected {len(selected_pairs)} pairs after filtering")
    
    return results_df, selected_pairs, filter_stats

def main():
    """Test fast analyzer with sample window."""
    
    print("🧪 Testing Fast Window Analyzer")
    print("=" * 40)
    
    # Test with 3-month window
    window_start = "2023-10-01"
    window_end = "2024-01-01"
    
    results_df, selected_pairs, filter_stats = analyze_window_fast(window_start, window_end)
    
    if len(results_df) > 0:
        print(f"\n📊 Fast Window Analysis Results:")
        print(f"   Total pairs analyzed: {len(results_df)}")
        print(f"   Selected pairs: {len(selected_pairs)}")
        print(f"   Selection rate: {len(selected_pairs)/len(results_df)*100:.1f}%")
        
        if len(selected_pairs) > 0:
            print(f"   Best Sharpe: {selected_pairs['sharpe_ratio'].max():.2f}")
            print(f"   Best Drawdown: {selected_pairs['max_drawdown'].max():.2%}")
    else:
        print("❌ No valid results")

if __name__ == "__main__":
    main()
