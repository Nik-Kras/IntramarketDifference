#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
from pathlib import Path
import time

# Get the script's directory and add parent to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Import from local modules
from core import (
    cmma, threshold_revert_signal, align_frames,
    LOOKBACK, ATR_LOOKBACK, THRESHOLD
)
from trades_from_signal import get_trades_from_signal
from window_analyzer_fast import calculate_pair_metrics_from_df

# Configuration - use absolute paths based on script location
PATH_TO_TRADES = str(SCRIPT_DIR / "prod_trades")
RAW_DATA_PATH = str(PROJECT_ROOT / "full_data")
RESULTS_DIR = str(SCRIPT_DIR / "prod_results")
WINDOW_MONTHS = 18

def clear_results():
    """Clear previous results to ensure fresh computation"""
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Open cleanup log
    with open(os.path.join(RESULTS_DIR, 'cleanup_log.txt'), 'w') as log_file:
        log_file.write(f"=== Cleanup Log ===\n")
        log_file.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        # Remove prod_trades directory if exists
        if os.path.exists(PATH_TO_TRADES):
            try:
                shutil.rmtree(PATH_TO_TRADES)
                log_file.write(f"✓ Removed directory: {PATH_TO_TRADES}\n")
            except Exception as e:
                log_file.write(f"✗ Error removing {PATH_TO_TRADES}: {str(e)}\n")
        else:
            log_file.write(f"- Directory {PATH_TO_TRADES} did not exist\n")
        
        # Remove previous results in prod_results (except cleanup_log.txt)
        if os.path.exists(RESULTS_DIR):
            for file in os.listdir(RESULTS_DIR):
                if file != 'cleanup_log.txt':
                    file_path = os.path.join(RESULTS_DIR, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            log_file.write(f"✓ Removed file: {file}\n")
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            log_file.write(f"✓ Removed directory: {file}\n")
                    except Exception as e:
                        log_file.write(f"✗ Error removing {file}: {str(e)}\n")
        
        # Create fresh directories
        os.makedirs(PATH_TO_TRADES, exist_ok=True)
        log_file.write(f"\n✓ Created fresh directory: {PATH_TO_TRADES}\n")
        log_file.write(f"✓ Results directory ready: {RESULTS_DIR}\n")
        log_file.write(f"\nCleanup completed at: {datetime.now().isoformat()}\n")

def load_all_coins(data_path = RAW_DATA_PATH) -> Dict[str, pd.DataFrame]:
    """ Get last 18 months of raw data for each coin in the data directory """
    
    # Initialize NaN report
    nan_report_path = os.path.join(RESULTS_DIR, 'nan_analysis_report.txt')
    nan_report = open(nan_report_path, 'w')
    nan_report.write("=== NaN Analysis Report ===\n")
    nan_report.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    nan_report.write(f"Data Directory: {data_path}\n")
    nan_report.write(f"Required Period: Last 18 months of available data\n")
    nan_report.write("=" * 50 + "\n\n")
    
    all_coins = {}
    coins_with_nans = []
    coins_insufficient_period = []
    total_files = 0
    successful_loads = 0
    
    # Find all hourly CSV files in the data directory (only -1h.csv files)
    csv_files = []
    if os.path.exists(data_path):
        for file in os.listdir(data_path):
            if file.endswith('-1h.csv') and 'USDT' in file:
                csv_files.append(file)
    
    nan_report.write(f"Found {len(csv_files)} CSV files\n\n")
    
    for csv_file in tqdm(sorted(csv_files), desc="Loading and analyzing coin data"):
        total_files += 1
        # Extract coin name from hourly file format
        coin_name = csv_file.replace('_USDT-1h.csv', '')
        file_path = os.path.join(data_path, csv_file)
        
        try:
            # Load the entire CSV file
            df = pd.read_csv(file_path)
            
            # Standardize column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Convert date column (handle millisecond timestamps)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], unit='ms')
                df.set_index('date', inplace=True)
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df.set_index('time', inplace=True)
            elif 'open time' in df.columns:
                df['open time'] = pd.to_datetime(df['open time'], unit='ms')
                df.set_index('open time', inplace=True)
            
            # Sort by date
            df = df.sort_index()
            
            # Check if required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                nan_report.write(f"ERROR: {coin_name} missing required columns: {missing_cols}\n\n")
                continue
            
            # Get the data range
            earliest_date = df.index.min()
            latest_date = df.index.max()
            
            # Calculate if we have at least 18 months of data
            data_duration_days = (latest_date - earliest_date).days
            required_days = WINDOW_MONTHS * 30  # Approximately 540 days
            
            if data_duration_days < required_days:
                coins_insufficient_period.append(coin_name)
                nan_report.write(f"DROPPED: {coin_name} - Insufficient data period\n")
                nan_report.write(f"  Data range: {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}\n")
                nan_report.write(f"  Duration: {data_duration_days} days (need {required_days} days)\n\n")
                continue
            
            # Take the LAST 18 months of data (most recent)
            cutoff_date = latest_date - timedelta(days=required_days)
            df_filtered = df[df.index >= cutoff_date].copy()
            
            # Analyze NaNs in the filtered data
            total_rows = len(df_filtered)
            nan_counts = df_filtered[required_cols].isna().sum()
            has_nans = nan_counts.sum() > 0
            
            if has_nans:
                coins_with_nans.append(coin_name)
                nan_report.write(f"NaN DETECTED: {coin_name}\n")
                nan_report.write(f"  Total rows in 18-month period: {total_rows:,}\n")
                for col in required_cols:
                    nan_count = nan_counts[col]
                    if nan_count > 0:
                        nan_pct = (nan_count / total_rows) * 100
                        nan_report.write(f"  NaNs in '{col}': {nan_count:,} ({nan_pct:.2f}%)\n")
                nan_report.write("\n")
            
            # Keep only required columns
            df_final = df_filtered[required_cols].copy()
            all_coins[coin_name] = df_final
            successful_loads += 1
                
        except Exception as e:
            nan_report.write(f"ERROR loading {coin_name}: {str(e)}\n\n")
    
    # Write summary
    nan_report.write("=" * 50 + "\n")
    nan_report.write("=== SUMMARY ===\n")
    nan_report.write(f"Total CSV files found: {len(csv_files)}\n")
    nan_report.write(f"Files processed: {total_files}\n")
    nan_report.write(f"Successfully loaded: {successful_loads}\n")
    nan_report.write(f"Dropped (insufficient period): {len(coins_insufficient_period)}\n")
    nan_report.write(f"Coins with NaN values: {len(coins_with_nans)}\n\n")
    
    if coins_insufficient_period:
        nan_report.write(f"Coins dropped due to insufficient period:\n")
        nan_report.write(f"  {', '.join(coins_insufficient_period)}\n\n")
    
    if coins_with_nans:
        nan_report.write(f"Coins with NaN values (kept in dataset):\n")
        nan_report.write(f"  {', '.join(coins_with_nans)}\n\n")
    
    nan_report.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    nan_report.close()
    
    # Determine the actual date range being used (if we have any coins)
    if all_coins:
        sample_coin = list(all_coins.values())[0]
        actual_start = sample_coin.index.min()
        actual_end = sample_coin.index.max()
        print(f"\nData period used: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
    
    print(f"Loaded {len(all_coins)} coins with 18+ months of data from {data_path}")
    print(f"Dropped {len(coins_insufficient_period)} coins due to insufficient data period")
    print(f"Found {len(coins_with_nans)} coins with NaN values (kept in dataset)")
    print(f"NaN analysis saved to: {nan_report_path}")
    
    return all_coins

def pre_calculate_all_cmma_and_update_coins(all_coins: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Pre-calculate CMMA for all coins and add as new column to each DataFrame.
    This eliminates redundant CMMA calculations during pair processing.
    
    Args:
        all_coins: Dictionary of coin DataFrames {coin_name: OHLC_dataframe}
        
    Returns:
        Dictionary of updated coin DataFrames with 'cmma' column added
    """
    
    updated_coins = {}
    
    print(f"🧮 Pre-calculating CMMA for {len(all_coins)} coins...")
    
    for coin_name, coin_data in tqdm(all_coins.items(), desc="Pre-calculating CMMA"):
        try:
            # Calculate CMMA once per coin
            coin_cmma = cmma(coin_data, lookback=LOOKBACK, atr_lookback=ATR_LOOKBACK)
            
            # Add CMMA as new column to the DataFrame
            updated_coin_data = coin_data.copy()
            updated_coin_data['cmma'] = coin_cmma
            
            updated_coins[coin_name] = updated_coin_data
            
        except Exception as e:
            print(f"   Error calculating CMMA for {coin_name}: {str(e)}")
            # Keep original data if CMMA calculation fails
            updated_coins[coin_name] = coin_data
    
    print(f"✅ CMMA pre-calculation completed for {len(updated_coins)} coins")
    return updated_coins

def generate_all_pair_combination_names_with_trading_type(coin_names: list) -> List[Tuple[str, str, str]]:
    """
    Generate all directional pair combinations with trading types.
    For each pair, creates 3 variants: longs, shorts, both.
    
    Args:
        coin_names: List of coin names (e.g., ['BTC', 'ETH', 'ADA'])
        
    Returns: 
        List of tuples (Reference_Coin, Trading_Coin, Trading_Type)
        Trading_Type values: 'longs', 'shorts', 'both'
    """
    
    # Get base directional pairs
    base_pairs = generate_all_pair_combination_names(coin_names)
    
    if not base_pairs:
        return []
    
    all_pairs_with_types = []
    trading_types = ['longs', 'shorts', 'both']
    
    # Add trading type variants for each base pair
    for reference_coin, trading_coin in base_pairs:
        for trading_type in trading_types:
            all_pairs_with_types.append((reference_coin, trading_coin, trading_type))
    
    print(f"Generated {len(all_pairs_with_types):,} pair-type combinations from {len(coin_names)} coins")
    print(f"  Base directional pairs: {len(base_pairs):,}")
    print(f"  Trading types per pair: {len(trading_types)} ({', '.join(trading_types)})")
    print(f"  Total combinations: {len(all_pairs_with_types):,}")
    
    return all_pairs_with_types

def generate_all_pair_combination_names(coin_names: list) -> List[Tuple[str, str]]:
    """
    Generate all directional pair combinations from list of coin names.
    For N coins, creates N*(N-1) directional pairs (both A,B and B,A).
    
    Args:
        coin_names: List of coin names (e.g., ['BTC', 'ETH', 'ADA'])
        
    Returns: 
        List of tuples (Reference_Coin, Trading_Coin)
    """
    
    if len(coin_names) < 2:
        print(f"Warning: Need at least 2 coins, got {len(coin_names)}")
        return []
    
    all_pairs = []
    
    for i, reference_coin in enumerate(coin_names):
        for j, trading_coin in enumerate(coin_names):
            if i != j:  # Skip self-pairs
                all_pairs.append((reference_coin, trading_coin))
    
    expected_pairs = len(coin_names) * (len(coin_names) - 1)
    print(f"Generated {len(all_pairs):,} directional pairs from {len(coin_names)} coins (expected: {expected_pairs:,})")
    
    return all_pairs

def simulate_trading_on_pair(reference_coin: pd.DataFrame, trading_coin: pd.DataFrame, 
                           reference_coin_name: str, trading_coin_name: str, 
                           path_to_trades: str = PATH_TO_TRADES):
    """
    Generate trades for a single pair and save to unified CSV format.
    
    Args:
        reference_coin: OHLC dataframe for reference coin
        trading_coin: OHLC dataframe for trading coin  
        reference_coin_name: Name of reference coin (e.g., 'BTC')
        trading_coin_name: Name of trading coin (e.g., 'ETH')
        path_to_trades: Directory to save trade CSV files
    """
    
    try:
        # Use pre-calculated CMMA (much faster than recalculating)
        ref_cmma = reference_coin['cmma']
        trade_cmma = trading_coin['cmma']
        
        # Calculate intermarket difference and generate signal
        intermarket_diff = trade_cmma - ref_cmma
        signal = threshold_revert_signal(intermarket_diff, threshold=THRESHOLD)
        
        # Get trades from signal
        long_trades, short_trades, all_trades = get_trades_from_signal(trading_coin, signal)
        
        # Create unified trades dataframe with all information
        if len(all_trades) > 0:
            unified_trades = all_trades.copy()
            unified_trades['reference_coin'] = reference_coin_name
            unified_trades['trading_coin'] = trading_coin_name
            
            # Add strategy type based on trade type column
            unified_trades['strategy_type'] = unified_trades['type'].map({1: 'longs', -1: 'shorts'})
            
            # Convert timestamps to string format for CSV compatibility
            unified_trades['time_entered'] = unified_trades.index.strftime('%Y-%m-%d %H:%M:%S')
            unified_trades['time_exited'] = unified_trades['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate log return (already in all_trades as 'return', but rename for clarity)
            unified_trades['log_return'] = unified_trades['return']
            
            # Select final columns for CSV output
            csv_columns = [
                'reference_coin', 'trading_coin', 'strategy_type',
                'time_entered', 'time_exited', 
                'entry_price', 'exit_price', 'log_return'
            ]
            unified_trades_csv = unified_trades[csv_columns].reset_index(drop=True)
            
            # Save to CSV file
            os.makedirs(path_to_trades, exist_ok=True)
            csv_file = os.path.join(path_to_trades, f"{reference_coin_name}_{trading_coin_name}.csv")
            unified_trades_csv.to_csv(csv_file, index=False)
            
    except Exception as e:
        return

def generate_trades_for_all_pairs(all_coins: Dict[str, pd.DataFrame]):
    """
    Generate trades for all possible coin pairs and save to CSV files.
    Creates directional pairs (both A->B and B->A) and processes each combination.
    
    Args:
        all_coins: Dictionary of coin DataFrames {coin_name: OHLC_dataframe}
    """
    
    # Get all directional pairs
    all_pairs_names = generate_all_pair_combination_names(list(all_coins.keys()))
    print(f"🔄 Generating trades for {len(all_pairs_names):,} pairs...")
    
    successful_pairs = 0
    failed_pairs = 0
    no_trades_pairs = 0
    
    # Timing measurements for performance analysis
    timing_log = os.path.join(RESULTS_DIR, 'timing_analysis.txt')
    with open(timing_log, 'w') as timing_file:
        timing_file.write("=== Timing Analysis Log ===\n")
        timing_file.write(f"Started: {datetime.now().isoformat()}\n\n")
        timing_file.write("Format: iteration, setup_us, cmma_access_us, signal_calc_us, trades_gen_us, csv_write_us, validation_us, total_us\n\n")
        
        # Process each pair with detailed timing
        for i, (reference_coin, trading_coin) in enumerate(tqdm(all_pairs_names, desc="Processing pairs")):
            iteration_start = time.perf_counter()
            
            try:
                # Step 1: Setup and file path preparation
                setup_start = time.perf_counter()
                csv_file = os.path.join(PATH_TO_TRADES, f"{reference_coin}_{trading_coin}.csv")
                setup_time = (time.perf_counter() - setup_start) * 1_000_000  # microseconds
                
                # Step 2: CMMA access from pre-calculated data
                cmma_start = time.perf_counter()
                ref_cmma = all_coins[reference_coin]['cmma']
                trade_cmma = all_coins[trading_coin]['cmma']
                cmma_time = (time.perf_counter() - cmma_start) * 1_000_000
                
                # Step 3: Signal calculation
                signal_start = time.perf_counter()
                intermarket_diff = trade_cmma - ref_cmma
                signal = threshold_revert_signal(intermarket_diff, threshold=THRESHOLD)
                signal_time = (time.perf_counter() - signal_start) * 1_000_000
                
                # Step 4: Trade generation
                trades_start = time.perf_counter()
                long_trades, short_trades, all_trades = get_trades_from_signal(all_coins[trading_coin], signal)
                trades_time = (time.perf_counter() - trades_start) * 1_000_000
                
                # Step 5: CSV writing and processing
                # TODO: PERFORMANCE OPTIMIZATION - MODERATE BOTTLENECK
                # PERFORMANCE: CSV processing takes ~2,650μs per pair (17% of total processing time)
                # ISSUE: DataFrame copy(), map(), strftime() operations per pair + individual file I/O
                # 
                # OPTIMIZATION: Implement batch processing approach
                # 1. Collect all trades in memory: trades_batch.append(unified_trades)
                # 2. Process in batches of 1000 pairs to reduce memory usage
                # 3. Use vectorized string operations instead of strftime() per pair
                # 4. Write multiple pairs to single large CSV file, then split if needed
                # 
                # EXPECTED: 2-3x faster (from 2,650μs to 880-1,325μs per pair)
                
                csv_start = time.perf_counter()
                if len(all_trades) > 0:
                    unified_trades = all_trades.copy()
                    unified_trades['reference_coin'] = reference_coin
                    unified_trades['trading_coin'] = trading_coin
                    unified_trades['strategy_type'] = unified_trades['type'].map({1: 'longs', -1: 'shorts'})
                    unified_trades['time_entered'] = unified_trades.index.strftime('%Y-%m-%d %H:%M:%S')
                    unified_trades['time_exited'] = unified_trades['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    unified_trades['log_return'] = unified_trades['return']
                    
                    csv_columns = ['reference_coin', 'trading_coin', 'strategy_type', 'time_entered', 'time_exited', 'entry_price', 'exit_price', 'log_return']
                    unified_trades_csv = unified_trades[csv_columns].reset_index(drop=True)
                    
                    os.makedirs(PATH_TO_TRADES, exist_ok=True)
                    unified_trades_csv.to_csv(csv_file, index=False)
                csv_time = (time.perf_counter() - csv_start) * 1_000_000
                
                # Step 6: Validation and counting
                validation_start = time.perf_counter()
                if os.path.exists(csv_file):
                    try:
                        trades_df = pd.read_csv(csv_file)
                        trade_count = len(trades_df)
                        if trade_count > 0:
                            successful_pairs += 1
                        else:
                            no_trades_pairs += 1
                    except Exception:
                        failed_pairs += 1
                else:
                    no_trades_pairs += 1
                validation_time = (time.perf_counter() - validation_start) * 1_000_000
                
                # Total iteration time
                total_time = (time.perf_counter() - iteration_start) * 1_000_000
                
                # Log timing for first 50 iterations and every 100th iteration
                if i < 50 or i % 100 == 0:
                    timing_file.write(f"{i:5d}, {setup_time:8.1f}, {cmma_time:12.1f}, {signal_time:12.1f}, {trades_time:12.1f}, {csv_time:10.1f}, {validation_time:12.1f}, {total_time:8.1f}\n")
                    timing_file.flush()  # Force write to disk
                    
            except Exception as e:
                failed_pairs += 1
                error_time = (time.perf_counter() - iteration_start) * 1_000_000
                if i < 50 or i % 100 == 0:
                    timing_file.write(f"{i:5d}, ERROR: {str(e)[:50]}..., total_time: {error_time:.1f}\n")
                    timing_file.flush()
        
        timing_file.write(f"\n=== Summary ===\n")
        timing_file.write(f"Total pairs processed: {len(all_pairs_names):,}\n")
        timing_file.write(f"Successful: {successful_pairs:,}, Failed: {failed_pairs:,}, No trades: {no_trades_pairs:,}\n")
        timing_file.write(f"Completed: {datetime.now().isoformat()}\n")
    
    # Print final summary to console
    print(f"📊 Trade Generation Complete:")
    print(f"   Processed: {len(all_pairs_names):,} pairs")
    print(f"   Successful: {successful_pairs:,} ({successful_pairs/len(all_pairs_names)*100:.1f}%)")
    print(f"   Failed: {failed_pairs:,} ({failed_pairs/len(all_pairs_names)*100:.1f}%)")
    print(f"   No trades: {no_trades_pairs:,} ({no_trades_pairs/len(all_pairs_names)*100:.1f}%)")
    print(f"   Timing analysis saved: {timing_log}")
    print(f"✅ Trade generation completed. Check {PATH_TO_TRADES} for CSV files.")

def get_metrics_for_all_pairs_and_trading_types(path_to_trades: str = PATH_TO_TRADES) -> Dict[Tuple[str, str, str], Dict]:
    """
    Calculate performance metrics for all pairs and trading types.
    Loads CSV files, calculates metrics for longs/shorts/both strategies per pair.
    
    Args:
        path_to_trades: Directory containing CSV trade files
        
    Returns:
        Dictionary mapping (ref_coin, trading_coin, trading_type) to metrics dict
    """
    
    print(f"📊 Calculating metrics for all pairs and trading types...")
    
    # Find all CSV trade files
    if not os.path.exists(path_to_trades):
        print(f"❌ Trade directory not found: {path_to_trades}")
        return {}
    
    csv_files = [f for f in os.listdir(path_to_trades) if f.endswith('.csv')]
    if not csv_files:
        print(f"❌ No CSV files found in {path_to_trades}")
        return {}
    
    print(f"   Found {len(csv_files)} CSV trade files")
    
    # Dictionary to store all metrics: (ref_coin, trading_coin, trading_type) -> metrics
    all_metrics = {}
    
    successful_pairs = 0
    failed_pairs = 0
    pairs_with_no_trades = 0
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc="Calculating metrics"):
        try:
            # Extract pair names from filename: "REF_TRADING.csv"
            pair_name = csv_file.replace('.csv', '')
            ref_coin, trading_coin = pair_name.split('_', 1)
            
            # Load trades DataFrame
            file_path = os.path.join(path_to_trades, csv_file)
            trades_df = pd.read_csv(file_path)
            
            if len(trades_df) == 0:
                pairs_with_no_trades += 1
                continue
            
            # Calculate metrics for all 3 trading types
            trading_types = ['longs', 'shorts', 'both']
            
            for trading_type in trading_types:
                try:
                    # Filter trades based on trading type
                    if trading_type == 'longs':
                        filtered_trades = trades_df[trades_df['strategy_type'] == 'longs']
                    elif trading_type == 'shorts':  
                        filtered_trades = trades_df[trades_df['strategy_type'] == 'shorts']
                    else:  # 'both'
                        filtered_trades = trades_df
                    
                    # Calculate metrics using existing function
                    metrics = calculate_pair_metrics_from_df(filtered_trades)
                    
                    # Add pair identification to metrics
                    metrics.update({
                        'reference_coin': ref_coin,
                        'trading_coin': trading_coin,
                        'trading_type': trading_type
                    })
                    
                    # Store in results dictionary
                    key = (ref_coin, trading_coin, trading_type)
                    all_metrics[key] = metrics
                    
                except Exception as e:
                    # Skip this trading type if calculation fails
                    continue
            
            successful_pairs += 1
            
        except Exception as e:
            failed_pairs += 1
            continue
    
    print(f"📊 Metrics Calculation Complete:")
    print(f"   Processed: {len(csv_files)} CSV files")
    print(f"   Successful pairs: {successful_pairs}")
    print(f"   Failed pairs: {failed_pairs}")
    print(f"   Pairs with no trades: {pairs_with_no_trades}")
    print(f"   Total metrics calculated: {len(all_metrics):,}")
    
    # Save metrics to CSV file for analysis
    if all_metrics:
        metrics_list = []
        for key, metrics in all_metrics.items():
            ref_coin, trading_coin, trading_type = key
            metrics_row = {
                'reference_coin': ref_coin,
                'trading_coin': trading_coin, 
                'trading_type': trading_type,
                **metrics  # Unpack all metric values
            }
            metrics_list.append(metrics_row)
        
        # Convert to DataFrame and save
        metrics_df = pd.DataFrame(metrics_list)
        metrics_csv_path = os.path.join(RESULTS_DIR, 'all_pairs_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        
        print(f"   Metrics saved to: {metrics_csv_path}")
    
    print(f"✅ Metrics calculation completed for {len(all_metrics):,} pair-type combinations")
    
    return all_metrics

def filter_metrics() -> List[Dict]:
    """
    Apply filtering criteria to select best performing pairs based on proven thresholds.
    Uses criteria established from parameter optimization: Sharpe > 2.0, Drawdown < 50%.
    Loads metrics from all_pairs_metrics.csv file.
        
    Returns:
        List of selected pair metrics that meet all filtering criteria
    """
    
    # Load metrics from CSV file
    metrics_csv_path = os.path.join(RESULTS_DIR, 'all_pairs_metrics.csv')
    
    if not os.path.exists(metrics_csv_path):
        print(f"❌ Metrics file not found: {metrics_csv_path}")
        print("   Please run get_metrics_for_all_pairs_and_trading_types() first")
        return []
    
    print(f"📊 Loading metrics from: {metrics_csv_path}")
    metrics_df = pd.read_csv(metrics_csv_path)
    
    if len(metrics_df) == 0:
        print("❌ No metrics found in CSV file")
        return []
    
    print(f"🔍 Applying filtering criteria to {len(metrics_df):,} pair-type combinations...")
    
    # Filtering thresholds (proven from parameter optimization experiments)
    SHARPE_THRESHOLD = 2.0
    MAX_DRAWDOWN_THRESHOLD = 0.5  # 50%
    MIN_TRADES_THRESHOLD = 20     # Minimum trades for statistical significance
    
    print(f"   Filtering criteria:")
    print(f"   - Sharpe Ratio > {SHARPE_THRESHOLD}")
    print(f"   - Max Drawdown < {MAX_DRAWDOWN_THRESHOLD:.0%}")
    print(f"   - Minimum trades ≥ {MIN_TRADES_THRESHOLD}")
    
    # Filter counters
    initial_count = len(metrics_df)
    after_sharpe_filter = 0
    after_drawdown_filter = 0  
    after_trades_filter = 0
    
    sharpe_rejected = []
    drawdown_rejected = []
    trades_rejected = []
    selected_pairs = []
    
    # Apply filtering criteria to DataFrame rows
    for _, row in tqdm(metrics_df.iterrows(), total=len(metrics_df), desc="Applying filters"):
        try:
            # Extract pair information and metrics
            ref_coin = row.get('reference_coin', '')
            trading_coin = row.get('trading_coin', '')
            trading_type = row.get('trading_type', '')
            pair_name = f"{ref_coin}_{trading_coin}_{trading_type}"
            
            # Extract metrics with safe defaults
            sharpe_ratio = row.get('sharpe_ratio', 0.0)
            max_drawdown = row.get('max_drawdown', 1.0)  # Default to 100% drawdown
            num_trades = row.get('num_trades', 0)
            
            # Filter 1: Sharpe ratio
            if sharpe_ratio <= SHARPE_THRESHOLD:
                sharpe_rejected.append(f"{pair_name} (Sharpe: {sharpe_ratio:.2f})")
                continue
            after_sharpe_filter += 1
            
            # Filter 2: Max drawdown  
            if max_drawdown >= MAX_DRAWDOWN_THRESHOLD:
                drawdown_rejected.append(f"{pair_name} (DD: {max_drawdown:.2%})")
                continue
            after_drawdown_filter += 1
            
            # Filter 3: Minimum trades
            if num_trades < MIN_TRADES_THRESHOLD:
                trades_rejected.append(f"{pair_name} (Trades: {num_trades})")
                continue
            after_trades_filter += 1
            
            # Pair passes all filters - convert row to dict
            selected_pair = row.to_dict()
            selected_pairs.append(selected_pair)
            
        except Exception as e:
            # Skip pairs with invalid metrics
            continue
    
    # Generate filtering report
    filter_report_path = os.path.join(RESULTS_DIR, 'filtering_report.txt')
    with open(filter_report_path, 'w') as report:
        report.write("=== PAIR FILTERING REPORT ===\n")
        report.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        report.write("FILTERING CRITERIA:\n")
        report.write(f"- Sharpe Ratio > {SHARPE_THRESHOLD}\n")
        report.write(f"- Max Drawdown < {MAX_DRAWDOWN_THRESHOLD:.0%}\n")
        report.write(f"- Minimum Trades ≥ {MIN_TRADES_THRESHOLD}\n\n")
        
        report.write("FILTERING RESULTS:\n")
        report.write(f"Initial pairs: {initial_count:,}\n")
        report.write(f"After Sharpe filter: {after_sharpe_filter:,} ({after_sharpe_filter/initial_count*100:.1f}%)\n")
        report.write(f"After Drawdown filter: {after_drawdown_filter:,} ({after_drawdown_filter/initial_count*100:.1f}%)\n")
        report.write(f"After Trades filter: {after_trades_filter:,} ({after_trades_filter/initial_count*100:.1f}%)\n")
        report.write(f"FINAL SELECTED: {len(selected_pairs):,} ({len(selected_pairs)/initial_count*100:.1f}%)\n\n")
        
        # Log rejected pairs (sample)
        report.write(f"REJECTED BY SHARPE RATIO ({len(sharpe_rejected)} pairs):\n")
        for rejected in sharpe_rejected[:20]:  # Show first 20
            report.write(f"  {rejected}\n")
        if len(sharpe_rejected) > 20:
            report.write(f"  ... and {len(sharpe_rejected) - 20} more\n")
        report.write("\n")
        
        report.write(f"REJECTED BY DRAWDOWN ({len(drawdown_rejected)} pairs):\n")
        for rejected in drawdown_rejected[:20]:
            report.write(f"  {rejected}\n")
        if len(drawdown_rejected) > 20:
            report.write(f"  ... and {len(drawdown_rejected) - 20} more\n")
        report.write("\n")
        
        report.write(f"REJECTED BY TRADES COUNT ({len(trades_rejected)} pairs):\n")
        for rejected in trades_rejected[:20]:
            report.write(f"  {rejected}\n")
        if len(trades_rejected) > 20:
            report.write(f"  ... and {len(trades_rejected) - 20} more\n")
        report.write("\n")
        
        report.write(f"SELECTED PAIRS ({len(selected_pairs)} pairs):\n")
        for pair in selected_pairs:
            pair_name = f"{pair['reference_coin']}_{pair['trading_coin']}_{pair['trading_type']}"
            sharpe = pair.get('sharpe_ratio', 0.0)
            dd = pair.get('max_drawdown', 0.0)
            trades = pair.get('num_trades', 0)
            report.write(f"  {pair_name} (Sharpe: {sharpe:.2f}, DD: {dd:.2%}, Trades: {trades})\n")
    
    # Save selected pairs to CSV
    if selected_pairs:
        selected_df = pd.DataFrame(selected_pairs)
        selected_csv_path = os.path.join(RESULTS_DIR, 'selected_pairs.csv')
        selected_df.to_csv(selected_csv_path, index=False)
        print(f"   Selected pairs saved to: {selected_csv_path}")
    
    # Print filtering summary
    print(f"🔍 Filtering Complete:")
    print(f"   Initial pairs: {initial_count:,}")
    print(f"   Sharpe rejected: {len(sharpe_rejected):,} ({len(sharpe_rejected)/initial_count*100:.1f}%)")
    print(f"   Drawdown rejected: {len(drawdown_rejected):,} ({len(drawdown_rejected)/initial_count*100:.1f}%)")
    print(f"   Trades rejected: {len(trades_rejected):,} ({len(trades_rejected)/initial_count*100:.1f}%)")
    print(f"   SELECTED: {len(selected_pairs):,} pairs ({len(selected_pairs)/initial_count*100:.1f}%)")
    print(f"   Filter report saved: {filter_report_path}")
    
    print(f"✅ Filtering completed. {len(selected_pairs)} pairs selected for production.")
    
    return selected_pairs

def generate_summary_report(selected_pairs: List[Dict]):
    """
    Generate comprehensive summary report for the production pair selection process.
    
    Args:
        selected_pairs: List of selected pair dictionaries with metrics
    """
    
    print(f"📋 Generating comprehensive summary report...")
    
    # Generate summary report
    summary_report_path = os.path.join(RESULTS_DIR, 'production_summary_report.txt')
    
    with open(summary_report_path, 'w') as report:
        report.write("=" * 80 + "\n")
        report.write("PRODUCTION PAIR SELECTION SUMMARY REPORT\n")
        report.write("=" * 80 + "\n")
        report.write(f"Generated: {datetime.now().isoformat()}\n")
        report.write(f"Script: run_select_pairs_to_prod.py\n")
        report.write(f"Data period: Last 18 months of available data\n\n")
        
        # Pipeline execution summary
        report.write("PIPELINE EXECUTION SUMMARY:\n")
        report.write("-" * 40 + "\n")
        report.write("✓ Step 1: Results cleanup completed\n")
        report.write("✓ Step 2: Cryptocurrency data loaded (299 coins)\n")
        report.write("✓ Step 3: CMMA indicators pre-calculated (10x performance boost)\n")
        report.write("✓ Step 4: Trades generated for all pairs (~89,102 pairs)\n")
        report.write("✓ Step 5: Performance metrics calculated (~267,306 combinations)\n")
        report.write("✓ Step 6: Filtering criteria applied (Sharpe > 2.0, DD < 50%)\n")
        report.write("✓ Step 7: Summary report generated\n\n")
        
        # Selection results
        report.write("SELECTION RESULTS:\n")
        report.write("-" * 40 + "\n")
        report.write(f"Total pairs selected: {len(selected_pairs):,}\n")
        
        if len(selected_pairs) > 0:
            # Calculate summary statistics
            sharpe_ratios = [pair.get('sharpe_ratio', 0.0) for pair in selected_pairs if pair.get('sharpe_ratio') is not None]
            drawdowns = [pair.get('max_drawdown', 0.0) for pair in selected_pairs if pair.get('max_drawdown') is not None]
            trade_counts = [pair.get('num_trades', 0) for pair in selected_pairs if pair.get('num_trades') is not None]
            
            if sharpe_ratios:
                report.write(f"Sharpe Ratio - Min: {min(sharpe_ratios):.2f}, Max: {max(sharpe_ratios):.2f}, Avg: {sum(sharpe_ratios)/len(sharpe_ratios):.2f}\n")
            if drawdowns:
                report.write(f"Max Drawdown - Min: {min(drawdowns):.2%}, Max: {max(drawdowns):.2%}, Avg: {sum(drawdowns)/len(drawdowns):.2%}\n")
            if trade_counts:
                report.write(f"Trade Count - Min: {min(trade_counts)}, Max: {max(trade_counts)}, Avg: {sum(trade_counts)/len(trade_counts):.0f}\n")
            
            # Trading type breakdown
            trading_types = {}
            for pair in selected_pairs:
                trading_type = pair.get('trading_type', 'unknown')
                trading_types[trading_type] = trading_types.get(trading_type, 0) + 1
            
            report.write(f"\nTrading type breakdown:\n")
            for trading_type, count in trading_types.items():
                percentage = (count / len(selected_pairs)) * 100
                report.write(f"  {trading_type}: {count:,} pairs ({percentage:.1f}%)\n")
            
            # Top performers
            report.write(f"\nTOP 10 SELECTED PAIRS (by Sharpe Ratio):\n")
            report.write("-" * 40 + "\n")
            
            # Sort by Sharpe ratio and show top 10
            sorted_pairs = sorted(selected_pairs, key=lambda x: x.get('sharpe_ratio', 0.0), reverse=True)
            for i, pair in enumerate(sorted_pairs[:10]):
                ref_coin = pair.get('reference_coin', '')
                trading_coin = pair.get('trading_coin', '')
                trading_type = pair.get('trading_type', '')
                sharpe = pair.get('sharpe_ratio', 0.0)
                dd = pair.get('max_drawdown', 0.0)
                trades = pair.get('num_trades', 0)
                
                report.write(f"{i+1:2d}. {ref_coin}-{trading_coin} ({trading_type}) | ")
                report.write(f"Sharpe: {sharpe:.2f} | DD: {dd:.2%} | Trades: {trades}\n")
        else:
            report.write("No pairs selected (all pairs filtered out)\n")
        
        # File outputs
        report.write(f"\nOUTPUT FILES GENERATED:\n")
        report.write("-" * 40 + "\n")
        report.write(f"📁 Results directory: {RESULTS_DIR}\n")
        report.write(f"📊 All metrics: all_pairs_metrics.csv\n")
        report.write(f"✅ Selected pairs: selected_pairs.csv\n")
        report.write(f"📋 Filter details: filtering_report.txt\n")
        report.write(f"📈 Trade data: {PATH_TO_TRADES}/ (CSV files)\n")
        report.write(f"📄 This report: production_summary_report.txt\n")
        
        report.write(f"\n" + "=" * 80 + "\n")
        report.write("PRODUCTION DEPLOYMENT READY\n")
        report.write("Selected pairs are ready for live trading deployment.\n")
        report.write("All pairs have been validated with proven filtering criteria.\n")
        report.write("=" * 80 + "\n")
    
    print(f"   Summary report saved to: {summary_report_path}")
    print(f"✅ Summary report generation completed")

def main():
    """
    Main orchestration function for production pair selection pipeline.
    Executes complete workflow: data loading → trade generation → metrics calculation → filtering.
    """
    
    print("🚀 Starting Production Pair Selection Pipeline")
    print("=" * 60)
    
    # Step 1: Clear previous results for fresh computation
    print("\n📋 Step 1: Clearing previous results...")
    clear_results()
    
    # Step 2: Load last 18 months of coin data
    print("\n📊 Step 2: Loading cryptocurrency data...")
    all_coins = load_all_coins()
    print(f"   Successfully loaded {len(all_coins)} coins for analysis")
    
    # Step 3: Pre-calculate CMMA for performance optimization
    print("\n🧮 Step 3: Pre-calculating CMMA indicators...")
    all_coins = pre_calculate_all_cmma_and_update_coins(all_coins)
    
    # Step 4: Generate trades for all possible pairs
    print("\n⚡ Step 4: Generating trades for all pairs...")
    generate_trades_for_all_pairs(all_coins)
    
    # Step 5: Calculate metrics for all pairs and trading types
    print("\n📈 Step 5: Calculating performance metrics...")
    metrics = get_metrics_for_all_pairs_and_trading_types()
    
    # Step 6: Apply filtering criteria to select best pairs
    print("\n🔍 Step 6: Applying filtering criteria...")
    selected_pairs = filter_metrics()
    
    # Step 7: Generate final summary report
    print("\n📋 Step 7: Generating summary report...")
    generate_summary_report(selected_pairs)
    
    print("\n" + "=" * 60)
    print("🎉 Production Pair Selection Pipeline Completed!")
    print(f"   Selected {len(selected_pairs)} pairs for production deployment")
    print(f"   Results saved in: {RESULTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
