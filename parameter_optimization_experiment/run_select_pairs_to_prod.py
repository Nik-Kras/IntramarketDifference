import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
from pathlib import Path

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
        # Calculate CMMA for both coins (no alignment needed since all coins have same 18-month period)
        ref_cmma = cmma(reference_coin, lookback=LOOKBACK, atr_lookback=ATR_LOOKBACK)
        trade_cmma = cmma(trading_coin, lookback=LOOKBACK, atr_lookback=ATR_LOOKBACK)
        
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
    
    # Process each pair with progress tracking
    for reference_coin, trading_coin in tqdm(all_pairs_names, desc="Processing pairs"):
        try:
            # Track before processing
            csv_file = os.path.join(PATH_TO_TRADES, f"{reference_coin}_{trading_coin}.csv")
            
            # Generate trades for this pair (silently)
            simulate_trading_on_pair(
                reference_coin=all_coins[reference_coin], 
                trading_coin=all_coins[trading_coin],
                reference_coin_name=reference_coin,
                trading_coin_name=trading_coin
            )
            
            # Check if trades were generated
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
                
        except Exception as e:
            failed_pairs += 1
    
    # Print final summary to console
    print(f"📊 Trade Generation Complete:")
    print(f"   Processed: {len(all_pairs_names):,} pairs")
    print(f"   Successful: {successful_pairs:,} ({successful_pairs/len(all_pairs_names)*100:.1f}%)")
    print(f"   Failed: {failed_pairs:,} ({failed_pairs/len(all_pairs_names)*100:.1f}%)")
    print(f"   No trades: {no_trades_pairs:,} ({no_trades_pairs/len(all_pairs_names)*100:.1f}%)")
    print(f"✅ Trade generation completed. Check {PATH_TO_TRADES} for CSV files.")

def get_metrics_for_all_pairs_and_trading_types(path_to_trades: str = PATH_TO_TRADES) -> Dict[Tuple[str, str, str], Dict]:
    """ For all pairs + trading types returns a set of metrics and saves them to the table """
    ...

def filter_metrics(metrics: dict) -> Tuple[str, str, str]:
    ...

def main():

    clear_results()

    # Load last 18 months of the coins
    all_coins = load_all_coins()
    print(f"Loaded {len(all_coins)} coins for analysis")

    # Generates trades for all possible pairs and saves them
    generate_trades_for_all_pairs(all_coins)

    # # Evaluates each pair including their trading type
    # metrics = get_metrics_for_all_pairs_and_trading_types()

    # # From all mrtrics, select pairs with the best results based on pre-defined criteria
    # selected_pairs = filter_metrics(metrics)

if __name__ == "__main__":
    main()
