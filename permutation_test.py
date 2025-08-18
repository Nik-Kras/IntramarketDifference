import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import functions from run_all.py
from run_all import load_one_csv, extract_coin_name, calculate_max_drawdown

# Configuration
COMMISSION_RATE = 0.002  # 0.2% per trade
N_PERMUTATIONS = 500
DATA_DIR = "data"
RESULTS_CSV = "pair_backtest_results.csv"
OUTPUT_DIR = "permutations"

def load_trading_data():
    """Load all trading coin data."""
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    data = {}
    for p in paths:
        try:
            coin_name = extract_coin_name(p)
            data[coin_name] = load_one_csv(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")
    return data

def generate_random_trade(coin_data: pd.DataFrame, direction: int, exit_hour_mean: float, exit_hour_std: float) -> float:
    """
    Generate a single random trade return.
    
    Args:
        coin_data: Historical price data for the trading coin
        direction: 1 for long, -1 for short
        exit_hour_mean: Mean exit hour from original algorithm
        exit_hour_std: Std exit hour from original algorithm
    
    Returns:
        Trade return (including commission)
    """
    try:
        # Randomly select entry point
        entry_idx = np.random.randint(0, len(coin_data) - 1)
        entry_time = coin_data.index[entry_idx]
        entry_price = coin_data.iloc[entry_idx]['close']
        
        # Sample exit hour from Gaussian distribution (clamp to 0-23 range)
        exit_hour_offset = np.random.normal(exit_hour_mean, max(exit_hour_std, 0.1))  # min std to avoid zero
        exit_hour_offset = max(1, min(168, exit_hour_offset))  # 1-168 hours (1 week max)
        
        # Calculate exit time
        exit_time = entry_time + pd.Timedelta(hours=exit_hour_offset)
        
        # Find closest available exit time in data
        available_times = coin_data.index[coin_data.index >= exit_time]
        if len(available_times) == 0:
            return 0.0  # No exit time available
            
        exit_time_actual = available_times[0]
        exit_price = coin_data.loc[exit_time_actual, 'close']
        
        # Calculate raw return
        if direction == 1:  # Long trade
            raw_return = (exit_price - entry_price) / entry_price
        else:  # Short trade
            raw_return = (entry_price - exit_price) / entry_price
        
        # Apply commission (0.2% entry + 0.2% exit = 0.4% total)
        return raw_return - 2 * COMMISSION_RATE
        
    except Exception:
        return 0.0

def generate_random_sequence(coin_data: pd.DataFrame, n_trades: int, exit_hour_mean: float, exit_hour_std: float) -> np.ndarray:
    """Generate a sequence of random trades."""
    returns = []
    
    for _ in range(n_trades):
        # Randomly choose long or short
        direction = np.random.choice([1, -1])
        
        # Generate trade return
        trade_return = generate_random_trade(coin_data, direction, exit_hour_mean, exit_hour_std)
        returns.append(trade_return)
    
    return np.array(returns)

def calculate_profit_factor(returns: np.ndarray) -> float:
    """Calculate profit factor from returns array."""
    if len(returns) == 0:
        return np.nan
    
    gains = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0]).sum()
    
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    
    return gains / losses if losses > 0 else np.nan

def run_permutation_test(results_df: pd.DataFrame, trading_data: dict):
    """Run permutation test for all trading coins."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Group by Trading Coin
    grouped = results_df.groupby('Trading Coin')
    
    all_results = []
    
    for trading_coin in tqdm(grouped.groups.keys(), desc="Processing coins"):
        if trading_coin not in trading_data:
            print(f"Warning: No data found for {trading_coin}")
            continue
            
        coin_data = trading_data[trading_coin]
        coin_group = grouped.get_group(trading_coin)
        
        # Create coin directory
        coin_dir = os.path.join(OUTPUT_DIR, trading_coin)
        os.makedirs(coin_dir, exist_ok=True)
        
        # Filter valid rows
        valid_rows = []
        for _, row in coin_group.iterrows():
            if (not pd.isna(row['Number of Trades']) and 
                not pd.isna(row['Mean Exit Hour']) and 
                not pd.isna(row['STD Exit Hour']) and
                int(row['Number of Trades']) > 0):
                valid_rows.append(row)
        
        if len(valid_rows) == 0:
            print(f"No valid data for {trading_coin}")
            continue
        
        # Use the first valid row to get trading characteristics for this coin
        # (all rows for same trading coin should have similar exit hour patterns)
        reference_row = valid_rows[0]
        n_trades = int(reference_row['Number of Trades'])
        exit_hour_mean = reference_row['Mean Exit Hour']
        exit_hour_std = max(reference_row['STD Exit Hour'], 0.1)  # Ensure minimum std
        
        # Generate ONE set of 500 random sequences for this trading coin
        random_profit_factors = []
        random_drawdowns = []
        
        print(f"Generating {N_PERMUTATIONS} random sequences for {trading_coin}...")
        for _ in range(N_PERMUTATIONS):
            random_returns = generate_random_sequence(coin_data, n_trades, exit_hour_mean, exit_hour_std)
            
            # Calculate metrics
            pf = calculate_profit_factor(random_returns)
            dd = calculate_max_drawdown(pd.Series(random_returns))
            
            if not pd.isna(pf) and not np.isinf(pf):
                random_profit_factors.append(pf)
            if not pd.isna(dd):
                random_drawdowns.append(dd)
        
        if len(random_profit_factors) == 0 or len(random_drawdowns) == 0:
            print(f"No valid random sequences generated for {trading_coin}")
            continue
        
        # Now compare all reference coin pairs against this single distribution
        coin_results = []
        for row in valid_rows:
            # Calculate quantiles for this reference coin pair
            # For profit factor: lower quantile = better performance (algorithm beats more random trades)
            pf_quantile = 100 - stats.percentileofscore(random_profit_factors, row['Profit Factor'])
            # For drawdown: lower quantile = better performance (algorithm has lower drawdown than more random trades)  
            dd_quantile = 100 - stats.percentileofscore(random_drawdowns, row['Max DrawDown'])
            
            coin_results.append({
                'reference_coin': row['Reference Coin'],
                'algo_profit_factor': row['Profit Factor'],
                'algo_drawdown': row['Max DrawDown'],
                'profit_factor_quantile': pf_quantile,
                'drawdown_quantile': dd_quantile
            })
            
            all_results.append({
                'Trading Coin': trading_coin,
                'Reference Coin': row['Reference Coin'],
                'Algorithm Profit Factor': row['Profit Factor'],
                'Algorithm Drawdown': row['Max DrawDown'],
                'Profit Factor Quantile': pf_quantile,
                'Drawdown Quantile': dd_quantile
            })
        
        # Create visualizations and save results
        create_visualizations(trading_coin, coin_results, random_profit_factors, random_drawdowns, coin_dir)
        save_coin_results(trading_coin, coin_results, coin_dir)
    
    # Save aggregated results
    if len(all_results) > 0:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv('permutation_test_results.csv', index=False)
        print(f"Saved aggregated results to permutation_test_results.csv")
    else:
        print("No valid results to save")

def create_visualizations(trading_coin: str, coin_results: list, random_profit_factors: list, random_drawdowns: list, output_dir: str):
    """Create distribution plots for profit factor and drawdown."""
    
    # Select top-10 reference coins by drawdown (lower is better) and profit factor (higher is better)
    sorted_by_drawdown = sorted(coin_results, key=lambda x: x['algo_drawdown'])[:10]
    sorted_by_pf = sorted(coin_results, key=lambda x: x['algo_profit_factor'], reverse=True)[:10]
    
    # Create profit factor distribution plot
    plt.figure(figsize=(12, 8))
    
    plt.hist(random_profit_factors, bins=50, alpha=0.7, color='lightblue', label='Random Trades')
    
    # Add vertical lines for top-10 by profit factor
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_by_pf)))
    for i, result in enumerate(sorted_by_pf):
        plt.axvline(result['algo_profit_factor'], color=colors[i], linewidth=2, 
                   label=f"{result['reference_coin']}: {result['profit_factor_quantile']:.1f}%")
    
    plt.xlabel('Profit Factor')
    plt.ylabel('Frequency')
    plt.title(f'{trading_coin} - Profit Factor Distribution (Top 10 by Performance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'profit_factor_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create drawdown distribution plot
    plt.figure(figsize=(12, 8))
    
    plt.hist(random_drawdowns, bins=50, alpha=0.7, color='lightcoral', label='Random Trades')
    
    # Add vertical lines for top-10 by drawdown (best = lowest)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_by_drawdown)))
    for i, result in enumerate(sorted_by_drawdown):
        plt.axvline(result['algo_drawdown'], color=colors[i], linewidth=2,
                   label=f"{result['reference_coin']}: {result['drawdown_quantile']:.1f}%")
    
    plt.xlabel('Max Drawdown')
    plt.ylabel('Frequency')
    plt.title(f'{trading_coin} - Drawdown Distribution (Top 10 by Performance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_coin_results(trading_coin: str, coin_results: list, output_dir: str):
    """Save coin results to JSON file."""
    
    # Prepare data for JSON (remove numpy arrays)
    json_results = []
    for result in coin_results:
        json_results.append({
            'reference_coin': result['reference_coin'],
            'algo_profit_factor': float(result['algo_profit_factor']) if not pd.isna(result['algo_profit_factor']) else None,
            'algo_drawdown': float(result['algo_drawdown']) if not pd.isna(result['algo_drawdown']) else None,
            'profit_factor_quantile': float(result['profit_factor_quantile']),
            'drawdown_quantile': float(result['drawdown_quantile'])
        })
    
    # Save to JSON
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'trading_coin': trading_coin,
            'results': json_results
        }, f, indent=2)

def main():
    """Main execution function."""
    print("Loading trading results...")
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: {RESULTS_CSV} not found. Please run run_all.py first.")
        return
    
    results_df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(results_df)} trading results")
    
    print("Loading trading data...")
    trading_data = load_trading_data()
    print(f"Loaded data for {len(trading_data)} coins")
    
    print("Running permutation tests...")
    run_permutation_test(results_df, trading_data)
    
    print(f"Permutation test completed! Results saved in {OUTPUT_DIR}/ directory")
    print(f"Aggregated results saved to permutation_test_results.csv")

if __name__ == "__main__":
    main()