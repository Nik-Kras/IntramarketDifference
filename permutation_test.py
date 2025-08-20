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
from run_all import load_one_csv, extract_coin_name, calculate_max_drawdown, cmma, threshold_revert_signal, align_frames, MIN_OVERLAP

# Configuration
COMMISSION_RATE = 0.002  # 0.2% per trade
N_PERMUTATIONS = 500
DATA_DIR = "data"
RESULTS_CSV = "pair_backtest_results.csv"
OUTPUT_DIR = "permutations"

# Trading algorithm parameters (from run_all.py)
LOOKBACK = 24               # MA window for cmma
ATR_LOOKBACK = 168          # ATR window for cmma
THRESHOLD = 0.25            # signal threshold

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
    """Calculate profit factor from returns array (assumes simple returns, not log returns)."""
    if len(returns) == 0:
        return np.nan
    
    gains = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0]).sum()
    
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    
    return gains / losses if losses > 0 else np.nan

def run_algorithm_simulation(reference_coin_data: pd.DataFrame, trading_coin_data: pd.DataFrame, 
                           initial_capital: float = 1000.0) -> tuple[pd.Series, pd.Series, float, float]:
    """
    Run the actual trading algorithm and return equity curve.
    Match the exact logic from run_all.py run_pair function.
    
    Args:
        reference_coin_data: Reference coin data (e.g., ETH data when ETH is reference)
        trading_coin_data: Trading coin data (e.g., BTC data when BTC is being traded)
    
    Returns:
        equity_curve: Time series of portfolio value
        returns_series: Time series of returns
        final_profit_factor: Profit factor of the simulation
        final_drawdown: Maximum drawdown of the simulation
    """
    # Align the data frames (same as run_all.py)
    ref_df, traded_df = align_frames(reference_coin_data, trading_coin_data)
    
    if len(ref_df) < MIN_OVERLAP:  # Use same minimum as run_all.py
        return pd.Series(), pd.Series(), np.nan, np.nan
    
    # Exact copy of run_all.py logic
    traded_df = traded_df.copy()
    traded_df["diff"] = np.log(traded_df["close"]).diff()
    traded_df["next_return"] = traded_df["diff"].shift(-1)
    
    # cmma indicators
    ref_cmma = cmma(ref_df, LOOKBACK, ATR_LOOKBACK)
    trd_cmma = cmma(traded_df, LOOKBACK, ATR_LOOKBACK)
    intermarket_diff = trd_cmma - ref_cmma
    
    # signal & returns (exact copy from run_all.py)
    traded_df["sig"] = threshold_revert_signal(intermarket_diff, THRESHOLD)
    rets = traded_df["sig"] * traded_df["next_return"]
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(rets) == 0:
        return pd.Series(), pd.Series(), np.nan, np.nan
    
    # Calculate metrics exactly like run_all.py (convert log returns to simple returns first)
    simple_rets = np.exp(rets) - 1  # Convert log returns to simple returns
    gains = simple_rets[simple_rets > 0].sum()
    losses = simple_rets[simple_rets < 0].abs().sum()
    profit_factor = np.inf if losses == 0 and gains > 0 else (gains / losses if losses > 0 else np.nan)
    
    # Calculate equity curve from log returns
    # rets contains log returns, so we use exp(cumsum) for proper compounding
    equity_curve = initial_capital * np.exp(rets.cumsum())
    
    # Max drawdown
    drawdown = calculate_max_drawdown(rets)
    
    return equity_curve, rets, profit_factor, drawdown

def run_permutation_test(results_df: pd.DataFrame, trading_data: dict):
    """Run permutation test for all trading coins."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Group by Trading Coin
    grouped = results_df.groupby('Trading Coin')
    
    all_results = []
    total_coins = len(grouped.groups)
    
    print(f"Found {total_coins} trading coins to process")
    
    for coin_idx, trading_coin in enumerate(grouped.groups.keys(), 1):
        print(f"\n[{coin_idx}/{total_coins}] Processing {trading_coin}...")
        
        if trading_coin not in trading_data:
            print(f"  ❌ Warning: No data found for {trading_coin}")
            continue
            
        coin_data = trading_data[trading_coin]
        coin_group = grouped.get_group(trading_coin)
        print(f"  📊 Found {len(coin_group)} reference coin pairs")
        
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
            print(f"  ❌ No valid data for {trading_coin}")
            continue
        
        print(f"  ✅ Using {len(valid_rows)} valid pairs for analysis")
        
        # Use the first valid row to get trading characteristics for this coin
        # (all rows for same trading coin should have similar exit hour patterns)
        reference_row = valid_rows[0]
        n_trades = int(reference_row['Number of Trades'])
        exit_hour_mean = reference_row['Mean Exit Hour']
        exit_hour_std = max(reference_row['STD Exit Hour'], 0.1)  # Ensure minimum std
        
        # Generate ONE set of 500 random sequences for this trading coin
        random_profit_factors = []
        random_drawdowns = []
        
        print(f"  🎲 Generating {N_PERMUTATIONS} random sequences...")
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
            print(f"  ❌ No valid random sequences generated for {trading_coin}")
            continue
        
        print(f"  ✅ Generated {len(random_profit_factors)} valid random profit factors")
        
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
        print(f"  📈 Creating permutation test visualizations...")
        create_visualizations(trading_coin, coin_results, random_profit_factors, random_drawdowns, coin_dir)
        
        # Generate equity curves for top-10 performers
        print(f"  💰 Generating equity curves for top performers...")
        create_equity_curves(trading_coin, coin_results, trading_data, coin_dir)
        
        # Generate distribution plots for this trading coin
        print(f"  📊 Creating performance distribution plots...")
        create_coin_distributions(trading_coin, results_df, coin_dir)
        
        print(f"  💾 Saving results...")
        save_coin_results(trading_coin, coin_results, coin_dir)
        
        print(f"  ✅ {trading_coin} completed successfully!")
    
    # Save aggregated results
    print(f"\n🔄 Finalizing results...")
    if len(all_results) > 0:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv('permutation_test_results.csv', index=False)
        print(f"✅ Saved aggregated results to permutation_test_results.csv ({len(all_results)} entries)")
    else:
        print("❌ No valid results to save")
    
    print(f"\n🎉 Permutation test completed for {total_coins} trading coins!")

def create_visualizations(trading_coin: str, coin_results: list, random_profit_factors: list, random_drawdowns: list, output_dir: str):
    """Create distribution plots for profit factor and drawdown."""
    
    # Select top-10 reference coins by HIGHEST profit factors (for profit factor distribution)
    sorted_by_pf = sorted(coin_results, key=lambda x: x['algo_profit_factor'], reverse=True)[:10]
    
    # Select top-10 reference coins by LOWEST drawdowns (for drawdown distribution) 
    sorted_by_drawdown = sorted(coin_results, key=lambda x: x['algo_drawdown'])[-10:]
    
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

def create_equity_curves(trading_coin: str, coin_results: list, trading_data: dict, output_dir: str):
    """Generate equity curves for top-10 reference coins by overall performance."""
    
    if trading_coin not in trading_data:
        print(f"Warning: No data for {trading_coin} to generate equity curves")
        return
    
    trading_coin_data = trading_data[trading_coin]
    
    # Select top-10 reference coins by HIGHEST profit factors
    sorted_by_profit_factor = sorted(coin_results, key=lambda x: x['algo_profit_factor'], reverse=True)[:10]
    top_performers = sorted_by_profit_factor
    
    # Generate equity curves
    equity_curves = {}
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_performers)))
    
    for i, result in enumerate(top_performers):
        reference_coin = result['reference_coin']
        
        if reference_coin not in trading_data:
            continue
            
        reference_coin_data = trading_data[reference_coin]
        
        # Run algorithm simulation (reference_coin_data, trading_coin_data)
        equity_curve, returns_series, pf, dd = run_algorithm_simulation(
            reference_coin_data, trading_coin_data, initial_capital=1000.0
        )
        
        if len(equity_curve) > 0:
            # Plot equity curve
            plt.plot(equity_curve.index, equity_curve.values, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f'{reference_coin} (PF: {pf:.2f}, DD: {dd:.1%}, Q: {result["profit_factor_quantile"]:.1f}%)')
            
            equity_curves[reference_coin] = equity_curve
    
    # Format the plot
    plt.axhline(y=1000, color='black', linestyle='--', alpha=0.5, label='Initial Capital ($1000)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title(f'{trading_coin} - Equity Curves for Top 10 Reference Coins\n(Starting Capital: $1000)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_curves_top10.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual equity curve data to CSV
    if equity_curves:
        # Combine all equity curves into one DataFrame
        equity_df = pd.DataFrame(equity_curves)
        equity_df.index.name = 'Date'
        equity_df.to_csv(os.path.join(output_dir, 'equity_curves_data.csv'))
        
        print(f"    💰 Saved equity curves for {len(equity_curves)} reference coins")

def create_coin_distributions(trading_coin: str, results_df: pd.DataFrame, output_dir: str):
    """Create profit factor and drawdown distribution plots for a specific trading coin."""
    
    # Filter data for this trading coin
    coin_data = results_df[results_df['Trading Coin'] == trading_coin]
    
    # Filter out invalid data
    valid_data = coin_data.dropna(subset=['Profit Factor', 'Max DrawDown'])
    
    if len(valid_data) == 0:
        print(f"  No valid data for {trading_coin} distributions")
        return
    
    profit_factors = valid_data['Profit Factor'].values
    drawdowns = valid_data['Max DrawDown'].values
    
    # Create Profit Factor distribution plot
    plt.figure(figsize=(12, 8))
    
    # Plot histogram
    plt.hist(profit_factors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add statistics
    mean_pf = np.mean(profit_factors)
    median_pf = np.median(profit_factors)
    std_pf = np.std(profit_factors)
    
    # Add vertical lines for statistics
    plt.axvline(mean_pf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pf:.3f}')
    plt.axvline(median_pf, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_pf:.3f}')
    plt.axvline(1.0, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Break-even (1.0)')
    
    # Format plot
    plt.xlabel('Profit Factor')
    plt.ylabel('Frequency')
    plt.title(f'{trading_coin} - Profit Factor Distribution\n'
             f'Mean: {mean_pf:.3f}, Median: {median_pf:.3f}, Std: {std_pf:.3f}\n'
             f'Total Pairs: {len(valid_data)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with summary statistics
    textstr = f'Above 1.0: {np.sum(profit_factors > 1.0)} ({100*np.sum(profit_factors > 1.0)/len(profit_factors):.1f}%)\n'
    textstr += f'Above 1.2: {np.sum(profit_factors > 1.2)} ({100*np.sum(profit_factors > 1.2)/len(profit_factors):.1f}%)\n'
    textstr += f'Above 1.5: {np.sum(profit_factors > 1.5)} ({100*np.sum(profit_factors > 1.5)/len(profit_factors):.1f}%)'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'profit_factor_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create Drawdown distribution plot
    plt.figure(figsize=(12, 8))
    
    # Convert to percentage for better readability
    drawdowns_pct = drawdowns * 100
    
    # Plot histogram
    plt.hist(drawdowns_pct, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    
    # Add statistics
    mean_dd = np.mean(drawdowns_pct)
    median_dd = np.median(drawdowns_pct)
    std_dd = np.std(drawdowns_pct)
    
    # Add vertical lines for statistics
    plt.axvline(mean_dd, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dd:.1f}%')
    plt.axvline(median_dd, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_dd:.1f}%')
    
    # Add reference lines for common drawdown thresholds
    plt.axvline(-10, color='green', linestyle='-', linewidth=1, alpha=0.6, label='10% DD')
    plt.axvline(-20, color='yellow', linestyle='-', linewidth=1, alpha=0.6, label='20% DD')
    plt.axvline(-50, color='red', linestyle='-', linewidth=1, alpha=0.6, label='50% DD')
    
    # Format plot
    plt.xlabel('Maximum Drawdown (%)')
    plt.ylabel('Frequency')
    plt.title(f'{trading_coin} - Maximum Drawdown Distribution\n'
             f'Mean: {mean_dd:.1f}%, Median: {median_dd:.1f}%, Std: {std_dd:.1f}%\n'
             f'Total Pairs: {len(valid_data)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with summary statistics
    textstr = f'Better than -10%: {np.sum(drawdowns_pct > -10)} ({100*np.sum(drawdowns_pct > -10)/len(drawdowns_pct):.1f}%)\n'
    textstr += f'Better than -20%: {np.sum(drawdowns_pct > -20)} ({100*np.sum(drawdowns_pct > -20)/len(drawdowns_pct):.1f}%)\n'
    textstr += f'Worse than -50%: {np.sum(drawdowns_pct < -50)} ({100*np.sum(drawdowns_pct < -50)/len(drawdowns_pct):.1f}%)'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    📊 Saved distribution plots ({len(valid_data)} pairs analyzed)")

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
    print("🚀 Starting Permutation Test Analysis")
    print("=" * 50)
    
    print("📂 Loading trading results...")
    if not os.path.exists(RESULTS_CSV):
        print(f"❌ Error: {RESULTS_CSV} not found. Please run run_all.py first.")
        return
    
    results_df = pd.read_csv(RESULTS_CSV)
    print(f"✅ Loaded {len(results_df)} trading results")
    
    print("\n📂 Loading historical trading data...")
    trading_data = load_trading_data()
    print(f"✅ Loaded price data for {len(trading_data)} coins")
    
    print(f"\n🔄 Creating output directory: {OUTPUT_DIR}/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n🎲 Starting permutation tests with {N_PERMUTATIONS} random sequences per coin...")
    print("=" * 50)
    
    run_permutation_test(results_df, trading_data)
    
    print("\n" + "=" * 50)
    print(f"🎉 Analysis completed! Results saved in:")
    print(f"   📁 {OUTPUT_DIR}/ directory (individual coin results)")
    print(f"   📄 permutation_test_results.csv (aggregated results)")
    print("=" * 50)

if __name__ == "__main__":
    main()