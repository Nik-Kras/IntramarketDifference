import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
RESULTS_CSV = "pair_backtest_results.csv"
OUTPUT_DIR = "permutations"

def create_distribution_plots():
    """Create profit factor and drawdown distribution plots for each trading coin."""
    
    # Load results
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: {RESULTS_CSV} not found. Please run run_all.py first.")
        return
    
    print(f"Loading results from {RESULTS_CSV}...")
    df = pd.read_csv(RESULTS_CSV)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Group by Trading Coin
    grouped = df.groupby('Trading Coin')
    
    print(f"Found {len(grouped.groups)} trading coins to process...")
    
    for trading_coin, coin_data in grouped:
        print(f"Processing {trading_coin}...")
        
        # Create coin directory
        coin_dir = os.path.join(OUTPUT_DIR, trading_coin)
        os.makedirs(coin_dir, exist_ok=True)
        
        # Filter out invalid data
        valid_data = coin_data.dropna(subset=['Profit Factor', 'Max DrawDown'])
        
        if len(valid_data) == 0:
            print(f"  No valid data for {trading_coin}")
            continue
        
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
        plt.savefig(os.path.join(coin_dir, 'profit_factor_dist.png'), dpi=300, bbox_inches='tight')
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
        plt.savefig(os.path.join(coin_dir, 'drawdown_dist.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved distribution plots for {trading_coin} ({len(valid_data)} pairs)")
    
    print(f"\nCompleted! Distribution plots saved in {OUTPUT_DIR}/ subdirectories")
    print("Each trading coin folder now contains:")
    print("  - profit_factor_dist.png: Profit factor distribution")
    print("  - drawdown_dist.png: Maximum drawdown distribution")

def create_summary_stats():
    """Create a summary CSV with statistics for each trading coin."""
    
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: {RESULTS_CSV} not found.")
        return
    
    df = pd.read_csv(RESULTS_CSV)
    grouped = df.groupby('Trading Coin')
    
    summary_stats = []
    
    for trading_coin, coin_data in grouped:
        valid_data = coin_data.dropna(subset=['Profit Factor', 'Max DrawDown'])
        
        if len(valid_data) == 0:
            continue
        
        profit_factors = valid_data['Profit Factor'].values
        drawdowns = valid_data['Max DrawDown'].values * 100  # Convert to percentage
        
        stats = {
            'Trading Coin': trading_coin,
            'Total Pairs': len(valid_data),
            'PF Mean': np.mean(profit_factors),
            'PF Median': np.median(profit_factors),
            'PF Std': np.std(profit_factors),
            'PF Min': np.min(profit_factors),
            'PF Max': np.max(profit_factors),
            'PF Above 1.0 (%)': 100 * np.sum(profit_factors > 1.0) / len(profit_factors),
            'PF Above 1.2 (%)': 100 * np.sum(profit_factors > 1.2) / len(profit_factors),
            'PF Above 1.5 (%)': 100 * np.sum(profit_factors > 1.5) / len(profit_factors),
            'DD Mean (%)': np.mean(drawdowns),
            'DD Median (%)': np.median(drawdowns),
            'DD Std (%)': np.std(drawdowns),
            'DD Best (%)': np.max(drawdowns),  # Closest to 0
            'DD Worst (%)': np.min(drawdowns), # Most negative
            'DD Better than -10% (%)': 100 * np.sum(drawdowns > -10) / len(drawdowns),
            'DD Better than -20% (%)': 100 * np.sum(drawdowns > -20) / len(drawdowns),
            'DD Worse than -50% (%)': 100 * np.sum(drawdowns < -50) / len(drawdowns)
        }
        
        summary_stats.append(stats)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats)
    
    # Sort by mean profit factor (descending)
    summary_df = summary_df.sort_values('PF Mean', ascending=False)
    
    # Save to CSV
    summary_df.to_csv('trading_coin_summary_stats.csv', index=False, float_format='%.3f')
    print(f"\nSaved summary statistics to trading_coin_summary_stats.csv")

def main():
    """Main execution function."""
    print("Creating distribution plots for each trading coin...")
    create_distribution_plots()
    
    print("\nCreating summary statistics...")
    create_summary_stats()

if __name__ == "__main__":
    main()