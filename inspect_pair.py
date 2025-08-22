#!/usr/bin/env python3
"""
Pair Inspection Script - inspect_pair.py

Performs detailed analysis of a specific trading-reference coin pair,
similar to permutation_test.py but focused on one pair combination.

Usage:
    python inspect_pair.py TRADING_COIN REFERENCE_COIN [TRADE_TYPE]

Arguments:
    TRADING_COIN  - The coin being traded (e.g., ETH)
    REFERENCE_COIN - The reference coin for signals (e.g., BTC)  
    TRADE_TYPE    - Optional: 'both', 'long', or 'short' (default: 'both')

Examples:
    python inspect_pair.py ETH BTC          # Analyze both long and short trades
    python inspect_pair.py ETH BTC long     # Analyze only long trades
    python inspect_pair.py ETH BTC short    # Analyze only short trades

Author: IntramarketDifference Analysis
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import functions from existing modules
from run_all import load_one_csv, calculate_max_drawdown
from permutation_test import (
    generate_random_sequence, calculate_profit_factor, 
    analyze_trades_by_type, get_equity_curve_by_type
)

# Configuration
DATA_DIR = "data"
TRADES_DIR = "trades"
OUTPUT_DIR_BASE = "inspection"
N_PERMUTATIONS = 500
COMMISSION_RATE = 0.002
INITIAL_CAPITAL = 1000.0

# Trading algorithm parameters
LOOKBACK = 24
ATR_LOOKBACK = 168
THRESHOLD = 0.25


def load_coin_data(coin_name: str) -> pd.DataFrame:
    """Load data for a specific coin."""
    coin_file = f"{coin_name}USDT_IS.csv"
    coin_path = os.path.join(DATA_DIR, coin_file)
    
    if not os.path.exists(coin_path):
        raise FileNotFoundError(f"Data file not found: {coin_path}")
    
    return load_one_csv(coin_path)


def load_pair_trades(trading_coin: str, reference_coin: str) -> list:
    """Load trade data for the specific pair."""
    filename = f"{reference_coin}_{trading_coin}_trades.json"
    filepath = os.path.join(TRADES_DIR, trading_coin, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trade data not found: {filepath}")
    
    with open(filepath, 'r') as f:
        trades_data = json.load(f)
    
    return trades_data


def create_pair_distributions(trading_coin: str, reference_coin: str, trades_data: list, 
                            random_profit_factors: list, random_drawdowns: list, 
                            output_dir: str, trade_type_filter: str = 'both'):
    """Create distribution plots comparing algorithm performance vs random."""
    
    # Filter trade types based on user selection
    if trade_type_filter == 'both':
        trade_types = [('combined', 'both', 'Combined (Longs + Shorts)', 'blue')]
    elif trade_type_filter == 'long':
        trade_types = [('longs', 'long', 'Longs Only', 'green')]
    elif trade_type_filter == 'short':
        trade_types = [('shorts', 'short', 'Shorts Only', 'red')]
    else:
        # Default to all types for backwards compatibility
        trade_types = [
            ('combined', 'both', 'Combined (Longs + Shorts)', 'blue'),
            ('longs', 'long', 'Longs Only', 'green'), 
            ('shorts', 'short', 'Shorts Only', 'red')
        ]
    
    for trade_type_name, trade_filter, title_suffix, base_color in trade_types:
        # Get metrics for this trade type
        profit_factor, max_drawdown, num_trades = analyze_trades_by_type(trades_data, trade_filter)
        
        if num_trades == 0 or pd.isna(profit_factor):
            print(f"    ⚠️ No valid {trade_type_name} trades for {trading_coin}-{reference_coin}")
            continue
        
        # Calculate quantiles
        pf_quantile = 100 - stats.percentileofscore(random_profit_factors, profit_factor)
        dd_quantile = 100 - stats.percentileofscore(random_drawdowns, max_drawdown)
        
        print(f"    📊 {title_suffix}:")
        print(f"        Profit Factor: {profit_factor:.2f} (Top {pf_quantile:.1f}%)")
        print(f"        Max Drawdown: {max_drawdown:.1%} (Top {dd_quantile:.1f}%)")
        print(f"        Number of Trades: {num_trades}")
        
        # Create profit factor distribution plot
        plt.figure(figsize=(12, 8))
        plt.hist(random_profit_factors, bins=50, alpha=0.7, color='lightblue', 
                label=f'Random Trades (n={len(random_profit_factors)})')
        plt.axvline(profit_factor, color=base_color, linewidth=3, 
                   label=f'{reference_coin}→{trading_coin}: {profit_factor:.2f} (Top {pf_quantile:.1f}%)')
        
        plt.xlabel('Profit Factor')
        plt.ylabel('Frequency')
        plt.title(f'{trading_coin} vs {reference_coin} - Profit Factor Distribution\n{title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'profit_factor_distribution_{trade_type_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create drawdown distribution plot
        plt.figure(figsize=(12, 8))
        plt.hist(random_drawdowns, bins=50, alpha=0.7, color='lightcoral', 
                label=f'Random Trades (n={len(random_drawdowns)})')
        plt.axvline(max_drawdown, color=base_color, linewidth=3,
                   label=f'{reference_coin}→{trading_coin}: {max_drawdown:.1%} (Top {dd_quantile:.1f}%)')
        
        plt.xlabel('Max Drawdown')
        plt.ylabel('Frequency')
        plt.title(f'{trading_coin} vs {reference_coin} - Drawdown Distribution\n{title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'drawdown_distribution_{trade_type_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def create_equity_curves(trading_coin: str, reference_coin: str, trades_data: list, 
                        trading_coin_data: pd.DataFrame, reference_coin_data: pd.DataFrame, 
                        output_dir: str, trade_type_filter: str = 'both'):
    """Create equity curves for different trade types."""
    
    # Filter trade types based on user selection
    if trade_type_filter == 'both':
        trade_types = [('combined', 'both', 'Combined (Longs + Shorts)', 'blue')]
    elif trade_type_filter == 'long':
        trade_types = [('longs', 'long', 'Longs Only', 'green')]
    elif trade_type_filter == 'short':
        trade_types = [('shorts', 'short', 'Shorts Only', 'red')]
    else:
        # Default to all types for backwards compatibility
        trade_types = [
            ('combined', 'both', 'Combined (Longs + Shorts)', 'blue'),
            ('longs', 'long', 'Longs Only', 'green'), 
            ('shorts', 'short', 'Shorts Only', 'red')
        ]
    
    plt.figure(figsize=(15, max(4 * len(trade_types), 6)))
    
    for i, (trade_type_name, trade_filter, title_suffix, color) in enumerate(trade_types):
        equity_curve, pf, dd = get_equity_curve_by_type(
            reference_coin_data, trading_coin_data, trading_coin, reference_coin, 
            trade_filter, initial_capital=INITIAL_CAPITAL
        )
        
        if len(equity_curve) > 0:
            plt.subplot(len(trade_types), 1, i + 1)
            plt.plot(equity_curve.index, equity_curve.values, color=color, linewidth=2)
            plt.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', alpha=0.5, 
                       label=f'Initial Capital (${INITIAL_CAPITAL})')
            
            plt.title(f'{trading_coin} vs {reference_coin} - {title_suffix}\n'
                     f'Final Value: ${equity_curve.iloc[-1]:.0f}, PF: {pf:.2f}, DD: {dd:.1%}')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Save individual equity curve data
            equity_curve.to_csv(os.path.join(output_dir, f'equity_curve_{trade_type_name}.csv'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_curves_all_types.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_trade_analysis(trading_coin: str, reference_coin: str, trades_data: list, 
                         output_dir: str, trade_type_filter: str = 'both'):
    """Create detailed trade analysis."""
    
    if not trades_data:
        print("    ⚠️ No trade data available for analysis")
        return
    
    # Convert to DataFrame for analysis
    trades_df = pd.DataFrame(trades_data)
    
    # Filter by trade type if specified
    if trade_type_filter == 'long':
        trades_df = trades_df[trades_df['trade_type'] == 'long']
    elif trade_type_filter == 'short':
        trades_df = trades_df[trades_df['trade_type'] == 'short']
    # 'both' keeps all trades
    
    if len(trades_df) == 0:
        print(f"    ⚠️ No {trade_type_filter} trades available for analysis")
        return {}
    
    trades_df['time_entered'] = pd.to_datetime(trades_df['time_entered'])
    trades_df['time_exited'] = pd.to_datetime(trades_df['time_exited'])
    trades_df['holding_period_hours'] = (trades_df['time_exited'] - trades_df['time_entered']).dt.total_seconds() / 3600
    
    # Create comprehensive analysis
    _, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Returns distribution
    returns = trades_df['log_return'].dropna()
    axes[0, 0].hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.3f}')
    axes[0, 0].axvline(returns.median(), color='orange', linestyle='--', label=f'Median: {returns.median():.3f}')
    axes[0, 0].set_title('Trade Returns Distribution')
    axes[0, 0].set_xlabel('Log Return')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Returns by trade type
    long_returns = trades_df[trades_df['trade_type'] == 'long']['log_return'].dropna()
    short_returns = trades_df[trades_df['trade_type'] == 'short']['log_return'].dropna()
    
    axes[0, 1].hist(long_returns, bins=20, alpha=0.7, color='green', label=f'Longs (n={len(long_returns)})', edgecolor='black')
    axes[0, 1].hist(short_returns, bins=20, alpha=0.7, color='red', label=f'Shorts (n={len(short_returns)})', edgecolor='black')
    axes[0, 1].set_title('Returns by Trade Type')
    axes[0, 1].set_xlabel('Log Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Holding periods
    holding_periods = trades_df['holding_period_hours'].dropna()
    axes[0, 2].hist(holding_periods, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 2].axvline(holding_periods.mean(), color='red', linestyle='--', 
                      label=f'Mean: {holding_periods.mean():.1f}h')
    axes[0, 2].axvline(holding_periods.median(), color='orange', linestyle='--', 
                      label=f'Median: {holding_periods.median():.1f}h')
    axes[0, 2].set_title('Holding Period Distribution')
    axes[0, 2].set_xlabel('Hours')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Returns over time
    trades_df_sorted = trades_df.sort_values('time_exited')
    axes[1, 0].scatter(trades_df_sorted['time_exited'], trades_df_sorted['log_return'], 
                      alpha=0.6, s=20, color='blue')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_title('Trade Returns Over Time')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Log Return')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Cumulative returns
    cumulative_returns = trades_df_sorted['log_return'].fillna(0).cumsum()
    axes[1, 1].plot(trades_df_sorted['time_exited'], np.exp(cumulative_returns) * INITIAL_CAPITAL, 
                   linewidth=2, color='green')
    axes[1, 1].axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Cumulative Portfolio Value')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Portfolio Value ($)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Monthly returns heatmap
    trades_df_sorted['year_month'] = trades_df_sorted['time_exited'].dt.to_period('M')
    monthly_returns = trades_df_sorted.groupby('year_month')['log_return'].sum()
    
    if len(monthly_returns) > 1:
        # Create a simple bar chart instead of heatmap for simplicity
        axes[1, 2].bar(range(len(monthly_returns)), monthly_returns.values, alpha=0.7)
        axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 2].set_title('Monthly Returns')
        axes[1, 2].set_xlabel('Month')
        axes[1, 2].set_ylabel('Log Return')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Set x-axis labels to show some month labels
        if len(monthly_returns) > 10:
            step = len(monthly_returns) // 10
            axes[1, 2].set_xticks(range(0, len(monthly_returns), step))
            axes[1, 2].set_xticklabels([str(monthly_returns.index[i]) for i in range(0, len(monthly_returns), step)])
        else:
            axes[1, 2].set_xticks(range(len(monthly_returns)))
            axes[1, 2].set_xticklabels([str(idx) for idx in monthly_returns.index])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trade_analysis_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed statistics
    stats_data = {
        'pair': f"{reference_coin} → {trading_coin}",
        'trade_type_filter': trade_type_filter,
        'total_trades': len(trades_df),
        'long_trades': len(long_returns),
        'short_trades': len(short_returns),
        'mean_return': float(returns.mean()),
        'median_return': float(returns.median()),
        'std_return': float(returns.std()),
        'mean_long_return': float(long_returns.mean()) if len(long_returns) > 0 else None,
        'mean_short_return': float(short_returns.mean()) if len(short_returns) > 0 else None,
        'mean_holding_period_hours': float(holding_periods.mean()),
        'median_holding_period_hours': float(holding_periods.median()),
        'total_log_return': float(returns.sum()),
        'total_simple_return': float(np.exp(returns.sum()) - 1),
        'win_rate': float((returns > 0).mean()),
        'best_trade': float(returns.max()),
        'worst_trade': float(returns.min()),
    }
    
    with open(os.path.join(output_dir, 'trade_statistics.json'), 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    return stats_data


def main():
    """Main execution function."""
    # Parse command line arguments
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python inspect_pair.py TRADING_COIN REFERENCE_COIN [TRADE_TYPE]")
        print("Examples:")
        print("  python inspect_pair.py ETH BTC          # Analyze both long and short trades")
        print("  python inspect_pair.py ETH BTC long     # Analyze only long trades")
        print("  python inspect_pair.py ETH BTC short    # Analyze only short trades")
        sys.exit(1)
    
    trading_coin = sys.argv[1].upper()
    reference_coin = sys.argv[2].upper()
    trade_type_filter = sys.argv[3].lower() if len(sys.argv) == 4 else 'both'
    
    # Validate trade type
    if trade_type_filter not in ['both', 'long', 'short']:
        print(f"❌ Error: Invalid trade type '{trade_type_filter}'. Must be 'both', 'long', or 'short'")
        sys.exit(1)
    
    print(f"🔍 Starting Detailed Pair Inspection")
    print(f"📊 Trading Coin: {trading_coin}")
    print(f"📈 Reference Coin: {reference_coin}")
    print(f"🎯 Trade Type Filter: {trade_type_filter}")
    print("=" * 60)
    
    # Create output directory with trade type
    output_dir = os.path.join(OUTPUT_DIR_BASE, f"{reference_coin}_{trading_coin}_{trade_type_filter}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Load coin data
        print(f"\n📂 Loading coin data...")
        trading_coin_data = load_coin_data(trading_coin)
        reference_coin_data = load_coin_data(reference_coin)
        print(f"✅ Loaded {len(trading_coin_data)} records for {trading_coin}")
        print(f"✅ Loaded {len(reference_coin_data)} records for {reference_coin}")
        
        # Load trade data
        print(f"\n💼 Loading trade data...")
        trades_data = load_pair_trades(trading_coin, reference_coin)
        print(f"✅ Loaded {len(trades_data)} trades for {reference_coin}→{trading_coin}")
        
        # Generate random baseline for comparison
        print(f"\n🎲 Generating {N_PERMUTATIONS} random trade sequences for comparison...")
        
        # Use first trade's characteristics for random generation
        if trades_data:
            # Get average holding period from actual trades
            trades_df_temp = pd.DataFrame(trades_data)
            trades_df_temp['time_entered'] = pd.to_datetime(trades_df_temp['time_entered'])
            trades_df_temp['time_exited'] = pd.to_datetime(trades_df_temp['time_exited'])
            holding_periods = (trades_df_temp['time_exited'] - trades_df_temp['time_entered']).dt.total_seconds() / 3600
            
            exit_hour_mean = float(holding_periods.mean())
            exit_hour_std = max(float(holding_periods.std()), 0.1)
            n_trades = len(trades_data)
        else:
            # Default values if no trades
            exit_hour_mean = 24.0
            exit_hour_std = 12.0
            n_trades = 100
        
        random_profit_factors = []
        random_drawdowns = []
        
        for _ in tqdm(range(N_PERMUTATIONS), desc="Generating random sequences"):
            random_returns = generate_random_sequence(trading_coin_data, n_trades, exit_hour_mean, exit_hour_std)
            
            pf = calculate_profit_factor(random_returns)
            dd = calculate_max_drawdown(pd.Series(random_returns))
            
            if not pd.isna(pf) and not np.isinf(pf):
                random_profit_factors.append(pf)
            if not pd.isna(dd):
                random_drawdowns.append(dd)
        
        print(f"✅ Generated {len(random_profit_factors)} random profit factors")
        print(f"✅ Generated {len(random_drawdowns)} random drawdowns")
        
        # Create visualizations and analysis
        print(f"\n📊 Creating distribution plots...")
        create_pair_distributions(trading_coin, reference_coin, trades_data, 
                                random_profit_factors, random_drawdowns, output_dir, trade_type_filter)
        
        print(f"\n📈 Creating equity curves...")
        create_equity_curves(trading_coin, reference_coin, trades_data, 
                           trading_coin_data, reference_coin_data, output_dir, trade_type_filter)
        
        print(f"\n🔬 Creating detailed trade analysis...")
        trade_stats = create_trade_analysis(trading_coin, reference_coin, trades_data, 
                                           output_dir, trade_type_filter)
        
        # Save summary report
        print(f"\n📋 Creating summary report...")
        
        # Calculate metrics for the specified trade type
        target_pf, target_dd, target_trades = analyze_trades_by_type(trades_data, trade_type_filter)
        
        # Also get breakdown by individual trade types for report
        long_pf, long_dd, long_trades = analyze_trades_by_type(trades_data, 'long')
        short_pf, short_dd, short_trades = analyze_trades_by_type(trades_data, 'short')
        
        # Calculate quantiles vs random for target trade type
        if target_pf and not pd.isna(target_pf) and random_profit_factors:
            target_pf_quantile = 100 - stats.percentileofscore(random_profit_factors, target_pf)
            target_dd_quantile = 100 - stats.percentileofscore(random_drawdowns, target_dd)
        else:
            target_pf_quantile = target_dd_quantile = np.nan
        
        # Helper function to safely format values
        def safe_format(value, format_str, default="N/A"):
            if value is None or pd.isna(value):
                return default
            try:
                return format_str.format(value)
            except:
                return default
        
        def safe_percent(value, decimals=1, default="N/A"):
            if value is None or pd.isna(value):
                return default
            try:
                return f"{value*100:.{decimals}f}%"
            except:
                return default
        
        summary_report = f"""
# Pair Inspection Report: {reference_coin} → {trading_coin}

## Analysis Focus: {trade_type_filter.upper()} Trades

## {trade_type_filter.title()} Performance  
- **Profit Factor**: {safe_format(target_pf, "{:.3f}", "N/A")} (Top {safe_format(target_pf_quantile, "{:.1f}%", "N/A")} vs random)
- **Max Drawdown**: {safe_percent(target_dd)} (Top {safe_format(target_dd_quantile, "{:.1f}%", "N/A")} vs random)
- **Total Trades**: {target_trades or 0}
- **Total Return**: {safe_percent(trade_stats.get('total_simple_return'), 1)}
- **Win Rate**: {safe_percent(trade_stats.get('win_rate'), 1)}

## Performance by Trade Type
### Long Trades
- **Profit Factor**: {safe_format(long_pf, "{:.3f}", "N/A")}
- **Max Drawdown**: {safe_percent(long_dd)}
- **Number of Trades**: {long_trades or 0}
- **Avg Return**: {safe_format(trade_stats.get('mean_long_return'), "{:.4f}", "N/A")}

### Short Trades  
- **Profit Factor**: {safe_format(short_pf, "{:.3f}", "N/A")}
- **Max Drawdown**: {safe_percent(short_dd)}
- **Number of Trades**: {short_trades or 0}
- **Avg Return**: {safe_format(trade_stats.get('mean_short_return'), "{:.4f}", "N/A")}

## Trade Characteristics
- **Mean Holding Period**: {safe_format(trade_stats.get('mean_holding_period_hours'), "{:.1f} hours", "N/A")}
- **Median Holding Period**: {safe_format(trade_stats.get('median_holding_period_hours'), "{:.1f} hours", "N/A")}
- **Best Trade**: {safe_percent(trade_stats.get('best_trade'), 2)}
- **Worst Trade**: {safe_percent(trade_stats.get('worst_trade'), 2)}
- **Return Std**: {safe_format(trade_stats.get('std_return'), "{:.4f}", "N/A")}

## Files Generated
- `profit_factor_distribution_{trade_type_filter}.png` - Profit factor vs random comparison
- `drawdown_distribution_{trade_type_filter}.png` - Drawdown vs random comparison  
- `equity_curves_all_types.png` - Portfolio value over time
- `equity_curve_{trade_type_filter}.csv` - Equity curve data
- `trade_analysis_detailed.png` - Comprehensive trade analysis
- `trade_statistics.json` - Detailed statistics in JSON format
- `summary_report.md` - This summary report

## Random Baseline Statistics
- **Random Profit Factors**: Mean={np.mean(random_profit_factors):.2f}, Std={np.std(random_profit_factors):.2f}
- **Random Drawdowns**: Mean={np.mean(random_drawdowns):.1%}, Std={np.std(random_drawdowns):.1%}
- **Sample Size**: {N_PERMUTATIONS} random sequences

---
Generated by IntramarketDifference Pair Inspection Tool
"""
        
        with open(os.path.join(output_dir, 'summary_report.md'), 'w') as f:
            f.write(summary_report)
        
        print(f"\n🎉 Analysis completed!")
        print(f"📁 All results saved in: {output_dir}")
        print(f"📋 Check summary_report.md for key findings")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure the coin names are correct and data files exist.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()