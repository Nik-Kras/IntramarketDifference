#!/usr/bin/env python3
"""
Optimal Window OOS Analysis Script

Performs comprehensive analysis using the optimal 18-month window:
- Analyzes In-Sample (18mo) to select best pairs
- Loads all OOS trades for selected pairs in chronological order
- Runs portfolio simulation with full visualizations
- Creates weekly trade frequency analysis and top-10 coin ranking
- Enables equity spike correlation analysis
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

# Import functions from existing modules
from window_analyzer_fast import analyze_window_fast as analyze_window, generate_selection_report
from window_portfolio_simulator import WindowPortfolioSimulator

# Configuration
OPTIMAL_WINDOW_MONTHS = 18
TRADES_DIR = "in_sample/trades"
OOS_TRADES_DIR = "out_of_sample/trades"
RESULTS_DIR = "results/optimal_window"
INITIAL_PORTFOLIO_VALUE = 1000.0

def calculate_optimal_window_dates() -> Tuple[str, str]:
    """Calculate 18-month In-Sample window dates."""
    oos_start_date = pd.to_datetime("2024-01-01")
    window_start = oos_start_date - pd.DateOffset(months=OPTIMAL_WINDOW_MONTHS)
    window_end = oos_start_date
    
    return window_start.strftime('%Y-%m-%d'), window_end.strftime('%Y-%m-%d')

def run_in_sample_analysis() -> pd.DataFrame:
    """Run 18-month In-Sample analysis to select optimal pairs."""
    
    print("📊 Step 1: In-Sample Analysis (18-month optimal window)")
    print("=" * 60)
    
    window_start, window_end = calculate_optimal_window_dates()
    print(f"Window period: {window_start} to {window_end}")
    
    # Check if results already exist
    selected_pairs_file = os.path.join(RESULTS_DIR, 'selected_pairs.csv')
    if os.path.exists(selected_pairs_file):
        print(f"📂 Loading existing selected pairs from: {selected_pairs_file}")
        selected_pairs = pd.read_csv(selected_pairs_file)
        print(f"✅ Loaded {len(selected_pairs)} pre-selected optimal pairs")
        return selected_pairs
    
    # Use the fast analyzer to get selected pairs
    print("🔄 Running In-Sample analysis (this may take several minutes)...")
    _, selected_pairs, filter_stats = analyze_window(window_start, window_end, TRADES_DIR)
    
    if len(selected_pairs) == 0:
        raise ValueError("No pairs selected from 18-month window analysis!")
    
    print(f"✅ In-Sample analysis complete!")
    print(f"   Selected {len(selected_pairs)} optimal pairs from filtering")
    
    # Save In-Sample results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    selected_pairs.to_csv(os.path.join(RESULTS_DIR, 'selected_pairs.csv'), 
                         index=False, float_format='%.6f')
    
    # Generate selection report
    generate_selection_report(filter_stats, window_start, window_end, 
                            os.path.join(RESULTS_DIR, 'selection_report.txt'))
    
    return selected_pairs

def load_chronological_oos_trades(selected_pairs: pd.DataFrame) -> pd.DataFrame:
    """Load all OOS trades for selected pairs in chronological order."""
    
    print("\n📂 Step 2: Loading chronological OOS trade data")
    print("=" * 60)
    
    # Validation check
    if len(selected_pairs) > 10000:
        print(f"⚠️  WARNING: Processing {len(selected_pairs)} pairs - this seems too many!")
        print("   Expected: ~3,000-5,000 selected pairs after filtering")
        print("   Actual: This suggests filtering was not applied properly")
    
    print(f"🎯 Processing {len(selected_pairs)} selected pairs for OOS trade loading...")
    
    all_oos_trades = []
    pairs_with_data = 0
    pairs_without_data = 0
    
    for _, pair_row in selected_pairs.iterrows():
        trading_coin = pair_row['trading_coin']
        reference_coin = pair_row['reference_coin']
        strategy_type = pair_row['trading_type']
        
        # Check for CSV first (fast), then JSON fallback
        csv_file = os.path.join(OOS_TRADES_DIR, 
                               f"{reference_coin}_{trading_coin}", 
                               f"{strategy_type}_trades_fast.csv")
        json_file = os.path.join(OOS_TRADES_DIR, 
                                f"{reference_coin}_{trading_coin}", 
                                f"{strategy_type}_trades.json")
        
        trades_file = csv_file if os.path.exists(csv_file) else json_file
        
        if os.path.exists(trades_file):
            try:
                if trades_file.endswith('.csv'):
                    trades_df = pd.read_csv(trades_file)
                    trades_df = trades_df.dropna(subset=['time_entered', 'time_exited', 'log_return'])
                else:
                    # Load JSON format
                    with open(trades_file, 'r') as f:
                        trades_data = json.load(f)
                    trades_df = pd.DataFrame(trades_data['trades'])
                
                if len(trades_df) > 0:
                    # Add pair identification columns
                    trades_df['trading_coin'] = trading_coin
                    trades_df['reference_coin'] = reference_coin
                    trades_df['trade_type'] = trades_df.get('trade_type', strategy_type)
                    
                    # Ensure datetime conversion
                    trades_df['time_entered'] = pd.to_datetime(trades_df['time_entered'])
                    trades_df['time_exited'] = pd.to_datetime(trades_df['time_exited'])
                    
                    all_oos_trades.append(trades_df)
                    pairs_with_data += 1
                else:
                    pairs_without_data += 1
                    
            except Exception as e:
                print(f"   Error loading {trades_file}: {e}")
                pairs_without_data += 1
        else:
            pairs_without_data += 1
    
    if not all_oos_trades:
        raise ValueError("No OOS trade data found for any selected pairs!")
    
    # Combine all trades and sort chronologically
    combined_trades_df = pd.concat(all_oos_trades, ignore_index=True)
    combined_trades_df = combined_trades_df.sort_values('time_entered').reset_index(drop=True)
    
    print(f"✅ Loaded {len(combined_trades_df):,} chronological OOS trades")
    print(f"   Pairs with data: {pairs_with_data}")
    print(f"   Pairs without data: {pairs_without_data}")
    print(f"   Date range: {combined_trades_df['time_entered'].min()} to {combined_trades_df['time_entered'].max()}")
    
    # Save chronological trades
    output_columns = ['trading_coin', 'reference_coin', 'trade_type', 
                     'time_entered', 'time_exited', 'log_return']
    
    chronological_file = os.path.join(RESULTS_DIR, 'chronological_oos_trades.csv')
    combined_trades_df[output_columns].to_csv(chronological_file, index=False)
    print(f"💾 Saved chronological trades to: {chronological_file}")
    
    return combined_trades_df

def run_portfolio_simulation(selected_pairs: pd.DataFrame) -> Dict:
    """Run portfolio simulation and generate visualizations."""
    
    print("\n💰 Step 3: Portfolio Simulation")
    print("=" * 60)
    
    # Initialize portfolio simulator
    simulator = WindowPortfolioSimulator(
        initial_capital=INITIAL_PORTFOLIO_VALUE,
        n_pairs=len(selected_pairs)
    )
    
    # Run portfolio simulation
    portfolio_metrics = simulator.simulate_portfolio(selected_pairs)
    
    if not portfolio_metrics:
        raise ValueError("Portfolio simulation failed!")
    
    # Extract key metrics
    portfolio_total_return = 1 + portfolio_metrics['total_return']
    portfolio_sharpe = portfolio_metrics['sharpe_ratio']
    portfolio_drawdown = portfolio_metrics['max_drawdown']
    
    print(f"✅ Portfolio simulation completed:")
    print(f"   Total Return: {portfolio_total_return:.2f}x ({(portfolio_total_return-1)*100:.1f}%)")
    print(f"   Sharpe Ratio: {portfolio_sharpe:.2f}")
    print(f"   Max Drawdown: {portfolio_drawdown:.2%}")
    print(f"   Trades Executed: {portfolio_metrics.get('num_trades', 0):,}")
    
    # Generate all visualizations (same as window experiments)
    print("🎨 Generating portfolio visualizations...")
    simulator.create_visualizations(RESULTS_DIR)
    
    # Save portfolio summary
    portfolio_summary = {
        'window_name': '18mo_optimal',
        'months_back': OPTIMAL_WINDOW_MONTHS,
        'pairs_selected': len(selected_pairs),
        'portfolio_return': portfolio_total_return,
        'portfolio_sharpe': portfolio_sharpe,
        'portfolio_drawdown': portfolio_drawdown,
        'trades_executed': portfolio_metrics.get('num_trades', 0),
        'simulation_date': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(RESULTS_DIR, 'portfolio_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(portfolio_summary, f, indent=2)
    
    print(f"💾 Portfolio summary saved to: {summary_file}")
    print("✅ All portfolio visualizations generated")
    
    return portfolio_metrics

def create_weekly_trade_matrix(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Create weekly trade count matrix: [Date_Week x Trading_Coin]."""
    
    print("\n📊 Step 4: Weekly Trade Frequency Analysis")
    print("=" * 60)
    
    # Convert to datetime if needed
    trades_df['time_entered'] = pd.to_datetime(trades_df['time_entered'])
    
    # Create weekly period grouping
    trades_df['week_start'] = trades_df['time_entered'].dt.to_period('W').dt.start_time
    
    # Count trades per week per trading coin
    weekly_counts = trades_df.groupby(['week_start', 'trading_coin']).size().reset_index(name='trade_count')
    
    # Pivot to create matrix format
    weekly_matrix = weekly_counts.pivot(index='week_start', columns='trading_coin', values='trade_count')
    weekly_matrix = weekly_matrix.fillna(0).astype(int)
    
    print(f"✅ Created weekly matrix: {weekly_matrix.shape[0]} weeks × {weekly_matrix.shape[1]} coins")
    print(f"   Date range: {weekly_matrix.index.min()} to {weekly_matrix.index.max()}")
    
    # Save weekly matrix
    matrix_file = os.path.join(RESULTS_DIR, 'weekly_trade_counts_by_coin.csv')
    weekly_matrix.to_csv(matrix_file)
    print(f"💾 Weekly trade matrix saved to: {matrix_file}")
    
    return weekly_matrix

def create_weekly_top10_ranking(weekly_matrix: pd.DataFrame) -> pd.DataFrame:
    """Create weekly top-10 coin ranking by trade count."""
    
    print("\n🏆 Step 5: Weekly Top-10 Coin Ranking Analysis") 
    print("=" * 60)
    
    top10_records = []
    
    for week_date, week_row in weekly_matrix.iterrows():
        # Get non-zero trades and sort by count
        week_trades = week_row[week_row > 0].sort_values(ascending=False)
        
        # Create record for this week
        week_record = {'week_start': week_date}
        
        # Fill top-10 slots
        for rank in range(1, 11):
            if rank <= len(week_trades):
                coin = week_trades.index[rank-1]
                count = week_trades.iloc[rank-1]
                week_record[f'coin_top_{rank}'] = coin
                week_record[f'coin_top_{rank}_count'] = count
            else:
                # Fill empty slots
                week_record[f'coin_top_{rank}'] = ''
                week_record[f'coin_top_{rank}_count'] = 0
        
        # Add total trades for the week
        week_record['total_trades_week'] = week_row.sum()
        week_record['active_coins_week'] = (week_row > 0).sum()
        
        top10_records.append(week_record)
    
    top10_df = pd.DataFrame(top10_records)
    
    print(f"✅ Created top-10 ranking: {len(top10_df)} weeks")
    print(f"   Average weekly trades: {top10_df['total_trades_week'].mean():.1f}")
    print(f"   Average active coins/week: {top10_df['active_coins_week'].mean():.1f}")
    
    # Show sample of most active weeks
    most_active_weeks = top10_df.nlargest(3, 'total_trades_week')
    print(f"\nMost active weeks:")
    for _, week in most_active_weeks.iterrows():
        print(f"   {week['week_start'].strftime('%Y-%m-%d')}: {week['total_trades_week']} trades "
              f"(top coin: {week['coin_top_1']} with {week['coin_top_1_count']} trades)")
    
    # Save top-10 ranking
    ranking_file = os.path.join(RESULTS_DIR, 'weekly_top10_coins_traded.csv')
    top10_df.to_csv(ranking_file, index=False)
    print(f"💾 Top-10 ranking saved to: {ranking_file}")
    
    return top10_df

def create_top10_coin_timeseries_plots(weekly_matrix: pd.DataFrame, top10_ranking: pd.DataFrame):
    """Create individual time series plots for top-10 most active coins."""
    
    print("\n📈 Step 6: Creating Top-10 Coin Time Series Visualizations")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Get top-10 coins from the first week (highest overall activity)
    first_week = top10_ranking.iloc[0]
    top10_coins = []
    for i in range(1, 11):
        coin = first_week[f'coin_top_{i}']
        if coin and coin != '':
            top10_coins.append(coin)
    
    print(f"🎯 Creating time series plots for top-10 coins: {', '.join(top10_coins)}")
    
    # Set up date range for X-axis
    date_start = pd.to_datetime("2024-01-01")
    date_end = pd.to_datetime("2025-08-01")
    
    for coin in top10_coins:
        try:
            print(f"   Creating plot for {coin}...")
            
            # Extract data for this coin
            if coin in weekly_matrix.columns:
                coin_data = weekly_matrix[coin]
                dates = weekly_matrix.index
                
                # Filter to our date range
                mask = (dates >= date_start) & (dates <= date_end)
                filtered_dates = dates[mask]
                filtered_data = coin_data[mask]
                
                # Create the plot
                plt.figure(figsize=(12, 6))
                plt.plot(filtered_dates, filtered_data, linewidth=2, color='#2E86AB', marker='o', markersize=4)
                
                # Customize the plot
                plt.title(f'{coin} - Weekly Trade Count', fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Number of Trades per Week', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Format X-axis dates
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.xticks(rotation=45)
                
                # Set Y-axis to start from 0
                plt.ylim(bottom=0)
                
                # Add some styling
                plt.tight_layout()
                
                # Save the plot
                plot_filename = os.path.join(RESULTS_DIR, f'{coin}_weekly_trades.png')
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Saved: {plot_filename}")
                
            else:
                print(f"   ⚠️  Coin {coin} not found in weekly matrix")
                
        except Exception as e:
            print(f"   ❌ Error creating plot for {coin}: {str(e)}")
            continue
    
    print(f"✅ Created time series plots for {len(top10_coins)} top coins")
    print(f"💾 All plots saved to: {RESULTS_DIR}/")

def create_monthly_top5_histogram(weekly_matrix: pd.DataFrame):
    """Create monthly histogram showing top-5 trading coins per month with grouped bars."""
    
    print("\n📊 Step 7: Creating Monthly Top-5 Coins Histogram")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    
    # Convert weekly data to monthly aggregation
    weekly_matrix_monthly = weekly_matrix.copy()
    weekly_matrix_monthly.index = pd.to_datetime(weekly_matrix_monthly.index)
    
    # Group by month and sum trades
    monthly_data = weekly_matrix_monthly.groupby(pd.Grouper(freq='ME')).sum()
    
    print(f"🗓️  Aggregated {len(weekly_matrix)} weeks into {len(monthly_data)} months")
    
    # Get top-5 coins for each month
    monthly_top5_data = []
    for month_date, month_row in monthly_data.iterrows():
        # Get non-zero trades and sort by count
        month_trades = month_row[month_row > 0].sort_values(ascending=False)
        total_trades = int(month_row.sum())
        
        # Get top-5 coins
        top5_coins = []
        top5_counts = []
        for i in range(min(5, len(month_trades))):
            coin = month_trades.index[i]
            count = int(month_trades.iloc[i])
            top5_coins.append(coin)
            top5_counts.append(count)
        
        # Pad with empty slots if needed
        while len(top5_coins) < 5:
            top5_coins.append('')
            top5_counts.append(0)
        
        monthly_top5_data.append({
            'month': month_date,
            'month_label': month_date.strftime('%Y-%m'),
            'total_trades': total_trades,
            'top5_coins': top5_coins,
            'top5_counts': top5_counts
        })
    
    print(f"📈 Processing {len(monthly_top5_data)} months for histogram visualization...")
    
    # Create the grouped histogram plot
    _, ax = plt.subplots(figsize=(20, 10))
    
    # Setup data for plotting
    months = [data['month_label'] for data in monthly_top5_data]
    total_trades_per_month = [data['total_trades'] for data in monthly_top5_data]
    
    # Colors for the 5 positions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Width of individual bars
    bar_width = 0.15
    x_positions = range(len(months))
    
    # Plot grouped bars for each top position
    for pos in range(5):
        position_counts = [data['top5_counts'][pos] for data in monthly_top5_data]
        position_coins = [data['top5_coins'][pos] for data in monthly_top5_data]
        
        # Calculate x positions for this group
        x_offset = (pos - 2) * bar_width  # Center around 0
        x_pos = [x + x_offset for x in x_positions]
        
        # Create bars
        bars = ax.bar(x_pos, position_counts, bar_width, 
                     label=f'Top {pos+1}', color=colors[pos], alpha=0.8)
        
        # Add coin names and count labels on bars (only for significant counts)
        for i, (bar, count, coin) in enumerate(zip(bars, position_counts, position_coins)):
            if count > 0 and coin:
                # Add coin name on top of bar
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                       coin, ha='center', va='bottom', fontsize=7, rotation=90, fontweight='bold')
                # Add count label in the middle of the bar
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                       f'{count:,}', ha='center', va='center', fontsize=6, rotation=90, 
                       color='white', fontweight='bold')
    
    # Add total trades numbers on top of each month group
    for i, total in enumerate(total_trades_per_month):
        # Position the total label at a fixed height near the top of the chart
        ax.text(i, 9500, f'{total:,}', 
               ha='center', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Customize the plot
    ax.set_title('Monthly Top-5 Trading Coins Distribution', fontsize=18, fontweight='bold', pad=30)
    ax.set_xlabel('Month', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Trades', fontsize=14, fontweight='bold')
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(months, rotation=45, ha='right')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    
    # Set y-axis with reasonable limits
    ax.set_ylim(bottom=0, top=10000)
    
    # Add subtle styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(RESULTS_DIR, 'monthly_top5_coins_histogram.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Monthly top-5 histogram created")
    print(f"💾 Saved: {plot_filename}")
    
    # Create summary statistics
    avg_total_trades = sum(total_trades_per_month) / len(total_trades_per_month)
    max_month_trades = max(total_trades_per_month)
    max_month_idx = total_trades_per_month.index(max_month_trades)
    max_month_name = months[max_month_idx]
    
    print(f"📊 Monthly Trading Statistics:")
    print(f"   Average trades per month: {avg_total_trades:,.0f}")
    print(f"   Most active month: {max_month_name} with {max_month_trades:,} trades")
    print(f"   Date range: {months[0]} to {months[-1]}")
    print(f"   Total months analyzed: {len(months)}")

def create_all_coins_histogram(weekly_matrix: pd.DataFrame):
    """Create histogram showing all trading coins and their total trade counts during OOS period."""
    
    print("\n📊 Step 8: Creating All Trading Coins Distribution Histogram")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Sum all trades per coin across entire OOS period
    total_trades_per_coin = weekly_matrix.sum(axis=0)
    
    # Remove coins with zero trades
    total_trades_per_coin = total_trades_per_coin[total_trades_per_coin > 0]
    
    # Sort coins by trade count (descending)
    total_trades_per_coin = total_trades_per_coin.sort_values(ascending=False)
    
    # Get statistics
    num_coins = len(total_trades_per_coin)
    total_trades = int(total_trades_per_coin.sum())
    max_trades = int(total_trades_per_coin.max())
    min_trades = int(total_trades_per_coin.min())
    median_trades = int(total_trades_per_coin.median())
    mean_trades = int(total_trades_per_coin.mean())
    
    # Get top and bottom coins
    top_5_coins = total_trades_per_coin.head(5)
    bottom_5_coins = total_trades_per_coin.tail(5)
    
    print(f"📊 Trading Coin Statistics:")
    print(f"   Total unique coins: {num_coins}")
    print(f"   Total trades: {total_trades:,}")
    print(f"   Most traded: {total_trades_per_coin.index[0]} ({max_trades:,} trades)")
    print(f"   Least traded: {total_trades_per_coin.index[-1]} ({min_trades:,} trades)")
    print(f"   Median trades per coin: {median_trades:,}")
    print(f"   Mean trades per coin: {mean_trades:,}")
    
    # Create the histogram plot
    _, ax = plt.subplots(figsize=(20, 10))
    
    # Prepare data for plotting
    coins = total_trades_per_coin.index.tolist()
    counts = total_trades_per_coin.values
    
    # Create color gradient - darker for higher values
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(coins)))
    
    # Create bars
    ax.bar(range(len(coins)), counts, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of bars (only for top 20 and bottom 5)
    for i, (coin, count) in enumerate(zip(coins, counts)):
        if i < 20 or i >= len(coins) - 5:
            ax.text(i, count + max_trades*0.01, f'{count:,}', 
                   ha='center', va='bottom', fontsize=7, rotation=90)
            # Add coin name below for top 10
            if i < 10:
                ax.text(i, -max_trades*0.02, coin, 
                       ha='center', va='top', fontsize=8, rotation=45, fontweight='bold')
    
    # Customize the plot
    title = f'Trading Coins Distribution - OOS Period\n({num_coins} Unique Coins, {total_trades:,} Total Trades)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Trading Coins (sorted by trade count)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Number of Trades', fontsize=12, fontweight='bold')
    
    # Set x-axis
    ax.set_xlim(-1, len(coins))
    ax.set_xticks([])  # Remove x-ticks since we have too many coins
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text box with statistics
    stats_text = (
        f'Statistics:\n'
        f'• Total Coins: {num_coins}\n'
        f'• Total Trades: {total_trades:,}\n'
        f'• Max: {max_trades:,} ({total_trades_per_coin.index[0]})\n'
        f'• Min: {min_trades:,} ({total_trades_per_coin.index[-1]})\n'
        f'• Median: {median_trades:,}\n'
        f'• Mean: {mean_trades:,}\n\n'
        f'Top 5 Most Traded:\n'
    )
    for coin, count in top_5_coins.items():
        stats_text += f'• {coin}: {count:,}\n'
    
    stats_text += f'\nBottom 5 Least Traded:\n'
    for coin, count in bottom_5_coins.items():
        stats_text += f'• {coin}: {count:,}\n'
    
    # Add text box to plot
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add reference lines
    ax.axhline(y=median_trades, color='red', linestyle='--', alpha=0.5, label=f'Median: {median_trades:,}')
    ax.axhline(y=mean_trades, color='orange', linestyle='--', alpha=0.5, label=f'Mean: {mean_trades:,}')
    
    # Add legend for reference lines
    ax.legend(loc='upper center', frameon=True, shadow=True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(RESULTS_DIR, 'all_coins_trade_distribution.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ All coins distribution histogram created")
    print(f"💾 Saved: {plot_filename}")

def create_top20_coins_histogram(weekly_matrix: pd.DataFrame):
    """Create focused histogram showing only top-20 trading coins during OOS period."""
    
    print("\n📊 Step 9: Creating Top-20 Trading Coins Histogram")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Sum all trades per coin across entire OOS period
    total_trades_per_coin = weekly_matrix.sum(axis=0)
    
    # Remove coins with zero trades
    total_trades_per_coin = total_trades_per_coin[total_trades_per_coin > 0]
    
    # Sort coins by trade count and take top 20
    total_trades_per_coin = total_trades_per_coin.sort_values(ascending=False).head(20)
    
    # Get statistics for top 20
    total_trades_top20 = int(total_trades_per_coin.sum())
    total_trades_all = int(weekly_matrix.sum().sum())
    percentage_of_total = (total_trades_top20 / total_trades_all) * 100
    
    print(f"📊 Top-20 Trading Coins Statistics:")
    print(f"   Total trades in top-20: {total_trades_top20:,}")
    print(f"   Percentage of all trades: {percentage_of_total:.1f}%")
    print(f"   Most traded: {total_trades_per_coin.index[0]} ({total_trades_per_coin.iloc[0]:,.0f} trades)")
    print(f"   20th most traded: {total_trades_per_coin.index[-1]} ({total_trades_per_coin.iloc[-1]:,.0f} trades)")
    
    # Create the histogram plot
    _, ax = plt.subplots(figsize=(16, 10))
    
    # Prepare data for plotting
    coins = total_trades_per_coin.index.tolist()
    counts = total_trades_per_coin.values
    
    # Create color gradient
    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(coins)))
    
    # Create bars with more spacing
    x_positions = range(len(coins))
    bars = ax.bar(x_positions, counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels on top of each bar
    for bar, count in zip(bars, counts):
        # Add count value on top
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
               f'{count:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Add percentage of total
        pct = (count / total_trades_all) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
               f'{pct:.1f}%', ha='center', va='center', fontsize=9, 
               color='white', fontweight='bold')
    
    # Customize the plot
    title = f'Top-20 Trading Coins - OOS Period\n({percentage_of_total:.1f}% of Total {total_trades_all:,} Trades)'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Trading Coins', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Number of Trades', fontsize=14, fontweight='bold')
    
    # Set x-axis with coin names
    ax.set_xticks(x_positions)
    ax.set_xticklabels(coins, rotation=45, ha='right', fontsize=11, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add cumulative percentage line
    cumulative_pct = np.cumsum(counts) / total_trades_all * 100
    ax2 = ax.twinx()
    ax2.plot(x_positions, cumulative_pct, 'k--', linewidth=2, marker='o', 
             markersize=6, label='Cumulative %', alpha=0.7)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    
    # Add reference line at 80% cumulative
    ax2.axhline(y=80, color='red', linestyle=':', alpha=0.5, label='80% Mark')
    
    # Find where we hit 80%
    idx_80 = np.where(cumulative_pct >= 80)[0]
    if len(idx_80) > 0:
        coins_for_80 = idx_80[0] + 1
        ax2.text(idx_80[0], 80, f'  {coins_for_80} coins = 80% of trades', 
                va='center', fontsize=10, color='red', fontweight='bold')
    
    # Add legends
    ax2.legend(loc='center right', frameon=True, shadow=True)
    
    # Add statistics box
    stats_text = (
        f'Summary:\n'
        f'• Top 20 coins: {total_trades_top20:,} trades\n'
        f'• % of total: {percentage_of_total:.1f}%\n'
        f'• Average: {total_trades_top20/20:,.0f} trades/coin\n'
        f'• Range: {total_trades_per_coin.iloc[0]:,.0f} - {total_trades_per_coin.iloc[-1]:,.0f}'
    )
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Remove top spine
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(RESULTS_DIR, 'top20_coins_trade_distribution.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Top-20 coins distribution histogram created")
    print(f"💾 Saved: {plot_filename}")

def main():
    """Main execution function."""
    
    print("🎯 Optimal Window (18mo) OOS Analysis")
    print("=" * 80)
    print(f"Target: Comprehensive analysis of optimal 18-month window configuration")
    print(f"Output directory: {RESULTS_DIR}")
    
    try:
        # Step 1: Run 18-month In-Sample analysis
        selected_pairs = run_in_sample_analysis()
        
        # Validation checkpoint
        print(f"\n✅ STEP 1 COMPLETE: {len(selected_pairs)} pairs selected")
        if len(selected_pairs) == 0:
            print("⚠️  WARNING: No pairs selected due to small dataset")
            print("   With only 2 coins (HOT and RUNE), filtering is too strict")
            print("   Consider adding more coins to the dataset for meaningful analysis")
            return
        if len(selected_pairs) > 10000:
            raise ValueError(f"Too many pairs selected ({len(selected_pairs)}) - filtering may have failed")
        
        # Step 2: Load chronological OOS trades
        oos_trades_df = load_chronological_oos_trades(selected_pairs)
        
        # Validation checkpoint  
        print(f"\n✅ STEP 2 COMPLETE: {len(oos_trades_df):,} chronological trades loaded")
        
        # Step 3: Run portfolio simulation with visualizations
        portfolio_metrics = run_portfolio_simulation(selected_pairs)
        
        # Step 4: Create weekly trade frequency matrix
        weekly_matrix = create_weekly_trade_matrix(oos_trades_df)
        
        # Step 5: Create top-10 weekly coin ranking
        top10_ranking = create_weekly_top10_ranking(weekly_matrix)
        
        # Step 6: Create individual time series plots for top-10 coins
        create_top10_coin_timeseries_plots(weekly_matrix, top10_ranking)
        
        # Step 7: Create monthly top-5 histogram
        create_monthly_top5_histogram(weekly_matrix)
        
        # Step 8: Create all coins distribution histogram
        create_all_coins_histogram(weekly_matrix)
        
        # Step 9: Create top-20 coins histogram
        create_top20_coins_histogram(weekly_matrix)
        
        print("\n🎉 OPTIMAL WINDOW ANALYSIS COMPLETE!")
        print(f"📁 All results saved to: {RESULTS_DIR}/")
        print("\n📊 Generated Files:")
        print("   • selected_pairs.csv - IS-selected optimal pairs")
        print("   • chronological_oos_trades.csv - All OOS trades in time order")
        print("   • weekly_trade_counts_by_coin.csv - Weekly trade matrix")
        print("   • weekly_top10_coins_traded.csv - Top-10 ranking per week")
        print("   • {COIN}_weekly_trades.png - Individual time series for top-10 coins")
        print("   • monthly_top5_coins_histogram.png - Monthly top-5 coins distribution")
        print("   • all_coins_trade_distribution.png - All trading coins histogram")
        print("   • top20_coins_trade_distribution.png - Top-20 coins focused view")
        print("   • portfolio_equity_curve.png - Portfolio performance")
        print("   • drawdown_curve.png - Risk analysis")
        print("   • budget_allocation.png - Capital allocation timeline")
        print("   • active_trades_timeline.png - Trade execution timeline")
        print("   • portfolio_summary.json - Complete metrics summary")
        
        # Display key insights
        print(f"\n🏆 OPTIMAL WINDOW INSIGHTS:")
        print(f"   Window length: {OPTIMAL_WINDOW_MONTHS} months")
        print(f"   Selected pairs: {len(selected_pairs)}")
        print(f"   Total OOS trades: {len(oos_trades_df):,}")
        print(f"   Weekly average trades: {top10_ranking['total_trades_week'].mean():.1f}")
        print(f"   Portfolio return: {portfolio_metrics.get('total_return', 0)*100:.1f}%")
        print(f"   Sharpe ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max drawdown: {portfolio_metrics.get('max_drawdown', 0):.1%}")
        
        print("\n💡 USE CASES:")
        print("   • Analyze equity spikes using weekly_top10_coins_traded.csv")
        print("   • Correlate portfolio movements with high-activity coins") 
        print("   • Identify risk concentration periods")
        print("   • Optimize trade scheduling and risk management")
        
    except Exception as e:
        print(f"❌ Error during optimal window analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()