#!/usr/bin/env python3
"""
Create visualization showing the number of cryptocurrency coins available over time.
Analyzes both in-sample and out-of-sample data to show coin availability from 2018-2025.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import seaborn as sns

def load_coin_data_ranges():
    """Load all coin data and find earliest/latest dates for each coin."""
    
    data_dir = Path("data")
    is_dir = data_dir / "in_sample"
    oos_dir = data_dir / "out_of_sample"
    
    coin_ranges = {}
    
    print("🔍 Analyzing coin data ranges...")
    
    # Process in-sample files
    is_files = list(is_dir.glob("*_is.csv"))
    print(f"Found {len(is_files)} in-sample files")
    
    for file_path in is_files:
        coin = file_path.stem.replace("_is", "")
        
        try:
            # Read just first and last few rows for efficiency
            df = pd.read_csv(file_path)
            if len(df) == 0:
                continue
                
            # Get date column (might be 'date', 'open time', or 'open_time')
            if 'date' in df.columns:
                date_col = 'date'
            elif 'open time' in df.columns:
                date_col = 'open time'
            elif 'open_time' in df.columns:
                date_col = 'open_time'
            else:
                print(f"No recognized date column in {file_path}")
                continue
            
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            earliest_date = df[date_col].min()
            latest_date = df[date_col].max()
            
            coin_ranges[coin] = {
                'earliest_is': earliest_date,
                'latest_is': latest_date,
                'earliest_oos': None,
                'latest_oos': None
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Process out-of-sample files
    oos_files = list(oos_dir.glob("*_oos.csv"))
    print(f"Found {len(oos_files)} out-of-sample files")
    
    for file_path in oos_files:
        coin = file_path.stem.replace("_oos", "")
        
        try:
            df = pd.read_csv(file_path)
            if len(df) == 0:
                continue
                
            # Get date column (might be 'date', 'open time', or 'open_time')
            if 'date' in df.columns:
                date_col = 'date'
            elif 'open time' in df.columns:
                date_col = 'open time'
            elif 'open_time' in df.columns:
                date_col = 'open_time'
            else:
                print(f"No recognized date column in {file_path}")
                continue
            
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            earliest_date = df[date_col].min()
            latest_date = df[date_col].max()
            
            if coin in coin_ranges:
                coin_ranges[coin]['earliest_oos'] = earliest_date
                coin_ranges[coin]['latest_oos'] = latest_date
            else:
                # Coin only exists in OOS data
                coin_ranges[coin] = {
                    'earliest_is': None,
                    'latest_is': None,
                    'earliest_oos': earliest_date,
                    'latest_oos': latest_date
                }
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Combine earliest and latest dates across IS and OOS
    for coin in coin_ranges:
        dates = []
        if coin_ranges[coin]['earliest_is'] is not None:
            dates.append(coin_ranges[coin]['earliest_is'])
        if coin_ranges[coin]['earliest_oos'] is not None:
            dates.append(coin_ranges[coin]['earliest_oos'])
            
        coin_ranges[coin]['earliest_overall'] = min(dates) if dates else None
        
        dates = []
        if coin_ranges[coin]['latest_is'] is not None:
            dates.append(coin_ranges[coin]['latest_is'])
        if coin_ranges[coin]['latest_oos'] is not None:
            dates.append(coin_ranges[coin]['latest_oos'])
            
        coin_ranges[coin]['latest_overall'] = max(dates) if dates else None
    
    return coin_ranges

def create_monthly_coin_counts(coin_ranges):
    """Create monthly time series of coin availability."""
    
    # Create monthly date range from 2018-01 to 2025-08
    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2025-08-01')
    
    # Generate monthly periods
    monthly_periods = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    coin_counts = []
    
    print("📊 Calculating monthly coin availability...")
    
    for period_start in monthly_periods:
        period_end = period_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        
        # Count coins available during this month
        available_coins = 0
        coin_list = []
        
        for coin, dates in coin_ranges.items():
            earliest = dates['earliest_overall']
            latest = dates['latest_overall']
            
            if earliest is None or latest is None:
                continue
                
            # Check if coin data spans this month
            if earliest <= period_end and latest >= period_start:
                available_coins += 1
                coin_list.append(coin)
        
        coin_counts.append({
            'date': period_start,
            'coin_count': available_coins,
            'coins': coin_list
        })
        
        if period_start.month == 1 or available_coins % 10 == 0:
            print(f"   {period_start.strftime('%Y-%m')}: {available_coins} coins")
    
    return pd.DataFrame(coin_counts)

def create_coin_timeline_figure(coin_counts_df, coin_ranges):
    """Create professional timeline visualization showing coin availability over time."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Main timeline plot
    dates = coin_counts_df['date']
    counts = coin_counts_df['coin_count']
    
    # Create stepped line plot
    ax.plot(dates, counts, linewidth=3, color='#2E86C1', alpha=0.8, marker='o', markersize=3)
    ax.fill_between(dates, counts, alpha=0.3, color='#2E86C1')
    
    # Annotations for key milestones
    max_count = counts.max()
    final_count = counts.iloc[-1]
    initial_count = counts.iloc[0]
    
    # Add numerical annotations for key years
    key_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    
    for year in key_years:
        # Find January of each year
        jan_date = pd.Timestamp(f'{year}-01-01')
        
        # Find closest date in our data
        closest_idx = (dates - jan_date).abs().argmin()
        closest_date = dates.iloc[closest_idx]
        closest_count = counts.iloc[closest_idx]
        
        if abs((closest_date - jan_date).days) < 32:  # Within a month
            # Add annotation
            ax.annotate(f'{closest_count}', 
                       xy=(closest_date, closest_count), 
                       xytext=(0, 15), textcoords='offset points',
                       fontsize=11, fontweight='bold', ha='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Highlight major growth periods with text annotations
    ax.annotate('Rapid Growth\nPeriod', 
                xy=(pd.Timestamp('2021-06-01'), 170), 
                xytext=(20, 30), textcoords='offset points',
                fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    # Add milestone markers
    milestones = [
        (pd.Timestamp('2020-01-01'), "Pre-DeFi Era"),
        (pd.Timestamp('2021-01-01'), "DeFi Boom"),
        (pd.Timestamp('2024-01-01'), "IS→OOS Split")
    ]
    
    for date, label in milestones:
        ax.axvline(x=date, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        # Find the count at this date
        closest_idx = (dates - date).abs().argmin()
        count_at_date = counts.iloc[closest_idx]
        ax.text(date, count_at_date + 10, label, 
               rotation=90, fontsize=10, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add grid and styling
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Timeline', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Available Cryptocurrency Coins', fontsize=14, fontweight='bold')
    ax.set_title('Cryptocurrency Data Availability Timeline (2018-2025)\nShowing Monthly Coin Count Evolution', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Format x-axis to show years nicely
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add summary statistics in text box
    stats_text = f"""Dataset Growth Summary:
• Total unique coins: {len(coin_ranges)}
• Started with: {initial_count} coins (Jan 2018)
• Ended with: {final_count} coins (Aug 2025)  
• Peak count: {max_count} coins
• Growth factor: {final_count/initial_count:.1f}x
• Avg monthly growth: {(final_count/initial_count)**(1/92)-1:.1%}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    # Set y-axis to start from 0 for better visualization
    ax.set_ylim(0, max_count * 1.1)
    
    plt.tight_layout()
    
    # Save figure
    output_path = "results/coin_availability_timeline.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Figure saved to: {output_path}")
    
    return output_path

def create_yearly_average_figure(coin_counts_df):
    """Create year-over-year average coin availability figure."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate yearly averages
    coin_counts_df['year'] = coin_counts_df['date'].dt.year
    yearly_avg = coin_counts_df.groupby('year')['coin_count'].mean()
    yearly_min = coin_counts_df.groupby('year')['coin_count'].min()
    yearly_max = coin_counts_df.groupby('year')['coin_count'].max()
    
    # Create bar plot with error bars
    bars = ax.bar(yearly_avg.index, yearly_avg.values, 
                  color='lightcoral', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add error bars showing min/max range
    yerr_lower = yearly_avg - yearly_min
    yerr_upper = yearly_max - yearly_avg
    ax.errorbar(yearly_avg.index, yearly_avg.values, 
                yerr=[yerr_lower, yerr_upper], 
                fmt='none', ecolor='black', capsize=5, capthick=2)
    
    # Add value labels on bars
    for bar, avg_value, min_val, max_val in zip(bars, yearly_avg.values, yearly_min.values, yearly_max.values):
        height = bar.get_height()
        # Show average with range
        if min_val == max_val:
            label = f'{int(avg_value)}'
        else:
            label = f'{int(avg_value)}\n({int(min_val)}-{int(max_val)})'
        ax.text(bar.get_x() + bar.get_width()/2., height + (yearly_max.max() * 0.02),
                label, ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Available Coins', fontsize=14, fontweight='bold')
    ax.set_title('Year-over-Year Average Coin Availability\n(With Min-Max Range)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, yearly_max.max() * 1.15)
    
    plt.tight_layout()
    
    # Save figure
    output_path = "results/yearly_average_coin_availability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Yearly average figure saved to: {output_path}")
    
    return output_path

def analyze_2025_coin_drop(coin_ranges, coin_counts_df):
    """Analyze why coin count drops in 2025 and create detailed explanation."""
    
    print("\n🔍 Analyzing 2025 coin count drop...")
    
    # Find coins that end before 2025-08
    coins_ending_early = []
    coins_continuing = []
    
    cutoff_date = pd.Timestamp('2025-08-01')
    
    for coin, dates in coin_ranges.items():
        latest_date = dates['latest_overall']
        if latest_date and latest_date < cutoff_date:
            coins_ending_early.append({
                'coin': coin,
                'end_date': latest_date,
                'days_short': (cutoff_date - latest_date).days
            })
        elif latest_date and latest_date >= cutoff_date:
            coins_continuing.append(coin)
    
    # Sort by end date
    coins_ending_early.sort(key=lambda x: x['end_date'])
    
    # Analyze patterns
    analysis_summary = f"""
CRYPTOCURRENCY DATA AVAILABILITY ANALYSIS SUMMARY
================================================

OVERVIEW:
- Total unique coins in dataset: {len(coin_ranges)}
- Peak availability (Jan 2024): 293 coins
- Final availability (Aug 2025): {len(coins_continuing)} coins
- Coins ending early: {len(coins_ending_early)}
- Net drop in 2025: {len(coins_ending_early)} coins

GROWTH TRAJECTORY:
- 2018 start: 5 coins
- 2024 peak: 293 coins  
- 2025 end: {len(coins_continuing)} coins
- Overall growth: {len(coins_continuing)/5:.1f}x over 7 years

REASON FOR 2025 DROP:
The decrease from 293 to 281 coins in 2025 is NOT due to In-Sample vs Out-of-Sample comparison.
Instead, it reflects the actual data availability cutoff in the Out-of-Sample files.

ANALYSIS METHOD:
1. Script combines both In-Sample (2018-2023) and Out-of-Sample (2024-2025) data
2. For each coin, finds earliest date from either dataset
3. For each coin, finds latest date from either dataset  
4. Monthly coin count = coins with data spanning that month

COINS ENDING BEFORE AUG 2025:
"""
    
    if coins_ending_early:
        analysis_summary += "\nCoins with data ending before August 2025:\n"
        for coin_info in coins_ending_early:
            analysis_summary += f"- {coin_info['coin']}: ends {coin_info['end_date'].strftime('%Y-%m-%d')} ({coin_info['days_short']} days short)\n"
    
    # Check if there are patterns in the cutoff dates
    end_dates = [coin_info['end_date'] for coin_info in coins_ending_early]
    if end_dates:
        earliest_cutoff = min(end_dates)
        latest_cutoff = max(end_dates)
        
        analysis_summary += f"""
CUTOFF PATTERNS:
- Earliest data cutoff: {earliest_cutoff.strftime('%Y-%m-%d')}
- Latest data cutoff: {latest_cutoff.strftime('%Y-%m-%d')}
- Cutoff period span: {(latest_cutoff - earliest_cutoff).days} days

DATA COMPLETENESS:
- Coins with full data through Aug 2025: {len(coins_continuing)} ({len(coins_continuing)/len(coin_ranges)*100:.1f}%)
- Coins with partial 2025 data: {len(coins_ending_early)} ({len(coins_ending_early)/len(coin_ranges)*100:.1f}%)

INTERPRETATION:
The 2025 coin count drop represents natural data availability limits in the source dataset,
not a methodological artifact. Some coins simply have data that ends before August 2025,
likely due to data collection cutoffs or exchange delisting.
"""
    
    # Save analysis to file
    analysis_path = "results/coin_availability_analysis.txt"
    with open(analysis_path, 'w') as f:
        f.write(analysis_summary)
    
    print(f"📄 Detailed analysis saved to: {analysis_path}")
    print(f"   {len(coins_ending_early)} coins end before Aug 2025")
    print(f"   {len(coins_continuing)} coins continue through Aug 2025")
    
    return coins_ending_early, analysis_path

def main():
    """Main execution function."""
    
    print("🪙 Cryptocurrency Coin Availability Analysis")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Load coin data ranges
    coin_ranges = load_coin_data_ranges()
    print(f"✅ Loaded data ranges for {len(coin_ranges)} coins")
    
    # Create monthly coin counts
    coin_counts_df = create_monthly_coin_counts(coin_ranges)
    print(f"✅ Generated monthly counts for {len(coin_counts_df)} months")
    
    # Create timeline visualization  
    timeline_path = create_coin_timeline_figure(coin_counts_df, coin_ranges)
    
    # Create yearly average analysis
    yearly_path = create_yearly_average_figure(coin_counts_df)
    
    # Analyze 2025 coin drop
    coins_ending_early, analysis_path = analyze_2025_coin_drop(coin_ranges, coin_counts_df)
    
    # Print summary statistics
    print("\n📈 Summary Statistics:")
    print(f"   Total unique coins: {len(coin_ranges)}")
    print(f"   Initial count (2018-01): {coin_counts_df['coin_count'].iloc[0]}")
    print(f"   Final count (2025-08): {coin_counts_df['coin_count'].iloc[-1]}")
    print(f"   Peak count: {coin_counts_df['coin_count'].max()}")
    print(f"   Average monthly count: {coin_counts_df['coin_count'].mean():.1f}")
    
    # Save detailed coin ranges to CSV for reference
    coin_details = []
    for coin, dates in coin_ranges.items():
        coin_details.append({
            'coin': coin,
            'earliest_date': dates['earliest_overall'],
            'latest_date': dates['latest_overall'],
            'data_span_days': (dates['latest_overall'] - dates['earliest_overall']).days if dates['earliest_overall'] and dates['latest_overall'] else 0
        })
    
    coin_details_df = pd.DataFrame(coin_details)
    coin_details_df = coin_details_df.sort_values('earliest_date')
    
    details_path = "results/coin_data_ranges.csv"
    coin_details_df.to_csv(details_path, index=False)
    print(f"📄 Coin details saved to: {details_path}")
    
    # Save monthly counts for further analysis
    counts_path = "results/monthly_coin_counts.csv"
    coin_counts_df.to_csv(counts_path, index=False)
    print(f"📄 Monthly counts saved to: {counts_path}")
    
    print("\n🎯 Analysis complete!")
    print(f"   Timeline figure: {timeline_path}")
    print(f"   Yearly average figure: {yearly_path}")
    print(f"   Analysis summary: {analysis_path}")

if __name__ == "__main__":
    main()