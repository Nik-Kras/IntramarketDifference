#!/usr/bin/env python3
"""
Data Preparation Script for New In-Sample/Out-of-Sample Experiment

This script merges data from two sources:
- data/{COIN}USDT_IS.csv (contains data up to 2022-12-31)
- OOS/{COIN}USDT_OOS.csv (contains data from 2023-01-01 to 2025-03-31)

Creates:
- data/in_sample/{COIN}_is.csv (2022-01-01 to 2024-01-01)
- data/out_of_sample/{COIN}_oos.csv (2024-01-01 to 2025-01-01)
"""

import os
import glob
import pandas as pd
from datetime import datetime
from tqdm import tqdm


def load_and_clean_csv(filepath: str) -> pd.DataFrame:
    """Load and standardize CSV file format."""
    
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        # Handle datetime column
        if "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"])
            df = df.set_index("open_time")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])  
            df = df.set_index("date")
        else:
            print(f"⚠️  No datetime column found in {filepath}")
            return None
        
        # Ensure required columns exist
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️  Missing column '{col}' in {filepath}")
                return None
        
        df = df.sort_index().dropna()
        return df
        
    except Exception as e:
        print(f"⚠️  Error loading {filepath}: {e}")
        return None


def get_available_coins() -> set:
    """Get list of coins available in both IS and OOS data sources."""
    
    # Get coins from IS data
    is_files = glob.glob("../data/*USDT_IS.csv")
    is_coins = {os.path.basename(f).replace("USDT_IS.csv", "") for f in is_files}
    
    # Get coins from OOS data  
    oos_files = glob.glob("../OOS/*USDT_OOS.csv")
    oos_coins = {os.path.basename(f).replace("USDT_OOS.csv", "") for f in oos_files}
    
    # Return intersection (coins available in both sources)
    common_coins = is_coins.intersection(oos_coins)
    
    print(f"📊 Data availability:")
    print(f"   IS data coins: {len(is_coins)}")
    print(f"   OOS data coins: {len(oos_coins)}")
    print(f"   Common coins: {len(common_coins)}")
    
    return common_coins


def merge_coin_data(coin: str) -> tuple:
    """
    Merge IS and OOS data for a single coin.
    
    Returns:
        (in_sample_df, out_of_sample_df) or (None, None) if failed
    """
    
    # Load IS data (up to 2022-12-31)
    is_file = f"../data/{coin}USDT_IS.csv"
    is_df = load_and_clean_csv(is_file)
    
    # Load OOS data (from 2023-01-01)
    oos_file = f"../OOS/{coin}USDT_OOS.csv"
    oos_df = load_and_clean_csv(oos_file)
    
    if is_df is None or oos_df is None:
        return None, None
    
    # Merge the dataframes
    combined_df = pd.concat([is_df, oos_df]).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    # Create In-Sample data: 2022-01-01 to 2024-01-01
    in_sample_start = pd.Timestamp('2022-01-01')
    in_sample_end = pd.Timestamp('2024-01-01')
    in_sample_df = combined_df[(combined_df.index >= in_sample_start) & 
                              (combined_df.index < in_sample_end)].copy()
    
    # Create Out-of-Sample data: 2024-01-01 to 2025-01-01
    out_of_sample_start = pd.Timestamp('2024-01-01')
    out_of_sample_end = pd.Timestamp('2025-01-01')
    out_of_sample_df = combined_df[(combined_df.index >= out_of_sample_start) & 
                                  (combined_df.index < out_of_sample_end)].copy()
    
    return in_sample_df, out_of_sample_df


def main():
    """Main execution function."""
    
    print("🔄 Data Preparation for New In-Sample/Out-of-Sample Experiment")
    print("=" * 70)
    
    # Create output directories
    os.makedirs("data/in_sample", exist_ok=True)
    os.makedirs("data/out_of_sample", exist_ok=True)
    
    # Get available coins
    coins = get_available_coins()
    
    if not coins:
        print("❌ No common coins found between IS and OOS data sources!")
        return
    
    print(f"\n🔄 Processing {len(coins)} coins...")
    
    # Process each coin
    successful_coins = []
    failed_coins = []
    
    for coin in tqdm(sorted(coins), desc="Processing coins"):
        
        # Merge data for this coin
        in_sample_df, out_of_sample_df = merge_coin_data(coin)
        
        if in_sample_df is None or out_of_sample_df is None:
            failed_coins.append(coin)
            continue
        
        # Check data quality
        if len(in_sample_df) < 100 or len(out_of_sample_df) < 100:
            failed_coins.append(f"{coin} (insufficient data)")
            continue
        
        # Save In-Sample data
        in_sample_file = f"data/in_sample/{coin}_is.csv"
        in_sample_df.to_csv(in_sample_file)
        
        # Save Out-of-Sample data
        out_of_sample_file = f"data/out_of_sample/{coin}_oos.csv"
        out_of_sample_df.to_csv(out_of_sample_file)
        
        successful_coins.append({
            'coin': coin,
            'in_sample_records': len(in_sample_df),
            'out_of_sample_records': len(out_of_sample_df),
            'in_sample_period': f"{in_sample_df.index.min().date()} to {in_sample_df.index.max().date()}",
            'out_of_sample_period': f"{out_of_sample_df.index.min().date()} to {out_of_sample_df.index.max().date()}"
        })
    
    # Summary
    print(f"\n📊 DATA PREPARATION SUMMARY:")
    print(f"   ✅ Successfully processed: {len(successful_coins)} coins")
    print(f"   ❌ Failed: {len(failed_coins)} coins")
    
    if failed_coins:
        print(f"\n⚠️  Failed coins:")
        for coin in failed_coins[:10]:  # Show first 10 failed coins
            print(f"     - {coin}")
        if len(failed_coins) > 10:
            print(f"     ... and {len(failed_coins) - 10} more")
    
    if successful_coins:
        print(f"\n📈 Data ranges for successfully processed coins:")
        sample_coin = successful_coins[0]
        print(f"   In-Sample Period: {sample_coin['in_sample_period']}")
        print(f"   Out-of-Sample Period: {sample_coin['out_of_sample_period']}")
        print(f"   Average In-Sample Records: {sum(c['in_sample_records'] for c in successful_coins) // len(successful_coins)}")
        print(f"   Average Out-of-Sample Records: {sum(c['out_of_sample_records'] for c in successful_coins) // len(successful_coins)}")
    
    # Save processing summary
    summary_file = "data/data_preparation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Data Preparation Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Successfully processed coins: {len(successful_coins)}\n")
        f.write(f"Failed coins: {len(failed_coins)}\n\n")
        
        f.write("Successful coins:\n")
        for coin_info in successful_coins:
            f.write(f"  {coin_info['coin']}: IS={coin_info['in_sample_records']} records, "
                   f"OOS={coin_info['out_of_sample_records']} records\n")
        
        if failed_coins:
            f.write(f"\nFailed coins:\n")
            for coin in failed_coins:
                f.write(f"  {coin}\n")
    
    print(f"\n💾 Summary saved to: {summary_file}")
    print(f"📁 Data files created in:")
    print(f"   - data/in_sample/")
    print(f"   - data/out_of_sample/")
    
    print(f"\n✅ Data preparation completed!")
    print(f"Ready to run the experiment with {len(successful_coins)} coins")


if __name__ == "__main__":
    main()