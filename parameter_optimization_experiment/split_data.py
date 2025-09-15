#!/usr/bin/env python3
"""
Data Splitting for Parameter Optimization Experiment

Splits full_data into In-Sample (before 2024-01-01) and Out-of-Sample (2024-01-01 onwards).
Filters out coins with insufficient data and generates a detailed report.
"""

import os
import glob
import pandas as pd
from datetime import datetime
import numpy as np

# Configuration
FULL_DATA_DIR = "/Users/nikitakrasnytskyi/Desktop/IntramarketDifference/full_data"
OUTPUT_DATA_DIR = "data"
SPLIT_DATE = "2024-01-01"
MIN_IN_SAMPLE_POINTS = 8736   # Minimum data points required for in-sample (1 year)
MIN_OUT_SAMPLE_POINTS = 8736  # Minimum data points required for out-of-sample (1 year)

def clean_coin_name(filename):
    """Extract coin name from filename."""
    # Remove path and extension, then extract coin symbol
    basename = os.path.basename(filename)
    
    # New naming format: {COIN}_USDT_USDT-1h-futures.csv
    if basename.endswith("_USDT_USDT-1h-futures.csv"):
        return basename.replace("_USDT_USDT-1h-futures.csv", "")
    # Keep old format support for backward compatibility
    elif basename.endswith("_USDT-1h.csv"):
        return basename.replace("_USDT-1h.csv", "")
    elif basename.endswith("USDT-1h.csv"):
        return basename.replace("USDT-1h.csv", "").replace("_", "")
    else:
        return None

def load_and_analyze_coin(filepath):
    """Load coin data and analyze date range and quality."""
    try:
        df = pd.read_csv(filepath)
        
        # Handle timestamp conversion (Unix timestamps in milliseconds)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            df = df.set_index('date')
        else:
            # Assume first column is timestamp
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], unit='ms')
            df = df.set_index(df.columns[0])
        
        # Standardize column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        # Check for required columns
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                return None, f"Missing required column: {col}"
        
        # Clean data
        df = df.sort_index().dropna()
        
        if len(df) == 0:
            return None, "No valid data after cleaning"
        
        # Get date range
        start_date = df.index.min()
        end_date = df.index.max()
        date_range_days = (end_date - start_date).total_seconds() / (24 * 3600)
        
        return df, {
            'start_date': start_date,
            'end_date': end_date,
            'total_points': len(df),
            'date_range_days': date_range_days
        }
        
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def split_coin_data(df, coin_name, split_date_str):
    """Split coin data into in-sample and out-of-sample."""
    split_date = pd.Timestamp(split_date_str)
    
    # Split data
    in_sample = df[df.index < split_date]
    out_of_sample = df[df.index >= split_date]
    
    # Check minimum requirements
    if len(in_sample) < MIN_IN_SAMPLE_POINTS:
        return None, None, f"Insufficient in-sample data: {len(in_sample)} < {MIN_IN_SAMPLE_POINTS}"
    
    if len(out_of_sample) < MIN_OUT_SAMPLE_POINTS:
        return None, None, f"Insufficient out-of-sample data: {len(out_of_sample)} < {MIN_OUT_SAMPLE_POINTS}"
    
    return in_sample, out_of_sample, None

def main():
    print("🔄 Data Splitting for Parameter Optimization Experiment")
    print("=" * 60)
    print(f"📂 Source: {FULL_DATA_DIR}")
    print(f"🎯 Output: {OUTPUT_DATA_DIR}")
    print(f"📅 Split Date: {SPLIT_DATE}")
    print(f"📊 Min In-Sample Points: {MIN_IN_SAMPLE_POINTS}")
    print(f"📊 Min Out-of-Sample Points: {MIN_OUT_SAMPLE_POINTS}")
    
    # Create output directories
    in_sample_dir = os.path.join(OUTPUT_DATA_DIR, "in_sample")
    out_sample_dir = os.path.join(OUTPUT_DATA_DIR, "out_of_sample")
    os.makedirs(in_sample_dir, exist_ok=True)
    os.makedirs(out_sample_dir, exist_ok=True)
    
    # Find all CSV files - support both old and new naming formats
    csv_files_old = glob.glob(os.path.join(FULL_DATA_DIR, "*_USDT-1h.csv"))
    csv_files_new = glob.glob(os.path.join(FULL_DATA_DIR, "*_USDT_USDT-1h-futures.csv"))
    csv_files = csv_files_old + csv_files_new
    print(f"\n📋 Found {len(csv_files)} CSV files (old format: {len(csv_files_old)}, new format: {len(csv_files_new)})")
    
    # Process each coin
    successful_coins = []
    rejected_coins = []
    processing_stats = {
        'total_files': len(csv_files),
        'successful': 0,
        'rejected': 0,
        'rejected_reasons': {}
    }
    
    for filepath in csv_files:
        coin_name = clean_coin_name(filepath)
        if coin_name is None:
            rejected_coins.append({
                'file': os.path.basename(filepath),
                'reason': 'Invalid filename format'
            })
            continue
        
        print(f"Processing {coin_name}...", end=" ")
        
        # Load and analyze
        df, result = load_and_analyze_coin(filepath)
        
        if df is None:
            print(f"❌ {result}")
            rejected_coins.append({
                'coin': coin_name,
                'file': os.path.basename(filepath),
                'reason': result
            })
            processing_stats['rejected_reasons'][result] = processing_stats['rejected_reasons'].get(result, 0) + 1
            continue
        
        # Split data
        in_sample, out_sample, error = split_coin_data(df, coin_name, SPLIT_DATE)
        
        if error:
            print(f"❌ {error}")
            rejected_coins.append({
                'coin': coin_name,
                'file': os.path.basename(filepath),
                'reason': error
            })
            processing_stats['rejected_reasons'][error] = processing_stats['rejected_reasons'].get(error, 0) + 1
            continue
        
        # Save split data
        try:
            in_sample_file = os.path.join(in_sample_dir, f"{coin_name}_is.csv")
            out_sample_file = os.path.join(out_sample_dir, f"{coin_name}_oos.csv")
            
            in_sample.to_csv(in_sample_file, float_format='%.8f')
            out_sample.to_csv(out_sample_file, float_format='%.8f')
            
            successful_coins.append({
                'coin': coin_name,
                'in_sample_points': len(in_sample),
                'out_sample_points': len(out_sample),
                'in_sample_start': in_sample.index.min(),
                'in_sample_end': in_sample.index.max(),
                'out_sample_start': out_sample.index.min(),
                'out_sample_end': out_sample.index.max(),
                'total_points': len(df)
            })
            
            print(f"✅ IS:{len(in_sample)} OOS:{len(out_sample)}")
            
        except Exception as e:
            print(f"❌ Save error: {str(e)}")
            rejected_coins.append({
                'coin': coin_name,
                'file': os.path.basename(filepath),
                'reason': f"Save error: {str(e)}"
            })
            continue
    
    # Update stats
    processing_stats['successful'] = len(successful_coins)
    processing_stats['rejected'] = len(rejected_coins)
    
    # Generate detailed report
    report_lines = []
    report_lines.append("DATA SPLITTING REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Split Date: {SPLIT_DATE}")
    report_lines.append(f"Minimum In-Sample Points: {MIN_IN_SAMPLE_POINTS}")
    report_lines.append(f"Minimum Out-of-Sample Points: {MIN_OUT_SAMPLE_POINTS}")
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("SUMMARY")
    report_lines.append("-" * 30)
    report_lines.append(f"Total files processed: {processing_stats['total_files']}")
    report_lines.append(f"Successfully split: {processing_stats['successful']}")
    report_lines.append(f"Rejected: {processing_stats['rejected']}")
    report_lines.append(f"Success rate: {processing_stats['successful'] / processing_stats['total_files'] * 100:.1f}%")
    report_lines.append("")
    
    # Successful coins details
    if successful_coins:
        report_lines.append("SUCCESSFUL COINS")
        report_lines.append("-" * 30)
        report_lines.append(f"{'Coin':<12} {'IS Points':<10} {'OOS Points':<11} {'IS Start':<12} {'IS End':<12} {'OOS Start':<12} {'OOS End'}")
        report_lines.append("-" * 85)
        
        for coin in successful_coins:
            report_lines.append(
                f"{coin['coin']:<12} {coin['in_sample_points']:<10} {coin['out_sample_points']:<11} "
                f"{coin['in_sample_start'].strftime('%Y-%m-%d'):<12} {coin['in_sample_end'].strftime('%Y-%m-%d'):<12} "
                f"{coin['out_sample_start'].strftime('%Y-%m-%d'):<12} {coin['out_sample_end'].strftime('%Y-%m-%d')}"
            )
        
        # Statistics
        is_points = [c['in_sample_points'] for c in successful_coins]
        oos_points = [c['out_sample_points'] for c in successful_coins]
        
        report_lines.append("")
        report_lines.append("IN-SAMPLE STATISTICS")
        report_lines.append("-" * 30)
        report_lines.append(f"Average points: {np.mean(is_points):.0f}")
        report_lines.append(f"Median points: {np.median(is_points):.0f}")
        report_lines.append(f"Min points: {np.min(is_points)}")
        report_lines.append(f"Max points: {np.max(is_points)}")
        
        report_lines.append("")
        report_lines.append("OUT-OF-SAMPLE STATISTICS")
        report_lines.append("-" * 30)
        report_lines.append(f"Average points: {np.mean(oos_points):.0f}")
        report_lines.append(f"Median points: {np.median(oos_points):.0f}")
        report_lines.append(f"Min points: {np.min(oos_points)}")
        report_lines.append(f"Max points: {np.max(oos_points)}")
    
    # Rejected coins
    if rejected_coins:
        report_lines.append("")
        report_lines.append("REJECTED COINS")
        report_lines.append("-" * 30)
        
        # Group by rejection reason
        for reason, count in processing_stats['rejected_reasons'].items():
            report_lines.append(f"\n{reason} ({count} coins):")
            rejected_by_reason = [c for c in rejected_coins if c['reason'] == reason]
            for coin_info in rejected_by_reason:
                if 'coin' in coin_info:
                    report_lines.append(f"  - {coin_info['coin']}")
                else:
                    report_lines.append(f"  - {coin_info['file']}")
    
    report_lines.append("")
    report_lines.append("FILES CREATED")
    report_lines.append("-" * 30)
    report_lines.append(f"In-sample files: {len(successful_coins)} (saved to {in_sample_dir}/)")
    report_lines.append(f"Out-of-sample files: {len(successful_coins)} (saved to {out_sample_dir}/)")
    
    # Save report
    report_file = os.path.join(OUTPUT_DATA_DIR, "data_splitting_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Print summary
    print(f"\n📊 PROCESSING COMPLETE:")
    print(f"   ✅ Successfully processed: {processing_stats['successful']} coins")
    print(f"   ❌ Rejected: {processing_stats['rejected']} coins")
    print(f"   📁 In-sample files saved to: {in_sample_dir}/")
    print(f"   📁 Out-of-sample files saved to: {out_sample_dir}/")
    print(f"   📋 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()