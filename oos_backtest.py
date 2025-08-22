#!/usr/bin/env python3
"""
Out-of-Sample Backtesting Script - oos_backtest.py

Tests selected pairs from user_custom_filter.csv on OOS data.
Uses the exact same trading algorithm as in run_all.py for consistency.

To change test period, modify OOS_TEST_PERIOD variable:
- '2023': Test on 2023 data only
- '2024': Test on 2024 data only  
- 'both': Test on combined 2023+2024 data

Author: IntramarketDifference Analysis
"""

import os
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from trades_from_signal import get_trades_from_signal

# Configuration - Same as original backtesting
LOOKBACK = 24               # MA window for cmma
ATR_LOOKBACK = 168          # ATR window for cmma  
THRESHOLD = 0.25            # signal threshold
MIN_OVERLAP = 500           # skip pairs with tiny overlap

# OOS Test Period Configuration
# The script now runs BOTH 2023 and 2024 separately and calculates metrics for each year
# This provides detailed year-by-year analysis

# Paths
INPUT_CSV = "filtered_results/user_custom_filter.csv"
OOS_DATA_DIR = "OOS"
OUTPUT_DIR = "oos_experiments"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "oos_backtest_results.csv")

# -----------------------
# Extract trading algorithm from run_all.py 
# -----------------------
def cmma(ohlc: pd.DataFrame, lookback: int, atr_lookback: int = 168) -> pd.Series:
    """Close-minus-MA normalized by ATR * sqrt(L)."""
    atr = ta.atr(ohlc["high"], ohlc["low"], ohlc["close"], atr_lookback)
    ma = ohlc["close"].rolling(lookback).mean()
    ind = (ohlc["close"] - ma) / (atr * lookback ** 0.5)
    return ind

def threshold_revert_signal(ind: pd.Series, threshold: float) -> np.ndarray:
    """Same signal generation logic as run_all.py"""
    signal = np.zeros(len(ind))
    position = 0
    values = ind.values  # keep NaNs as-is
    for i in range(len(values)):
        v = values[i]
        if not np.isnan(v):
            if v > threshold:
                position = 1
            if v < -threshold:
                position = -1
            if position == 1 and v <= 0:
                position = 0
            if position == -1 and v >= 0:
                position = 0
        # if NaN: do nothing (position persists)
        signal[i] = position
    return signal

def load_oos_csv(coin_name: str, year: int) -> pd.DataFrame:
    """Load OOS CSV file for a coin with specified test period."""
    filename = f"{coin_name}USDT_OOS.csv"
    filepath = os.path.join(OOS_DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"OOS file not found: {filepath}")
    
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
        raise ValueError(f"No datetime column found in {filepath}")
    
    df = df.sort_index()
    
    # Ensure required columns exist
    for col in ["high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {filepath}")
    
    # Filter to specific year
    df_filtered = df.loc[df.index.year == year]
    
    return df_filtered.dropna()

def align_frames(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two dataframes by common index."""
    ix = a.index.intersection(b.index)
    return a.loc[ix].copy(), b.loc[ix].copy()

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate maximum drawdown from cumulative returns."""
    if len(cumulative_returns) == 0:
        return np.nan
    
    cumulative = np.exp(cumulative_returns.cumsum())
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from returns."""
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    
    excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
    return float(excess_returns.mean() / returns.std() * np.sqrt(252))

def filter_trades_by_type(trades: pd.DataFrame, trade_type: str) -> pd.DataFrame:
    """Filter trades by type: 'long', 'short', or 'both'."""
    if trade_type == 'both':
        return trades
    elif trade_type == 'long':
        return trades[trades['type'] == 1]
    elif trade_type == 'short':
        return trades[trades['type'] == -1]
    else:
        raise ValueError(f"Unknown trade_type: {trade_type}")

def run_oos_backtest_single_year(trading_coin: str, reference_coin: str, trade_type: str, year: int) -> dict:
    """
    Run OOS backtest for a specific pair using specified test period data.
    Uses exact same algorithm as run_all.py.
    """
    
    # DEBUG: Enable detailed logging for DOGE 1INCH long (based on existing folder)
    DEBUG_MODE = (trading_coin == "DOGE" and reference_coin == "1INCH" and trade_type == "long")
    
    if DEBUG_MODE:
        print(f"\n🔍 DEBUG MODE: Analyzing {trading_coin}-{reference_coin} {trade_type} for {year}")
    
    try:
        # Load OOS data for both coins
        ref_df = load_oos_csv(reference_coin, year)
        traded_df = load_oos_csv(trading_coin, year)
        
        if DEBUG_MODE:
            print(f"📊 Loaded {len(ref_df)} records for {reference_coin}")
            print(f"📊 Loaded {len(traded_df)} records for {trading_coin}")
        
        # Align dataframes
        ref_df, traded_df = align_frames(ref_df, traded_df)
        
        if DEBUG_MODE:
            print(f"📊 After alignment: {len(ref_df)} common records")
        
        if len(ref_df) < MIN_OVERLAP:
            return {
                'trading_coin': trading_coin,
                'reference_coin': reference_coin,
                'trade_type': trade_type,
                'year': year,
                'oos_num_trades': 0,
                'oos_profit_factor': np.nan,
                'oos_total_return': np.nan,
                'oos_max_drawdown': np.nan,
                'oos_sharpe_ratio': np.nan,
                'error': 'Insufficient data overlap'
            }
        
        # Calculate next returns (same as run_all.py)
        traded_df = traded_df.copy()
        traded_df["diff"] = np.log(traded_df["close"]).diff()
        traded_df["next_return"] = traded_df["diff"].shift(-1)
        
        # Calculate CMMA indicators
        ref_cmma = cmma(ref_df, LOOKBACK, ATR_LOOKBACK)
        trd_cmma = cmma(traded_df, LOOKBACK, ATR_LOOKBACK)
        intermarket_diff = trd_cmma - ref_cmma
        
        if DEBUG_MODE:
            print(f"📈 CMMA indicators calculated:")
            print(f"   Reference CMMA range: {ref_cmma.min():.4f} to {ref_cmma.max():.4f}")
            print(f"   Trading CMMA range: {trd_cmma.min():.4f} to {trd_cmma.max():.4f}")
            print(f"   Intermarket diff range: {intermarket_diff.min():.4f} to {intermarket_diff.max():.4f}")
            print(f"   Non-NaN values: {intermarket_diff.notna().sum()}/{len(intermarket_diff)}")
        
        # Generate signal
        traded_df["sig"] = threshold_revert_signal(intermarket_diff, THRESHOLD)
        
        if DEBUG_MODE:
            signal_counts = traded_df["sig"].value_counts().sort_index()
            print(f"📊 Signal distribution: {dict(signal_counts)}")
            print(f"   Long signals (1): {signal_counts.get(1, 0)}")
            print(f"   Flat signals (0): {signal_counts.get(0, 0)}")
            print(f"   Short signals (-1): {signal_counts.get(-1, 0)}")
        
        # Calculate base returns (signal * next_return)
        rets = traded_df["sig"] * traded_df["next_return"]
        rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
        
        if DEBUG_MODE:
            print(f"📊 Base returns calculated:")
            print(f"   Total returns: {len(rets)}")
            print(f"   Non-zero returns: {(rets != 0).sum()}")
            print(f"   Returns range: {rets.min():.6f} to {rets.max():.6f}")
            print(f"   Returns mean: {rets.mean():.6f}")
        
        # Get individual trades for analysis
        long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, traded_df["sig"].values)
        
        if DEBUG_MODE:
            print(f"📊 Trades analysis:")
            print(f"   Long trades: {len(long_trades)}")
            print(f"   Short trades: {len(short_trades)}")
            print(f"   All trades: {len(all_trades)}")
        
        # CRITICAL: Apply trade type filtering - use actual trades for all calculations
        if trade_type == 'long':
            # Long-only strategy: use only long trades
            filtered_trades = long_trades
            if DEBUG_MODE:
                print(f"🎯 Long-only filter applied:")
                print(f"   Long trades selected: {len(long_trades)}")
        elif trade_type == 'short':
            # Short-only strategy: use only short trades
            filtered_trades = short_trades
            if DEBUG_MODE:
                print(f"🎯 Short-only filter applied:")
                print(f"   Short trades selected: {len(short_trades)}")
        else:  # 'both' or 'combined'
            # Combined strategy: use all trades
            filtered_trades = all_trades
            if DEBUG_MODE:
                print(f"🎯 No filter applied - using all trades")
                print(f"   All trades selected: {len(all_trades)}")
        
        # Use existing trade 'return' field directly (already calculated correctly)
        trades_list = []
        for _, trade in filtered_trades.iterrows():
            trade_type_str = 'long' if trade['type'] == 1 else 'short'
            
            trades_list.append({
                'time_entered': trade.name,  # entry_time is the index
                'time_exited': trade['exit_time'],
                'return': trade['return'],  # Use existing simple return field
                'trade_type': trade_type_str
            })
        
        # Extract simple returns directly from trades
        trade_returns = [t['return'] for t in trades_list if t['return'] is not None]
        
        if DEBUG_MODE:
            print(f"📊 Trade-based returns:")
            print(f"   Number of trades: {len(filtered_trades)}")
            print(f"   Valid returns: {len(trade_returns)}")
            if len(trade_returns) > 0:
                print(f"   Returns range: {min(trade_returns):.6f} to {max(trade_returns):.6f}")
                print(f"   Returns mean: {np.mean(trade_returns):.6f}")
                print(f"   Cumulative return: {sum(trade_returns):.6f}")
        
        # Calculate metrics - permutation_test.py style
        if len(filtered_trades) == 0 or len(trade_returns) == 0:
            return {
                'trading_coin': trading_coin,
                'reference_coin': reference_coin,
                'trade_type': trade_type,
                'year': year,
                'oos_num_trades': 0,
                'oos_profit_factor': np.nan,
                'oos_total_return': np.nan,
                'oos_max_drawdown': np.nan,
                'oos_sharpe_ratio': np.nan,
                'error': 'No trades generated'
            }
        
        # Number of trades
        num_trades = len(filtered_trades)
        
        # Use existing simple returns directly for all metrics
        if len(trade_returns) > 0:
            # Use simple returns directly (already calculated in trade['return'])
            simple_returns = np.array(trade_returns)
            
            # Calculate profit factor using simple returns
            gains = simple_returns[simple_returns > 0].sum()
            losses = simple_returns[simple_returns < 0].sum()
            profit_factor = np.inf if losses == 0 and gains > 0 else (gains / abs(losses) if losses < 0 else np.nan)
            
            # Create time series for equity curve calculation
            trade_times = []
            trade_simple_returns = []
            
            for trade in trades_list:
                if trade['time_exited'] and trade['return'] is not None:
                    trade_times.append(pd.to_datetime(trade['time_exited']))
                    trade_simple_returns.append(trade['return'])
            
            if trade_times:
                # Create series and sort by time
                returns_series = pd.Series(trade_simple_returns, index=trade_times).sort_index()
                
                # Calculate equity curve using simple returns compounding: (1 + r1) * (1 + r2) * ...
                equity_curve = 1000 * (1 + returns_series).cumprod()  # Starting with $1000
                
                # Total return from equity curve
                total_return = (equity_curve.iloc[-1] / 1000) - 1 if len(equity_curve) > 0 else np.nan
                
                # Max drawdown using simple returns series
                max_drawdown = calculate_max_drawdown(returns_series)
                
                # Sharpe ratio using returns series  
                sharpe_ratio = calculate_sharpe_ratio(returns_series)
            else:
                total_return = np.nan
                max_drawdown = np.nan
                sharpe_ratio = np.nan
                equity_curve = pd.Series()
        else:
            profit_factor = np.nan
            total_return = np.nan
            max_drawdown = np.nan
            sharpe_ratio = np.nan
            equity_curve = pd.Series()
        
        if DEBUG_MODE:
            print(f"📊 Final metrics calculated:")
            print(f"   Num trades: {num_trades}")
            print(f"   Profit factor: {profit_factor}")
            print(f"   Total return: {total_return}")
            print(f"   Max drawdown: {max_drawdown}")
            print(f"   Sharpe ratio: {sharpe_ratio}")
            print(f"   Gains: {gains}")
            print(f"   Losses: {losses}")
            
            # Show equity curve calculation preview using simple returns compounding
            if len(equity_curve) > 0:
                print(f"💰 Equity curve preview (time-based, simple returns):")
                print(f"   Starting value: $1000.00")
                print(f"   Final value: ${equity_curve.iloc[-1]:.2f}")
                print(f"   Total return check: {(equity_curve.iloc[-1] / 1000) - 1:.6f} (should match {total_return:.6f})")
                print(f"   Time span: {equity_curve.index[0]} to {equity_curve.index[-1]}")
        
        return {
            'trading_coin': trading_coin,
            'reference_coin': reference_coin,
            'trade_type': trade_type,
            'year': year,
            'oos_num_trades': int(num_trades),
            'oos_profit_factor': float(profit_factor) if pd.notna(profit_factor) else np.nan,
            'oos_total_return': float(total_return) if pd.notna(total_return) else np.nan,
            'oos_max_drawdown': float(max_drawdown) if pd.notna(max_drawdown) else np.nan,
            'oos_sharpe_ratio': float(sharpe_ratio) if pd.notna(sharpe_ratio) else np.nan,
            'error': None
        }
        
    except Exception as e:
        return {
            'trading_coin': trading_coin,
            'reference_coin': reference_coin,
            'trade_type': trade_type,
            'year': year,
            'oos_num_trades': 0,
            'oos_profit_factor': np.nan,
            'oos_total_return': np.nan,
            'oos_max_drawdown': np.nan,
            'oos_sharpe_ratio': np.nan,
            'error': str(e)
        }

def run_oos_backtest_both_years(trading_coin: str, reference_coin: str, trade_type: str) -> dict:
    """Run OOS backtests for both 2023 and 2024 and return combined results."""
    
    # Run backtests for both years
    result_2023 = run_oos_backtest_single_year(trading_coin, reference_coin, trade_type, 2023)
    result_2024 = run_oos_backtest_single_year(trading_coin, reference_coin, trade_type, 2024)
    
    # Create combined result with year-specific columns
    combined_result = {
        'trading_coin': trading_coin,
        'reference_coin': reference_coin,
        'trade_type': trade_type
    }
    
    # Add 2023 metrics
    combined_result['oos_num_trades_2023'] = result_2023['oos_num_trades']
    combined_result['oos_profit_factor_2023'] = result_2023['oos_profit_factor']
    combined_result['oos_total_return_2023'] = result_2023['oos_total_return']
    combined_result['oos_max_drawdown_2023'] = result_2023['oos_max_drawdown']
    combined_result['oos_sharpe_ratio_2023'] = result_2023['oos_sharpe_ratio']
    combined_result['oos_error_2023'] = result_2023['error']
    
    # Add 2024 metrics
    combined_result['oos_num_trades_2024'] = result_2024['oos_num_trades']
    combined_result['oos_profit_factor_2024'] = result_2024['oos_profit_factor']
    combined_result['oos_total_return_2024'] = result_2024['oos_total_return']
    combined_result['oos_max_drawdown_2024'] = result_2024['oos_max_drawdown']
    combined_result['oos_sharpe_ratio_2024'] = result_2024['oos_sharpe_ratio']
    combined_result['oos_error_2024'] = result_2024['error']
    
    return combined_result


def save_detailed_results_both_years(trading_coin: str, reference_coin: str, trade_type: str):
    """Save detailed results including trades JSON and equity curves for both 2023 and 2024."""
    
    # DEBUG: Enable detailed logging for DOGE 1INCH long (based on existing folder)
    DEBUG_MODE = (trading_coin == "DOGE" and reference_coin == "1INCH" and trade_type == "long")
    
    try:
        # Create output directory - FIXED: Trading coin comes first in folder name
        pair_name = f"{trading_coin}_{reference_coin}_{trade_type}"
        output_dir = os.path.join(OUTPUT_DIR, pair_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process both years
        for year in [2023, 2024]:
            try:
                # Load data for this year
                ref_df = load_oos_csv(reference_coin, year)
                traded_df = load_oos_csv(trading_coin, year)
                ref_df, traded_df = align_frames(ref_df, traded_df)
                
                if len(ref_df) < MIN_OVERLAP:
                    print(f"⚠️  Insufficient data for {trading_coin}-{reference_coin} in {year}")
                    continue
                
                # Calculate indicators
                traded_df["diff"] = np.log(traded_df["close"]).diff()
                traded_df["next_return"] = traded_df["diff"].shift(-1)
                ref_cmma = cmma(ref_df, LOOKBACK, ATR_LOOKBACK)
                trd_cmma = cmma(traded_df, LOOKBACK, ATR_LOOKBACK)
                intermarket_diff = trd_cmma - ref_cmma
                traded_df["sig"] = threshold_revert_signal(intermarket_diff, THRESHOLD)
                
                # Get individual trades
                long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, traded_df["sig"].values)
                
                # Filter trades by type
                if trade_type == 'long':
                    filtered_trades = long_trades
                elif trade_type == 'short':
                    filtered_trades = short_trades
                else:  # 'both' or 'combined'
                    filtered_trades = all_trades
                
                # Save trades to JSON
                trades_data = []
                if len(filtered_trades) > 0:
                    for entry_time, trade in filtered_trades.iterrows():
                        # Calculate log return for this specific trade
                        if trade['type'] == 1:  # Long trade
                            log_return = np.log(trade['exit_price'] / trade['entry_price'])
                        else:  # Short trade  
                            log_return = np.log(trade['entry_price'] / trade['exit_price'])
                            
                        trade_record = {
                            'time_entered': entry_time.isoformat() if pd.notna(entry_time) else None,
                            'time_exited': trade['exit_time'].isoformat() if pd.notna(trade['exit_time']) else None,
                            'log_return': float(log_return) if pd.notna(log_return) else None,
                            'trade_type': 'long' if trade['type'] == 1 else 'short'
                        }
                        trades_data.append(trade_record)
                
                # Save trades JSON for this year
                trades_file = os.path.join(output_dir, f'oos_trades_{year}.json')
                with open(trades_file, 'w') as f:
                    json.dump(trades_data, f, indent=2)
                
                # Create equity curve using permutation_test.py methodology (SAME AS METRICS)
                # Filter trades by type to match the metrics calculation
                if trade_type == 'long':
                    chart_trades = long_trades
                elif trade_type == 'short':
                    chart_trades = short_trades
                else:  # 'both' or 'combined'
                    chart_trades = all_trades
                
                # Convert to trades list format using existing 'return' field
                chart_trades_list = []
                for _, trade in chart_trades.iterrows():
                    trade_type_str = 'long' if trade['type'] == 1 else 'short'
                    
                    chart_trades_list.append({
                        'time_entered': trade.name,
                        'time_exited': trade['exit_time'],
                        'return': trade['return'],  # Use existing simple return field
                        'trade_type': trade_type_str
                    })
                
                # Create time series exactly like metrics calculation
                if chart_trades_list:
                    trade_times = []
                    trade_simple_returns = []
                    
                    for trade in chart_trades_list:
                        if trade['time_exited'] and trade['return'] is not None:
                            trade_times.append(pd.to_datetime(trade['time_exited']))
                            trade_simple_returns.append(trade['return'])
                    
                    if trade_times:
                        # Create series and sort by time
                        returns_series = pd.Series(trade_simple_returns, index=trade_times).sort_index()
                        
                        # Calculate equity curve using simple returns compounding
                        equity_curve = 1000 * (1 + returns_series).cumprod()  # Starting with $1000
                        
                        if DEBUG_MODE:
                            print(f"💰 EQUITY CURVE DEBUG ({year}) - simple returns:")
                            print(f"   Number of trades: {len(chart_trades_list)}")
                            print(f"   Time series length: {len(returns_series)}")
                            print(f"   First 5 returns: {returns_series.head().values}")
                            print(f"   Last 5 returns: {returns_series.tail().values}")
                            print(f"   Cumulative simple return: {((1 + returns_series).prod() - 1):.6f}")
                            print(f"   Starting equity: $1000.00")
                            print(f"   Final equity: ${equity_curve.iloc[-1]:.2f}")
                            print(f"   Total return: {(equity_curve.iloc[-1] / 1000) - 1:.6f}")
                            print(f"   Time span: {equity_curve.index[0]} to {equity_curve.index[-1]}")
                    else:
                        equity_curve = pd.Series()
                else:
                    equity_curve = pd.Series()
                
                if len(equity_curve) > 0:
                    
                    plt.figure(figsize=(12, 8))
                    plt.plot(equity_curve.index, equity_curve.values, linewidth=1.5, color='blue')
                    plt.title(f'OOS Equity Curve: {trading_coin}-{reference_coin} ({trade_type})\n{year} Performance')
                    plt.xlabel('Date')
                    plt.ylabel('Portfolio Value ($)')
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    equity_file = os.path.join(output_dir, f'oos_equity_curve_{year}.png')
                    plt.savefig(equity_file, dpi=300, bbox_inches='tight')
                    plt.close()
                
            except Exception as e:
                print(f"⚠️  Could not process {trading_coin}-{reference_coin} for {year}: {e}")
        
        # Save combined metrics JSON
        result_2023 = run_oos_backtest_single_year(trading_coin, reference_coin, trade_type, 2023)
        result_2024 = run_oos_backtest_single_year(trading_coin, reference_coin, trade_type, 2024)
        
        combined_metrics = {
            'pair_info': {
                'trading_coin': trading_coin,
                'reference_coin': reference_coin,
                'trade_type': trade_type,
                'test_period': '2023 and 2024 (separate)'
            },
            'oos_metrics_2023': {
                'num_trades': result_2023['oos_num_trades'],
                'profit_factor': result_2023['oos_profit_factor'],
                'total_cumulative_return': result_2023['oos_total_return'],
                'max_drawdown': result_2023['oos_max_drawdown'],
                'sharpe_ratio': result_2023['oos_sharpe_ratio'],
                'error': result_2023['error']
            },
            'oos_metrics_2024': {
                'num_trades': result_2024['oos_num_trades'],
                'profit_factor': result_2024['oos_profit_factor'],
                'total_cumulative_return': result_2024['oos_total_return'],
                'max_drawdown': result_2024['oos_max_drawdown'],
                'sharpe_ratio': result_2024['oos_sharpe_ratio'],
                'error': result_2024['error']
            }
        }
        
        metrics_file = os.path.join(output_dir, 'oos_metrics_combined.json')
        with open(metrics_file, 'w') as f:
            json.dump(combined_metrics, f, indent=2)
            
    except Exception as e:
        print(f"Warning: Could not save detailed results for {trading_coin}-{reference_coin}: {e}")

def analyze_oos_performance(results_df: pd.DataFrame):
    """Analyze and summarize OOS performance with detailed statistics for both years."""
    
    print(f"\n" + "=" * 80)
    print("OUT-OF-SAMPLE PERFORMANCE ANALYSIS (2023 & 2024)")
    print("=" * 80)
    
    # Filter valid results for both years
    valid_2023 = results_df[pd.isna(results_df['oos_error_2023'])]
    valid_2024 = results_df[pd.isna(results_df['oos_error_2024'])]
    valid_both = results_df[pd.isna(results_df['oos_error_2023']) & pd.isna(results_df['oos_error_2024'])]
    
    total_pairs = len(results_df)
    valid_pairs_2023 = len(valid_2023)
    valid_pairs_2024 = len(valid_2024)
    valid_pairs_both = len(valid_both)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total pairs tested: {total_pairs}")
    print(f"   Valid 2023 backtests: {valid_pairs_2023} ({100 * valid_pairs_2023 / total_pairs:.1f}%)")
    print(f"   Valid 2024 backtests: {valid_pairs_2024} ({100 * valid_pairs_2024 / total_pairs:.1f}%)")
    print(f"   Valid both years: {valid_pairs_both} ({100 * valid_pairs_both / total_pairs:.1f}%)")
    
    if valid_pairs_both == 0:
        print("\n❌ No pairs have valid results for both years!")
        return
    
    # Performance Statistics for Each Year
    print(f"\n📈 OOS PERFORMANCE METRICS:")
    
    # 2023 Statistics
    print(f"\n   2023 PERFORMANCE:")
    print(f"     Average Profit Factor: {valid_2023['oos_profit_factor_2023'].mean():.3f}")
    print(f"     Average Total Return: {valid_2023['oos_total_return_2023'].mean():.3f} ({valid_2023['oos_total_return_2023'].mean()*100:.1f}%)")
    print(f"     Average Max Drawdown: {valid_2023['oos_max_drawdown_2023'].mean():.3f} ({valid_2023['oos_max_drawdown_2023'].mean()*100:.1f}%)")
    print(f"     Average Sharpe Ratio: {valid_2023['oos_sharpe_ratio_2023'].mean():.3f}")
    print(f"     Average Number of Trades: {valid_2023['oos_num_trades_2023'].mean():.0f}")
    
    # 2024 Statistics
    print(f"\n   2024 PERFORMANCE:")
    print(f"     Average Profit Factor: {valid_2024['oos_profit_factor_2024'].mean():.3f}")
    print(f"     Average Total Return: {valid_2024['oos_total_return_2024'].mean():.3f} ({valid_2024['oos_total_return_2024'].mean()*100:.1f}%)")
    print(f"     Average Max Drawdown: {valid_2024['oos_max_drawdown_2024'].mean():.3f} ({valid_2024['oos_max_drawdown_2024'].mean()*100:.1f}%)")
    print(f"     Average Sharpe Ratio: {valid_2024['oos_sharpe_ratio_2024'].mean():.3f}")
    print(f"     Average Number of Trades: {valid_2024['oos_num_trades_2024'].mean():.0f}")
    
    # Filter Analysis
    print(f"\n🎯 OOS FILTER ANALYSIS:")
    print(f"   Filter criteria applied:")
    print(f"   1. Total Return > 0% (positive PnL)")
    print(f"   2. Sharpe Ratio > 1.0") 
    print(f"   3. Max Drawdown better than -40%")
    
    print(f"\n📋 FILTER RESULTS:")
    
    # 2023 Filter Results
    positive_return_2023 = results_df['oos_filter_positive_return_2023'].sum()
    sharpe_above_1_2023 = results_df['oos_filter_sharpe_above_1_2023'].sum()
    drawdown_better_40_2023 = results_df['oos_filter_drawdown_better_40pct_2023'].sum()
    all_filters_2023 = results_df['oos_passes_all_filters_2023'].sum()
    
    print(f"\n   2023 FILTER RESULTS:")
    print(f"     Positive Return: {positive_return_2023}/{valid_pairs_2023} ({100*positive_return_2023/valid_pairs_2023:.1f}%)")
    print(f"     Sharpe > 1.0: {sharpe_above_1_2023}/{valid_pairs_2023} ({100*sharpe_above_1_2023/valid_pairs_2023:.1f}%)")
    print(f"     Drawdown > -40%: {drawdown_better_40_2023}/{valid_pairs_2023} ({100*drawdown_better_40_2023/valid_pairs_2023:.1f}%)")
    print(f"     ALL FILTERS: {all_filters_2023}/{valid_pairs_2023} ({100*all_filters_2023/valid_pairs_2023:.1f}%)")
    
    # 2024 Filter Results
    positive_return_2024 = results_df['oos_filter_positive_return_2024'].sum()
    sharpe_above_1_2024 = results_df['oos_filter_sharpe_above_1_2024'].sum()
    drawdown_better_40_2024 = results_df['oos_filter_drawdown_better_40pct_2024'].sum()
    all_filters_2024 = results_df['oos_passes_all_filters_2024'].sum()
    
    print(f"\n   2024 FILTER RESULTS:")
    print(f"     Positive Return: {positive_return_2024}/{valid_pairs_2024} ({100*positive_return_2024/valid_pairs_2024:.1f}%)")
    print(f"     Sharpe > 1.0: {sharpe_above_1_2024}/{valid_pairs_2024} ({100*sharpe_above_1_2024/valid_pairs_2024:.1f}%)")
    print(f"     Drawdown > -40%: {drawdown_better_40_2024}/{valid_pairs_2024} ({100*drawdown_better_40_2024/valid_pairs_2024:.1f}%)")
    print(f"     ALL FILTERS: {all_filters_2024}/{valid_pairs_2024} ({100*all_filters_2024/valid_pairs_2024:.1f}%)")
    
    # Both Years Filter Results
    both_years_filters = results_df['oos_passes_all_filters_both_years'].sum()
    print(f"\n   BOTH YEARS (COMBINED) FILTER RESULTS:")
    print(f"     Passes ALL filters in BOTH years: {both_years_filters}/{valid_pairs_both} ({100*both_years_filters/valid_pairs_both:.1f}%)")
    
    # Performance comparison: In-Sample vs Out-of-Sample
    print(f"\n📊 IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON:")
    
    is_avg_pf = valid_both['profit_factor'].mean()
    oos_avg_pf_2023 = valid_both['oos_profit_factor_2023'].mean()
    oos_avg_pf_2024 = valid_both['oos_profit_factor_2024'].mean()
    is_avg_return = valid_both['total_cumulative_return'].mean()
    oos_avg_return_2023 = valid_both['oos_total_return_2023'].mean()
    oos_avg_return_2024 = valid_both['oos_total_return_2024'].mean()
    is_avg_dd = valid_both['max_drawdown'].mean()
    oos_avg_dd_2023 = valid_both['oos_max_drawdown_2023'].mean()
    oos_avg_dd_2024 = valid_both['oos_max_drawdown_2024'].mean()
    is_avg_sharpe = valid_both['sharpe_ratio'].mean()
    oos_avg_sharpe_2023 = valid_both['oos_sharpe_ratio_2023'].mean()
    oos_avg_sharpe_2024 = valid_both['oos_sharpe_ratio_2024'].mean()
    
    print(f"   PROFIT FACTOR:")
    print(f"     IS={is_avg_pf:.2f} → OOS 2023={oos_avg_pf_2023:.2f} (Δ: {oos_avg_pf_2023-is_avg_pf:+.2f})")
    print(f"     IS={is_avg_pf:.2f} → OOS 2024={oos_avg_pf_2024:.2f} (Δ: {oos_avg_pf_2024-is_avg_pf:+.2f})")
    print(f"   TOTAL RETURN:")
    print(f"     IS={is_avg_return:.2f} → OOS 2023={oos_avg_return_2023:.2f} (Δ: {oos_avg_return_2023-is_avg_return:+.2f})")
    print(f"     IS={is_avg_return:.2f} → OOS 2024={oos_avg_return_2024:.2f} (Δ: {oos_avg_return_2024-is_avg_return:+.2f})")
    print(f"   MAX DRAWDOWN:")
    print(f"     IS={is_avg_dd:.2%} → OOS 2023={oos_avg_dd_2023:.2%} (Δ: {oos_avg_dd_2023-is_avg_dd:+.2%})")
    print(f"     IS={is_avg_dd:.2%} → OOS 2024={oos_avg_dd_2024:.2%} (Δ: {oos_avg_dd_2024-is_avg_dd:+.2%})")
    print(f"   SHARPE RATIO:")
    print(f"     IS={is_avg_sharpe:.2f} → OOS 2023={oos_avg_sharpe_2023:.2f} (Δ: {oos_avg_sharpe_2023-is_avg_sharpe:+.2f})")
    print(f"     IS={is_avg_sharpe:.2f} → OOS 2024={oos_avg_sharpe_2024:.2f} (Δ: {oos_avg_sharpe_2024-is_avg_sharpe:+.2f})")
    
    # Top performers in OOS - 2023
    if all_filters_2023 > 0:
        top_oos_pairs_2023 = valid_both[valid_both['oos_passes_all_filters_2023']].copy()
        top_oos_pairs_2023 = top_oos_pairs_2023.sort_values('oos_profit_factor_2023', ascending=False)
        
        print(f"\n🏆 TOP 2023 OOS PERFORMERS (passing all filters):")
        print(f"   {'Rank':<4} {'Pair':<20} {'Type':<6} {'OOS_PF':<8} {'OOS_Ret':<9} {'OOS_DD':<9} {'OOS_Sharpe':<10}")
        print(f"   {'-'*4} {'-'*20} {'-'*6} {'-'*8} {'-'*9} {'-'*9} {'-'*10}")
        
        for i, (_, row) in enumerate(top_oos_pairs_2023.head(5).iterrows(), 1):
            pair_name = f"{row['trading_coin']}-{row['reference_coin']}"
            print(f"   {i:<4} {pair_name:<20} {row['trade_type']:<6} "
                  f"{row['oos_profit_factor_2023']:<8.2f} {row['oos_total_return_2023']:<9.2f} "
                  f"{row['oos_max_drawdown_2023']:<9.2%} {row['oos_sharpe_ratio_2023']:<10.2f}")
    else:
        print(f"\n❌ No pairs passed all 2023 OOS filters!")
    
    # Top performers in OOS - 2024  
    if all_filters_2024 > 0:
        top_oos_pairs_2024 = valid_both[valid_both['oos_passes_all_filters_2024']].copy()
        top_oos_pairs_2024 = top_oos_pairs_2024.sort_values('oos_profit_factor_2024', ascending=False)
        
        print(f"\n🏆 TOP 2024 OOS PERFORMERS (passing all filters):")
        print(f"   {'Rank':<4} {'Pair':<20} {'Type':<6} {'OOS_PF':<8} {'OOS_Ret':<9} {'OOS_DD':<9} {'OOS_Sharpe':<10}")
        print(f"   {'-'*4} {'-'*20} {'-'*6} {'-'*8} {'-'*9} {'-'*9} {'-'*10}")
        
        for i, (_, row) in enumerate(top_oos_pairs_2024.head(5).iterrows(), 1):
            pair_name = f"{row['trading_coin']}-{row['reference_coin']}"
            print(f"   {i:<4} {pair_name:<20} {row['trade_type']:<6} "
                  f"{row['oos_profit_factor_2024']:<8.2f} {row['oos_total_return_2024']:<9.2f} "
                  f"{row['oos_max_drawdown_2024']:<9.2%} {row['oos_sharpe_ratio_2024']:<10.2f}")
    else:
        print(f"\n❌ No pairs passed all 2024 OOS filters!")
    
    # Trade type analysis - for valid results with both years
    print(f"\n📊 TRADE TYPE ANALYSIS (Both Years):")
    
    # 2023 analysis
    print(f"\n   2023 OOS RESULTS:")
    valid_2023 = results_df[results_df['oos_error_2023'].isna()]
    trade_type_stats_2023 = valid_2023.groupby('trade_type').agg({
        'oos_profit_factor_2023': 'mean',
        'oos_total_return_2023': 'mean', 
        'oos_max_drawdown_2023': 'mean',
        'oos_sharpe_ratio_2023': 'mean',
        'oos_passes_all_filters_2023': 'sum'
    }).round(3)
    
    trade_type_counts_2023 = valid_2023['trade_type'].value_counts()
    
    for trade_type in trade_type_stats_2023.index:
        count = trade_type_counts_2023[trade_type]
        stats = trade_type_stats_2023.loc[trade_type]
        pass_count = int(stats['oos_passes_all_filters_2023'])
        
        print(f"     {trade_type.upper()}: {count} pairs")
        print(f"       Avg Profit Factor: {stats['oos_profit_factor_2023']:.3f}")
        print(f"       Avg Return: {stats['oos_total_return_2023']:.3f}")
        print(f"       Avg Drawdown: {stats['oos_max_drawdown_2023']:.2%}")
        print(f"       Avg Sharpe: {stats['oos_sharpe_ratio_2023']:.3f}")
        print(f"       Passing all filters: {pass_count}/{count} ({100*pass_count/count:.1f}%)")
        
    # 2024 analysis  
    print(f"\n   2024 OOS RESULTS:")
    valid_2024 = results_df[results_df['oos_error_2024'].isna()]
    trade_type_stats_2024 = valid_2024.groupby('trade_type').agg({
        'oos_profit_factor_2024': 'mean',
        'oos_total_return_2024': 'mean', 
        'oos_max_drawdown_2024': 'mean',
        'oos_sharpe_ratio_2024': 'mean',
        'oos_passes_all_filters_2024': 'sum'
    }).round(3)
    
    trade_type_counts_2024 = valid_2024['trade_type'].value_counts()
    
    for trade_type in trade_type_stats_2024.index:
        count = trade_type_counts_2024[trade_type]
        stats = trade_type_stats_2024.loc[trade_type]
        pass_count = int(stats['oos_passes_all_filters_2024'])
        
        print(f"     {trade_type.upper()}: {count} pairs")
        print(f"       Avg Profit Factor: {stats['oos_profit_factor_2024']:.3f}")
        print(f"       Avg Return: {stats['oos_total_return_2024']:.3f}")
        print(f"       Avg Drawdown: {stats['oos_max_drawdown_2024']:.2%}")
        print(f"       Avg Sharpe: {stats['oos_sharpe_ratio_2024']:.3f}")
        print(f"       Passing all filters: {pass_count}/{count} ({100*pass_count/count:.1f}%)")
    
    # Save summary to file
    summary_file = os.path.join(OUTPUT_DIR, "oos_performance_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("OUT-OF-SAMPLE PERFORMANCE ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Periods: 2023 & 2024 (Out-of-Sample)\n")
        f.write(f"Algorithm: Intermarket Difference Trading Strategy\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write(f"Total pairs tested: {total_pairs}\n")
        f.write(f"Valid 2023 backtests: {valid_pairs_2023}\n") 
        f.write(f"Valid 2024 backtests: {valid_pairs_2024}\n")
        f.write(f"Valid both years backtests: {valid_pairs_both}\n")
        f.write(f"Failed backtests: {total_pairs - max(valid_pairs_2023, valid_pairs_2024)}\n\n")
        
        f.write("OOS PERFORMANCE METRICS (2023):\n")
        f.write(f"Average Profit Factor: {valid_2023['oos_profit_factor_2023'].mean():.3f}\n")
        f.write(f"Average Total Return: {valid_2023['oos_total_return_2023'].mean():.3f}\n")
        f.write(f"Average Max Drawdown: {valid_2023['oos_max_drawdown_2023'].mean():.3f}\n")
        f.write(f"Average Sharpe Ratio: {valid_2023['oos_sharpe_ratio_2023'].mean():.3f}\n")
        f.write(f"Average Number of Trades: {valid_2023['oos_num_trades_2023'].mean():.0f}\n\n")
        
        f.write("OOS PERFORMANCE METRICS (2024):\n")
        f.write(f"Average Profit Factor: {valid_2024['oos_profit_factor_2024'].mean():.3f}\n")
        f.write(f"Average Total Return: {valid_2024['oos_total_return_2024'].mean():.3f}\n")
        f.write(f"Average Max Drawdown: {valid_2024['oos_max_drawdown_2024'].mean():.3f}\n")
        f.write(f"Average Sharpe Ratio: {valid_2024['oos_sharpe_ratio_2024'].mean():.3f}\n")
        f.write(f"Average Number of Trades: {valid_2024['oos_num_trades_2024'].mean():.0f}\n\n")
        
        f.write("OOS FILTER RESULTS:\n")
        f.write("Filter criteria:\n")
        f.write("1. Total Return > 0% (positive PnL)\n")
        f.write("2. Sharpe Ratio > 1.0\n") 
        f.write("3. Max Drawdown better than -40%\n\n")
        
        f.write("2023 FILTER RESULTS:\n")
        f.write(f"Filter 1 - Positive Return: {positive_return_2023}/{valid_pairs_2023} ({100*positive_return_2023/valid_pairs_2023:.1f}%)\n")
        f.write(f"Filter 2 - Sharpe > 1.0: {sharpe_above_1_2023}/{valid_pairs_2023} ({100*sharpe_above_1_2023/valid_pairs_2023:.1f}%)\n")
        f.write(f"Filter 3 - Drawdown > -40%: {drawdown_better_40_2023}/{valid_pairs_2023} ({100*drawdown_better_40_2023/valid_pairs_2023:.1f}%)\n")
        f.write(f"ALL FILTERS COMBINED: {all_filters_2023}/{valid_pairs_2023} ({100*all_filters_2023/valid_pairs_2023:.1f}%)\n\n")
        
        f.write("2024 FILTER RESULTS:\n")
        f.write(f"Filter 1 - Positive Return: {positive_return_2024}/{valid_pairs_2024} ({100*positive_return_2024/valid_pairs_2024:.1f}%)\n")
        f.write(f"Filter 2 - Sharpe > 1.0: {sharpe_above_1_2024}/{valid_pairs_2024} ({100*sharpe_above_1_2024/valid_pairs_2024:.1f}%)\n")
        f.write(f"Filter 3 - Drawdown > -40%: {drawdown_better_40_2024}/{valid_pairs_2024} ({100*drawdown_better_40_2024/valid_pairs_2024:.1f}%)\n")
        f.write(f"ALL FILTERS COMBINED: {all_filters_2024}/{valid_pairs_2024} ({100*all_filters_2024/valid_pairs_2024:.1f}%)\n\n")
        
        f.write("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON:\n")
        f.write(f"PROFIT FACTOR:\n")
        f.write(f"  IS={is_avg_pf:.2f} → OOS 2023={oos_avg_pf_2023:.2f} (Δ: {oos_avg_pf_2023-is_avg_pf:+.2f})\n")
        f.write(f"  IS={is_avg_pf:.2f} → OOS 2024={oos_avg_pf_2024:.2f} (Δ: {oos_avg_pf_2024-is_avg_pf:+.2f})\n")
        f.write(f"TOTAL RETURN:\n")
        f.write(f"  IS={is_avg_return:.2f} → OOS 2023={oos_avg_return_2023:.2f} (Δ: {oos_avg_return_2023-is_avg_return:+.2f})\n")
        f.write(f"  IS={is_avg_return:.2f} → OOS 2024={oos_avg_return_2024:.2f} (Δ: {oos_avg_return_2024-is_avg_return:+.2f})\n")
        f.write(f"MAX DRAWDOWN:\n")
        f.write(f"  IS={is_avg_dd:.2%} → OOS 2023={oos_avg_dd_2023:.2%} (Δ: {oos_avg_dd_2023-is_avg_dd:+.2%})\n")
        f.write(f"  IS={is_avg_dd:.2%} → OOS 2024={oos_avg_dd_2024:.2%} (Δ: {oos_avg_dd_2024-is_avg_dd:+.2%})\n")
        f.write(f"SHARPE RATIO:\n")
        f.write(f"  IS={is_avg_sharpe:.2f} → OOS 2023={oos_avg_sharpe_2023:.2f} (Δ: {oos_avg_sharpe_2023-is_avg_sharpe:+.2f})\n")
        f.write(f"  IS={is_avg_sharpe:.2f} → OOS 2024={oos_avg_sharpe_2024:.2f} (Δ: {oos_avg_sharpe_2024-is_avg_sharpe:+.2f})\n\n")
        
        if all_filters_2023 > 0:
            f.write(f"TOP 2023 OOS PERFORMERS (passing all filters):\n")
            f.write(f"Rank  Pair                 Type   OOS_PF   OOS_Ret   OOS_DD    OOS_Sharpe\n")
            f.write(f"----  ----                 ----   ------   -------   ------    ----------\n")
            
            for i, (_, row) in enumerate(top_oos_pairs_2023.head(5).iterrows(), 1):
                pair_name = f"{row['trading_coin']}-{row['reference_coin']}"
                f.write(f"{i:<4}  {pair_name:<20} {row['trade_type']:<6} "
                       f"{row['oos_profit_factor_2023']:<8.2f} {row['oos_total_return_2023']:<9.2f} "
                       f"{row['oos_max_drawdown_2023']:<9.2%} {row['oos_sharpe_ratio_2023']:<10.2f}\n")
        else:
            f.write("No pairs passed all 2023 OOS filters!\n\n")
            
        if all_filters_2024 > 0:
            f.write(f"TOP 2024 OOS PERFORMERS (passing all filters):\n")
            f.write(f"Rank  Pair                 Type   OOS_PF   OOS_Ret   OOS_DD    OOS_Sharpe\n")
            f.write(f"----  ----                 ----   ------   -------   ------    ----------\n")
            
            for i, (_, row) in enumerate(top_oos_pairs_2024.head(5).iterrows(), 1):
                pair_name = f"{row['trading_coin']}-{row['reference_coin']}"
                f.write(f"{i:<4}  {pair_name:<20} {row['trade_type']:<6} "
                       f"{row['oos_profit_factor_2024']:<8.2f} {row['oos_total_return_2024']:<9.2f} "
                       f"{row['oos_max_drawdown_2024']:<9.2%} {row['oos_sharpe_ratio_2024']:<10.2f}\n")
        else:
            f.write("No pairs passed all 2024 OOS filters!\n")
    
    print(f"\n💾 Detailed summary saved to: {summary_file}")
    print(f"\n🎉 OOS Analysis completed!")


def main():
    """Main execution function."""
    
    print("🚀 Starting Out-of-Sample Backtesting")
    print("=" * 60)
    
    # Load selected pairs
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    
    print(f"📂 Loading selected pairs from {INPUT_CSV}...")
    selected_pairs = pd.read_csv(INPUT_CSV)
    print(f"✅ Loaded {len(selected_pairs)} selected pairs")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    print(f"\n🔄 Running OOS backtests for 2023 and 2024 separately...")
    
    # Process each pair
    results = []
    failed_pairs = 0
    
    for index, row in tqdm(selected_pairs.iterrows(), total=len(selected_pairs), desc="Processing pairs"):
        trading_coin = row['trading_coin']
        reference_coin = row['reference_coin'] 
        trade_type = row['trade_type']
        
        # Run OOS backtests for both years
        oos_result = run_oos_backtest_both_years(trading_coin, reference_coin, trade_type)
        
        # Check for errors in either year
        has_error_2023 = oos_result.get('oos_error_2023') is not None
        has_error_2024 = oos_result.get('oos_error_2024') is not None
        
        if has_error_2023 and has_error_2024:
            failed_pairs += 1
            print(f"\n⚠️  Failed: {trading_coin}-{reference_coin} ({trade_type}) - both years failed")
        elif has_error_2023 or has_error_2024:
            print(f"\n⚠️  Partial failure: {trading_coin}-{reference_coin} ({trade_type}) - one year failed")
        else:
            # Save detailed results for both years
            try:
                save_detailed_results_both_years(trading_coin, reference_coin, trade_type)
            except Exception as e:
                print(f"\n⚠️  Could not save detailed results for {trading_coin}-{reference_coin}: {e}")
        
        # Combine original data with OOS results
        combined_result = row.to_dict()
        combined_result.update(oos_result)
        results.append(combined_result)
    
    # Create final results DataFrame
    results_df = pd.DataFrame(results)
    
    # Add OOS filter columns for both years
    # 2023 filters
    results_df['oos_filter_positive_return_2023'] = results_df['oos_total_return_2023'] > 0
    results_df['oos_filter_sharpe_above_1_2023'] = results_df['oos_sharpe_ratio_2023'] > 1.0
    results_df['oos_filter_drawdown_better_40pct_2023'] = results_df['oos_max_drawdown_2023'] > -0.4
    results_df['oos_passes_all_filters_2023'] = (
        results_df['oos_filter_positive_return_2023'] & 
        results_df['oos_filter_sharpe_above_1_2023'] & 
        results_df['oos_filter_drawdown_better_40pct_2023']
    )
    
    # 2024 filters
    results_df['oos_filter_positive_return_2024'] = results_df['oos_total_return_2024'] > 0
    results_df['oos_filter_sharpe_above_1_2024'] = results_df['oos_sharpe_ratio_2024'] > 1.0
    results_df['oos_filter_drawdown_better_40pct_2024'] = results_df['oos_max_drawdown_2024'] > -0.4
    results_df['oos_passes_all_filters_2024'] = (
        results_df['oos_filter_positive_return_2024'] & 
        results_df['oos_filter_sharpe_above_1_2024'] & 
        results_df['oos_filter_drawdown_better_40pct_2024']
    )
    
    # Combined filters (both years must pass)
    results_df['oos_passes_all_filters_both_years'] = (
        results_df['oos_passes_all_filters_2023'] & 
        results_df['oos_passes_all_filters_2024']
    )
    
    # Save results CSV with new filter columns
    results_df.to_csv(OUTPUT_CSV, index=False, float_format='%.4f')
    
    print(f"\n✅ OOS backtesting completed!")
    print(f"📊 Processed: {len(results)} pairs")
    if failed_pairs > 0:
        print(f"⚠️  Failed: {failed_pairs} pairs")
    print(f"💾 Results saved to: {OUTPUT_CSV}")
    print(f"📁 Detailed results in: {OUTPUT_DIR}/")
    
    # Comprehensive OOS Analysis
    analyze_oos_performance(results_df)

if __name__ == "__main__":
    main()