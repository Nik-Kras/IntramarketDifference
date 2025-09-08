#!/usr/bin/env python3
"""
Core Trading Algorithm and Utilities
====================================

Shared functions for the intermarket difference trading strategy.
Contains trading algorithm, data loading, metrics calculation, and utility functions.

This module eliminates code duplication between in-sample and out-of-sample scripts.
"""

import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import sys

# Add parent directory to path to import trading functions
sys.path.append('..')
from parameter_optimization_experiment.trades_from_signal import get_trades_from_signal


# -----------------------
# Algorithm Parameters
# -----------------------
LOOKBACK = 24               # MA window for cmma
ATR_LOOKBACK = 168          # ATR window for cmma  
THRESHOLD = 0.25            # signal threshold
MIN_OVERLAP = 500           # minimum data points for in-sample (can be overridden)


# -----------------------
# Core Trading Algorithm
# -----------------------
def cmma(ohlc: pd.DataFrame, lookback: int, atr_lookback: int = 168) -> pd.Series:
    """Close-minus-MA normalized by ATR * sqrt(L)."""
    atr = ta.atr(ohlc["high"], ohlc["low"], ohlc["close"], atr_lookback)
    ma = ohlc["close"].rolling(lookback).mean()
    ind = (ohlc["close"] - ma) / (atr * lookback ** 0.5)
    return ind


def threshold_revert_signal(ind: pd.Series, threshold: float) -> np.ndarray:
    """Generate mean reversion signals based on threshold crossings."""
    signal = np.zeros(len(ind))
    position = 0
    values = ind.values
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
        signal[i] = position
    return signal


# -----------------------
# Data Loading Functions  
# -----------------------
def load_csv_with_cache(coin: str, data_dir: str, suffix: str, cache_dict: dict) -> pd.DataFrame:
    """Load CSV file for a coin with caching support."""
    # Check cache first
    cache_key = f"{coin}_{suffix}"
    if cache_key in cache_dict:
        return cache_dict[cache_key]
    
    filepath = os.path.join(data_dir, f"{coin}_{suffix}.csv")
    
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        
        # Ensure required columns
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                return None
        
        df = df.sort_index().dropna()
        
        # Cache the loaded data
        cache_dict[cache_key] = df
        
        return df
        
    except Exception:
        return None


def align_frames(a: pd.DataFrame, b: pd.DataFrame) -> tuple:
    """Align two dataframes by common index."""
    ix = a.index.intersection(b.index)
    return a.loc[ix].copy(), b.loc[ix].copy()


# -----------------------
# Metrics Calculation
# -----------------------
def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate maximum drawdown from cumulative returns."""
    if len(cumulative_returns) == 0:
        return np.nan
    
    cumulative = np.exp(cumulative_returns.cumsum())
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def calculate_metrics_from_trades(trades_df: pd.DataFrame, data_points: int) -> dict:
    """
    Calculate performance metrics from trade DataFrame.
    
    Args:
        trades_df: DataFrame with trade data from get_trades_from_signal
        data_points: Number of data points in the time series
        
    Returns:
        Dictionary with performance metrics
    """
    if len(trades_df) == 0:
        return {
            'total_cumulative_return': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'last_year_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'num_trades': 0,
            'data_points': data_points
        }
    
    # Convert trades to returns series
    trade_returns = []
    trade_times = []
    
    for _, trade in trades_df.iterrows():
        if pd.notna(trade['exit_price']) and pd.notna(trade['entry_price']):
            if trade['type'] == 1:  # Long trade
                log_return = np.log(trade['exit_price'] / trade['entry_price'])
            else:  # Short trade
                log_return = np.log(trade['entry_price'] / trade['exit_price'])
            
            trade_returns.append(log_return)
            trade_times.append(trade['exit_time'])
    
    if len(trade_returns) == 0:
        return {
            'total_cumulative_return': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'last_year_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'num_trades': 0,
            'data_points': data_points
        }
    
    # Create returns series
    returns_series = pd.Series(trade_returns, index=trade_times)
    returns_series = returns_series.sort_index().replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns_series) == 0:
        return {
            'total_cumulative_return': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'last_year_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'num_trades': 0,
            'data_points': data_points
        }
    
    # Calculate metrics - total_cumulative_return based on equity curve
    # Equity curve: starting from $1000 (same as plots), final value after all trades
    initial_budget = 1000.0
    portfolio_values = initial_budget * np.exp(returns_series.cumsum())
    total_return = portfolio_values.iloc[-1] / initial_budget
    
    # Profit factor
    gains = returns_series[returns_series > 0].sum()
    losses = returns_series[returns_series < 0].sum()
    profit_factor = gains / abs(losses) if losses < 0 else np.inf if gains > 0 else 0
    
    # Max drawdown
    max_drawdown = calculate_max_drawdown(returns_series)
    
    # Last year drawdown (last 25% of data)
    last_quarter_start = int(len(returns_series) * 0.75)
    last_quarter_rets = returns_series.iloc[last_quarter_start:]
    if len(last_quarter_rets) > 0:
        last_year_drawdown = calculate_max_drawdown(last_quarter_rets)
    else:
        last_year_drawdown = 0
    
    # Sharpe ratio (assuming trades are roughly daily frequency)
    # Use conservative estimate of trade frequency for annualization
    trade_frequency = max(1, len(returns_series) * 365 / (data_points / 24))  # Convert hourly data points to days
    sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(trade_frequency) if returns_series.std() > 0 else 0
    
    # Volatility (annualized)
    volatility = returns_series.std() * np.sqrt(trade_frequency)
    
    return {
        'total_cumulative_return': total_return,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'last_year_drawdown': last_year_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'volatility': volatility,
        'num_trades': len(returns_series),
        'data_points': data_points
    }


# -----------------------
# Backtesting Functions
# -----------------------
def run_pair_backtest_core(ref_df: pd.DataFrame, traded_df: pd.DataFrame, 
                          ref_coin: str, trading_coin: str, 
                          min_overlap: int = MIN_OVERLAP) -> list:
    """
    Core pair backtesting logic.
    
    Args:
        ref_df: Reference coin OHLC data
        traded_df: Trading coin OHLC data  
        ref_coin: Reference coin symbol
        trading_coin: Trading coin symbol
        min_overlap: Minimum data overlap required
        
    Returns:
        List of metrics dictionaries (one for each trading type: both, longs, shorts)
    """
    
    if ref_df is None or traded_df is None:
        return []
    
    # Align data
    ref_df, traded_df = align_frames(ref_df, traded_df)
    
    if len(ref_df) < min_overlap:
        return []
    
    try:
        # Calculate returns
        traded_df = traded_df.copy()
        traded_df["diff"] = np.log(traded_df["close"]).diff()
        traded_df["next_return"] = traded_df["diff"].shift(-1)
        
        # Calculate CMMA indicators
        ref_cmma = cmma(ref_df, LOOKBACK, ATR_LOOKBACK)
        trd_cmma = cmma(traded_df, LOOKBACK, ATR_LOOKBACK)
        intermarket_diff = trd_cmma - ref_cmma
        
        # Generate signal
        traded_df["sig"] = threshold_revert_signal(intermarket_diff, THRESHOLD)
        
        # Get detailed trades
        long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, traded_df["sig"].values)
        
        data_points = len(traded_df)
        
        # Calculate metrics for each trading type
        results = []
        
        # 1. Both longs and shorts
        both_metrics = calculate_metrics_from_trades(all_trades, data_points)
        both_metrics.update({
            'reference_coin': ref_coin,
            'trading_coin': trading_coin,
            'trading_type': 'both'
        })
        results.append(both_metrics)
        
        # 2. Longs only
        longs_metrics = calculate_metrics_from_trades(long_trades, data_points)
        longs_metrics.update({
            'reference_coin': ref_coin,
            'trading_coin': trading_coin,
            'trading_type': 'longs'
        })
        results.append(longs_metrics)
        
        # 3. Shorts only
        shorts_metrics = calculate_metrics_from_trades(short_trades, data_points)
        shorts_metrics.update({
            'reference_coin': ref_coin,
            'trading_coin': trading_coin,
            'trading_type': 'shorts'
        })
        results.append(shorts_metrics)
        
        return results
        
    except Exception:
        return []


def run_single_trading_type_backtest(ref_df: pd.DataFrame, traded_df: pd.DataFrame,
                                   ref_coin: str, trading_coin: str, trading_type: str,
                                   min_overlap: int = MIN_OVERLAP) -> dict:
    """
    Run backtest for a single trading type (for OOS validation).
    
    Args:
        ref_df: Reference coin OHLC data
        traded_df: Trading coin OHLC data
        ref_coin: Reference coin symbol
        trading_coin: Trading coin symbol
        trading_type: 'both', 'longs', or 'shorts'
        min_overlap: Minimum data overlap required
        
    Returns:
        Metrics dictionary or None if failed
    """
    
    if ref_df is None or traded_df is None:
        return None
    
    # Align data
    ref_df, traded_df = align_frames(ref_df, traded_df)
    
    if len(ref_df) < min_overlap:
        return None
    
    try:
        # Calculate returns
        traded_df = traded_df.copy()
        traded_df["diff"] = np.log(traded_df["close"]).diff()
        traded_df["next_return"] = traded_df["diff"].shift(-1)
        
        # Calculate CMMA indicators
        ref_cmma = cmma(ref_df, LOOKBACK, ATR_LOOKBACK)
        trd_cmma = cmma(traded_df, LOOKBACK, ATR_LOOKBACK)
        intermarket_diff = trd_cmma - ref_cmma
        
        # Generate signal
        traded_df["sig"] = threshold_revert_signal(intermarket_diff, THRESHOLD)
        
        # Get detailed trades
        long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, traded_df["sig"].values)
        
        data_points = len(traded_df)
        
        # Select trades based on trading type
        if trading_type == 'longs':
            selected_trades = long_trades
        elif trading_type == 'shorts':
            selected_trades = short_trades
        else:  # 'both'
            selected_trades = all_trades
        
        # Calculate metrics
        metrics = calculate_metrics_from_trades(selected_trades, data_points)
        metrics.update({
            'reference_coin': ref_coin,
            'trading_coin': trading_coin,
            'trading_type': trading_type
        })
        
        return metrics
        
    except Exception:
        return None


# -----------------------
# Equity Curve Generation
# -----------------------
def generate_equity_curve_from_trades(trades_df: pd.DataFrame, initial_capital: float = 1000.0) -> pd.Series:
    """
    Generate equity curve from trade DataFrame.
    
    Args:
        trades_df: DataFrame with trade data (with 'exit_time' column or time index)
        initial_capital: Starting capital amount
        
    Returns:
        Pandas Series with equity curve values indexed by time
    """
    if len(trades_df) == 0:
        return None
        
    # Convert trades to returns series
    trade_returns = []
    trade_times = []
    
    for idx, trade in trades_df.iterrows():
        if pd.notna(trade['exit_price']) and pd.notna(trade['entry_price']):
            if trade['type'] == 1:  # Long trade
                log_return = np.log(trade['exit_price'] / trade['entry_price'])
            else:  # Short trade
                log_return = np.log(trade['entry_price'] / trade['exit_price'])
            
            trade_returns.append(log_return)
            # Use exit_time column if available, otherwise use index
            if 'exit_time' in trade.index:
                trade_times.append(trade['exit_time'])
            else:
                trade_times.append(idx)
    
    if len(trade_returns) == 0:
        return None
    
    # Create returns series
    returns_series = pd.Series(trade_returns, index=trade_times)
    returns_series = returns_series.sort_index().replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns_series) == 0:
        return None
    
    # Generate equity curve
    portfolio_values = initial_capital * np.exp(returns_series.cumsum())
    
    return portfolio_values


# -----------------------
# Utility Functions
# -----------------------
def get_available_coins(data_dir: str, suffix: str) -> list:
    """Get list of available coins from data directory."""
    import glob
    files = glob.glob(os.path.join(data_dir, f"*_{suffix}.csv"))
    coins = [os.path.basename(f).replace(f"_{suffix}.csv", "") for f in files]
    return sorted(coins)


def create_trade_record(trade_row, entry_time) -> dict:
    """Create standardized trade record for JSON export."""
    # Calculate log return
    if trade_row['type'] == 1:  # Long trade
        log_return = np.log(trade_row['exit_price'] / trade_row['entry_price'])
    else:  # Short trade  
        log_return = np.log(trade_row['entry_price'] / trade_row['exit_price'])
    
    return {
        'time_entered': entry_time.isoformat() if pd.notna(entry_time) else None,
        'time_exited': trade_row['exit_time'].isoformat() if pd.notna(trade_row['exit_time']) else None,
        'log_return': float(log_return) if pd.notna(log_return) else None,
        'trade_type': 'long' if trade_row['type'] == 1 else 'short'
    }
