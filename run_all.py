import os
import glob
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import statsmodels.api as sm
from itertools import permutations
from trades_from_signal import get_trades_from_signal
from tqdm import tqdm

# -----------------------
# Config
# -----------------------
DATA_DIR = "data"           # where your 9+ CSVs are
OUT_CSV  = "pair_backtest_results.csv"
TRADES_DIR = "trades"       # where to save detailed trade JSON files

LOOKBACK = 24               # MA window for cmma
ATR_LOOKBACK = 168          # ATR window for cmma
THRESHOLD = 0.25            # signal threshold
MIN_OVERLAP = 500           # skip pairs with tiny overlap


# -----------------------
# Indicators & signal
# -----------------------
def cmma(ohlc: pd.DataFrame, lookback: int, atr_lookback: int = 168) -> pd.Series:
    """Close-minus-MA normalized by ATR * sqrt(L)."""
    atr = ta.atr(ohlc["high"], ohlc["low"], ohlc["close"], atr_lookback)
    ma = ohlc["close"].rolling(lookback).mean()
    ind = (ohlc["close"] - ma) / (atr * lookback ** 0.5)
    return ind

def threshold_revert_signal(ind: pd.Series, threshold: float) -> np.ndarray:
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

# -----------------------
# Data helpers
# -----------------------
def load_one_csv(path: str) -> pd.DataFrame:
    """Load a CSV, standardize columns, set datetime index."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "open time" in df.columns:
        df["open time"] = pd.to_datetime(df["open time"])
        df = df.set_index("open time")
    else:
        raise ValueError(f"No datetime column found in {path} (expected 'date' or 'time').")
    df = df.sort_index()
    # basic sanity
    for col in ["high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df.dropna()

def align_frames(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ix = a.index.intersection(b.index)
    return a.loc[ix].copy(), b.loc[ix].copy()


# -----------------------
# Stats: Beta & R^2
# -----------------------
def beta_r2_traded_on_ref(traded_close: pd.Series, ref_close: pd.Series) -> tuple[float, float]:
    """
    OLS on log prices: log(traded) ~ const + beta * log(reference)
    Returns (beta, R^2).
    """
    y, x = traded_close.align(ref_close, join="inner")
    y_log = np.log(y).dropna()
    x_log = np.log(x).dropna()
    y_log, x_log = y_log.align(x_log, join="inner")
    if len(y_log) < 100:
        return np.nan, np.nan
    X = sm.add_constant(x_log)
    model = sm.OLS(y_log, X).fit()
    return float(model.params[1]), float(model.rsquared)


# -----------------------
# Backtest one ordered pair
# -----------------------
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

def calculate_exit_hour_stats(traded_df: pd.DataFrame) -> tuple[float, float]:
    """Calculate mean and std of exit hours from trades."""
    # Get trades using the existing function
    try:
        long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, traded_df["sig"].values)
        
        if len(all_trades) == 0:
            return np.nan, np.nan
        
        # Extract exit hours (0-23)
        exit_hours = all_trades['exit_time'].dt.hour
        
        if len(exit_hours) == 0:
            return np.nan, np.nan
            
        mean_exit_hour = float(exit_hours.mean())
        std_exit_hour = float(exit_hours.std())
        
        return mean_exit_hour, std_exit_hour
        
    except Exception as e:
        return np.nan, np.nan

def save_trades_to_json(reference_name: str, traded_name: str, traded_df: pd.DataFrame):
    """Save detailed trade data to JSON file."""
    try:
        # Get individual trades using existing function
        long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, traded_df["sig"].values)
        
        if len(all_trades) == 0:
            return  # No trades to save
        
        # Prepare trade data for JSON
        trades_data = []
        
        for entry_time, trade in all_trades.iterrows():
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
        
        # Create directory structure - ensure both root and coin directories exist
        if not os.path.exists(TRADES_DIR):
            os.makedirs(TRADES_DIR, exist_ok=True)
            print(f"Created trades directory: {TRADES_DIR}")
        
        coin_dir = os.path.join(TRADES_DIR, traded_name)
        if not os.path.exists(coin_dir):
            os.makedirs(coin_dir, exist_ok=True)
            print(f"Created coin directory: {coin_dir}")
        
        # Save to JSON file with pair name format
        filename = f"{reference_name}_{traded_name}_trades.json"
        filepath = os.path.join(coin_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(trades_data, f, indent=2)
            
    except Exception as e:
        print(f"Warning: Could not save trades for {reference_name}-{traded_name}: {e}")
        import traceback
        traceback.print_exc()

def run_pair(reference_name: str,
             traded_name: str,
             ref_df: pd.DataFrame,
             traded_df: pd.DataFrame,
             lookback: int = LOOKBACK,
             atr_lookback: int = ATR_LOOKBACK,
             threshold: float = THRESHOLD) -> dict:
    """
    Reproduce your logic:
      intermarket_diff = cmma(traded) - cmma(reference)
      signal on intermarket_diff -> trade TRaded only
      rets = signal * next_return(traded)
    Report all required metrics: Profit Factor, Total Return, Number of Trades, 
    Max DrawDown, Sharpe Ratio, plus Beta/R^2 (Traded~Reference).
    """
    ref_df, traded_df = align_frames(ref_df, traded_df)
    if len(ref_df) < MIN_OVERLAP:
        return {
            "Reference Coin": reference_name,
            "Trading Coin": traded_name,
            "Number of Trades": 0,
            "Profit Factor": np.nan,
            "Total Cummulative Return": np.nan,
            "Max DrawDown": np.nan,
            "Sharpe Ratio": np.nan,
            "Mean Exit Hour": np.nan,
            "STD Exit Hour": np.nan,
        }

    # next_return on traded asset (log-return 1-step ahead)
    traded_df = traded_df.copy()
    traded_df["diff"] = np.log(traded_df["close"]).diff()
    traded_df["next_return"] = traded_df["diff"].shift(-1)

    # cmma indicators
    ref_cmma = cmma(ref_df, lookback, atr_lookback)
    trd_cmma = cmma(traded_df, lookback, atr_lookback)
    intermarket_diff = trd_cmma - ref_cmma

    # signal & returns
    traded_df["sig"] = threshold_revert_signal(intermarket_diff, threshold)
    rets = traded_df["sig"] * traded_df["next_return"]
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()

    # Calculate number of trades (position changes)
    signal_changes = traded_df["sig"].diff().fillna(0)
    num_trades = int((signal_changes != 0).sum())

    # profit factor (convert log returns to simple returns first)
    simple_rets = np.exp(rets) - 1  # Convert log returns to simple returns
    gains = simple_rets[simple_rets > 0].sum()
    losses = simple_rets[simple_rets < 0].abs().sum()
    profit_factor = np.inf if losses == 0 and gains > 0 else (gains / losses if losses > 0 else np.nan)

    total_return = rets.cumsum().iloc[-1] if len(rets) else np.nan
    
    # Max drawdown
    max_drawdown = calculate_max_drawdown(rets)
    
    # Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(rets)
    
    # Exit hour statistics
    mean_exit_hour, std_exit_hour = calculate_exit_hour_stats(traded_df)
    
    # Save detailed trade data to JSON
    save_trades_to_json(reference_name, traded_name, traded_df)

    return {
        "Reference Coin": reference_name,
        "Trading Coin": traded_name,
        "Number of Trades": num_trades,
        "Profit Factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
        "Total Cummulative Return": float(total_return) if pd.notna(total_return) else np.nan,
        "Max DrawDown": float(max_drawdown) if pd.notna(max_drawdown) else np.nan,
        "Sharpe Ratio": float(sharpe_ratio) if pd.notna(sharpe_ratio) else np.nan,
        "Mean Exit Hour": float(mean_exit_hour) if pd.notna(mean_exit_hour) else np.nan,
        "STD Exit Hour": float(std_exit_hour) if pd.notna(std_exit_hour) else np.nan,
    }

def extract_coin_name(filename: str) -> str:
    """From 'ETHUSDT_IS.csv' -> 'ETH'."""
    base = os.path.basename(filename)
    if base.endswith(".csv"):
        base = base[:-4]
    if base.endswith("_IS"):
        base = base[:-3]
    if base.upper().endswith("USDT"):
        base = base[:-4]
    return base.upper()

# -----------------------
# Main: load, pair, run, save
# -----------------------
def main():
    # Load all CSVs under data/
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not paths:
        raise SystemExit(f"No CSV files found in {DATA_DIR}/")

    data = {}
    for p in paths:
        try:
            coin_name = extract_coin_name(p)
            data[coin_name] = load_one_csv(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")

    names = list(data.keys())
    print(f"Loaded {len(names)} assets:", names)

    # Run all ordered pairs (Reference -> Traded), excluding self-pairs
    rows = []
    pair_list = list(permutations(names, 2))
    for ref, traded in tqdm(pair_list, desc="Simulating pairs", unit="pair"):
        res = run_pair(ref, traded, data[ref], data[traded],
                       lookback=LOOKBACK, atr_lookback=ATR_LOOKBACK, threshold=THRESHOLD)
        rows.append(res)

    df = pd.DataFrame(rows)

    # Save CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} experiments to {OUT_CSV}")

if __name__ == "__main__":
    main()
