import os
import glob
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
    Report Profit Factor + Total Return, plus Beta/R^2 (Traded~Reference).
    """
    ref_df, traded_df = align_frames(ref_df, traded_df)
    if len(ref_df) < MIN_OVERLAP:
        return {
            "reference": reference_name,
            "traded": traded_name,
            "profit_factor": np.nan,
            "total_return": np.nan,
            "beta": np.nan,
            "r2": np.nan,
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

    # profit factor
    gains = rets[rets > 0].sum()
    losses = rets[rets < 0].abs().sum()
    profit_factor = np.inf if losses == 0 and gains > 0 else (gains / losses if losses > 0 else np.nan)

    total_return = rets.cumsum().iloc[-1] if len(rets) else np.nan

    # beta & r2
    beta, r2 = beta_r2_traded_on_ref(traded_df["close"], ref_df["close"])

    return {
        "reference": reference_name,
        "traded": traded_name,
        "threshold": threshold,
        "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
        "total_return": float(total_return) if pd.notna(total_return) else np.nan,
        "beta": beta,
        "r2": r2,
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
