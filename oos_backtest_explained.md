# OOS Backtest System - Complete Explanation

## 🎯 **What Does OOS Backtest Do?**

The Out-of-Sample (OOS) backtest validates trading strategies that performed well in historical data (2018-2022) by testing them on **new, unseen data** (2023-2024).

### **The Complete Pipeline:**

```
1. IN-SAMPLE (2018-2022) → 2. FILTER BEST → 3. OOS TEST (2023-2024)
   All pairs tested      →   Top performers  →   Validate on new data
   19,000+ combinations  →   952 selected    →   Check if still works
```

---

## 🔧 **How It Works (Step by Step)**

### **INPUT: Filtered Pairs**
- **File:** `user_custom_filter.csv` 
- **Contains:** 952 pairs that passed strict filters
- **Each row:** `DOGE-AAVE (long)` = Test long-only strategy for this pair

### **PROCESS: Same Algorithm, New Data**
1. **Load OOS data** for both coins (2023 & 2024 separately)
2. **Apply identical trading algorithm** used in original backtest
3. **Filter by strategy type** (long-only, short-only, or combined)
4. **Calculate same metrics** (Profit Factor, Sharpe, Drawdown, etc.)
5. **Generate equity curves** and save detailed trade data

### **OUTPUT: Comprehensive Results**
- **CSV:** All pairs with both IS and OOS metrics side-by-side
- **JSON:** Detailed trade data for each pair and year
- **PNG:** Equity curves showing actual performance
- **Analysis:** Correlation between IS and OOS performance

---

## 📊 **Key Functions Explained**

### **1. Core Trading Algorithm (Identical to Original)**
```python
def cmma(ohlc, lookback, atr_lookback):
    """Calculate momentum indicator"""
    # Same exact formula as run_all.py

def threshold_revert_signal(ind, threshold):
    """Generate trading signals"""  
    # Same exact logic as run_all.py

def run_oos_backtest_single_year(coin1, coin2, trade_type, year):
    """Run backtest for one year"""
    # 1. Load data for specified year
    # 2. Calculate indicators 
    # 3. Generate signals
    # 4. Filter by trade type (long/short/both)
    # 5. Calculate metrics
    # 6. Return results
```

### **2. Strategy Type Filtering (Critical Part)**
```python
if trade_type == 'long':
    # Long-only strategy: only returns from long positions
    rets = rets[signal == 1]
elif trade_type == 'short':  
    # Short-only strategy: only returns from short positions
    rets = rets[signal == -1]
else:
    # Combined strategy: all returns (original algorithm)
    rets = all_returns
```

**Why this matters:** The filtered pairs specify which strategy variant worked best:
- `DOGE-AAVE (long)` → Test only long positions
- `LINK-BNT (short)` → Test only short positions

### **3. Year-by-Year Analysis**
```python
def run_oos_backtest_both_years(coin1, coin2, trade_type):
    """Test strategy on both 2023 and 2024 separately"""
    result_2023 = run_oos_backtest_single_year(coin1, coin2, trade_type, 2023)
    result_2024 = run_oos_backtest_single_year(coin1, coin2, trade_type, 2024)
    
    return {
        'oos_profit_factor_2023': result_2023['profit_factor'],
        'oos_profit_factor_2024': result_2024['profit_factor'],
        # ... other metrics for both years
    }
```

---

## 📁 **File Structure Explained**

### **Input Files:**
- `filtered_results/user_custom_filter.csv` - Selected pairs to test
- `OOS/BTCUSDT_OOS.csv` - Out-of-sample price data (2023-2025)

### **Output Structure:**
```
oos_experiments/
├── oos_backtest_results.csv          # Main results table
├── rank_correlation_analysis.csv     # Signal vs noise analysis
├── DOGE_AAVE_long/                   # Individual pair results
│   ├── oos_metrics_combined.json    # Performance metrics  
│   ├── oos_trades_2023.json         # Individual trades (2023)
│   ├── oos_trades_2024.json         # Individual trades (2024)
│   ├── oos_equity_curve_2023.png    # Visual performance (2023)
│   └── oos_equity_curve_2024.png    # Visual performance (2024)
└── ... (one folder per pair)
```

---

## 🎯 **What Makes This System Correct**

### **1. Algorithm Consistency** ✅
- **Identical indicators:** Same CMMA calculation
- **Identical signals:** Same threshold logic  
- **Identical metrics:** Same profit factor, drawdown formulas

### **2. Strategy Replication** ✅
- **Respects trade types:** Tests exact strategy variant that was selected
- **Proper filtering:** Long-only tests only long positions, etc.
- **No data leakage:** Uses completely separate OOS data

### **3. Comprehensive Validation** ✅
- **Multiple timeframes:** Tests 2023 and 2024 separately
- **Multiple metrics:** Profit factor, Sharpe, drawdown, trade count
- **Visual verification:** Equity curves must match JSON metrics
- **Statistical analysis:** Correlation between IS and OOS performance

---

## 🔍 **How to Interpret Results**

### **Good OOS Performance (Signal):**
- **High correlation** between IS and OOS metrics (ρ > 0.3)
- **Consistent performance** across both 2023 and 2024
- **Reasonable drawdowns** (not too different from IS)

### **Poor OOS Performance (Noise):**
- **Low correlation** between IS and OOS metrics (ρ ≈ 0)
- **Inconsistent results** between years
- **Extreme degradation** compared to IS performance

### **Example Result Interpretation:**
```
DOGE-AAVE (long):
  IS: PF=2.27, Return=35x, DD=-67%
  OOS 2023: PF=0.95, Return=-41%, DD=-47%  ❌ Strategy failed OOS
  OOS 2024: PF=1.10, Return=81%, DD=-50%   ✅ Some recovery
  → CONCLUSION: Weak persistence, risky for live trading
```

---

## 🎉 **Why This System is Valuable**

1. **Prevents Overfitting:** Tests if IS performance was genuine or just luck
2. **Risk Management:** Identifies strategies likely to fail in live trading  
3. **Strategy Selection:** Helps choose most robust pairs for actual trading
4. **Performance Expectation:** Sets realistic expectations for future returns

**Bottom Line:** The OOS backtest is the final validation step before considering any strategy for live trading. It's the difference between curve-fitted backtests and genuine alpha discovery.