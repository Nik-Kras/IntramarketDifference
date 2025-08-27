# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an **Intermarket Difference Trading Analysis** project that implements and analyzes a cryptocurrency pairs trading strategy. The core concept is to use the difference between two coins' momentum indicators (CMMA - Close Minus Moving Average normalized by ATR) to generate trading signals on one coin based on another coin's behavior.

## Core Algorithm

The trading strategy works as follows:
1. Calculate CMMA indicator for both reference coin and trading coin
2. Compute the intermarket difference: `trading_coin_cmma - reference_coin_cmma`
3. Generate mean-reversion signals when this difference crosses thresholds (+0.25/-0.25)
4. Trade the trading coin based on these signals
5. Exit positions when the difference returns to zero

## Key Components

### Core Algorithm Files
- `run_all.py` - Main backtesting engine that tests all coin pair combinations
- `intermarket.py` - Original prototype focusing on ETH-BTC pair analysis
- `permutation_test.py` - Statistical validation using Monte Carlo permutation testing
- `trades_from_signal.py` - Trade execution logic and P&L calculation
- `create_distributions.py` - Visualization and summary statistics generation

### Strategy Validation & Selection Files
- `custom_filter.py` - Applies multi-criteria filters to select best performing pairs
- `filter_pairs.py` - Core filtering engine with flexible parameter system
- `selecting_pairs.py` - Pair selection utilities and analysis
- `oos_backtest.py` - Out-of-Sample validation on unseen 2023-2024 data

### Analysis & Inspection Tools
- `inspect_pair.py` - Detailed analysis of specific trading pairs
- `rank_correlation_analysis.py` - Correlation analysis between in-sample and OOS performance
- `eth_btc_comparison.py` - Comparative analysis tools for specific pairs

### Data Structure
- `data/` directory contains hourly OHLC data for ~140 cryptocurrencies in CSV format
  - Files follow naming convention: `{COIN}USDT_IS.csv` (e.g., `BTCUSDT_IS.csv`)
  - Each CSV has columns: date/open time, high, low, close (volume not used)
- `OOS/` directory contains Out-of-Sample data (2023-2025) for validation
  - Files follow naming convention: `{COIN}USDT_OOS.csv`
  - Used exclusively for strategy validation, never for optimization

### Results Structure
- `pair_backtest_results.csv` - Main results file from `run_all.py`
- `all_pairs_metrics.csv` - Enhanced results with additional metrics and quantiles
- `permutation_test_results.csv` - Statistical significance results
- `trading_coin_summary_stats.csv` - Aggregated statistics per trading coin
- `trades/` directory - Individual trade data in JSON format
  - `trades/{COIN}/` - Trade data for each trading coin
  - `{REF_COIN}_{TRADING_COIN}_trades.json` - Detailed trade records per pair
- `permutations/` directory - Individual coin analysis with visualizations
  - Each coin gets its own subdirectory with distribution plots and equity curves
- `filtered_results/` directory - Filtered pair selections for OOS testing
  - `user_custom_filter.csv` - Top performing pairs selected by multi-criteria filters
  - `user_custom_filter_summary.txt` - Filter application summary
- `oos_experiments/` directory - Out-of-Sample validation results
  - `oos_backtest_results.csv` - OOS performance metrics for selected pairs
  - Individual pair folders with 2023/2024 results and visualizations
- `inspection/` directory - Detailed pair analysis results
  - Per-pair subdirectories with comprehensive analysis reports

## Common Commands

### Run Complete Analysis Pipeline
```bash
# Step 1: Generate all pair combinations backtest results
python run_all.py

# Step 2: Create distribution visualizations for each coin
python create_distributions.py  

# Step 3: Run Monte Carlo permutation tests for statistical validation
python permutation_test.py

# Step 4: Apply filtering criteria to select best pairs
python custom_filter.py

# Step 5: Validate selected pairs on out-of-sample data
python oos_backtest.py
```

### Advanced Analysis Workflows
```bash
# Inspect specific trading pair in detail
python inspect_pair.py

# Analyze correlation between in-sample and OOS performance  
python rank_correlation_analysis.py

# Run comparative analysis on selected pairs
python selecting_pairs.py
```

### Quick ETH-BTC Analysis
```bash
# Run original prototype analysis
python intermarket.py
```

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
pip install pandas numpy matplotlib pandas_ta statsmodels scipy tqdm
```

### Data Requirements
- Ensure all CSV files in `data/` directory have proper datetime columns (`date` or `open time`)
- Required OHLC columns: `high`, `low`, `close`
- Files should be cleaned of missing data before processing

## Algorithm Parameters

### Default Configuration (from run_all.py)
- `LOOKBACK = 24` - Moving average window for CMMA calculation
- `ATR_LOOKBACK = 168` - ATR window for volatility normalization  
- `THRESHOLD = 0.25` - Signal generation threshold
- `MIN_OVERLAP = 500` - Minimum data points required for pair analysis

### Permutation Test Settings
- `N_PERMUTATIONS = 500` - Number of random sequences per coin
- `COMMISSION_RATE = 0.002` - 0.2% commission per trade side

## Key Metrics Calculated

### Trading Performance
- **Profit Factor**: Gross gains / Gross losses
- **Total Cumulative Return**: Log return cumulative sum  
- **Number of Trades**: Position changes in signal
- **Max Drawdown**: Peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return metric
- **Last Year Drawdown**: Maximum drawdown in final year of data
- **Volatility**: Standard deviation of returns

### Exit Timing Analysis  
- **Mean Exit Hour**: Average holding period
- **STD Exit Hour**: Standard deviation of holding periods

### Statistical Validation
- **Profit Factor Quantile**: Percentile rank vs random trading
- **Drawdown Quantile**: Percentile rank vs random trading

### Out-of-Sample Validation Metrics
- **OOS Profit Factor (2023/2024)**: Performance on unseen data by year
- **OOS Sharpe Ratio (2023/2024)**: Risk-adjusted returns on OOS data
- **OOS Max Drawdown (2023/2024)**: Worst-case performance on OOS data
- **IS-OOS Correlation**: Correlation between in-sample and out-of-sample metrics

## Output Files

### Main Results
- `pair_backtest_results.csv` - All pair combinations with performance metrics
- `permutation_test_results.csv` - Statistical significance of each pair
- `trading_coin_summary_stats.csv` - Aggregated statistics per trading coin

### Trade Data (per coin in trades/{COIN}/)
- `{REF_COIN}_{TRADING_COIN}_trades.json` - Individual trade records with:
  - `time_entered`, `time_exited` - Entry/exit timestamps
  - `log_return` - Individual trade log return
  - `trade_type` - "long" or "short"

### Visualizations (per coin in permutations/{COIN}/)
**Separate Analysis by Trade Type:**
- `profit_factor_distribution_combined.png` - Combined longs+shorts vs random
- `profit_factor_distribution_longs.png` - Longs-only vs random
- `profit_factor_distribution_shorts.png` - Shorts-only vs random
- `drawdown_distribution_combined.png` - Combined drawdown vs random
- `drawdown_distribution_longs.png` - Longs-only drawdown vs random
- `drawdown_distribution_shorts.png` - Shorts-only drawdown vs random
- `equity_curves_combined.png` - Top 10 combined performers
- `equity_curves_longs.png` - Top 10 long performers
- `equity_curves_shorts.png` - Top 10 short performers

**Results Data:**
- `results_combined.json` - Combined analysis quantile data
- `results_longs.json` - Longs-only analysis quantile data
- `results_shorts.json` - Shorts-only analysis quantile data

**Legacy Files:**
- `profit_factor_dist.png` - All reference coins distribution  
- `drawdown_dist.png` - All reference coins drawdown distribution

## Architecture Notes

### Data Pipeline Flow
1. **Data Loading** (`load_one_csv`) - Standardizes CSV format and datetime indexing
2. **Feature Engineering** (`cmma`) - Calculates momentum indicator normalized by volatility
3. **Signal Generation** (`threshold_revert_signal`) - Mean reversion logic with state tracking
4. **Backtesting** (`run_pair`) - Simulates trading with next-bar execution
5. **Statistical Analysis** (`run_permutation_test`) - Validates against random baseline

### Key Design Patterns
- **Vectorized Operations**: Uses pandas/numpy for efficient time series processing
- **State Machine**: Signal generation tracks position state (long/flat/short)
- **Alignment**: Automatically handles different date ranges across coin pairs
- **Robust Error Handling**: Gracefully handles missing data and edge cases

### Statistical Methodology
The permutation testing framework:
1. Extracts actual trade count and timing patterns from algorithm
2. Generates random trades with same count and timing characteristics  
3. Compares algorithm performance vs random distribution
4. Reports percentile rankings (lower = better performance)
5. Selects top performers for visualization: highest profit factors and lowest drawdowns

## Out-of-Sample Validation System

### Purpose
The OOS backtesting system validates that in-sample (2018-2022) performance generalizes to unseen market conditions. This prevents overfitting and provides realistic expectations for live trading.

### Workflow
1. **Filter Selection**: `custom_filter.py` applies multi-criteria filters to select top ~1000 pairs
2. **OOS Testing**: `oos_backtest.py` tests selected pairs on 2023-2024 data separately
3. **Correlation Analysis**: `rank_correlation_analysis.py` measures IS-OOS performance persistence
4. **Strategy Selection**: Choose pairs with high IS-OOS correlation and consistent performance

### Key Validation Principles
- **Identical Algorithm**: Same CMMA calculation, signal generation, and metrics
- **Strategy Type Consistency**: Respects long-only, short-only, or combined strategies from filtering
- **Temporal Separation**: OOS data (2023-2024) completely separate from IS data (2018-2022)
- **No Optimization**: Parameters never adjusted based on OOS results

### Default Filter Criteria (custom_filter.py)
- Sharpe Ratio > 1.0
- Max Drawdown > -75%
- Last Year Drawdown > -40%
- Profit Factor Quantile <= 5% (top 5%)
- Drawdown Quantile <= 5% (top 5%)

## Development Notes

### Adding New Coins
1. Place CSV file in `data/` directory following naming convention
2. Ensure proper datetime and OHLC columns
3. Re-run `run_all.py` to include in analysis

### Modifying Strategy Parameters
- Edit constants at top of `run_all.py` 
- Consider parameter ranges tested in commented section of `intermarket.py`
- Validate changes using permutation testing

### Performance Optimization
- Current implementation processes ~19,000 pair combinations
- Uses tqdm for progress tracking
- Consider parallel processing for large-scale analysis

### Dependencies and Environment
- **Python 3.9+** required (based on `__pycache__` files)
- **Required packages**: pandas, numpy, matplotlib, pandas_ta, statsmodels, scipy, tqdm
- **Virtual environment**: `venv/` directory present - activate with `source venv/bin/activate`

### File Structure Understanding
```
data/{COIN}USDT_IS.csv     # Raw OHLC data (140+ coins)
run_all.py                 # Main backtester → pair_backtest_results.csv
create_distributions.py    # Visualizations → permutations/{COIN}/
permutation_test.py        # Statistical validation → permutation_test_results.csv
trades/{COIN}/             # Individual trade records (JSON)
```

### Signal Logic Details
The `threshold_revert_signal` function implements a state machine:
- **Long Entry**: When intermarket difference > +0.25
- **Short Entry**: When intermarket difference < -0.25  
- **Long Exit**: When in long position and difference <= 0
- **Short Exit**: When in short position and difference >= 0
- **Position Persistence**: Signal maintains last state during NaN values

### Trading Execution Model
- **Entry**: Signal changes trigger position entry at next bar's close
- **Returns**: Log returns calculated as `signal[t] * log_return[t+1]`
- **Commission**: 0.2% per trade side applied in permutation testing only
- **Position Sizing**: Fixed 1x leverage (no sizing optimization)

### Known Limitations and Considerations
- **Look-ahead bias**: Signals use same-bar prices but trade at next bar
- **Survivorship bias**: Only includes coins with complete data histories
- **Commission impact**: Main backtests exclude commission costs
- **Data quality**: Assumes clean OHLC data without gaps or errors
- **Market regime**: Results may not generalize across different market conditions

### Recent Bug Fixes
- Fixed drawdown sorting in `permutation_test.py:302` - now correctly selects worst performing (highest drawdown) reference coins for visualization instead of best performing ones