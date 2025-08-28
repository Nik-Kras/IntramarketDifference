# New In-Sample/Out-of-Sample Experiment

This directory contains a complete experimental framework for testing cryptocurrency pairs trading strategies with proper in-sample/out-of-sample validation.

## 📊 **Experiment Overview**

**In-Sample Period**: 2022-01-01 to 2024-01-01 (2 years)  
**Out-of-Sample Period**: 2024-01-01 to 2025-01-01 (1 year)  
**Algorithm**: CMMA-based intermarket difference mean reversion  
**Filter Criteria**: Sharpe Ratio > 1.0, Max Drawdown > -50%

## 🗂️ **Directory Structure**

```
new_in_sample_out_of_sample_experiment/
├── prepare_data.py                     # Data preparation script
├── run_in_sample_experiment.py         # In-sample backtesting (2022-2024)
├── apply_custom_filters.py             # Apply filtering criteria
├── run_out_of_sample_validation.py     # OOS validation (2024-2025)
├── simulate_portfolio_2024.py          # Portfolio simulation
├── README.md                           # This file
├── data/
│   ├── in_sample/                      # Merged IS data (2022-2024)
│   ├── out_of_sample/                  # Merged OOS data (2024-2025)
│   └── data_preparation_summary.txt    # Data preparation summary
├── results/
│   ├── in_sample/
│   │   ├── in_sample_results.csv       # All pair results
│   │   ├── distributions/              # Distribution plots by coin
│   │   └── trades/                     # Individual trade data
│   ├── out_of_sample/
│   │   └── oos_validation_results.csv  # OOS validation results
│   ├── filtered_pairs/
│   │   ├── new_experiment_filtered_pairs.csv    # Selected pairs
│   │   └── new_experiment_filtered_pairs_summary.txt
│   └── pair_figures/                   # Individual pair visualizations
└── portfolio_simulation/
    ├── portfolio_metrics.json          # Portfolio performance metrics
    ├── portfolio_equity_curve.png      # Equity curve
    ├── budget_allocation.png           # Budget allocation over time
    ├── active_trades_timeline.png      # Active trades timeline
    └── drawdown_curve.png              # Drawdown visualization
```

## 🚀 **How to Run the Complete Experiment**

### Step 1: Prepare Data
```bash
cd new_in_sample_out_of_sample_experiment
python prepare_data.py
```
- Merges data from `../data/` and `../OOS/` directories
- Creates in-sample (2022-2024) and out-of-sample (2024-2025) datasets
- Saves prepared data in `data/in_sample/` and `data/out_of_sample/`

### Step 2: Run In-Sample Experiment
```bash
python run_in_sample_experiment.py
```
- Tests ~55,000 unique pair combinations on 2022-2024 data
- Uses identical algorithm as original experiment
- Creates distribution plots and saves detailed trade data
- Output: `results/in_sample/in_sample_results.csv`

### Step 3: Apply Custom Filters
```bash
python apply_custom_filters.py
```
- Filters pairs with Sharpe Ratio > 1.0 AND Max Drawdown > -50%
- Saves selected pairs for out-of-sample validation
- Output: `results/filtered_pairs/new_experiment_filtered_pairs.csv`

### Step 4: Out-of-Sample Validation
```bash
python run_out_of_sample_validation.py
```
- Tests selected pairs on 2024 out-of-sample data
- Creates individual equity curves and drawdown plots for each pair
- Saves performance metrics and visualizations
- Output: `results/out_of_sample/oos_validation_results.csv`

### Step 5: Portfolio Simulation
```bash
# Default: $10,000 budget, 1% allocation per trade
python simulate_portfolio_2024.py

# Custom budget and allocation
python simulate_portfolio_2024.py --initial_budget 50000 --allocation_fraction 0.005
```
- Simulates trading all validated pairs as a portfolio
- Proper budget allocation with trade limits
- Visualizes equity curve, budget allocation, active trades, and drawdown
- Output: `portfolio_simulation/` directory with metrics and charts

## ⚙️ **Algorithm Parameters**

All scripts use identical parameters:
- **LOOKBACK**: 24 (Moving average window)
- **ATR_LOOKBACK**: 168 (ATR window for volatility normalization)
- **THRESHOLD**: 0.25 (Signal generation threshold)
- **MIN_OVERLAP**: 500 for IS, 100 for OOS (Minimum data overlap requirement)

## 📈 **Filter Criteria**

Custom filters applied to in-sample results:
1. **Sharpe Ratio > 1.0**: Risk-adjusted performance threshold
2. **Max Drawdown > -50%**: Maximum acceptable drawdown

## 💰 **Portfolio Simulation Features**

- **Budget Management**: Tracks available vs allocated capital
- **Trade Limits**: Maximum concurrent trades = 1 / allocation_fraction
- **Dynamic Sizing**: Trade size = current_portfolio_value / max_trades
- **Proper Accounting**: No over-allocation, realistic trade rejection
- **Rich Visualizations**: Equity, budget allocation, active trades, drawdown

## 📊 **Key Output Files**

1. **In-Sample Results**: `results/in_sample/in_sample_results.csv`
2. **Filtered Pairs**: `results/filtered_pairs/new_experiment_filtered_pairs.csv`
3. **OOS Validation**: `results/out_of_sample/oos_validation_results.csv`
4. **Portfolio Metrics**: `portfolio_simulation/portfolio_metrics.json`

## 🎯 **Expected Results**

The experiment will provide:
- **In-Sample Performance**: Historical performance on 2022-2024 data
- **Filter Effectiveness**: How many pairs pass the quality filters
- **Out-of-Sample Validation**: Performance persistence on unseen 2024 data
- **Portfolio Simulation**: Realistic trading performance with proper risk management

## 📝 **Notes**

- All scripts use the exact same trading algorithm for consistency
- Data preparation handles missing data and ensures proper datetime indexing
- Portfolio simulation includes realistic budget constraints and trade limits
- Visualizations provide comprehensive performance analysis

## 🔧 **Troubleshooting**

- Ensure `../data/` and `../OOS/` directories contain the required CSV files
- Run scripts in order - each depends on the previous step's output
- Check data preparation summary for any failed coin processing
- Verify sufficient disk space for results and visualizations