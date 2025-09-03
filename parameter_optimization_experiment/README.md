# Parameter Optimization Experiment

Tests optimal In-Sample window lengths for the intermarket difference trading strategy to determine the best backtesting period before live deployment.

## Objective

Test different In-Sample window lengths (3mo, 6mo, 9mo, etc.) to find the optimal amount of historical data needed for:
- Reliable pair selection (Sharpe > 2.0, Drawdown < 50%)
- Strong Out-of-Sample performance on 2024 data
- Optimal portfolio returns with dynamic budget allocation

## Execution Order

Run these scripts in sequence:

### 1. Data Preparation
```bash
python split_data.py
```

- **Purpose**: Split full dataset into In-Sample (pre-2024) and Out-of-Sample (2024+) periods
- **Output**: `data/in_sample/` and `data/out_of_sample/` directories

### 2. Generate In-Sample Trade Data
```bash
python run_all_pairs_backtest.py
```

- **Purpose**: Generate all directional pair combinations and detailed trade records for In-Sample data
- **Output**: `in_sample/trades/` directory with JSON trade files for each pair
- **Duration**: ~2-3 hours (processes ~85,500 pairs)

### 3. Generate Out-of-Sample Trade Data
```bash
python run_all_pairs_backtest_oos.py
```

- **Purpose**: Generate all directional pair combinations and detailed trade records for Out-of-Sample data (2024+)
- **Output**: `out_of_sample/trades/` directory with JSON trade files for each pair
- **Duration**: ~1 hour (processes ~85,500 pairs)

### 4. Convert Trade Data to CSV (Performance Optimization)
```bash
python convert_trades_to_csv.py
```

- **Purpose**: Convert JSON trade files to CSV format for 10x faster loading
  - Processes In-Sample and Out-of-Sample trade directories
  - Creates `both_trades_fast.csv` files alongside existing JSON files
  - Enables vectorized pandas operations for window filtering
- **Output**: CSV files in trade directories (`*_fast.csv`)
- **Duration**: ~10-15 minutes (one-time conversion)

### 5. Run Window Experiments
```bash
python run_window_experiment_updated.py
```

- **Purpose**: Test 20 different In-Sample window lengths (3mo to 60mo in equal 3-month steps)
  - Uses optimized CSV loading for fast trade filtering
  - Applies selection criteria (Sharpe > 2.0, Drawdown < 50%)
  - Loads pre-generated OOS trades for portfolio simulation
  - Simulates portfolio with dynamic budget allocation (current_portfolio_value / max_trades)
- **Output**: `results/window_experiments/` with results for each window
- **Duration**: ~1-2 hours (100x performance improvement with CSV optimization)

### 6. Analyze Results
```bash
python analyze_window_results.py
```

- **Purpose**: Create comprehensive analysis and visualizations
  - Performance comparison across all window lengths
  - Correlation analysis and ranking systems
  - Master summary with recommendations
- **Output**: `results/window_analysis/` with plots and detailed report

## Key Components

### Essential Scripts (Window Experiment Workflow)
- `split_data.py` - Data preparation (In-Sample/Out-of-Sample split)
- `run_all_pairs_backtest.py` - Generate In-Sample trade data
- `run_all_pairs_backtest_oos.py` - Generate Out-of-Sample trade data
- `convert_trades_to_csv.py` - Convert JSON trade files to CSV for 10x performance boost
- `window_analyzer_fast.py` - Optimized CSV-based trade filtering and metrics calculation
- `window_portfolio_simulator.py` - Portfolio simulation with dynamic budget allocation
- `run_window_experiment_updated.py` - Main orchestrator for all window experiments
- `analyze_window_results.py` - Master summary analysis and visualization

## Expected Results

The experiment will identify the optimal In-Sample window length by comparing:
- **Portfolio Returns**: Actual simulated returns on Out-of-Sample 2024 data
- **Risk-Adjusted Performance**: Sharpe ratios from portfolio simulation
- **Risk Management**: Maximum drawdown during OOS period
- **Trade Execution**: Number of trades executed and concurrent trade utilization
- **Pair Selection Stability**: Consistency of selected pairs across different window lengths

## Output Structure

```
results/window_experiments/
├── 3mo/
│   ├── selected_pairs.csv           # In-Sample filtered pairs
│   ├── valid_oos_pairs.csv          # Pairs with OOS data
│   ├── selection_report.txt         # Filtering details
│   ├── portfolio_summary.json       # Portfolio metrics
│   ├── portfolio_equity_curve.png   # Equity curve visualization
│   ├── budget_allocation.png        # Budget allocation timeline
│   ├── active_trades_timeline.png   # Active trades over time
│   └── drawdown_curve.png           # Drawdown visualization
├── 6mo/
│   └── [same structure]
├── ...
├── master_window_summary.csv        # All window results
└── master_experiment_report.txt     # Detailed analysis
```

## Cleanup Recommendations

**Remove these outdated files:**
- `run_window_experiment.py` (replaced by `run_window_experiment_updated.py`)
- `simulate_portfolio_2024.py` (functionality moved to `window_portfolio_simulator.py`)

**Archive these legacy files if not needed:**
- `apply_custom_filters.py`
- `analyze_oos_results.py` 
- `analyze_in_sample_results.py`

## Filter Criteria

- **Sharpe Ratio**: > 2.0 (strong risk-adjusted returns)
- **Max Drawdown**: < 50% (acceptable risk levels)
### Portfolio Simulation
- **Budget Formula**: `current_portfolio_value / max_trades` where `max_trades = N_pairs / 1.1`
- **Dynamic Rebalancing**: Trade amounts recalculated on every entry based on current portfolio value
- **Max Concurrent**: ~90% of selected pairs can trade simultaneously
- **Initial Capital**: $1,000

### Trading Parameters
- **LOOKBACK**: 24 (moving average window)
- **ATR_LOOKBACK**: 168 (volatility normalization window)
- **THRESHOLD**: 0.25 (signal generation threshold)

---

## Rolling Window Validation Experiment

Advanced temporal robustness testing that validates strategy performance across multiple time periods and market regimes. This experiment splits the 2018-2024 dataset into multiple overlapping In-Sample/Out-of-Sample combinations to test window length stability.

### Objective

Test different In-Sample window lengths (12mo, 15mo, 18mo, 21mo, 24mo) across multiple time periods to find optimal parameters that are:
- **Temporally Robust**: Consistent performance across different market regimes
- **Statistically Stable**: Low coefficient of variation in key metrics
- **Market Regime Agnostic**: Performance not dependent on specific time periods
- **Implementation Ready**: Reliable for live deployment with confidence intervals

### Execution Order

Run these scripts in sequence:

### 1. Time Period Setup
```bash
python rolling_period_manager.py
```

- **Purpose**: Generate rolling IS/OOS period combinations (2018-2024 dataset)
  - Creates 4 overlapping periods: each with 2-year IS + 1-year OOS
  - Period 1: IS(2018-2020) + OOS(2020-2021)  
  - Period 2: IS(2019-2021) + OOS(2021-2022)
  - Period 3: IS(2020-2022) + OOS(2022-2023)
  - Period 4: IS(2021-2023) + OOS(2023-2024)
- **Output**: Period definitions and directory structure under `results/rolling_experiments/`
- **Duration**: < 1 minute

### 2. Run Rolling Window Experiments
```bash
python run_rolling_window_experiment.py
```

- **Purpose**: Execute comprehensive rolling window validation
  - Tests 5 window lengths × 4 time periods = 20 total experiments
  - For each experiment: filters pairs, simulates OOS portfolio performance
  - Uses dynamic budget allocation with 90% utilization target
  - Tracks portfolio returns, Sharpe ratios, drawdown, and pair selection stability
- **Output**: 
  - `results/rolling_experiments/{period}/{window}mo/` - Individual experiment results
  - `results/rolling_experiments/master_rolling_summary.csv` - Aggregated results
- **Duration**: ~2-3 hours (5-10 minutes per window × 20 experiments)

### 3. Analyze Rolling Results
```bash
python analyze_rolling_results.py
```

- **Purpose**: Create comprehensive analysis and visualizations
  - Window stability analysis across different time periods
  - Market regime sensitivity assessment
  - Robustness metrics (coefficient of variation) and ranking
  - Executive summary with implementation recommendations
- **Output**: `results/rolling_analysis/` with plots and detailed report
  - `efficiency_frontier.png` - Risk-return analysis across periods
  - `portfolio_composition.png` - Diversification analysis  
  - `window_performance_comparison.png` - Average performance with error bars
  - `executive_summary.png` - Robustness insights
  - `comprehensive_analysis_report.txt` - Full validation report

## Key Components (Rolling Window Experiment)

### Core Infrastructure
- `rolling_period_manager.py` - Time period generation and data filtering utilities
- `run_rolling_window_experiment.py` - Main orchestrator for all rolling experiments
- `analyze_rolling_results.py` - Comprehensive analysis and visualization suite

### Analysis Modules
- `window_analyzer_fast.py` - Optimized pair filtering and metrics calculation
- `window_portfolio_simulator.py` - Portfolio simulation with dynamic budget allocation
- `visualization_utils.py` - Professional plotting utilities for rolling analysis

## Expected Results (Rolling Window Experiment)

The rolling window experiment will identify optimal parameters by comparing:
- **Temporal Consistency**: Coefficient of variation in Sharpe ratios across periods
- **Market Regime Robustness**: Performance stability across different market conditions  
- **Risk-Adjusted Returns**: Average Sharpe ratio with confidence intervals
- **Drawdown Stability**: Maximum drawdown consistency across time periods
- **Implementation Confidence**: Statistical confidence for live deployment

## Methodology Comparison

### Single Period Optimization (Original)
- **Approach**: Fixed IS(2018-2022) + OOS(2024) split
- **Focus**: Maximum Out-of-Sample performance
- **Risk**: Potential overfitting to specific market regime
- **Use Case**: Quick parameter selection for immediate deployment

### Rolling Window Validation (Advanced)  
- **Approach**: Multiple overlapping IS/OOS combinations
- **Focus**: Temporal robustness and regime-agnostic performance
- **Benefit**: Higher confidence in parameter stability
- **Use Case**: Production deployment with long-term reliability requirements

## Trading Parameters (Both Experiments)
- **LOOKBACK**: 24 (moving average window)
- **ATR_LOOKBACK**: 168 (volatility normalization window)  
- **THRESHOLD**: 0.25 (signal generation threshold)