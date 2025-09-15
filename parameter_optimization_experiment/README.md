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

### 5. Run Optimal Window Execution (Portfolio Backtesting + Plots)
```bash
python run_optimal_window_oos.py
```

## Filter Criteria

- **Sharpe Ratio**: > 2.0 (strong risk-adjusted returns)
- **Max Drawdown**: < 50% (acceptable risk levels)
