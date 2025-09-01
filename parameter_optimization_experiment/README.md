# Parameter Optimization Experiment

This experiment focuses on finding optimal parameters for the intermarket difference trading strategy through systematic testing across different window sizes for In-Sample and Out-of-Sample periods.

## Objective

Test various parameter combinations to optimize:
- LOOKBACK (Moving Average window for CMMA)
- ATR_LOOKBACK (ATR window for volatility normalization)
- THRESHOLD (Signal generation threshold)
- Time window splits between In-Sample and Out-of-Sample periods

## Copied Files

- `core.py` - Core trading algorithms and utilities (with equity curve bug fixes)
- `run_in_sample_experiment.py` - In-sample backtesting engine
- `run_out_of_sample_validation.py` - Out-of-sample validation
- `apply_custom_filters.py` - Multi-criteria filtering for pair selection
- `simulate_portfolio_2024.py` - Portfolio simulation with budget constraints
- `analyze_oos_results.py` - Results analysis tools
- `prepare_data.py` - Data preparation utilities
- `data/` - Complete dataset (in_sample and out_of_sample)

## Ready for Implementation

All core logic, visualization functions, and metrics calculations are ready for parameter optimization implementation.