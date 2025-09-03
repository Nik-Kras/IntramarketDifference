#!/usr/bin/env python3
"""
Rolling Window Experiment Orchestrator

Runs comprehensive rolling window validation experiment:
- Tests multiple IS/OOS time period splits
- Tests window lengths: 12, 15, 18, 21, 24 months for each period
- Validates window robustness across different market regimes
- Provides temporal stability analysis for parameter selection
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# Import our custom modules
from rolling_period_manager import RollingPeriodManager
from window_analyzer_fast import analyze_window_fast as analyze_window
from window_portfolio_simulator import WindowPortfolioSimulator

# Configuration Constants
BASE_TRADES_DIR = "in_sample/trades"
RESULTS_DIR = "results/rolling_experiments"
INITIAL_PORTFOLIO_VALUE = 1000.0
WINDOW_LENGTHS = [12, 15, 18, 21, 24]
IS_LENGTH_MONTHS = 24  # 2 years
OOS_LENGTH_MONTHS = 12  # 1 year
START_YEAR = 2018
END_YEAR = 2024

# Analysis Constants
CV_HIGH_THRESHOLD = 0.2  # High stability threshold for coefficient of variation
CV_MEDIUM_THRESHOLD = 0.5  # Medium stability threshold
SUCCESS_RATE_PRECISION = 1  # Decimal places for success rate display

class RollingWindowExperiment:
    """Orchestrates rolling window validation experiments.
    
    This class manages the complete rolling window validation workflow:
    1. Sets up time period splits using RollingPeriodManager
    2. Runs In-Sample analysis for each window/period combination
    3. Simulates Out-of-Sample portfolio performance
    4. Aggregates results and generates comprehensive reports
    
    The experiment tests window robustness across different market regimes
    to ensure parameter selections are not dependent on specific time periods.
    """
    
    def __init__(self):
        self.period_manager = RollingPeriodManager(
            start_year=START_YEAR,
            end_year=END_YEAR,
            is_length_months=IS_LENGTH_MONTHS,
            oos_length_months=OOS_LENGTH_MONTHS
        )
        self.results_dir = RESULTS_DIR
        
        # Create directory structure
        self.period_manager.create_period_directory_structure(RESULTS_DIR)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _create_error_result(self, period: Dict[str, str], window_months: int, error_msg: str) -> Dict[str, Any]:
        """Create standardized error result dictionary.
        
        Args:
            period: Period dictionary
            window_months: Window length in months
            error_msg: Error message description
            
        Returns:
            Standardized error result dictionary
        """
        return {
            'period_name': period['period_name'],
            'window_months': window_months,
            'pairs_selected': 0,
            'portfolio_return': 0.0,
            'portfolio_sharpe': 0.0,
            'portfolio_drawdown': 0.0,
            'trades_executed': 0,
            'error': error_msg
        }
    
    def filter_trades_by_date_range(self, trades_data: List[Dict[str, Any]], 
                                   start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Filter trades by date range.
        
        Args:
            trades_data: List of trade dictionaries
            start_date: Start date string (YYYY-MM-DD format)
            end_date: End date string (YYYY-MM-DD format)
            
        Returns:
            Filtered list of trade dictionaries
        """
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        filtered_trades = []
        for trade in trades_data:
            trade_entry = pd.to_datetime(trade['time_entered'])
            
            # Include trade if it starts within the period
            if start_dt <= trade_entry < end_dt:
                filtered_trades.append(trade)
        
        return filtered_trades
    
    def load_and_filter_trades_for_period(self, reference_coin: str, trading_coin: str,
                                         strategy_type: str, period: Dict[str, str], 
                                         data_type: str) -> List[Dict[str, Any]]:
        """Load and filter trades for specific period and data type.
        
        Args:
            reference_coin: Reference coin symbol
            trading_coin: Trading coin symbol
            strategy_type: Strategy type ('combined', 'longs', 'shorts')
            period: Period dictionary from RollingPeriodManager
            data_type: 'is' for In-Sample, 'oos' for Out-of-Sample
            
        Returns:
            List of filtered trade dictionaries
            
        Raises:
            ValueError: If data_type is invalid or required period keys are missing
        """
        # Input validation
        if data_type not in ['is', 'oos']:
            raise ValueError(f"data_type must be 'is' or 'oos', got '{data_type}'")
        
        required_keys = ['is_start', 'is_end', 'oos_start', 'oos_end']
        missing_keys = [key for key in required_keys if key not in period]
        if missing_keys:
            raise ValueError(f"Period dictionary missing required keys: {missing_keys}")
        
        if not reference_coin or not trading_coin or not strategy_type:
            raise ValueError("Coin symbols and strategy_type cannot be empty")
        
        # Construct file paths (prefer CSV, fallback to JSON)
        csv_file = os.path.join(BASE_TRADES_DIR, f"{reference_coin}_{trading_coin}", 
                               f"{strategy_type}_trades_fast.csv")
        json_file = os.path.join(BASE_TRADES_DIR, f"{reference_coin}_{trading_coin}", 
                                f"{strategy_type}_trades.json")
        
        # Try CSV first
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                df = df.dropna(subset=['time_entered', 'time_exited', 'log_return'])
                
                # Convert to dict format for consistency
                trades_data = df.to_dict('records')
            except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
                print(f"⚠️  Failed to load CSV {csv_file}: {e}")
                return []
        elif os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    trades_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️  Failed to load JSON {json_file}: {e}")
                return []
        else:
            return []
        
        # Filter by period
        if data_type == 'is':
            start_date = period['is_start']
            end_date = period['is_end']
        elif data_type == 'oos':
            start_date = period['oos_start']
            end_date = period['oos_end']
        else:
            raise ValueError("data_type must be 'is' or 'oos'")
        
        filtered_trades = self.filter_trades_by_date_range(trades_data, start_date, end_date)
        
        # Add missing fields that portfolio simulator expects
        for i, trade in enumerate(filtered_trades):
            trade['trade_id'] = f"{reference_coin}_{trading_coin}_{i}"
            trade['trading_coin'] = trading_coin
            trade['reference_coin'] = reference_coin
            trade['strategy_type'] = strategy_type
        
        return filtered_trades
    
    # =========================================================================
    # EXPERIMENT EXECUTION METHODS
    # =========================================================================
    
    def run_single_period_window_experiment(self, period: Dict[str, str], window_months: int) -> Dict[str, Any]:
        """Run experiment for single period and window combination.
        
        Args:
            period: Period dictionary from RollingPeriodManager
            window_months: Window length in months
            
        Returns:
            Dictionary with experiment results or error information
        """
        
        print(f"🔍 Window: {window_months}mo, Period: {period['period_name']}")
        
        # Create results directory
        period_dir = os.path.join(RESULTS_DIR, period['period_name'])
        window_dir = os.path.join(period_dir, f"{window_months}mo")
        os.makedirs(window_dir, exist_ok=True)
        
        # Get window data range within IS period
        window_start, window_end = self.period_manager.get_window_data_range(period, window_months)
        
        # Step 1: Run In-Sample analysis with window restriction
        print(f"   📊 IS Analysis: {window_start} to {window_end}")
        
        try:
            # Use analyze_window with window date range (it already supports date filtering)
            _, selected_pairs, _ = analyze_window(window_start, window_end, BASE_TRADES_DIR)
            
            if len(selected_pairs) == 0:
                return self._create_error_result(period, window_months, 'No pairs selected')
            
            # Save IS results
            selected_pairs.to_csv(os.path.join(window_dir, 'selected_pairs.csv'), 
                                 index=False, float_format='%.6f')
            
            # Save selected pairs summary for debugging
            pair_summary = selected_pairs[['reference_coin', 'trading_coin', 'trading_type', 
                                          'sharpe_ratio', 'total_cumulative_return', 'num_trades']].copy()
            pair_summary.to_csv(os.path.join(window_dir, 'selected_pairs_summary.csv'), 
                               index=False, float_format='%.6f')
            
            # Step 2: Simulate portfolio on OOS period
            print(f"   💰 OOS Simulation: {period['oos_start']} to {period['oos_end']}")
            
            # Load OOS trades for selected pairs and track detailed metrics
            valid_oos_pairs = []
            pair_oos_metrics = []  # Track detailed metrics for debugging
            
            for _, pair_row in selected_pairs.iterrows():
                oos_trades = self.load_and_filter_trades_for_period(
                    pair_row['reference_coin'], 
                    pair_row['trading_coin'],
                    pair_row['trading_type'], 
                    period, 
                    'oos'
                )
                
                if oos_trades:
                    pair_with_trades = pair_row.to_dict()
                    pair_with_trades['oos_trades'] = oos_trades
                    valid_oos_pairs.append(pair_with_trades)
                    
                    # Calculate OOS metrics for this pair
                    oos_returns = [trade.get('log_return', 0) for trade in oos_trades]
                    pair_metrics = {
                        'reference_coin': pair_row['reference_coin'],
                        'trading_coin': pair_row['trading_coin'],
                        'trading_type': pair_row['trading_type'],
                        'is_sharpe': pair_row.get('sharpe_ratio', 0),
                        'is_return': pair_row.get('total_cumulative_return', 0),
                        'is_trades': pair_row.get('num_trades', 0),
                        'oos_trades': len(oos_trades),
                        'oos_total_return': sum(oos_returns),
                        'oos_mean_return': sum(oos_returns) / len(oos_returns) if oos_returns else 0
                    }
                    pair_oos_metrics.append(pair_metrics)
            
            # Save pair OOS metrics for debugging
            if pair_oos_metrics:
                pair_oos_df = pd.DataFrame(pair_oos_metrics)
                pair_oos_df.to_csv(os.path.join(window_dir, 'pair_oos_metrics.csv'), 
                                  index=False, float_format='%.6f')
            
            if not valid_oos_pairs:
                error_result = self._create_error_result(period, window_months, 'No OOS trades available')
                error_result['pairs_selected'] = len(selected_pairs)  # Override to show actual pairs selected
                return error_result
            
            # Initialize portfolio simulator
            simulator = WindowPortfolioSimulator(
                initial_capital=INITIAL_PORTFOLIO_VALUE,
                n_pairs=len(valid_oos_pairs)
            )
            
            # Run portfolio simulation
            portfolio_metrics = simulator.simulate_portfolio_with_trades(
                valid_oos_pairs, period['oos_start'], period['oos_end']
            )
            
            if not portfolio_metrics:
                error_result = self._create_error_result(period, window_months, 'Portfolio simulation failed')
                error_result['pairs_selected'] = len(selected_pairs)  # Override to show actual pairs selected
                return error_result
            
            # Extract results
            result = {
                'period_name': period['period_name'],
                'window_months': window_months,
                'window_start': window_start,
                'window_end': window_end,
                'oos_start': period['oos_start'],
                'oos_end': period['oos_end'],
                'pairs_selected': len(selected_pairs),
                'pairs_with_oos': len(valid_oos_pairs),
                'portfolio_return': 1 + portfolio_metrics['total_return'],
                'portfolio_sharpe': portfolio_metrics['sharpe_ratio'],
                'portfolio_drawdown': portfolio_metrics['max_drawdown'],
                'trades_executed': portfolio_metrics.get('num_trades', 0),
                'simulation_metrics': portfolio_metrics
            }
            
            # Save detailed results
            with open(os.path.join(window_dir, 'experiment_result.json'), 'w') as f:
                json.dump(result, f, indent=2)
            
            # Create visualizations
            simulator.create_visualizations(window_dir)
            
            print(f"   ✅ Return: {result['portfolio_return']:.2f}x, "
                  f"Sharpe: {result['portfolio_sharpe']:.2f}, "
                  f"Drawdown: {result['portfolio_drawdown']:.2%}")
            
            return result
            
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, ValueError) as e:
            print(f"   ❌ Experiment error: {str(e)}")
            return self._create_error_result(period, window_months, str(e))
        except Exception as e:
            print(f"   ❌ Unexpected error: {str(e)}")
            return self._create_error_result(period, window_months, f"Unexpected: {str(e)}")
    
    def run_complete_rolling_experiment(self) -> List[Dict[str, Any]]:
        """Run complete rolling window validation experiment.
        
        Returns:
            List of experiment result dictionaries
        """
        
        print("🚀 Rolling Window Validation Experiment")
        print("=" * 60)
        
        # Print experiment overview
        self.period_manager.print_experiment_overview()
        
        periods = self.period_manager.get_periods()
        total_experiments = len(periods) * len(WINDOW_LENGTHS)
        
        print(f"\n🔄 Starting {total_experiments} total experiments...")
        print(f"📊 Structure: {len(periods)} periods × {len(WINDOW_LENGTHS)} windows")
        
        all_results = []
        experiment_count = 0
        
        # Nested loops: periods × windows
        for period in periods:
            print(f"\n{'='*80}")
            print(f"📅 PERIOD: {period['period_name']}")
            print(f"{'='*80}")
            print(f"IS Period: {period['is_start']} to {period['is_end']}")
            print(f"OOS Period: {period['oos_start']} to {period['oos_end']}")
            
            period_results = []
            
            for window_months in WINDOW_LENGTHS:
                experiment_count += 1
                print(f"\n🔬 Experiment {experiment_count}/{total_experiments}: "
                      f"{window_months}mo window")
                print(f"📊 Overall Progress: {(experiment_count-1)/total_experiments*100:.1f}%")
                
                try:
                    result = self.run_single_period_window_experiment(period, window_months)
                    period_results.append(result)
                    all_results.append(result)
                    
                except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, ValueError) as e:
                    print(f"❌ Experiment failed: {str(e)}")
                    error_result = self._create_error_result(period, window_months, f"Experiment failed: {str(e)}")
                    period_results.append(error_result)
                    all_results.append(error_result)
                except Exception as e:
                    print(f"❌ Unexpected error: {str(e)}")
                    error_result = self._create_error_result(period, window_months, f"Unexpected error: {str(e)}")
                    period_results.append(error_result)
                    all_results.append(error_result)
            
            # Save period summary
            period_summary_df = pd.DataFrame(period_results)
            period_csv = os.path.join(RESULTS_DIR, period['period_name'], 'period_summary.csv')
            period_summary_df.to_csv(period_csv, index=False, float_format='%.6f')
            
            print(f"\n✅ PERIOD COMPLETED: {period['period_name']}")
            valid_results = [r for r in period_results if 'error' not in r]
            if valid_results:
                best_window = max(valid_results, key=lambda x: x['portfolio_sharpe'])
                print(f"   🏆 Best window: {best_window['window_months']}mo "
                      f"(Sharpe: {best_window['portfolio_sharpe']:.2f})")
        
        return all_results
    
    # =========================================================================
    # ANALYSIS AND REPORTING METHODS
    # =========================================================================
    
    def create_master_rolling_summary(self, all_results: List[Dict[str, Any]]) -> None:
        """Create master summary of rolling experiment results.
        
        Args:
            all_results: List of all experiment result dictionaries
        """
        
        print(f"\n📋 Creating Master Rolling Summary...")
        
        # Filter valid results
        valid_results = [r for r in all_results if 'error' not in r]
        
        if not valid_results:
            print("❌ No valid results to summarize!")
            return
        
        # Create summary DataFrame
        results_df = pd.DataFrame(valid_results)
        
        # Save master results
        master_file = os.path.join(RESULTS_DIR, 'master_rolling_summary.csv')
        results_df.to_csv(master_file, index=False, float_format='%.6f')
        
        # Calculate window stability metrics
        window_performance = {}
        
        for window in WINDOW_LENGTHS:
            window_results = results_df[results_df['window_months'] == window]
            
            if len(window_results) > 0:
                window_performance[window] = {
                    'avg_return': window_results['portfolio_return'].mean(),
                    'avg_sharpe': window_results['portfolio_sharpe'].mean(), 
                    'avg_drawdown': window_results['portfolio_drawdown'].mean(),
                    'return_std': window_results['portfolio_return'].std(),
                    'sharpe_std': window_results['portfolio_sharpe'].std(),
                    'periods_tested': len(window_results),
                    'success_rate': len(window_results) / len(self.period_manager.get_periods()) * 100
                }
        
        # Create comprehensive report
        report_lines = [
            "ROLLING WINDOW VALIDATION EXPERIMENT RESULTS",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total experiments: {len(all_results)}",
            f"Successful experiments: {len(valid_results)}",
            f"Success rate: {len(valid_results)/len(all_results)*100:.1f}%",
            "",
            "EXPERIMENTAL DESIGN",
            "-" * 25,
            f"Time periods tested: {len(self.period_manager.get_periods())}",
            f"Window lengths tested: {WINDOW_LENGTHS}",
            f"IS period length: {self.period_manager.is_length_months} months",
            f"OOS period length: {self.period_manager.oos_length_months} months",
            "",
            "WINDOW STABILITY ANALYSIS",
            "-" * 30,
            f"{'Window':<8} {'Periods':<8} {'Avg Return':<12} {'Avg Sharpe':<11} {'Return Std':<12} {'Sharpe Std':<11} {'Success%':<9}",
            f"{'-'*8} {'-'*8} {'-'*12} {'-'*11} {'-'*12} {'-'*11} {'-'*9}"
        ]
        
        # Sort windows by average Sharpe ratio
        sorted_windows = sorted(window_performance.items(), 
                               key=lambda x: x[1]['avg_sharpe'], reverse=True)
        
        for window, metrics in sorted_windows:
            report_lines.append(
                f"{window}mo{'':<4} {metrics['periods_tested']:<8} "
                f"{metrics['avg_return']:<12.3f} {metrics['avg_sharpe']:<11.3f} "
                f"{metrics['return_std']:<12.3f} {metrics['sharpe_std']:<11.3f} "
                f"{metrics['success_rate']:<9.1f}"
            )
        
        # Find most robust window (lowest coefficient of variation for Sharpe)
        if sorted_windows:
            cv_analysis = []
            for window, metrics in sorted_windows:
                if metrics['avg_sharpe'] > 0 and metrics['sharpe_std'] > 0:
                    cv = metrics['sharpe_std'] / metrics['avg_sharpe']
                    cv_analysis.append((window, cv, metrics))
            
            cv_analysis.sort(key=lambda x: x[1])  # Sort by CV (lower = more stable)
            
            report_lines.extend([
                "",
                "ROBUSTNESS RANKING (by Sharpe CV)",
                "-" * 35,
                f"{'Rank':<5} {'Window':<8} {'CV':<8} {'Avg Sharpe':<11} {'Stability':<10}",
                f"{'-'*5} {'-'*8} {'-'*8} {'-'*11} {'-'*10}"
            ])
            
            for rank, (window, cv, metrics) in enumerate(cv_analysis, 1):
                stability = "High" if cv < CV_HIGH_THRESHOLD else "Medium" if cv < CV_MEDIUM_THRESHOLD else "Low"
                report_lines.append(
                    f"{rank:<5} {window}mo{'':<4} {cv:<8.3f} {metrics['avg_sharpe']:<11.3f} {stability:<10}"
                )
            
            # Recommendation
            if cv_analysis:
                most_robust = cv_analysis[0]
                best_performance = sorted_windows[0]
                
                report_lines.extend([
                    "",
                    "RECOMMENDATIONS",
                    "-" * 20,
                    f"Most Robust Window: {most_robust[0]}mo (CV: {most_robust[1]:.3f})",
                    f"Best Performance Window: {best_performance[0]}mo (Sharpe: {best_performance[1]['avg_sharpe']:.3f})",
                    "",
                    "DECISION CRITERIA:",
                    f"• Choose {most_robust[0]}mo for stability across market regimes",
                    f"• Choose {best_performance[0]}mo for maximum expected performance",
                    "",
                    "VALIDATION NOTES:",
                    "• Robustness measured by coefficient of variation in Sharpe ratio",
                    "• Lower CV indicates more consistent performance across periods",
                    "• Consider both stability and performance in final selection"
                ])
        
        # Save master report
        report_file = os.path.join(RESULTS_DIR, 'master_rolling_report.txt')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n📊 MASTER ROLLING SUMMARY:")
        print(f"   Valid experiments: {len(valid_results)}/{len(all_results)}")
        print(f"   Most robust window: {cv_analysis[0][0]}mo (CV: {cv_analysis[0][1]:.3f})")
        print(f"   Best performance: {sorted_windows[0][0]}mo (Sharpe: {sorted_windows[0][1]['avg_sharpe']:.3f})")
        print(f"   Results: {master_file}")
        print(f"   Report: {report_file}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution function for rolling window validation experiment.
    
    Orchestrates the complete rolling window validation experiment:
    1. Validates prerequisites (trade data availability)
    2. Runs experiments across all period/window combinations
    3. Generates comprehensive analysis reports
    4. Provides next steps for analysis
    """
    
    experiment_start = datetime.now()
    print("🎯 Rolling Window Validation Experiment")
    print("=" * 50)
    print("Testing window robustness across multiple time periods")
    print(f"🕐 Started: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not os.path.exists(BASE_TRADES_DIR):
        print(f"❌ Prerequisite missing: {BASE_TRADES_DIR}")
        print("Run run_all_pairs_backtest.py first to generate trade data.")
        return
    
    # Initialize experiment
    experiment = RollingWindowExperiment()
    
    # Run complete experiment
    all_results = experiment.run_complete_rolling_experiment()
    
    if not all_results:
        print("❌ No experiments completed!")
        return
    
    # Create master summary
    experiment.create_master_rolling_summary(all_results)
    
    experiment_end = datetime.now()
    total_duration = experiment_end - experiment_start
    
    print(f"\n🎯 ROLLING EXPERIMENT COMPLETED!")
    print(f"🕐 Started: {experiment_start.strftime('%H:%M:%S')}")
    print(f"🕐 Finished: {experiment_end.strftime('%H:%M:%S')}")
    print(f"⏱️  Total Duration: {total_duration}")
    print(f"📁 Results: {RESULTS_DIR}/")
    
    print(f"\n🔍 Next steps:")
    print(f"   1. Review master_rolling_report.txt for robustness analysis")
    print(f"   2. Run analyze_rolling_results.py for detailed visualizations")
    print(f"   3. Compare with original window experiment results")

if __name__ == "__main__":
    main()