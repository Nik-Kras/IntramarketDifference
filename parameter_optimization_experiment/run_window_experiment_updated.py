#!/usr/bin/env python3
"""
Updated Window Experiment Orchestrator

Runs parameter optimization experiment using the corrected portfolio simulator:
- Uses pre-generated OOS trades from run_all_pairs_backtest_oos.py
- Implements dynamic budget allocation: current_portfolio_value / max_trades
- Tests windows: 3mo, 6mo, 9mo, 12mo, 18mo, 24mo, 30mo, 36mo, 48mo, 60mo
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List

# Import our custom modules (use fast CSV-based analyzer)
from window_analyzer_fast import analyze_window_fast as analyze_window, generate_selection_report
from window_portfolio_simulator import WindowPortfolioSimulator

# Configuration
TRADES_DIR = "in_sample/trades"
OOS_TRADES_DIR = "out_of_sample/trades"
RESULTS_DIR = "results/window_experiments"
INITIAL_PORTFOLIO_VALUE = 1000.0  # $1k starting capital

# Window configurations - (months_back, window_name) - Equal 3-month steps
WINDOW_CONFIGS = [
    (3, "3mo"),
    (6, "6mo"), 
    (9, "9mo"),
    (12, "12mo"),
    (15, "15mo"),
    (18, "18mo"),
    (21, "21mo"),
    (24, "24mo"),
    (27, "27mo"),
    (30, "30mo"),
    (33, "33mo"),
    (36, "36mo"),
    (39, "39mo"),
    (42, "42mo"),
    (45, "45mo"),
    (48, "48mo"),
    (51, "51mo"),
    (54, "54mo"),
    (57, "57mo"),
    (60, "60mo")
]

def calculate_window_dates(months_back: int) -> tuple:
    """Calculate In-Sample window dates."""
    oos_start_date = pd.to_datetime("2024-01-01")
    window_start = oos_start_date - pd.DateOffset(months=months_back)
    window_end = oos_start_date
    
    return window_start.strftime('%Y-%m-%d'), window_end.strftime('%Y-%m-%d')

def check_oos_trades_exist(selected_pairs: pd.DataFrame) -> pd.DataFrame:
    """Check which selected pairs have OOS trade data available."""
    
    print(f"🔍 Checking OOS trade data for {len(selected_pairs)} pairs...")
    
    valid_pairs = []
    
    for _, pair_row in selected_pairs.iterrows():
        trading_coin = pair_row['trading_coin']
        reference_coin = pair_row['reference_coin']
        strategy_type = pair_row['trading_type']
        
        # Check if OOS trade file exists (prefer CSV, fallback to JSON)
        csv_file = os.path.join(OOS_TRADES_DIR, 
                               f"{reference_coin}_{trading_coin}", 
                               f"{strategy_type}_trades_fast.csv")
        json_file = os.path.join(OOS_TRADES_DIR, 
                                f"{reference_coin}_{trading_coin}", 
                                f"{strategy_type}_trades.json")
        
        trades_file = csv_file if os.path.exists(csv_file) else json_file
        
        if os.path.exists(trades_file):
            try:
                if trades_file.endswith('.csv'):
                    # Fast CSV loading
                    df = pd.read_csv(trades_file)
                    df = df.dropna(subset=['time_entered', 'time_exited', 'log_return'])
                    has_trades = len(df) > 0
                else:
                    # Fallback JSON loading
                    with open(trades_file, 'r') as f:
                        trades_data = json.load(f)
                    has_trades = len(trades_data) > 0
                
                if has_trades:
                    valid_pairs.append(pair_row.to_dict())
                    
            except Exception:
                continue
    
    valid_df = pd.DataFrame(valid_pairs)
    print(f"   Found OOS trades for {len(valid_df)} pairs ({len(valid_df)/len(selected_pairs)*100:.1f}%)")
    
    return valid_df

def run_single_window_experiment(months_back: int, window_name: str) -> Dict:
    """Run complete experiment for a single window configuration."""
    
    experiment_start_time = datetime.now()
    print(f"⏰ Started: {experiment_start_time.strftime('%H:%M:%S')}")
    print(f"🔍 Window: {months_back} months back from 2024-01-01")
    
    # Calculate window dates
    window_start, window_end = calculate_window_dates(months_back)
    
    # Create results directory for this window
    window_results_dir = os.path.join(RESULTS_DIR, window_name)
    os.makedirs(window_results_dir, exist_ok=True)
    
    # Step 1: Analyze In-Sample window and filter pairs
    print(f"\n📊 Step 1: In-Sample Analysis ({window_start} to {window_end})")
    _, selected_pairs, filter_stats = analyze_window(window_start, window_end, TRADES_DIR)
    
    if len(selected_pairs) == 0:
        print(f"❌ No pairs selected for {window_name} window!")
        return {
            'window_name': window_name,
            'months_back': months_back,
            'window_start': window_start,
            'window_end': window_end,
            'pairs_selected': 0,
            'portfolio_return': 0.0,
            'portfolio_sharpe': 0.0,
            'portfolio_drawdown': 0.0,
            'error': 'No pairs selected'
        }
    
    # Save In-Sample results
    selected_pairs.to_csv(os.path.join(window_results_dir, 'selected_pairs.csv'), 
                         index=False, float_format='%.6f')
    
    # Generate selection report
    generate_selection_report(filter_stats, window_start, window_end, 
                            os.path.join(window_results_dir, 'selection_report.txt'))
    
    print(f"   Selected {len(selected_pairs)} pairs after filtering")
    
    # Step 2: Check OOS trade data availability
    print(f"\n🧪 Step 2: Checking Out-of-Sample Trade Data")
    valid_pairs_for_oos = check_oos_trades_exist(selected_pairs)
    
    if len(valid_pairs_for_oos) == 0:
        print(f"❌ No OOS trade data for {window_name} window!")
        return {
            'window_name': window_name,
            'months_back': months_back,
            'window_start': window_start,
            'window_end': window_end,
            'pairs_selected': len(selected_pairs),
            'portfolio_return': 0.0,
            'portfolio_sharpe': 0.0,
            'portfolio_drawdown': 0.0,
            'error': 'No OOS trade data'
        }
    
    # Save valid pairs for OOS
    valid_pairs_for_oos.to_csv(os.path.join(window_results_dir, 'valid_oos_pairs.csv'), 
                              index=False, float_format='%.6f')
    
    print(f"   Found OOS data for {len(valid_pairs_for_oos)} pairs")
    
    # Step 3: Portfolio simulation
    print(f"\n💰 Step 3: Portfolio Simulation")
    
    # Initialize portfolio simulator with corrected parameters
    simulator = WindowPortfolioSimulator(
        initial_capital=INITIAL_PORTFOLIO_VALUE,
        n_pairs=len(valid_pairs_for_oos)
    )
    
    # Run actual portfolio simulation using loaded OOS trades
    portfolio_metrics = simulator.simulate_portfolio(valid_pairs_for_oos)
    
    if not portfolio_metrics:
        print(f"❌ Portfolio simulation failed for {window_name} window!")
        return {
            'window_name': window_name,
            'months_back': months_back,
            'window_start': window_start,
            'window_end': window_end,
            'pairs_selected': len(selected_pairs),
            'portfolio_return': 0.0,
            'portfolio_sharpe': 0.0,
            'portfolio_drawdown': 0.0,
            'error': 'Portfolio simulation failed'
        }
    
    # Extract metrics from simulation
    portfolio_total_return = 1 + portfolio_metrics['total_return']  # Convert to multiplier
    portfolio_sharpe = portfolio_metrics['sharpe_ratio']
    portfolio_drawdown = portfolio_metrics['max_drawdown']
    
    print(f"   Portfolio Return: {portfolio_total_return:.2f}x ({(portfolio_total_return-1)*100:.1f}%)")
    print(f"   Portfolio Sharpe: {portfolio_sharpe:.2f}")
    print(f"   Portfolio Drawdown: {portfolio_drawdown:.2%}")
    print(f"   Trades Executed: {portfolio_metrics.get('num_trades', 0):,}")
    
    # Save portfolio results
    portfolio_summary = {
        'window_name': window_name,
        'months_back': months_back,
        'window_start': window_start,
        'window_end': window_end,
        'pairs_selected': len(selected_pairs),
        'pairs_with_oos': len(valid_pairs_for_oos),
        'portfolio_return': portfolio_total_return,
        'portfolio_sharpe': portfolio_sharpe,
        'portfolio_drawdown': portfolio_drawdown,
        'trades_executed': portfolio_metrics.get('num_trades', 0),
        'max_concurrent_trades': simulator.max_trades,
        'simulation_metrics': portfolio_metrics
    }
    
    with open(os.path.join(window_results_dir, 'portfolio_summary.json'), 'w') as f:
        json.dump(portfolio_summary, f, indent=2)
    
    # Save portfolio visualizations
    simulator.create_visualizations(window_results_dir)
    
    return portfolio_summary

def run_all_window_experiments() -> List[Dict]:
    """Run the complete window experiment pipeline."""
    
    print("🚀 Parameter Optimization Window Experiment")
    print("=" * 50)
    print(f"Testing {len(WINDOW_CONFIGS)} different In-Sample window lengths")
    print(f"Initial portfolio: ${INITIAL_PORTFOLIO_VALUE:,.0f}")
    
    # Create main results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check prerequisites
    if not os.path.exists(TRADES_DIR):
        print(f"❌ Prerequisite missing: {TRADES_DIR}")
        print("Run run_all_pairs_backtest.py first to generate In-Sample trade data.")
        return []
    
    if not os.path.exists(OOS_TRADES_DIR):
        print(f"❌ Prerequisite missing: {OOS_TRADES_DIR}")
        print("Run run_all_pairs_backtest_oos.py first to generate Out-of-Sample trade data.")
        return []
    
    # Run experiments for each window
    all_results = []
    
    print(f"\n🔄 Starting {len(WINDOW_CONFIGS)} window experiments...")
    print(f"📅 Window range: {WINDOW_CONFIGS[0][1]} to {WINDOW_CONFIGS[-1][1]}")
    
    for i, (months_back, window_name) in enumerate(WINDOW_CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"🔄 EXPERIMENT {i}/{len(WINDOW_CONFIGS)}: {window_name} Window")
        print(f"{'='*80}")
        print(f"📊 Overall Progress: {i-1}/{len(WINDOW_CONFIGS)} completed ({(i-1)/len(WINDOW_CONFIGS)*100:.1f}%)")
        
        try:
            result = run_single_window_experiment(months_back, window_name)
            all_results.append(result)
            
            print(f"\n✅ COMPLETED {window_name} window experiment")
            if 'error' not in result:
                print(f"   📈 Portfolio Return: {result['portfolio_return']:.2f}x")
                print(f"   📊 Sharpe Ratio: {result['portfolio_sharpe']:.2f}")
                print(f"   📉 Max Drawdown: {result['portfolio_drawdown']:.2%}")
                print(f"   📋 Trades Executed: {result.get('trades_executed', 0):,}")
            
        except Exception as e:
            print(f"\n❌ ERROR in {window_name} window: {str(e)}")
            
            # Add error result
            window_start, window_end = calculate_window_dates(months_back)
            error_result = {
                'window_name': window_name,
                'months_back': months_back,
                'window_start': window_start,
                'window_end': window_end,
                'pairs_selected': 0,
                'portfolio_return': 0.0,
                'portfolio_sharpe': 0.0,
                'portfolio_drawdown': 0.0,
                'error': str(e)
            }
            all_results.append(error_result)
    
    return all_results

def create_master_summary(all_results: List[Dict]):
    """Create master summary analysis of all window experiments."""
    
    print(f"\n📋 Creating Master Summary...")
    
    # Filter out error results
    valid_results = [r for r in all_results if 'error' not in r]
    
    if not valid_results:
        print("❌ No valid results to summarize!")
        return
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(valid_results)
    
    # Save detailed summary
    summary_file = os.path.join(RESULTS_DIR, 'master_window_summary.csv')
    summary_df.to_csv(summary_file, index=False, float_format='%.6f')
    
    # Create text report
    report_lines = [
        "PARAMETER OPTIMIZATION WINDOW EXPERIMENT RESULTS",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total windows tested: {len(WINDOW_CONFIGS)}",
        f"Successful experiments: {len(valid_results)}",
        f"Initial portfolio value: ${INITIAL_PORTFOLIO_VALUE:,.0f}",
        "",
        "WINDOW PERFORMANCE SUMMARY",
        "-" * 30,
        f"{'Window':<8} {'Pairs':<6} {'Return':<8} {'Sharpe':<7} {'Drawdown':<10} {'Trades':<8}",
        f"{'-'*8} {'-'*6} {'-'*8} {'-'*7} {'-'*10} {'-'*8}"
    ]
    
    # Sort by portfolio return for ranking
    summary_df_sorted = summary_df.sort_values('portfolio_return', ascending=False)
    
    for _, row in summary_df_sorted.iterrows():
        trades_executed = row.get('trades_executed', 0)
        report_lines.append(
            f"{row['window_name']:<8} {row['pairs_selected']:<6} "
            f"{row['portfolio_return']:<8.2f} {row['portfolio_sharpe']:<7.2f} "
            f"{row['portfolio_drawdown']:<10.2%} {trades_executed:<8}"
        )
    
    # Add best performer details
    if len(valid_results) > 0:
        best = summary_df_sorted.iloc[0]
        report_lines.extend([
            "",
            "BEST PERFORMING WINDOW",
            "-" * 25,
            f"Window: {best['window_name']} ({best['months_back']} months)",
            f"Portfolio Return: {best['portfolio_return']:.2f}x ({(best['portfolio_return']-1)*100:.1f}%)",
            f"Portfolio Sharpe: {best['portfolio_sharpe']:.2f}",
            f"Portfolio Drawdown: {best['portfolio_drawdown']:.2%}",
            f"Pairs Selected: {best['pairs_selected']}",
            f"Trades Executed: {best.get('trades_executed', 0):,}",
            "",
            "SUMMARY STATISTICS",
            "-" * 20,
            f"Average portfolio return: {summary_df['portfolio_return'].mean():.2f}x",
            f"Average portfolio Sharpe: {summary_df['portfolio_sharpe'].mean():.2f}",
            f"Average portfolio drawdown: {summary_df['portfolio_drawdown'].mean():.2%}",
            f"Average pairs selected: {summary_df['pairs_selected'].mean():.0f}",
        ])
    
    # Save master report
    report_file = os.path.join(RESULTS_DIR, 'master_experiment_report.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📊 MASTER SUMMARY COMPLETE:")
    print(f"   Valid experiments: {len(valid_results)}/{len(WINDOW_CONFIGS)}")
    if len(valid_results) > 0:
        print(f"   Best window: {summary_df_sorted.iloc[0]['window_name']} "
              f"(Return: {summary_df_sorted.iloc[0]['portfolio_return']:.2f}x)")
    print(f"   Summary CSV: {summary_file}")
    print(f"   Detailed report: {report_file}")

def main():
    """Main execution function."""
    
    experiment_start = datetime.now()
    print("🎯 Parameter Optimization Window Experiment")
    print("=" * 50)
    print("Testing optimal In-Sample window lengths for live trading")
    print(f"🕐 Experiment started: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not os.path.exists(TRADES_DIR):
        print(f"❌ Prerequisite missing: {TRADES_DIR}")
        print("Run run_all_pairs_backtest.py first to generate In-Sample trade data.")
        return
    
    if not os.path.exists(OOS_TRADES_DIR):
        print(f"❌ Prerequisite missing: {OOS_TRADES_DIR}")
        print("Run run_all_pairs_backtest_oos.py first to generate Out-of-Sample trade data.")
        return
    
    # Run all window experiments
    all_results = run_all_window_experiments()
    
    if not all_results:
        print("❌ No experiments completed successfully!")
        return
    
    # Create master summary
    create_master_summary(all_results)
    
    experiment_end = datetime.now()
    total_duration = experiment_end - experiment_start
    
    print(f"\n🎯 EXPERIMENT COMPLETED!")
    print(f"🕐 Started: {experiment_start.strftime('%H:%M:%S')}")
    print(f"🕐 Finished: {experiment_end.strftime('%H:%M:%S')}")
    print(f"⏱️  Total Duration: {total_duration}")
    print(f"📁 All results saved to: {RESULTS_DIR}/")
    print(f"📊 Master summary: {RESULTS_DIR}/master_window_summary.csv")
    print(f"📋 Detailed report: {RESULTS_DIR}/master_experiment_report.txt")
    
    print(f"\n🔍 Next steps:")
    print(f"   1. Review master_experiment_report.txt for best window length")
    print(f"   2. Run analyze_window_results.py for detailed visualizations")
    print(f"   3. Select optimal window length for live trading deployment")

if __name__ == "__main__":
    main()