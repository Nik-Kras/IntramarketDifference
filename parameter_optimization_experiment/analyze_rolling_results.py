#!/usr/bin/env python3
"""
Rolling Window Results Analysis Script

Analyzes results from rolling window validation experiment:
- Window stability analysis across different time periods
- Market regime sensitivity assessment  
- Robustness metrics and ranking
- Temporal consistency evaluation
- Comprehensive visualization suite
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = "results/rolling_experiments"
ANALYSIS_DIR = "results/rolling_analysis"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_divide(numerator: pd.Series, denominator: pd.Series, default_value: float = float('nan')) -> pd.Series:
    """Safely divide two pandas Series, handling zero denominators."""
    return numerator.where(denominator != 0, default_value) / denominator.where(denominator != 0, 1)

def validate_dataframe(df: pd.DataFrame, required_columns: list, function_name: str) -> None:
    """Validate DataFrame structure and required columns."""
    if df.empty:
        raise ValueError(f"{function_name}: Input DataFrame is empty")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"{function_name}: Missing required columns: {missing_cols}")

# ============================================================================
# ANALYSIS FUNCTIONS  
# ============================================================================


def load_rolling_results() -> pd.DataFrame:
    """Load all rolling experiment results."""
    
    print("📂 Loading rolling experiment results...")
    
    master_file = os.path.join(RESULTS_DIR, 'master_rolling_summary.csv')
    
    if not os.path.exists(master_file):
        print(f"❌ Master rolling summary not found: {master_file}")
        print("Run run_rolling_window_experiment.py first.")
        return pd.DataFrame()
    
    df = pd.read_csv(master_file)
    
    # Filter out error results
    df_clean = df[~df.isin(['error']).any(axis=1)].copy()
    
    print(f"   Loaded {len(df)} total results ({len(df_clean)} valid)")
    
    return df_clean

def create_efficiency_frontier(df: pd.DataFrame) -> None:
    """Create efficiency frontier analysis across all periods."""
    
    validate_dataframe(df, ['portfolio_return', 'portfolio_sharpe'], 'create_efficiency_frontier')
    
    from visualization_utils import create_rolling_efficiency_frontier_plot
    
    print("📊 Creating efficiency frontier analysis...")
    
    create_rolling_efficiency_frontier_plot(
        df,
        os.path.join(ANALYSIS_DIR, 'efficiency_frontier.png')
    )

def analyze_portfolio_composition(df: pd.DataFrame) -> None:
    """Analyze diversification vs performance across all periods."""
    
    validate_dataframe(df, ['pairs_selected', 'portfolio_sharpe'], 'analyze_portfolio_composition')
    
    from visualization_utils import create_rolling_portfolio_composition_plot
    
    print("📊 Analyzing portfolio composition...")
    
    create_rolling_portfolio_composition_plot(
        df,
        os.path.join(ANALYSIS_DIR, 'portfolio_composition.png')
    )

def create_window_performance_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive window performance comparison across periods."""
    
    validate_dataframe(df, ['window_months', 'portfolio_return', 'portfolio_sharpe', 'portfolio_drawdown', 'pairs_selected'], 'create_window_performance_comparison')
    
    from visualization_utils import create_rolling_window_performance_plot
    
    print("📊 Creating window performance comparison...")
    
    # Calculate average metrics by window across all periods
    window_stats = df.groupby('window_months').agg({
        'portfolio_return': ['mean', 'std'],
        'portfolio_sharpe': ['mean', 'std'],
        'portfolio_drawdown': ['mean', 'std'],
        'pairs_selected': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    window_stats.columns = ['_'.join(col).strip() for col in window_stats.columns]
    window_stats = window_stats.reset_index()
    
    create_rolling_window_performance_plot(
        window_stats,
        os.path.join(ANALYSIS_DIR, 'window_performance_comparison.png')
    )
    
    # Calculate robustness ranking with safe division
    window_stats['sharpe_cv'] = safe_divide(window_stats['portfolio_sharpe_std'], window_stats['portfolio_sharpe_mean'])
    window_stats['return_cv'] = safe_divide(window_stats['portfolio_return_std'], window_stats['portfolio_return_mean'])
    
    # Composite robustness score (lower CV = more robust)
    window_stats['robustness_score'] = (window_stats['sharpe_cv'] + window_stats['return_cv']) / 2
    window_stats['robustness_rank'] = window_stats['robustness_score'].rank()
    
    # Performance ranking
    window_stats['performance_rank'] = window_stats['portfolio_sharpe_mean'].rank(ascending=False)
    
    # Final ranking (combine robustness and performance)
    window_stats['final_score'] = window_stats['robustness_rank'] + window_stats['performance_rank']
    window_stats['final_rank'] = window_stats['final_score'].rank()
    
    return window_stats.sort_values('final_rank')

def create_executive_summary(df_analysis: pd.DataFrame, df_results: pd.DataFrame) -> None:
    """Create executive summary for rolling experiment."""
    
    validate_dataframe(df_analysis, ['window_months', 'portfolio_sharpe_mean'], 'create_executive_summary')
    validate_dataframe(df_results, ['period_name'], 'create_executive_summary')
    
    from visualization_utils import create_rolling_executive_summary_plot
    
    print("📋 Creating executive summary...")
    
    # Get most robust window
    most_robust = df_analysis.iloc[0]
    best_performance = df_analysis.sort_values('portfolio_sharpe_mean', ascending=False).iloc[0]
    
    create_rolling_executive_summary_plot(
        most_robust,
        best_performance,
        df_results,
        os.path.join(ANALYSIS_DIR, 'executive_summary.png')
    )

def generate_comprehensive_report(df_analysis: pd.DataFrame, df_results: pd.DataFrame) -> None:
    """Generate comprehensive rolling validation report."""
    
    validate_dataframe(df_analysis, ['window_months', 'portfolio_sharpe_mean', 'sharpe_cv'], 'generate_comprehensive_report')
    validate_dataframe(df_results, ['period_name', 'window_months', 'portfolio_sharpe'], 'generate_comprehensive_report')
    
    print("📋 Generating comprehensive rolling validation report...")
    
    # Get key statistics
    most_robust = df_analysis.iloc[0]
    best_performance = df_analysis.sort_values('portfolio_sharpe_mean', ascending=False).iloc[0]
    total_periods = len(df_results['period_name'].unique())
    
    report_lines = [
        "=" * 80,
        "ROLLING WINDOW VALIDATION COMPREHENSIVE REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total experiments: {len(df_results)}",
        f"Time periods tested: {total_periods}",
        f"Window lengths tested: {sorted(df_results['window_months'].unique())}",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        f"Most Robust Window: {most_robust['window_months']:.0f} months",
        f"  Consistency Score (CV): {most_robust['sharpe_cv']:.3f}",
        f"  Average Sharpe: {most_robust['portfolio_sharpe_mean']:.2f} ± {most_robust['portfolio_sharpe_std']:.2f}",
        f"  Average Return: {most_robust['portfolio_return_mean']:.2f}x ± {most_robust['portfolio_return_std']:.2f}x",
        "",
        f"Best Performance Window: {best_performance['window_months']:.0f} months", 
        f"  Average Sharpe: {best_performance['portfolio_sharpe_mean']:.2f}",
        f"  Consistency Score (CV): {best_performance['sharpe_cv']:.3f}",
        "",
        "ROBUSTNESS ANALYSIS",
        "-" * 40,
        f"{'Rank':<5} {'Window':<8} {'Avg Sharpe':<11} {'Sharpe CV':<10} {'Avg Return':<11} {'Return CV':<10} {'Robustness':<10}",
        f"{'-'*5} {'-'*8} {'-'*11} {'-'*10} {'-'*11} {'-'*10} {'-'*10}",
    ]
    
    # Add robustness ranking table
    for _, row in df_analysis.iterrows():
        robustness = "High" if row['sharpe_cv'] < 0.2 else "Medium" if row['sharpe_cv'] < 0.5 else "Low"
        report_lines.append(
            f"{row['final_rank']:<5.0f} {row['window_months']:<8.0f} "
            f"{row['portfolio_sharpe_mean']:<11.3f} {row['sharpe_cv']:<10.3f} "
            f"{row['portfolio_return_mean']:<11.3f} {row['return_cv']:<10.3f} {robustness:<10}"
        )
    
    # Period-by-period analysis
    report_lines.extend([
        "",
        "PERIOD-BY-PERIOD PERFORMANCE",
        "-" * 40,
        f"{'Period':<25} {'Best Window':<12} {'Best Sharpe':<11} {'Worst Window':<12} {'Worst Sharpe':<11}",
        f"{'-'*25} {'-'*12} {'-'*11} {'-'*12} {'-'*11}",
    ])
    
    for period in df_results['period_name'].unique():
        period_data = df_results[df_results['period_name'] == period]
        best_idx = period_data['portfolio_sharpe'].idxmax()
        worst_idx = period_data['portfolio_sharpe'].idxmin()
        
        best_window = period_data.loc[best_idx]
        worst_window = period_data.loc[worst_idx]
        
        report_lines.append(
            f"{period:<25} {best_window['window_months']:<12.0f} "
            f"{best_window['portfolio_sharpe']:<11.2f} {worst_window['window_months']:<12.0f} "
            f"{worst_window['portfolio_sharpe']:<11.2f}"
        )
    
    # Market regime analysis
    report_lines.extend([
        "",
        "MARKET REGIME INSIGHTS",
        "-" * 40,
        "• Window performance varies significantly across time periods",
        "• Robustness measured by coefficient of variation in key metrics",
        "• Lower CV indicates more consistent performance across regimes",
        f"• {most_robust['window_months']:.0f}-month window shows highest consistency",
        "",
        "VALIDATION METHODOLOGY",
        "-" * 40,
        "• Rolling validation with 1-year OOS periods",
        "• 2-year In-Sample windows for pair selection",
        "• Identical filtering criteria across all periods",
        "• Dynamic budget allocation with 90% utilization target",
        "• Statistical robustness measured via coefficient of variation",
        "",
        "IMPLEMENTATION GUIDELINES",
        "-" * 40,
        f"1. RECOMMENDED: {most_robust['window_months']:.0f}-month window for stable performance",
        f"   Expected Sharpe: {most_robust['portfolio_sharpe_mean']:.2f} (±{most_robust['portfolio_sharpe_std']:.2f})",
        f"   Expected Return: {(most_robust['portfolio_return_mean']-1)*100:.1f}% (±{most_robust['portfolio_return_std']*100:.1f}%)",
        "",
        f"2. ALTERNATIVE: {best_performance['window_months']:.0f}-month window for maximum performance",
        f"   Higher expected Sharpe: {best_performance['portfolio_sharpe_mean']:.2f}",
        f"   But higher variability (CV: {best_performance['sharpe_cv']:.3f})",
        "",
        "3. RISK MANAGEMENT:",
        f"   Set maximum drawdown limit: {abs(most_robust['portfolio_drawdown_mean']) * 1.5:.1%}",
        "   Monitor performance weekly during first month",
        "   Re-validate quarterly or on regime changes",
        "",
        "NEXT STEPS",
        "-" * 40,
        "1. Review visualizations for detailed analysis",
        "2. Compare with original single-period optimization",
        "3. Select final window based on risk tolerance",
        "4. Implement with gradual position scaling",
        "5. Establish monitoring framework for live deployment"
    ])
    
    # Save comprehensive report
    report_file = os.path.join(ANALYSIS_DIR, 'comprehensive_analysis_report.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"   Comprehensive report saved: {report_file}")

def main() -> None:
    """Main execution function."""
    
    print("🔬 Rolling Window Validation Analysis")
    print("=" * 50)
    
    # Professional styling handled by visualization_utils
    
    # Create analysis directory
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    try:
        # Load rolling results
        df_results = load_rolling_results()
        
        if len(df_results) == 0:
            print("❌ No results to analyze!")
            return
        
        print(f"📊 Analyzing {len(df_results)} experiments across {len(df_results['period_name'].unique())} periods")
        
        # Create visualizations
        create_efficiency_frontier(df_results)
        analyze_portfolio_composition(df_results)
        df_analysis = create_window_performance_comparison(df_results)
        create_executive_summary(df_analysis, df_results)
        
        # Generate comprehensive report
        generate_comprehensive_report(df_analysis, df_results)
        
        print("\n✅ ROLLING ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {ANALYSIS_DIR}/")
        print("\n📊 Key Outputs:")
        print("   • efficiency_frontier.png - Risk-return analysis across periods")
        print("   • portfolio_composition.png - Diversification analysis")
        print("   • window_performance_comparison.png - Average performance with error bars")
        print("   • executive_summary.png - Robustness insights")
        print("   • comprehensive_analysis_report.txt - Full validation report")
        
        # Display recommendations
        if len(df_analysis) > 0:
            robust = df_analysis.iloc[0]
            best_perf = df_analysis.sort_values('portfolio_sharpe_mean', ascending=False).iloc[0]
            
            print(f"\n🏆 RECOMMENDATIONS:")
            print(f"   🛡️  Most Robust: {robust['window_months']:.0f}mo (CV: {robust['sharpe_cv']:.3f})")
            print(f"   🚀 Best Performance: {best_perf['window_months']:.0f}mo (Sharpe: {best_perf['portfolio_sharpe_mean']:.2f})")
            
            if robust['window_months'] == best_perf['window_months']:
                print(f"   ✨ OPTIMAL: {robust['window_months']:.0f}mo window combines robustness AND performance!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found error: {str(e)}")
    except pd.errors.EmptyDataError as e:
        print(f"❌ Empty data error: {str(e)}")
    except KeyError as e:
        print(f"❌ Missing required column: {str(e)}")
    except ValueError as e:
        print(f"❌ Data validation error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()