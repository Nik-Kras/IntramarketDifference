#!/usr/bin/env python3
"""
Window Results Analysis Script

Creates comprehensive visualizations and analysis of the parameter optimization experiment:
- Efficiency frontier analysis (without efficient frontier line)
- Diversification vs performance analysis
- Window performance comparison across all metrics
- Executive summary with key insights
- Merged comprehensive analysis report
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = "results/window_experiments"
ANALYSIS_DIR = "results/window_analysis"


def load_and_clean_results() -> pd.DataFrame:
    """Load results and handle data quality issues."""
    
    print("📂 Loading and cleaning window experiment results...")
    
    summary_file = os.path.join(RESULTS_DIR, 'master_window_summary.csv')
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Master summary not found: {summary_file}")
    
    df = pd.read_csv(summary_file)
    
    # Identify duplicate results (data quality issue)
    df['metrics_hash'] = df.apply(lambda row: hash((
        round(row['portfolio_return'], 6),
        round(row['portfolio_sharpe'], 6),
        round(row['portfolio_drawdown'], 6),
        row['trades_executed']
    )), axis=1)
    
    # Mark duplicates
    df['is_duplicate'] = df.duplicated(subset=['metrics_hash'], keep='first')
    
    # Find actual data cutoff
    unique_results = df[~df['is_duplicate']]
    if len(unique_results) < len(df):
        cutoff_month = unique_results['months_back'].max()
        print(f"⚠️  Data quality issue detected: Results identical for windows > {cutoff_month} months")
        print(f"   Likely cause: Historical data only goes back ~{cutoff_month} months")
        df['data_quality'] = df['months_back'].apply(lambda x: 'Full' if x <= cutoff_month else 'Truncated')
    else:
        df['data_quality'] = 'Full'
    
    print(f"   Loaded {len(df)} windows ({len(unique_results)} with unique results)")
    
    return df

def create_efficiency_frontier(df: pd.DataFrame):
    """Create risk-return efficiency frontier analysis (without efficient frontier line)."""
    
    from visualization_utils import create_efficiency_frontier_plot
    
    print("📊 Creating efficiency frontier analysis...")
    
    # Filter to unique results only
    df_unique = df[~df['is_duplicate']].copy()
    
    create_efficiency_frontier_plot(
        df_unique, 
        os.path.join(ANALYSIS_DIR, 'efficiency_frontier.png'),
        'Portfolio Efficiency Analysis'
    )

def analyze_portfolio_composition(df: pd.DataFrame):
    """Analyze portfolio composition - only diversification vs performance."""
    
    from visualization_utils import create_portfolio_composition_plot
    
    print("📊 Analyzing portfolio composition...")
    
    df_unique = df[~df['is_duplicate']].copy()
    
    create_portfolio_composition_plot(
        df_unique,
        os.path.join(ANALYSIS_DIR, 'portfolio_composition.png'),
        'Diversification vs Performance'
    )

def create_window_performance_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Create window performance comparison visualization."""
    
    from visualization_utils import create_window_performance_comparison_plot
    
    print("📊 Creating window performance comparison...")
    
    df_unique = df[~df['is_duplicate']].copy()
    
    create_window_performance_comparison_plot(
        df_unique,
        os.path.join(ANALYSIS_DIR, 'window_performance_comparison.png')
    )
    
    # Calculate simple composite ranking
    df_unique['return_rank'] = df_unique['portfolio_return'].rank(ascending=False)
    df_unique['sharpe_rank'] = df_unique['portfolio_sharpe'].rank(ascending=False)
    df_unique['drawdown_rank'] = df_unique['portfolio_drawdown'].rank(ascending=True)
    
    df_unique['composite_score'] = (df_unique['return_rank'] + 
                                   df_unique['sharpe_rank'] + 
                                   df_unique['drawdown_rank']) / 3
    
    df_unique['final_rank'] = df_unique['composite_score'].rank().astype(int)
    df_unique = df_unique.sort_values('final_rank')
    
    return df_unique

def create_executive_summary(df_analysis: pd.DataFrame):
    """Create executive summary with key insights."""
    
    from visualization_utils import create_executive_summary_plot
    
    print("📋 Creating executive summary...")
    
    # Get top performer
    top_window = df_analysis.iloc[0]
    df_valid = df_analysis[df_analysis['data_quality'] == 'Full'].sort_values('months_back')
    
    create_executive_summary_plot(
        top_window,
        df_valid,
        os.path.join(ANALYSIS_DIR, 'executive_summary.png')
    )

def analyze_pair_selection_patterns():
    """Analyze pair selection patterns across different windows."""
    
    print("🔍 Analyzing pair selection patterns...")
    
    # Load selected pairs from each window
    pair_analysis = {}
    
    for window_dir in os.listdir(RESULTS_DIR):
        window_path = os.path.join(RESULTS_DIR, window_dir)
        
        if not os.path.isdir(window_path) or window_dir.startswith('.'):
            continue
        
        selected_pairs_file = os.path.join(window_path, 'selected_pairs.csv')
        
        if os.path.exists(selected_pairs_file):
            try:
                pairs_df = pd.read_csv(selected_pairs_file)
                
                # Create unique pair identifiers
                pairs_df['pair_id'] = (pairs_df['reference_coin'] + '_' + 
                                     pairs_df['trading_coin'] + '_' + 
                                     pairs_df['trading_type'])
                
                pair_analysis[window_dir] = {
                    'pair_ids': set(pairs_df['pair_id'].tolist()),
                    'count': len(pairs_df),
                    'trading_coins': pairs_df['trading_coin'].value_counts().to_dict(),
                    'reference_coins': pairs_df['reference_coin'].value_counts().to_dict(),
                    'strategy_types': pairs_df['trading_type'].value_counts().to_dict()
                }
                
            except Exception:
                continue
    
    if not pair_analysis:
        return {}, {}
    
    # Find consistently selected pairs
    all_pairs = set()
    for window_data in pair_analysis.values():
        all_pairs.update(window_data['pair_ids'])
    
    # Count appearance frequency
    pair_frequency = {}
    for pair_id in all_pairs:
        frequency = sum(1 for window_data in pair_analysis.values() 
                       if pair_id in window_data['pair_ids'])
        pair_frequency[pair_id] = frequency
    
    # Create consistency analysis
    total_windows = len(pair_analysis)
    consistent_pairs = {pair_id: freq for pair_id, freq in pair_frequency.items() 
                       if freq >= total_windows * 0.7}  # Appears in 70%+ of windows
    
    return pair_analysis, consistent_pairs

def generate_merged_report(df_analysis: pd.DataFrame, pair_analysis: Dict, consistent_pairs: Dict):
    """Generate merged comprehensive report combining both detailed and enhanced reports."""
    
    print("📋 Generating merged comprehensive analysis report...")
    
    report_lines = [
        "=" * 80,
        "COMPREHENSIVE WINDOW OPTIMIZATION ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Experiments analyzed: {len(df_analysis)}",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
    ]
    
    # Get top performers
    top_3 = df_analysis.head(3)
    top_window = top_3.iloc[0]
    
    report_lines.extend([
        f"Optimal Window: {top_window['window_name']} ({top_window['months_back']} months)",
        f"Composite Score: {top_window['composite_score']:.3f}",
        f"Expected Annual Return: {(top_window['portfolio_return'] - 1) * 100:.1f}%",
        f"Risk-Adjusted Return (Sharpe): {top_window['portfolio_sharpe']:.2f}",
        f"Maximum Drawdown: {top_window['portfolio_drawdown']:.2%}",
        f"Selected Pairs: {top_window['pairs_selected']:,}",
        "",
        "DATA QUALITY ASSESSMENT",
        "-" * 40,
        f"Full Data Available: Up to {df_analysis[df_analysis['data_quality']=='Full']['months_back'].max()} months",
        f"Truncated Data Windows: {len(df_analysis[df_analysis['data_quality']!='Full'])} windows affected",
        "",
        "PERFORMANCE STATISTICS",
        "-" * 40,
        f"Return Range: {df_analysis['portfolio_return'].min():.2f}x to {df_analysis['portfolio_return'].max():.2f}x",
        f"Sharpe Range: {df_analysis['portfolio_sharpe'].min():.2f} to {df_analysis['portfolio_sharpe'].max():.2f}",
        f"Drawdown Range: {df_analysis['portfolio_drawdown'].min():.2%} to {df_analysis['portfolio_drawdown'].max():.2%}",
        f"Standard deviation of returns: {df_analysis['portfolio_return'].std():.3f}",
        f"Coefficient of variation: {df_analysis['portfolio_return'].std() / df_analysis['portfolio_return'].mean():.3f}",
        "",
        "PAIR SELECTION ANALYSIS",
        "-" * 40,
    ])
    
    if pair_analysis:
        all_pairs_count = len(set().union(*[data['pair_ids'] for data in pair_analysis.values()]))
        report_lines.extend([
            f"Total unique pairs across all windows: {all_pairs_count}",
            f"Consistently selected pairs (70%+ windows): {len(consistent_pairs)}",
            f"Selection consistency rate: {len(consistent_pairs) / all_pairs_count * 100:.1f}%",
        ])
    
    report_lines.extend([
        "",
        "DETAILED WINDOW RANKINGS",
        "-" * 40,
        f"{'Rank':<5} {'Window':<8} {'Months':<7} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Pairs':<6} {'Score':<8}",
        f"{'-'*5} {'-'*8} {'-'*7} {'-'*10} {'-'*8} {'-'*10} {'-'*6} {'-'*8}",
    ])
    
    for idx, row in df_analysis.iterrows():
        report_lines.append(
            f"{row['final_rank']:<5} {row['window_name']:<8} {row['months_back']:<7} "
            f"{row['portfolio_return']:<10.2f} {row['portfolio_sharpe']:<8.2f} "
            f"{row['portfolio_drawdown']:<10.2%} {row['pairs_selected']:<6} "
            f"{row['composite_score']:<8.3f}"
        )
    
    report_lines.extend([
        "",
        "TOP 3 RECOMMENDATIONS",
        "-" * 40,
    ])
    
    for i in range(min(3, len(top_3))):
        row = top_3.iloc[i]
        report_lines.extend([
            f"{i+1}. {row['window_name']} Window ({row['months_back']} months)",
            f"   Return: {row['portfolio_return']:.2f}x ({(row['portfolio_return']-1)*100:.1f}%)",
            f"   Sharpe: {row['portfolio_sharpe']:.2f}",
            f"   Drawdown: {row['portfolio_drawdown']:.2%}",
            f"   Pairs: {row['pairs_selected']}",
            ""
        ])
    
    report_lines.extend([
        "IMPLEMENTATION GUIDELINES",
        "-" * 40,
        f"1. Start with {top_window['window_name']} window configuration",
        f"2. Allocate capital across {top_window['pairs_selected']:,} selected pairs",
        f"3. Set maximum drawdown limit at {abs(top_window['portfolio_drawdown']) * 1.5:.1%}",
        f"4. Monitor Sharpe ratio weekly (alert if drops below 2.0)",
        f"5. Re-optimize selection quarterly using same framework",
        "",
        "RISK FACTORS",
        "-" * 40,
        "• Market regime changes may affect strategy performance",
        "• Increased pairs may lead to higher execution complexity",
        "• Historical performance does not guarantee future results",
        "• Consider transaction costs in final implementation",
        "",
        "METHODOLOGY NOTES",
        "-" * 40,
        "- All windows use identical filtering criteria (Sharpe > 2.0, Drawdown > -50%)",
        "- Out-of-Sample validation on 2024 data only",
        "- Portfolio simulation with dynamic budget allocation",
        "- Budget formula: 1.1 * portfolio_value / N_pairs (allows 90% concurrent trades)",
        "- Rankings based on composite score of return, Sharpe, and drawdown",
        "",
        "NEXT STEPS",
        "-" * 40,
        "1. Review visualizations in results/window_analysis/",
        "2. Validate top configuration with paper trading",
        "3. Implement gradual position scaling during deployment",
        "4. Set up monitoring dashboard for key metrics",
        "5. Document any deviations from recommended parameters",
    ])
    
    # Save report
    report_file = os.path.join(ANALYSIS_DIR, 'comprehensive_analysis_report.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"   Report saved: {report_file}")

def main():
    """Main execution function."""
    
    print("🔬 Window Experiment Analysis")
    print("=" * 40)
    
    # Professional styling handled by visualization_utils
    
    # Create analysis directory
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    try:
        # Load and clean data
        df = load_and_clean_results()
        
        # Create visualizations
        create_efficiency_frontier(df)
        analyze_portfolio_composition(df)
        df_analysis = create_window_performance_comparison(df)
        create_executive_summary(df_analysis)
        
        # Analyze pair patterns
        pair_analysis, consistent_pairs = analyze_pair_selection_patterns()
        
        # Generate merged report
        generate_merged_report(df_analysis, pair_analysis, consistent_pairs)
        
        print("\n✅ ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {ANALYSIS_DIR}/")
        print("\n📊 Key Outputs:")
        print("   • efficiency_frontier.png - Risk-return analysis")
        print("   • portfolio_composition.png - Diversification analysis")
        print("   • window_performance_comparison.png - Performance comparison")
        print("   • executive_summary.png - Key insights")
        print("   • comprehensive_analysis_report.txt - Full documentation")
        
        # Display top recommendation
        if len(df_analysis) > 0:
            top = df_analysis.iloc[0]
            print(f"\n🏆 RECOMMENDATION: Deploy {top['window_name']} window")
            print(f"   Expected Return: {(top['portfolio_return']-1)*100:.1f}%")
            print(f"   Sharpe Ratio: {top['portfolio_sharpe']:.2f}")
            print(f"   Max Drawdown: {top['portfolio_drawdown']:.1%}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()