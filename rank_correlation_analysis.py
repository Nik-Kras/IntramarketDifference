#!/usr/bin/env python3
"""
Rank Correlation Analysis: In-Sample vs Out-of-Sample
=====================================================

Computes Spearman rank correlations between In-Sample and Out-of-Sample metrics
to assess if selected pairs represent genuine signal or just noise.

If correlations are near zero (~0), it suggests we're selecting noise rather than
persistent trading edge.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
RESULTS_FILE = "oos_experiments/oos_backtest_results.csv"
OUTPUT_DIR = "oos_experiments"

def load_and_prepare_data():
    """Load OOS results and prepare for correlation analysis."""
    print("📂 Loading OOS backtest results...")
    
    df = pd.read_csv(RESULTS_FILE)
    print(f"✅ Loaded {len(df)} pairs")
    
    # Filter out pairs with errors in either year
    valid_df = df[df['oos_error_2023'].isna() & df['oos_error_2024'].isna()].copy()
    print(f"📊 Valid pairs (no errors): {len(valid_df)}")
    
    if len(valid_df) == 0:
        raise ValueError("No valid pairs found for correlation analysis!")
    
    return valid_df

def compute_correlations(df):
    """Compute Spearman rank correlations between IS and OOS metrics."""
    
    # Define metric pairs for correlation analysis
    metric_pairs = [
        # Profit Factor
        ('profit_factor', 'oos_profit_factor_2023', 'Profit Factor (2023)'),
        ('profit_factor', 'oos_profit_factor_2024', 'Profit Factor (2024)'),
        
        # Total Return
        ('total_cumulative_return', 'oos_total_return_2023', 'Total Return (2023)'),
        ('total_cumulative_return', 'oos_total_return_2024', 'Total Return (2024)'),
        
        # Sharpe Ratio
        ('sharpe_ratio', 'oos_sharpe_ratio_2023', 'Sharpe Ratio (2023)'),
        ('sharpe_ratio', 'oos_sharpe_ratio_2024', 'Sharpe Ratio (2024)'),
        
        # Max Drawdown
        ('max_drawdown', 'oos_max_drawdown_2023', 'Max Drawdown (2023)'),
        ('max_drawdown', 'oos_max_drawdown_2024', 'Max Drawdown (2024)'),
        
        # Number of Trades
        ('num_trades', 'oos_num_trades_2023', 'Num Trades (2023)'),
        ('num_trades', 'oos_num_trades_2024', 'Num Trades (2024)'),
    ]
    
    print("\n🔍 SPEARMAN RANK CORRELATIONS (In-Sample vs Out-of-Sample)")
    print("=" * 80)
    print(f"{'Metric':<25} {'Correlation':<12} {'P-value':<10} {'N':<6} {'Assessment'}")
    print("-" * 80)
    
    correlations = []
    
    for is_metric, oos_metric, label in metric_pairs:
        # Remove rows with NaN values in either metric
        valid_mask = df[is_metric].notna() & df[oos_metric].notna()
        valid_data = df[valid_mask]
        
        if len(valid_data) < 10:
            print(f"{label:<25} {'INSUFFICIENT DATA':<12} {'':<10} {len(valid_data):<6}")
            continue
            
        # Compute Spearman correlation
        correlation, p_value = spearmanr(valid_data[is_metric], valid_data[oos_metric])
        
        # Assessment based on correlation strength
        if abs(correlation) < 0.1:
            assessment = "NOISE ⚠️"
        elif abs(correlation) < 0.3:
            assessment = "WEAK"
        elif abs(correlation) < 0.5:
            assessment = "MODERATE"
        elif abs(correlation) < 0.7:
            assessment = "STRONG"
        else:
            assessment = "VERY STRONG ✅"
            
        print(f"{label:<25} {correlation:<12.4f} {p_value:<10.4f} {len(valid_data):<6} {assessment}")
        
        correlations.append({
            'metric': label,
            'is_metric': is_metric,
            'oos_metric': oos_metric,
            'correlation': correlation,
            'p_value': p_value,
            'n_samples': len(valid_data),
            'assessment': assessment
        })
    
    return pd.DataFrame(correlations)

def analyze_by_trade_type(df):
    """Analyze correlations separately by trade type (long vs short)."""
    
    print("\n\n📊 CORRELATION ANALYSIS BY TRADE TYPE")
    print("=" * 80)
    
    for trade_type in ['long', 'short']:
        subset = df[df['trade_type'] == trade_type]
        if len(subset) < 10:
            print(f"\n{trade_type.upper()}: Insufficient data ({len(subset)} pairs)")
            continue
            
        print(f"\n{trade_type.upper()} TRADES ({len(subset)} pairs):")
        print("-" * 50)
        
        # Key metrics for trade type analysis
        key_metrics = [
            ('profit_factor', 'oos_profit_factor_2023', 'PF 2023'),
            ('profit_factor', 'oos_profit_factor_2024', 'PF 2024'),
            ('sharpe_ratio', 'oos_sharpe_ratio_2023', 'Sharpe 2023'),
            ('sharpe_ratio', 'oos_sharpe_ratio_2024', 'Sharpe 2024'),
        ]
        
        for is_metric, oos_metric, label in key_metrics:
            valid_mask = subset[is_metric].notna() & subset[oos_metric].notna()
            valid_data = subset[valid_mask]
            
            if len(valid_data) < 5:
                continue
                
            correlation, p_value = spearmanr(valid_data[is_metric], valid_data[oos_metric])
            print(f"  {label:<12}: ρ={correlation:6.3f} (p={p_value:.3f}, n={len(valid_data)})")

def create_correlation_matrix_plot(df, correlations_df):
    """Create correlation matrix heatmap."""
    
    # Prepare correlation matrix
    metrics_2023 = [
        ('profit_factor', 'oos_profit_factor_2023'),
        ('total_cumulative_return', 'oos_total_return_2023'),
        ('sharpe_ratio', 'oos_sharpe_ratio_2023'),
        ('max_drawdown', 'oos_max_drawdown_2023')
    ]
    
    metrics_2024 = [
        ('profit_factor', 'oos_profit_factor_2024'),
        ('total_cumulative_return', 'oos_total_return_2024'),
        ('sharpe_ratio', 'oos_sharpe_ratio_2024'),
        ('max_drawdown', 'oos_max_drawdown_2024')
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2023 correlations
    corr_matrix_2023 = []
    labels = ['Profit Factor', 'Total Return', 'Sharpe Ratio', 'Max Drawdown']
    
    for is_metric, oos_metric in metrics_2023:
        valid_mask = df[is_metric].notna() & df[oos_metric].notna()
        valid_data = df[valid_mask]
        if len(valid_data) >= 10:
            corr, _ = spearmanr(valid_data[is_metric], valid_data[oos_metric])
            corr_matrix_2023.append(corr)
        else:
            corr_matrix_2023.append(np.nan)
    
    # 2024 correlations
    corr_matrix_2024 = []
    for is_metric, oos_metric in metrics_2024:
        valid_mask = df[is_metric].notna() & df[oos_metric].notna()
        valid_data = df[valid_mask]
        if len(valid_data) >= 10:
            corr, _ = spearmanr(valid_data[is_metric], valid_data[oos_metric])
            corr_matrix_2024.append(corr)
        else:
            corr_matrix_2024.append(np.nan)
    
    # Plot 2023
    corr_2023 = np.array(corr_matrix_2023).reshape(1, -1)
    sns.heatmap(corr_2023, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=labels, yticklabels=['IS vs OOS'],
                cbar_kws={'label': 'Spearman Correlation'}, ax=ax1,
                vmin=-1, vmax=1)
    ax1.set_title('2023 Out-of-Sample Correlations', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Metrics', fontsize=12)
    
    # Plot 2024
    corr_2024 = np.array(corr_matrix_2024).reshape(1, -1)
    sns.heatmap(corr_2024, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=labels, yticklabels=['IS vs OOS'],
                cbar_kws={'label': 'Spearman Correlation'}, ax=ax2,
                vmin=-1, vmax=1)
    ax2.set_title('2024 Out-of-Sample Correlations', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Metrics', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, "rank_correlations_heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Correlation heatmap saved: {plot_path}")
    
    plt.show()

def create_scatter_plots(df):
    """Create scatter plots for key IS vs OOS relationships."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Key relationships to plot
    relationships = [
        ('profit_factor', 'oos_profit_factor_2023', 'Profit Factor: IS vs OOS 2023'),
        ('profit_factor', 'oos_profit_factor_2024', 'Profit Factor: IS vs OOS 2024'),
        ('sharpe_ratio', 'oos_sharpe_ratio_2023', 'Sharpe Ratio: IS vs OOS 2023'),
        ('sharpe_ratio', 'oos_sharpe_ratio_2024', 'Sharpe Ratio: IS vs OOS 2024'),
    ]
    
    for i, (is_metric, oos_metric, title) in enumerate(relationships):
        ax = axes[i//2, i%2]
        
        # Filter valid data
        valid_mask = df[is_metric].notna() & df[oos_metric].notna()
        valid_data = df[valid_mask]
        
        if len(valid_data) < 10:
            ax.text(0.5, 0.5, 'Insufficient Data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue
        
        # Scatter plot
        ax.scatter(valid_data[is_metric], valid_data[oos_metric], 
                  alpha=0.6, s=30, c='blue')
        
        # Add trend line
        z = np.polyfit(valid_data[is_metric], valid_data[oos_metric], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_data[is_metric].min(), valid_data[is_metric].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        
        # Compute correlation
        correlation, p_value = spearmanr(valid_data[is_metric], valid_data[oos_metric])
        
        # Labels and title
        ax.set_xlabel(f'In-Sample {is_metric.replace("_", " ").title()}')
        ax.set_ylabel(f'Out-of-Sample {oos_metric.replace("_", " ").title()}')
        ax.set_title(f'{title}\nρ = {correlation:.3f} (p = {p_value:.3f}, n = {len(valid_data)})')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, "rank_correlations_scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Scatter plots saved: {plot_path}")
    
    plt.show()

def save_correlation_results(correlations_df):
    """Save detailed correlation results to CSV."""
    
    output_path = os.path.join(OUTPUT_DIR, "rank_correlation_analysis.csv")
    correlations_df.to_csv(output_path, index=False)
    print(f"💾 Detailed results saved: {output_path}")

def print_summary_assessment(correlations_df):
    """Print overall assessment of signal vs noise."""
    
    print("\n\n🎯 SIGNAL vs NOISE ASSESSMENT")
    print("=" * 50)
    
    # Count assessments
    assessments = correlations_df['assessment'].value_counts()
    total = len(correlations_df)
    
    noise_count = assessments.get('NOISE ⚠️', 0)
    weak_count = assessments.get('WEAK', 0)
    moderate_plus = total - noise_count - weak_count
    
    print(f"Total correlations analyzed: {total}")
    print(f"NOISE correlations (|ρ| < 0.1): {noise_count} ({100*noise_count/total:.1f}%)")
    print(f"WEAK correlations (0.1 ≤ |ρ| < 0.3): {weak_count} ({100*weak_count/total:.1f}%)")
    print(f"MODERATE+ correlations (|ρ| ≥ 0.3): {moderate_plus} ({100*moderate_plus/total:.1f}%)")
    
    # Overall assessment
    if noise_count > total * 0.6:
        overall = "⚠️  HIGH RISK OF NOISE - Consider strategy revision"
    elif noise_count > total * 0.4:
        overall = "⚠️  MODERATE NOISE RISK - Some persistence but mixed signals"
    elif moderate_plus > total * 0.3:
        overall = "✅ EVIDENCE OF SIGNAL - Strategy shows persistence"
    else:
        overall = "🔍 MIXED RESULTS - Requires further investigation"
    
    print(f"\nOVERALL ASSESSMENT: {overall}")
    
    # Specific metric insights
    print(f"\n📋 KEY INSIGHTS:")
    
    # Profit Factor insights
    pf_correlations = correlations_df[correlations_df['metric'].str.contains('Profit Factor')]
    if len(pf_correlations) > 0:
        avg_pf_corr = pf_correlations['correlation'].abs().mean()
        print(f"• Profit Factor persistence: Average |ρ| = {avg_pf_corr:.3f}")
    
    # Sharpe Ratio insights
    sharpe_correlations = correlations_df[correlations_df['metric'].str.contains('Sharpe')]
    if len(sharpe_correlations) > 0:
        avg_sharpe_corr = sharpe_correlations['correlation'].abs().mean()
        print(f"• Sharpe Ratio persistence: Average |ρ| = {avg_sharpe_corr:.3f}")
    
    # Year comparison
    corr_2023 = correlations_df[correlations_df['metric'].str.contains('2023')]['correlation'].abs().mean()
    corr_2024 = correlations_df[correlations_df['metric'].str.contains('2024')]['correlation'].abs().mean()
    print(f"• 2023 OOS performance: Average |ρ| = {corr_2023:.3f}")
    print(f"• 2024 OOS performance: Average |ρ| = {corr_2024:.3f}")
    
    if corr_2023 > corr_2024 + 0.1:
        print("• Better persistence in 2023 - strategy may be degrading")
    elif corr_2024 > corr_2023 + 0.1:
        print("• Better persistence in 2024 - strategy may be improving")
    else:
        print("• Similar persistence across years")

def main():
    """Main execution function."""
    
    print("🔍 RANK CORRELATION ANALYSIS: IN-SAMPLE vs OUT-OF-SAMPLE")
    print("=" * 70)
    
    # Load data
    df = load_and_prepare_data()
    
    # Compute correlations
    correlations_df = compute_correlations(df)
    
    # Analyze by trade type
    analyze_by_trade_type(df)
    
    # Create visualizations
    create_correlation_matrix_plot(df, correlations_df)
    create_scatter_plots(df)
    
    # Save results
    save_correlation_results(correlations_df)
    
    # Print summary assessment
    print_summary_assessment(correlations_df)
    
    print(f"\n🎉 Rank correlation analysis completed!")

if __name__ == "__main__":
    main()