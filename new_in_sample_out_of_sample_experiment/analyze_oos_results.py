#!/usr/bin/env python3
"""
Out-of-Sample Results Analysis
==============================

Analyzes correlations between in-sample and out-of-sample performance metrics.
Creates comprehensive scatter plots with marginal distributions for:
- Sharpe Ratio correlation
- Total Cumulative Return correlation

Each figure shows:
- Central scatter plot with linear regression line and R²
- Marginal distributions on X and Y axes
- Connected and aligned layout
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_FILE = "results/out_of_sample/oos_validation_results.csv"
OUTPUT_DIR = "results/out_of_sample"

def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare the OOS validation results."""
    
    print("📊 Loading OOS validation results...")
    
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")
    
    df = pd.read_csv(RESULTS_FILE)
    
    print(f"✅ Loaded {len(df)} validation results")
    
    # Check required columns
    required_cols = ['is_total_cumulative_return', 'is_sharpe_ratio', 
                    'oos_total_cumulative_return', 'oos_sharpe_ratio']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove infinite values and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    initial_count = len(df)
    
    # Clean data for analysis
    df_clean = df[required_cols].dropna()
    
    print(f"📊 Data summary:")
    print(f"   Initial pairs: {initial_count}")
    print(f"   Clean pairs: {len(df_clean)}")
    print(f"   Removed: {initial_count - len(df_clean)} ({100*(initial_count - len(df_clean))/initial_count:.1f}%)")
    
    return df_clean


def create_correlation_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                           title: str, xlabel: str, ylabel: str, 
                           output_file: str) -> None:
    """Create scatter plot with marginal distributions."""
    
    print(f"🎨 Creating correlation plot: {title}")
    
    # Remove any remaining infinite or NaN values for this specific pair
    plot_data = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(plot_data) == 0:
        print(f"⚠️  No valid data for {title}")
        return
    
    x_data = plot_data[x_col].values
    y_data = plot_data[y_col].values
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 3, 0.3], height_ratios=[1, 3, 0.3],
                         hspace=0.05, wspace=0.05)
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 1])
    
    # Marginal distribution plots
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    
    # ---- Calculate optimal axis limits ----
    # Add padding to the data ranges for better visualization
    x_range = x_data.max() - x_data.min()
    y_range = y_data.max() - y_data.min()
    x_padding = x_range * 0.05  # 5% padding
    y_padding = y_range * 0.05  # 5% padding
    
    x_min = x_data.min() - x_padding
    x_max = x_data.max() + x_padding
    y_min = y_data.min() - y_padding
    y_max = y_data.max() + y_padding
    
    # ---- Main Scatter Plot ----
    # Create scatter plot with transparency
    ax_main.scatter(x_data, y_data, alpha=0.6, s=30, color='steelblue', edgecolors='white', linewidth=0.5)
    
    # Set axis limits based on data range
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    
    # Calculate and plot linear regression
    X = x_data.reshape(-1, 1)
    y = y_data
    
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    
    # Calculate R²
    r2 = r2_score(y, y_pred)
    
    # Calculate Pearson correlation
    correlation, p_value = stats.pearsonr(x_data, y_data)
    
    # Plot regression line
    sort_idx = np.argsort(x_data)
    ax_main.plot(x_data[sort_idx], y_pred[sort_idx], 'red', linewidth=2, alpha=0.8)
    
    # Add regression statistics text
    stats_text = f'R² = {r2:.3f}\nρ = {correlation:.3f}\np = {p_value:.1e}' if p_value < 0.001 else f'R² = {r2:.3f}\nρ = {correlation:.3f}\np = {p_value:.3f}'
    ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=11, weight='bold')
    
    # Add diagonal reference line (perfect correlation) only within visible range
    diag_min = max(x_min, y_min)
    diag_max = min(x_max, y_max)
    if diag_min < diag_max:  # Only draw if there's overlap
        ax_main.plot([diag_min, diag_max], [diag_min, diag_max], '--', color='gray', alpha=0.5, linewidth=1)
    
    # Formatting main plot
    ax_main.set_xlabel(xlabel, fontsize=12, weight='bold')
    ax_main.set_ylabel(ylabel, fontsize=12, weight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.tick_params(labelsize=10)
    
    # ---- Top Marginal Distribution (X-axis) ----
    # Create histogram bins that align with the main plot x-axis
    x_bins = np.linspace(x_min, x_max, 30)
    ax_top.hist(x_data, bins=x_bins, alpha=0.7, color='steelblue', density=True, edgecolor='white', linewidth=0.5)
    
    # Add KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde_x = gaussian_kde(x_data)
        x_kde_range = np.linspace(x_min, x_max, 100)
        ax_top.plot(x_kde_range, kde_x(x_kde_range), color='darkred', linewidth=2, alpha=0.8)
    except:
        pass
    
    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(labelsize=9)
    ax_top.grid(True, alpha=0.3)
    
    # Add statistics to top plot
    mean_x = np.mean(x_data)
    std_x = np.std(x_data)
    ax_top.axvline(mean_x, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax_top.text(0.98, 0.85, f'μ = {mean_x:.3f}\nσ = {std_x:.3f}', 
               transform=ax_top.transAxes, ha='right', va='top',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
               fontsize=9)
    
    # ---- Right Marginal Distribution (Y-axis) ----
    # Create histogram bins that align with the main plot y-axis
    y_bins = np.linspace(y_min, y_max, 30)
    ax_right.hist(y_data, bins=y_bins, alpha=0.7, color='lightcoral', density=True, 
                 orientation='horizontal', edgecolor='white', linewidth=0.5)
    
    # Add KDE overlay
    try:
        kde_y = gaussian_kde(y_data)
        y_kde_range = np.linspace(y_min, y_max, 100)
        ax_right.plot(kde_y(y_kde_range), y_kde_range, color='darkblue', linewidth=2, alpha=0.8)
    except:
        pass
    
    ax_right.set_xlabel('Density', fontsize=10)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(labelsize=9)
    ax_right.grid(True, alpha=0.3)
    
    # Add statistics to right plot
    mean_y = np.mean(y_data)
    std_y = np.std(y_data)
    ax_right.axhline(mean_y, color='blue', linestyle='--', alpha=0.8, linewidth=2)
    ax_right.text(0.85, 0.02, f'μ = {mean_y:.3f}\nσ = {std_y:.3f}', 
                 transform=ax_right.transAxes, ha='right', va='bottom',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                 fontsize=9, rotation=90)
    
    # ---- Overall Title ----
    fig.suptitle(title, fontsize=16, weight='bold', y=0.98)
    
    # Add sample size info
    fig.text(0.02, 0.02, f'n = {len(plot_data)} pairs', fontsize=10, alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Saved: {output_path}")
    print(f"   Correlation: ρ = {correlation:.3f} (R² = {r2:.3f})")
    print(f"   Sample size: n = {len(plot_data)}")


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for the analysis."""
    
    print("\n" + "="*60)
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    
    metrics = {
        'is_total_cumulative_return': 'In-Sample Total Return',
        'oos_total_cumulative_return': 'Out-of-Sample Total Return',
        'is_sharpe_ratio': 'In-Sample Sharpe Ratio',
        'oos_sharpe_ratio': 'Out-of-Sample Sharpe Ratio'
    }
    
    for col, label in metrics.items():
        if col in df.columns:
            data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            print(f"\n{label}:")
            print(f"   Count: {len(data)}")
            print(f"   Mean:  {data.mean():.4f}")
            print(f"   Std:   {data.std():.4f}")
            print(f"   Min:   {data.min():.4f}")
            print(f"   25%:   {data.quantile(0.25):.4f}")
            print(f"   50%:   {data.median():.4f}")
            print(f"   75%:   {data.quantile(0.75):.4f}")
            print(f"   Max:   {data.max():.4f}")
    
    # Correlation analysis
    print("\n" + "-"*40)
    print("📈 CORRELATION ANALYSIS")
    print("-"*40)
    
    # Total Return Correlation
    return_data = df[['is_total_cumulative_return', 'oos_total_cumulative_return']].replace([np.inf, -np.inf], np.nan).dropna()
    if len(return_data) > 1:
        corr_return, p_return = stats.pearsonr(return_data['is_total_cumulative_return'], 
                                              return_data['oos_total_cumulative_return'])
        print(f"Total Return IS-OOS Correlation:")
        print(f"   Pearson ρ: {corr_return:.4f}")
        print(f"   P-value:   {p_return:.6f}")
        print(f"   Sample:    {len(return_data)} pairs")
    
    # Sharpe Ratio Correlation  
    sharpe_data = df[['is_sharpe_ratio', 'oos_sharpe_ratio']].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sharpe_data) > 1:
        corr_sharpe, p_sharpe = stats.pearsonr(sharpe_data['is_sharpe_ratio'], 
                                              sharpe_data['oos_sharpe_ratio'])
        print(f"\nSharpe Ratio IS-OOS Correlation:")
        print(f"   Pearson ρ: {corr_sharpe:.4f}")
        print(f"   P-value:   {p_sharpe:.6f}")
        print(f"   Sample:    {len(sharpe_data)} pairs")


def main():
    """Main analysis function."""
    
    print("🔍 Out-of-Sample Results Analysis")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Load data
        df = load_and_prepare_data()
        
        # Create correlation plots
        print("\n🎨 Generating correlation visualizations...")
        
        # 1. Sharpe Ratio Correlation
        create_correlation_plot(
            df=df,
            x_col='is_sharpe_ratio',
            y_col='oos_sharpe_ratio', 
            title='In-Sample vs Out-of-Sample Sharpe Ratio Correlation',
            xlabel='In-Sample Sharpe Ratio',
            ylabel='Out-of-Sample Sharpe Ratio',
            output_file='sharpe_ratio_correlation_analysis.png'
        )
        
        # 2. Total Return Correlation
        create_correlation_plot(
            df=df,
            x_col='is_total_cumulative_return',
            y_col='oos_total_cumulative_return',
            title='In-Sample vs Out-of-Sample Total Return Correlation', 
            xlabel='In-Sample Total Cumulative Return',
            ylabel='Out-of-Sample Total Cumulative Return',
            output_file='total_return_correlation_analysis.png'
        )
        
        # Print summary statistics
        print_summary_statistics(df)
        
        print(f"\n✅ Analysis completed!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        print(f"📊 Generated correlation plots:")
        print(f"   - sharpe_ratio_correlation_analysis.png")
        print(f"   - total_return_correlation_analysis.png")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()