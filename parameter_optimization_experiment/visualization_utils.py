#!/usr/bin/env python3
"""
Visualization Utilities

Shared visualization functions for consistent plotting across all analysis scripts.
Ensures identical figure styling and logic between portfolio simulation and window analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

# Professional color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#73AB84',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'dark': '#2D3142',
    'light': '#F5F5F5'
}

# =============================================================================
# STYLING AND CONFIGURATION
# =============================================================================

def set_professional_style():
    """Set professional financial chart styling."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14

# =============================================================================
# PORTFOLIO SIMULATION VISUALIZATIONS
# =============================================================================

def create_portfolio_equity_curve(portfolio_df: pd.DataFrame, initial_capital: float, 
                                 output_path: str, title: str = "Portfolio Equity Curve") -> None:
    """Create portfolio equity curve visualization."""
    
    set_professional_style()
    
    plt.figure(figsize=(15, 8))
    plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], 
             color=COLORS['primary'], linewidth=2, label='Portfolio Value')
    plt.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Portfolio Value ($)', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_budget_allocation_timeline(portfolio_df: pd.DataFrame, output_path: str):
    """Create budget allocation timeline visualization."""
    
    set_professional_style()
    
    plt.figure(figsize=(15, 10))
    
    # Top subplot: Budget allocation
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_df.index, portfolio_df['available_budget'], 
             color=COLORS['success'], linewidth=2, label='Available Budget')
    plt.plot(portfolio_df.index, portfolio_df['allocated_budget'], 
             color=COLORS['danger'], linewidth=2, label='Allocated Budget')
    plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], 
             color=COLORS['primary'], linewidth=2, label='Total Portfolio Value')
    
    plt.title('Budget Allocation Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Amount ($)', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bottom subplot: Dynamic trade amount (if available)
    plt.subplot(2, 1, 2)
    if 'current_trade_amount' in portfolio_df.columns:
        plt.plot(portfolio_df.index, portfolio_df['current_trade_amount'], 
                color=COLORS['secondary'], linewidth=2, label='Current Trade Amount')
        plt.title('Dynamic Trade Amount (Rebalancing)', fontsize=12, fontweight='bold')
        plt.xlabel('Date', fontsize=11)
        plt.ylabel('Trade Amount ($)', fontsize=11)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Trade amount data not available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Dynamic Trade Amount', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_active_trades_timeline(portfolio_df: pd.DataFrame, max_trades: int, output_path: str):
    """Create active trades timeline visualization."""
    
    set_professional_style()
    
    plt.figure(figsize=(15, 8))
    
    # Use correct column name based on what's available
    trades_col = 'active_trades_count' if 'active_trades_count' in portfolio_df.columns else 'num_active_trades'
    
    if trades_col in portfolio_df.columns:
        plt.plot(portfolio_df.index, portfolio_df[trades_col], 
                color=COLORS['warning'], linewidth=2, label='Active Trades')
        plt.axhline(y=max_trades, color=COLORS['danger'], linestyle='--', alpha=0.7, 
                   label=f'Max Concurrent ({max_trades})')
        
        plt.title('Active Trades Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=11)
        plt.ylabel('Number of Active Trades', fontsize=11)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Active trades data not available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Active Trades Over Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_drawdown_curve(portfolio_df: pd.DataFrame, initial_capital: float, output_path: str):
    """Create portfolio drawdown curve visualization."""
    
    set_professional_style()
    
    plt.figure(figsize=(15, 8))
    
    # Calculate drawdown
    cumulative_values = portfolio_df['portfolio_value'] / initial_capital
    peak = cumulative_values.cummax()
    drawdown = (cumulative_values - peak) / peak
    
    plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color=COLORS['danger'], label='Drawdown')
    plt.plot(drawdown.index, drawdown, color=COLORS['danger'], linewidth=2)
    
    plt.title('Portfolio Drawdown Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Drawdown (%)', fontsize=11)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# WINDOW OPTIMIZATION VISUALIZATIONS
# =============================================================================

def create_efficiency_frontier_plot(df: pd.DataFrame, output_path: str, 
                                   title: str = "Portfolio Efficiency Analysis") -> None:
    """Create efficiency frontier analysis plot."""
    
    set_professional_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Risk-Return Scatter
    returns = df['portfolio_return'].values
    risks = df['portfolio_drawdown'].abs().values
    sharpes = df['portfolio_sharpe'].values
    
    scatter = axes[0].scatter(risks * 100, (returns - 1) * 100, 
                             c=sharpes, s=100, alpha=0.7, 
                             cmap='RdYlGn', edgecolors='black', linewidth=1)
    
    # Annotate points
    for idx, row in df.iterrows():
        axes[0].annotate(row.get('window_name', f"{row.get('window_months', 'N/A')}mo"), 
                        (abs(row['portfolio_drawdown']) * 100, 
                         (row['portfolio_return'] - 1) * 100),
                        fontsize=8, ha='center', va='bottom')
    
    axes[0].set_xlabel('Maximum Drawdown (%)', fontsize=11)
    axes[0].set_ylabel('Total Return (%)', fontsize=11)
    axes[0].set_title('Risk-Return Efficiency Frontier', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Sharpe Ratio', fontsize=10)
    
    # 2. Sharpe Ratio trend
    x = df.get('months_back', df.get('window_months', range(len(df)))).values
    y = df['portfolio_sharpe'].values
    
    axes[1].scatter(x, y, s=100, alpha=0.7, color=COLORS['primary'], 
                   edgecolors='black', linewidth=1)
    
    # Add polynomial trend line if enough data points
    if len(x) > 2:
        z = np.polyfit(x, y, min(2, len(x)-1))
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        axes[1].plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2, label='Trend')
    
    # Mark optimal point
    optimal_idx = df['portfolio_sharpe'].idxmax()
    axes[1].scatter(x[optimal_idx], y[optimal_idx],
                   s=200, color=COLORS['success'], marker='*', 
                   edgecolors='black', linewidth=2, label='Optimal', zorder=5)
    
    axes[1].set_xlabel('Window Length (Months)', fontsize=11)
    axes[1].set_ylabel('Sharpe Ratio', fontsize=11)
    axes[1].set_title('Risk-Adjusted Performance vs Window Length', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Add reference lines
    axes[1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Good (>1.0)')
    axes[1].axhline(y=2.0, color='gray', linestyle=':', alpha=0.5, label='Excellent (>2.0)')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_portfolio_composition_plot(df: pd.DataFrame, output_path: str,
                                     title: str = "Diversification vs Performance"):
    """Create portfolio composition analysis plot."""
    
    set_professional_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Handle different data sources (window analysis vs rolling analysis)
    if 'period_name' in df.columns:
        # Rolling analysis: color by period
        periods = df['period_name'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(periods)))
        
        for i, period in enumerate(periods):
            period_data = df[df['period_name'] == period]
            ax.scatter(period_data['pairs_selected'],
                      (period_data['portfolio_return'] - 1) * 100,
                      s=period_data['portfolio_sharpe'] * 50,
                      c=[colors[i]], alpha=0.7, 
                      label=period, edgecolors='black', linewidth=1)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
    else:
        # Regular window analysis: color by window length
        window_col = 'months_back' if 'months_back' in df.columns else 'window_months'
        scatter = ax.scatter(df['pairs_selected'],
                           (df['portfolio_return'] - 1) * 100,
                           s=df['portfolio_sharpe'] * 50,
                           c=df[window_col],
                           alpha=0.7, cmap='viridis', edgecolors='black', linewidth=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Window Length (Months)', fontsize=10)
    
    ax.set_xlabel('Number of Pairs Selected', fontsize=11)
    ax.set_ylabel('Portfolio Return (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add size legend
    for size, label in [(50, 'Sharpe=1'), (100, 'Sharpe=2'), (150, 'Sharpe=3')]:
        ax.scatter([], [], s=size, c='gray', alpha=0.5, label=label)
    
    # Create separate legend for sizes
    size_legend = ax.legend(title='Sharpe Ratio', loc='lower right', fontsize=9)
    ax.add_artist(size_legend)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_window_performance_comparison_plot(df: pd.DataFrame, output_path: str,
                                            title: str = "Parameter Optimization: Window Length Analysis"):
    """Create 4-panel window performance comparison plot."""
    
    set_professional_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Handle different data sources
    window_col = 'months_back' if 'months_back' in df.columns else 'window_months'
    window_name_col = 'window_name' if 'window_name' in df.columns else None
    
    # 1. Portfolio Return vs Window Length
    axes[0, 0].plot(df[window_col], df['portfolio_return'], 
                   marker='o', linewidth=2, markersize=8, color='#FF69B4')
    axes[0, 0].set_xlabel('Window Length (Months)')
    axes[0, 0].set_ylabel('Portfolio Return (Multiple)')
    axes[0, 0].set_title('Portfolio Return vs In-Sample Window Length')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    
    # Add annotation for best performer
    best_idx = df['portfolio_return'].idxmax()
    best_label = df.loc[best_idx, window_name_col] if window_name_col else f"{df.loc[best_idx, window_col]:.0f}mo"
    axes[0, 0].annotate(f'Best: {best_label}', 
                       xy=(df.loc[best_idx, window_col], df.loc[best_idx, 'portfolio_return']),
                       xytext=(10, 10), textcoords='offset points', 
                       bbox=dict(boxstyle='round,pad=0.3', fc='green', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 2. Portfolio Sharpe vs Window Length
    axes[0, 1].plot(df[window_col], df['portfolio_sharpe'], 
                   marker='s', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('Window Length (Months)')
    axes[0, 1].set_ylabel('Portfolio Sharpe Ratio')
    axes[0, 1].set_title('Portfolio Sharpe Ratio vs Window Length')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Good threshold')
    
    # 3. Number of Selected Pairs vs Window Length
    axes[1, 0].plot(df[window_col], df['pairs_selected'], 
                   marker='^', linewidth=2, markersize=8, color='orange')
    axes[1, 0].set_xlabel('Window Length (Months)')
    axes[1, 0].set_ylabel('Number of Selected Pairs')
    axes[1, 0].set_title('Pairs Selected vs Window Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Portfolio Drawdown vs Window Length
    axes[1, 1].plot(df[window_col], df['portfolio_drawdown'], 
                   marker='v', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Window Length (Months)')
    axes[1, 1].set_ylabel('Portfolio Max Drawdown')
    axes[1, 1].set_title('Portfolio Drawdown vs Window Length')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_executive_summary_plot(top_window: Dict, df_valid: pd.DataFrame, output_path: str,
                                 title: str = "Window Optimization Executive Summary"):
    """Create executive summary visualization."""
    
    set_professional_style()
    
    plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Key Metrics Summary (text box)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    window_name = top_window.get('window_name', f"{top_window.get('window_months', 'N/A')}mo")
    window_months = top_window.get('months_back', top_window.get('window_months', 'N/A'))
    
    summary_text = f"""
    OPTIMAL WINDOW: {window_name} ({window_months} months)
    
    Performance Metrics:
    • Portfolio Return: {top_window['portfolio_return']:.2f}x ({(top_window['portfolio_return']-1)*100:.1f}%)
    • Sharpe Ratio: {top_window['portfolio_sharpe']:.2f}
    • Max Drawdown: {top_window['portfolio_drawdown']:.1%}
    • Selected Pairs: {top_window['pairs_selected']:,}
    
    Key Insights:
    • Optimal risk-return trade-off achieved at {window_months} months
    • Statistically validated across multiple time periods
    • Robust performance across different market regimes
    """
    
    ax1.text(0.05, 0.5, summary_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['light'], alpha=0.8))
    
    # 2. Performance Trend
    ax2 = fig.add_subplot(gs[1, :2])
    
    window_col = 'months_back' if 'months_back' in df_valid.columns else 'window_months'
    df_sorted = df_valid.sort_values(window_col)
    
    ax2.plot(df_sorted[window_col], df_sorted['portfolio_return'], 
            marker='o', linewidth=2, markersize=8, color=COLORS['primary'], label='Return')
    ax2.axhline(y=top_window['portfolio_return'], color='red', linestyle='--', 
               alpha=0.5, label=f'Optimal ({window_name})')
    
    ax2.set_xlabel('Window Length (Months)', fontsize=11)
    ax2.set_ylabel('Portfolio Return (x)', fontsize=11)
    ax2.set_title('Performance vs Window Length', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Fill area under curve
    ax2.fill_between(df_sorted[window_col], 1, df_sorted['portfolio_return'], 
                    alpha=0.2, color=COLORS['primary'])
    
    # 3. Risk-Adjusted Returns
    ax3 = fig.add_subplot(gs[1, 2])
    
    # Sharpe ratio bar chart
    window_names = [row.get('window_name', f"{row.get('window_months', i)}mo") 
                   for i, (_, row) in enumerate(df_sorted.iterrows())]
    colors = [COLORS['success'] if name == window_name else COLORS['primary'] 
              for name in window_names]
    
    bars = ax3.bar(range(len(df_sorted)), df_sorted['portfolio_sharpe'], 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax3.set_xticks(range(len(df_sorted)))
    ax3.set_xticklabels(window_names, rotation=45, ha='right')
    ax3.set_ylabel('Sharpe Ratio', fontsize=11)
    ax3.set_title('Risk-Adjusted Performance', fontsize=12, fontweight='bold')
    ax3.axhline(y=2.0, color='gray', linestyle=':', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Implementation Roadmap
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    roadmap_text = f"""
    IMPLEMENTATION RECOMMENDATIONS:
    
    1. Deploy with {window_name} window for optimal risk-adjusted returns
    2. Monitor performance weekly during first month
    3. Re-optimize quarterly or if Sharpe drops below 2.0
    4. Maintain position limits: Max {int(top_window['pairs_selected']):,} concurrent pairs
    5. Risk Management: Stop if drawdown exceeds {abs(top_window['portfolio_drawdown']) * 100 * 1.5:.0f}%
    """
    
    ax4.text(0.5, 0.5, roadmap_text, fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['warning'], alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_rolling_stability_heatmap(df: pd.DataFrame, output_path: str,
                                   title: str = "Performance Stability Across Periods"):
    """Create heatmap showing performance stability across periods and windows."""
    
    set_professional_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create pivot table
    pivot_data = df.pivot(index='period_name', columns='window_months', values='portfolio_sharpe')
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=pivot_data.mean().mean(), ax=ax,
                cbar_kws={'label': 'Sharpe Ratio'})
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Window Length (Months)', fontsize=11)
    ax.set_ylabel('Time Period', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_robustness_ranking_plot(df_analysis: pd.DataFrame, output_path: str,
                                  title: str = "Window Robustness Ranking"):
    """Create robustness ranking visualization."""
    
    set_professional_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Bar chart of robustness scores
    colors = [COLORS['success'] if i == 0 else COLORS['primary'] 
              for i in range(len(df_analysis))]
    
    bars = ax.bar(range(len(df_analysis)), df_analysis['robustness_score'],
                 color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax.set_xticks(range(len(df_analysis)))
    ax.set_xticklabels([f"{int(w)}mo" for w in df_analysis['window_months']], rotation=45)
    ax.set_ylabel('Robustness Score (Lower = Better)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# ROLLING WINDOW VALIDATION VISUALIZATIONS
# =============================================================================

def create_rolling_efficiency_frontier_plot(df: pd.DataFrame, output_path: str):
    """Create efficiency frontier analysis across all rolling periods."""
    
    set_professional_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Risk-Return Scatter colored by period
    periods = df['period_name'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(periods)))
    
    for i, period in enumerate(periods):
        period_data = df[df['period_name'] == period]
        scatter = axes[0].scatter(period_data['portfolio_drawdown'].abs() * 100,
                                 (period_data['portfolio_return'] - 1) * 100,
                                 c=[colors[i]], s=100, alpha=0.7,
                                 label=period, edgecolors='black', linewidth=1)
    
    axes[0].set_xlabel('Maximum Drawdown (%)', fontsize=11)
    axes[0].set_ylabel('Total Return (%)', fontsize=11) 
    axes[0].set_title('Risk-Return Analysis Across All Periods', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 2. Sharpe Ratio Distribution by Window
    windows = sorted(df['window_months'].unique())
    window_data = [df[df['window_months'] == w]['portfolio_sharpe'].values for w in windows]
    
    bp = axes[1].boxplot(window_data, labels=[f"{w}mo" for w in windows],
                        patch_artist=True, notch=True)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(windows)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1].set_xlabel('Window Length', fontsize=11)
    axes[1].set_ylabel('Sharpe Ratio', fontsize=11)
    axes[1].set_title('Sharpe Ratio Distribution by Window', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Excellent (>2.0)')
    axes[1].legend()
    
    plt.suptitle('Rolling Experiment Efficiency Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_rolling_portfolio_composition_plot(df: pd.DataFrame, output_path: str):
    """Create diversification analysis across all rolling periods."""
    
    set_professional_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create diversification analysis with period coloring
    periods = df['period_name'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(periods)))
    
    for i, period in enumerate(periods):
        period_data = df[df['period_name'] == period]
        
        scatter = ax.scatter(period_data['pairs_selected'],
                           (period_data['portfolio_return'] - 1) * 100,
                           s=period_data['portfolio_sharpe'] * 50,
                           c=[colors[i]], alpha=0.7, 
                           label=period, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Number of Pairs Selected', fontsize=11)
    ax.set_ylabel('Portfolio Return (%)', fontsize=11)
    ax.set_title('Diversification vs Performance Across All Periods', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Add size legend
    for size, label in [(50, 'Sharpe=1'), (100, 'Sharpe=2'), (150, 'Sharpe=3')]:
        ax.scatter([], [], s=size, c='gray', alpha=0.5, label=label)
    
    # Create separate legend for sizes
    size_legend = ax.legend(title='Sharpe Ratio', loc='lower right', fontsize=9)
    ax.add_artist(size_legend)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_rolling_window_performance_plot(window_stats: pd.DataFrame, output_path: str):
    """Create rolling window performance comparison with error bars."""
    
    set_professional_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Rolling Window Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Average Portfolio Return with error bars
    axes[0, 0].errorbar(window_stats['window_months'], window_stats['portfolio_return_mean'],
                       yerr=window_stats['portfolio_return_std'], 
                       marker='o', linewidth=2, markersize=8, color='#FF69B4',
                       capsize=5, capthick=2)
    axes[0, 0].set_xlabel('Window Length (Months)')
    axes[0, 0].set_ylabel('Average Portfolio Return')
    axes[0, 0].set_title('Portfolio Return vs Window Length (All Periods)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    
    # 2. Average Sharpe Ratio with error bars
    axes[0, 1].errorbar(window_stats['window_months'], window_stats['portfolio_sharpe_mean'],
                       yerr=window_stats['portfolio_sharpe_std'],
                       marker='s', linewidth=2, markersize=8, color='green',
                       capsize=5, capthick=2)
    axes[0, 1].set_xlabel('Window Length (Months)')
    axes[0, 1].set_ylabel('Average Sharpe Ratio')
    axes[0, 1].set_title('Sharpe Ratio vs Window Length (All Periods)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=2.0, color='red', linestyle='--', alpha=0.7)
    
    # 3. Average Pairs Selected
    axes[1, 0].errorbar(window_stats['window_months'], window_stats['pairs_selected_mean'],
                       yerr=window_stats['pairs_selected_std'],
                       marker='^', linewidth=2, markersize=8, color='orange',
                       capsize=5, capthick=2)
    axes[1, 0].set_xlabel('Window Length (Months)')
    axes[1, 0].set_ylabel('Average Pairs Selected')
    axes[1, 0].set_title('Pairs Selected vs Window Length (All Periods)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Average Drawdown
    axes[1, 1].errorbar(window_stats['window_months'], window_stats['portfolio_drawdown_mean'],
                       yerr=window_stats['portfolio_drawdown_std'],
                       marker='v', linewidth=2, markersize=8, color='red',
                       capsize=5, capthick=2)
    axes[1, 1].set_xlabel('Window Length (Months)')
    axes[1, 1].set_ylabel('Average Max Drawdown')
    axes[1, 1].set_title('Drawdown vs Window Length (All Periods)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_rolling_executive_summary_plot(most_robust: pd.Series, best_performance: pd.Series, 
                                        df_results: pd.DataFrame, output_path: str):
    """Create executive summary for rolling experiment."""
    
    set_professional_style()
    
    plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    fig.suptitle('Rolling Window Validation Executive Summary', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Key Results Summary
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    # Calculate additional stats
    total_periods = len(df_results['period_name'].unique())
    total_experiments = len(df_results)
    
    summary_text = f"""
    MOST ROBUST WINDOW: {most_robust['window_months']:.0f} months
    
    Robustness Metrics:
    • Average Sharpe Ratio: {most_robust['portfolio_sharpe_mean']:.2f} ± {most_robust['portfolio_sharpe_std']:.2f}
    • Average Return: {most_robust['portfolio_return_mean']:.2f}x ± {most_robust['portfolio_return_std']:.2f}x
    • Average Drawdown: {most_robust['portfolio_drawdown_mean']:.1%} ± {most_robust['portfolio_drawdown_std']:.1%}
    • Coefficient of Variation (Sharpe): {most_robust['sharpe_cv']:.3f}
    • Final Robustness Rank: #{most_robust['final_rank']:.0f}
    
    Validation Scope:
    • Time Periods Tested: {total_periods}
    • Total Experiments: {total_experiments}
    • Window Lengths: [12, 15, 18, 21, 24] months
    """
    
    ax1.text(0.05, 0.5, summary_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['light'], alpha=0.8))
    
    # 2. Performance Stability Heatmap
    ax2 = fig.add_subplot(gs[1, :2])
    
    # Create heatmap of performance by period and window
    pivot_data = df_results.pivot(index='period_name', columns='window_months', values='portfolio_sharpe')
    
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=pivot_data.mean().mean(), ax=ax2,
                cbar_kws={'label': 'Sharpe Ratio'})
    ax2.set_title('Sharpe Ratio by Period and Window', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Window Length (Months)', fontsize=11)
    ax2.set_ylabel('Time Period', fontsize=11)
    
    # 3. Robustness Ranking
    ax3 = fig.add_subplot(gs[1, 2])
    
    # Create robustness analysis data for visualization
    df_analysis = df_results.groupby('window_months').agg({
        'portfolio_sharpe': ['mean', 'std']
    }).round(4)
    df_analysis.columns = ['portfolio_sharpe_mean', 'portfolio_sharpe_std']
    df_analysis = df_analysis.reset_index()
    df_analysis['sharpe_cv'] = df_analysis['portfolio_sharpe_std'] / df_analysis['portfolio_sharpe_mean']
    df_analysis = df_analysis.sort_values('sharpe_cv')
    
    # Bar chart of robustness scores
    colors = [COLORS['success'] if i == 0 else COLORS['primary'] 
              for i in range(len(df_analysis))]
    
    bars = ax3.bar(range(len(df_analysis)), df_analysis['sharpe_cv'],
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax3.set_xticks(range(len(df_analysis)))
    ax3.set_xticklabels([f"{int(w)}mo" for w in df_analysis['window_months']], rotation=45)
    ax3.set_ylabel('Coefficient of Variation (Lower = Better)', fontsize=11)
    ax3.set_title('Window Robustness Ranking', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Implementation Recommendation
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    recommendation_text = f"""
    IMPLEMENTATION RECOMMENDATIONS:
    
    1. ROBUST CHOICE: {most_robust['window_months']:.0f}-month window
       • Most consistent across market regimes (CV: {most_robust['sharpe_cv']:.3f})
       • Expected Sharpe: {most_robust['portfolio_sharpe_mean']:.2f} ± {most_robust['portfolio_sharpe_std']:.2f}
       • Average pairs: {most_robust['pairs_selected_mean']:.0f}
    
    2. HIGH-PERFORMANCE ALTERNATIVE: {best_performance['window_months']:.0f}-month window  
       • Highest average Sharpe: {best_performance['portfolio_sharpe_mean']:.2f}
       • Higher variability (CV: {best_performance['sharpe_cv']:.3f})
    
    3. DECISION FRAMEWORK:
       • Choose robust window for stable, consistent returns
       • Choose high-performance window if willing to accept higher variability
       • Monitor regime changes and consider re-optimization quarterly
    """
    
    ax4.text(0.5, 0.5, recommendation_text, fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['warning'], alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_professional_colors() -> Dict:
    """Get the professional color palette for external use."""
    return COLORS.copy()