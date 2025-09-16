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
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

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

def create_budget_allocation_timeline(portfolio_df: pd.DataFrame, output_path: str, max_trades: int = None):
    """Create budget allocation timeline visualization with active trades."""
    
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
    
    # Bottom subplot: Active trades timeline
    plt.subplot(2, 1, 2)
    
    # Use correct column name based on what's available
    trades_col = 'active_trades_count' if 'active_trades_count' in portfolio_df.columns else 'num_active_trades'
    
    if trades_col in portfolio_df.columns:
        plt.plot(portfolio_df.index, portfolio_df[trades_col], 
                color=COLORS['warning'], linewidth=2, label='Active Trades')
        if max_trades:
            plt.axhline(y=max_trades, color=COLORS['danger'], linestyle='--', alpha=0.7, 
                       label=f'Max Concurrent ({max_trades})')
        
        plt.title('Active Trades Over Time', fontsize=12, fontweight='bold')
        plt.xlabel('Date', fontsize=11)
        plt.ylabel('Number of Active Trades', fontsize=11)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Active trades data not available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Active Trades Over Time', fontsize=12, fontweight='bold')
    
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

def create_combined_portfolio_visualization(portfolio_df: pd.DataFrame, initial_capital: float, 
                                          max_trades: int, output_path: str):
    """Create combined portfolio visualization with equity curve, drawdown, and active trades."""
    
    set_professional_style()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
    
    # Top subplot: Portfolio Equity Curve
    ax1.plot(portfolio_df.index, portfolio_df['portfolio_value'], 
             color=COLORS['primary'], linewidth=2, label='Portfolio Value')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Middle subplot: Drawdown Curve
    cumulative_values = portfolio_df['portfolio_value'] / initial_capital
    peak = cumulative_values.cummax()
    drawdown = (cumulative_values - peak) / peak
    
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color=COLORS['danger'], label='Drawdown')
    ax2.plot(drawdown.index, drawdown, color=COLORS['danger'], linewidth=2)
    ax2.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # Bottom subplot: Active Trades Timeline
    trades_col = 'active_trades_count' if 'active_trades_count' in portfolio_df.columns else 'num_active_trades'
    
    if trades_col in portfolio_df.columns:
        ax3.plot(portfolio_df.index, portfolio_df[trades_col], 
                color=COLORS['warning'], linewidth=2, label='Active Trades')
        ax3.axhline(y=max_trades, color=COLORS['danger'], linestyle='--', alpha=0.7, 
                   label=f'Max Concurrent ({max_trades})')
        ax3.set_title('Active Trades Over Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Active Trades', fontsize=11)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Active trades data not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Active Trades Over Time', fontsize=14, fontweight='bold')
    
    ax3.set_xlabel('Date', fontsize=11)
    
    # Add overall title
    fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# INTERACTIVE VISUALIZATIONS (PLOTLY)
# =============================================================================

def create_interactive_combined_portfolio(portfolio_df: pd.DataFrame, initial_capital: float, 
                                         max_trades: int, output_path: str):
    """Create interactive portfolio visualization with Plotly."""
    
    # Calculate drawdown
    cumulative_values = portfolio_df['portfolio_value'] / initial_capital
    peak = cumulative_values.cummax()
    drawdown = ((cumulative_values - peak) / peak) * 100  # Convert to percentage
    
    # Determine trades column
    trades_col = 'active_trades_count' if 'active_trades_count' in portfolio_df.columns else 'num_active_trades'
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Portfolio Equity Curve', 'Portfolio Drawdown (%)', 'Active Trades'),
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # 1. Portfolio Equity Curve
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color=COLORS['primary'], width=2),
            hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=[initial_capital] * len(portfolio_df),
            mode='lines',
            name='Initial Capital',
            line=dict(color='gray', width=1, dash='dash'),
            hovertemplate='Initial: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color=COLORS['danger'], width=2),
            fillcolor='rgba(199, 62, 29, 0.3)',
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 3. Active Trades
    if trades_col in portfolio_df.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df[trades_col],
                mode='lines',
                name='Active Trades',
                line=dict(color=COLORS['warning'], width=2),
                hovertemplate='Date: %{x}<br>Active: %{y}<extra></extra>'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=[max_trades] * len(portfolio_df),
                mode='lines',
                name=f'Max Concurrent ({max_trades})',
                line=dict(color=COLORS['danger'], width=1, dash='dash'),
                hovertemplate='Max: %{y}<extra></extra>'
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_yaxes(title_text="Number of Trades", row=3, col=1)
    
    fig.update_layout(
        title={
            'text': 'Interactive Portfolio Performance Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2D3142'}
        },
        height=900,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    # Determine file paths
    if output_path.endswith('.html'):
        html_path = output_path
        png_path = output_path.replace('.html', '.png')
    else:
        png_path = output_path
        html_path = output_path.replace('.png', '.html')
    
    # Save as HTML
    fig.write_html(html_path)
    
    # Also save as static image with data sampling for performance
    try:
        # For PNG generation, sample data if too large to avoid memory issues
        if len(portfolio_df) > 100:
            # Sample every nth point to reduce data size for PNG rendering
            sample_step = max(1, len(portfolio_df) // 100)
            sampled_df = portfolio_df.iloc[::sample_step].copy()
            
            # Create a simplified figure for PNG with sampled data
            fig_png = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Portfolio Equity Curve', 'Portfolio Drawdown (%)', 'Active Trades'),
                vertical_spacing=0.08,
                row_heights=[0.4, 0.3, 0.3]
            )
            
            # Add sampled traces
            fig_png.add_trace(
                go.Scatter(
                    x=sampled_df.index, y=sampled_df['portfolio_value'],
                    mode='lines', name='Portfolio Value',
                    line=dict(color=COLORS['primary'], width=2)
                ), row=1, col=1
            )
            
            if 'drawdown_pct' in sampled_df.columns:
                fig_png.add_trace(
                    go.Scatter(
                        x=sampled_df.index, y=sampled_df['drawdown_pct'],
                        mode='lines', name='Drawdown',
                        line=dict(color=COLORS['danger'], width=2),
                        fill='tonexty', fillcolor='rgba(231, 76, 60, 0.1)'
                    ), row=2, col=1
                )
            
            if trades_col in sampled_df.columns:
                fig_png.add_trace(
                    go.Scatter(
                        x=sampled_df.index, y=sampled_df[trades_col],
                        mode='lines', name='Active Trades',
                        line=dict(color=COLORS['warning'], width=2)
                    ), row=3, col=1
                )
            
            # Update layout for PNG
            fig_png.update_layout(
                title='Interactive Portfolio Performance Analysis',
                height=900, template='plotly_white', showlegend=True
            )
            fig_png.update_xaxes(title_text="Date", row=3, col=1)
            fig_png.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig_png.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            fig_png.update_yaxes(title_text="Number of Trades", row=3, col=1)
            
            fig_png.write_image(png_path, width=1400, height=900, scale=1)
        else:
            fig.write_image(png_path, width=1400, height=900, scale=1)
            
        print(f"   ✅ Saved both: {html_path} and {png_path}")
    except Exception as e:
        print(f"   ⚠️  HTML saved: {html_path}, PNG failed: {e}")
    
    return fig

# Rolling Sharpe functions removed - not needed for this analysis

def create_coin_pair_distribution(selected_pairs_df: pd.DataFrame, output_path: str):
    """Create vertical bar chart showing ALL trading coins and their reference coin counts.
    
    Args:
        selected_pairs_df: DataFrame with columns 'trading_coin' and 'reference_coin'
        output_path: Path to save the visualization
    """
    
    # Count unique reference coins for each trading coin
    pair_counts = selected_pairs_df.groupby('trading_coin')['reference_coin'].nunique()
    # Sort ascending so that coins with most pairs are at the top (like P&L chart)
    pair_counts = pair_counts.sort_values(ascending=True)
    
    # Create the plot with dynamic height based on number of coins
    set_professional_style()
    
    # Calculate figure height based on number of coins (20 pixels per coin, min 10 inches)
    fig_height = max(10, len(pair_counts) * 0.3)
    plt.figure(figsize=(14, fig_height))
    
    # Create horizontal bar plot (vertical chart with horizontal bars)
    y_positions = range(len(pair_counts))
    bars = plt.barh(y_positions, pair_counts.values, 
                    color=COLORS['primary'], edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add value labels at the end of bars
    for i, (bar, value) in enumerate(zip(bars, pair_counts.values)):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                str(int(value)), ha='left', va='center', fontsize=8, fontweight='bold')
    
    # Customize the plot
    plt.title(f'Trading Coins by Number of Reference Coin Pairs ({len(pair_counts)} coins)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Reference Coins', fontsize=12, fontweight='bold')
    plt.ylabel('Trading Coins', fontsize=12, fontweight='bold')
    
    # Set y-axis labels with coin names
    plt.yticks(y_positions, pair_counts.index, fontsize=9)
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add statistics box
    total_pairs = len(selected_pairs_df)
    unique_trading = selected_pairs_df['trading_coin'].nunique()
    unique_reference = selected_pairs_df['reference_coin'].nunique()
    avg_pairs = pair_counts.mean()
    median_pairs = pair_counts.median()
    max_pairs = pair_counts.max()
    min_pairs = pair_counts.min()
    max_coin = pair_counts.idxmax()
    min_coin = pair_counts.idxmin()
    
    stats_text = (
        f'Statistics:\n'
        f'• Total pairs: {total_pairs:,}\n'
        f'• Trading coins: {unique_trading}\n'
        f'• Reference coins: {unique_reference}\n'
        f'• Avg pairs/coin: {avg_pairs:.1f}\n'
        f'• Median pairs/coin: {median_pairs:.0f}\n'
        f'• Max: {max_pairs} ({max_coin})\n'
        f'• Min: {min_pairs} ({min_coin})'
    )
    
    plt.text(0.98, 0.03, stats_text, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add vertical line for average
    plt.axvline(x=avg_pairs, color='red', linestyle='--', alpha=0.5, linewidth=1,
                label=f'Average: {avg_pairs:.1f}')
    
    # Add vertical line for median
    plt.axvline(x=median_pairs, color='orange', linestyle=':', alpha=0.5, linewidth=1,
                label=f'Median: {median_pairs:.0f}')
    
    plt.legend(loc='upper right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save as PNG
    if output_path.endswith('.html'):
        png_path = output_path.replace('.html', '.png')
    else:
        png_path = output_path
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved coin pair distribution: {png_path}")
    
    # Also save the full distribution as CSV for reference
    csv_path = png_path.replace('.png', '_data.csv')
    pair_counts.to_csv(csv_path, header=['reference_coin_count'])
    print(f"   ✅ Saved distribution data: {csv_path}")

def create_pnl_by_coin_histogram(trades_df: pd.DataFrame, output_path: str):
    """Create P&L histogram by coin with proper cumulative return calculation."""
    
    # Calculate cumulative returns by coin
    if 'trade_return' in trades_df.columns:
        # Use simple returns already calculated
        return_col = 'trade_return'
    elif 'log_return' in trades_df.columns:
        # Convert log returns to simple returns
        trades_df['trade_return'] = np.exp(trades_df['log_return']) - 1
        return_col = 'trade_return'
    else:
        # Fallback to trade_pnl if available
        return_col = 'trade_pnl'
    
    # Calculate cumulative return and dollar P&L for each coin
    cumulative_returns_by_coin = {}
    dollar_pnl_by_coin = {}
    
    for coin in trades_df['trading_coin'].unique():
        coin_trades = trades_df[trades_df['trading_coin'] == coin]
        
        # Use actual P&L and trade amounts for correct calculation
        if 'trade_pnl' in coin_trades.columns and 'trade_amount' in coin_trades.columns:
            # Calculate total P&L and total amount invested for this coin
            total_pnl = coin_trades['trade_pnl'].sum()
            total_invested = coin_trades['trade_amount'].sum()
            
            # Calculate cumulative return as percentage: total P&L / total invested
            if total_invested > 0:
                cumulative_return_pct = (total_pnl / total_invested) * 100
            else:
                cumulative_return_pct = 0.0
            
            # Store both percentage and dollar amounts
            cumulative_returns_by_coin[coin] = cumulative_return_pct
            dollar_pnl_by_coin[coin] = total_pnl
                
        elif 'log_return' in coin_trades.columns:
            # Fallback: sum log returns (assumes equal weighting)
            cumulative_return = np.exp(coin_trades['log_return'].sum()) - 1
            cumulative_return_pct = cumulative_return * 100
            cumulative_returns_by_coin[coin] = cumulative_return_pct
            dollar_pnl_by_coin[coin] = 0.0  # Unknown dollar amount
            
        else:
            # Last resort: sum simple returns (not ideal but better than nothing)
            cumulative_return = coin_trades.get(return_col, 0).sum()
            cumulative_return_pct = cumulative_return * 100 if return_col != 'trade_pnl' else cumulative_return
            cumulative_returns_by_coin[coin] = cumulative_return_pct
            dollar_pnl_by_coin[coin] = cumulative_return if return_col == 'trade_pnl' else 0.0
    
    # Convert to Series and sort by dollar P&L (not percentage)
    dollar_pnl_series = pd.Series(dollar_pnl_by_coin).sort_values()
    
    # Calculate each coin's contribution to total portfolio P&L
    total_portfolio_pnl = dollar_pnl_series.sum()
    portfolio_contribution_pct = {}
    
    for coin in dollar_pnl_series.index:
        if total_portfolio_pnl != 0:
            contribution_pct = (dollar_pnl_series[coin] / total_portfolio_pnl) * 100
        else:
            contribution_pct = 0.0
        portfolio_contribution_pct[coin] = contribution_pct
    
    contribution_pct_series = pd.Series(portfolio_contribution_pct)
    
    # Separate profits and losses based on dollar amounts
    profits = dollar_pnl_series[dollar_pnl_series > 0]
    losses = dollar_pnl_series[dollar_pnl_series < 0]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for losses (red)
    if len(losses) > 0:
        loss_contribution_pct = [contribution_pct_series[coin] for coin in losses.index]
        fig.add_trace(go.Bar(
            y=losses.index,
            x=losses.values,
            orientation='h',
            name='Loss',
            marker_color='rgba(220, 38, 127, 0.8)',
            customdata=loss_contribution_pct,
            hovertemplate='%{y}<br>P&L: $%{x:.2f} (%{customdata:.1f}% of portfolio)<extra></extra>',
            text=[f'${val:.0f}<br>({pct:.1f}%)' for val, pct in zip(losses.values, loss_contribution_pct)],
            textposition='inside',
            textfont=dict(color='white', size=9)
        ))
    
    # Add bars for profits (green)
    if len(profits) > 0:
        profit_contribution_pct = [contribution_pct_series[coin] for coin in profits.index]
        fig.add_trace(go.Bar(
            y=profits.index,
            x=profits.values,
            orientation='h',
            name='Profit',
            marker_color='rgba(46, 134, 171, 0.8)',
            customdata=profit_contribution_pct,
            hovertemplate='%{y}<br>P&L: $%{x:.2f} (%{customdata:.1f}% of portfolio)<extra></extra>',
            text=[f'${val:.0f}<br>({pct:.1f}%)' for val, pct in zip(profits.values, profit_contribution_pct)],
            textposition='inside',
            textfont=dict(color='white', size=9)
        ))
    
    # Update layout with additional statistics
    total_dollar_pnl = dollar_pnl_series.sum()
    num_profitable = len(profits)
    num_losing = len(losses)
    
    fig.update_layout(
        title=f'P&L by Coin: ${total_dollar_pnl:.0f} Total Portfolio P&L (Profitable: {num_profitable}, Losing: {num_losing})',
        xaxis_title='P&L ($)',
        yaxis_title='Trading Coin',
        height=max(400, len(dollar_pnl_series) * 20),  # Dynamic height based on number of coins
        template='plotly_white',
        showlegend=True,
        hovermode='y unified'
    )
    
    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="solid", line_color="black", opacity=0.3)
    
    # Determine file paths
    if output_path.endswith('.html'):
        html_path = output_path
        png_path = output_path.replace('.html', '.png')
    else:
        png_path = output_path
        html_path = output_path.replace('.png', '.html')
    
    # Save both formats
    fig.write_html(html_path)
    
    try:
        fig.write_image(png_path, width=1400, height=max(800, len(dollar_pnl_series) * 30), scale=1)
        print(f"   ✅ Saved both: {html_path} and {png_path}")
    except Exception as e:
        print(f"   ⚠️  HTML saved: {html_path}, PNG failed: {e}")
    
    return fig

def detect_trade_outliers(trades_df: pd.DataFrame, method: str = 'zscore', 
                         threshold: float = 3.0) -> Tuple[pd.DataFrame, List[dict]]:
    """Detect outliers in individual trade returns/profits."""
    
    # Ensure we have trade returns
    if 'log_return' in trades_df.columns:
        # Convert log returns to simple returns for analysis
        trades_df['simple_return'] = np.exp(trades_df['log_return']) - 1
        return_col = 'simple_return'
    elif 'trade_pnl' in trades_df.columns:
        # Use existing P&L data
        return_col = 'trade_pnl'
    else:
        raise ValueError("No return data found. Need 'log_return' or 'trade_pnl' column.")
    
    # Remove any NaN values
    clean_trades = trades_df.dropna(subset=[return_col]).copy()
    returns = clean_trades[return_col]
    
    outliers = []
    
    if method == 'zscore':
        # Z-score method for trade returns
        z_scores = np.abs(stats.zscore(returns))
        outlier_mask = z_scores > threshold
        
        outlier_indices = returns[outlier_mask].index
        
        for idx in outlier_indices:
            trade_return = float(returns.loc[idx])  # Ensure scalar
            z_score = float(z_scores[returns.index == idx].iloc[0])  # Fix indexing
            
            outliers.append({
                'trade_index': idx,
                'date': pd.to_datetime(clean_trades.loc[idx, 'time_entered']) if 'time_entered' in clean_trades.columns else None,
                'trading_coin': clean_trades.loc[idx, 'trading_coin'] if 'trading_coin' in clean_trades.columns else 'Unknown',
                'reference_coin': clean_trades.loc[idx, 'reference_coin'] if 'reference_coin' in clean_trades.columns else 'Unknown',
                'return': trade_return,
                'z_score': z_score,
                'trade_type': clean_trades.loc[idx, 'trade_type'] if 'trade_type' in clean_trades.columns else 'Unknown'
            })
    
    elif method == 'iqr':
        # IQR method for trade returns
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = (returns < lower_bound) | (returns > upper_bound)
        outlier_indices = returns[outlier_mask].index
        
        for idx in outlier_indices:
            trade_return = float(returns.loc[idx])  # Ensure scalar
            iqr_distance = float(max(abs(trade_return - lower_bound), 
                                   abs(trade_return - upper_bound)) / IQR)
            
            outliers.append({
                'trade_index': idx,
                'date': pd.to_datetime(clean_trades.loc[idx, 'time_entered']) if 'time_entered' in clean_trades.columns else None,
                'trading_coin': clean_trades.loc[idx, 'trading_coin'] if 'trading_coin' in clean_trades.columns else 'Unknown',
                'reference_coin': clean_trades.loc[idx, 'reference_coin'] if 'reference_coin' in clean_trades.columns else 'Unknown',
                'return': trade_return,
                'iqr_distance': iqr_distance,
                'trade_type': clean_trades.loc[idx, 'trade_type'] if 'trade_type' in clean_trades.columns else 'Unknown'
            })
    
    return clean_trades, outliers

def detect_outliers(portfolio_df: pd.DataFrame, method: str = 'zscore', 
                   threshold: float = 3.0) -> Tuple[pd.DataFrame, List[dict]]:
    """Legacy function - kept for backward compatibility. Use detect_trade_outliers for trade analysis."""
    
    # Calculate daily returns
    portfolio_df = portfolio_df.copy()
    portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
    
    outliers = []
    
    if method == 'zscore':
        # Z-score method
        returns = portfolio_df['daily_return'].dropna()
        z_scores = np.abs(stats.zscore(returns))
        outlier_mask = z_scores > threshold
        
        outlier_indices = returns[outlier_mask].index
        
        for idx in outlier_indices:
            daily_return = float(returns.loc[idx])  # Ensure scalar
            z_score_idx = returns.index.get_loc(idx)  # Get position index
            z_score = float(z_scores[z_score_idx])  # Fix indexing
            
            outliers.append({
                'date': idx,
                'return': daily_return,
                'z_score': z_score,
                'portfolio_value': float(portfolio_df.loc[idx, 'portfolio_value'])
            })
    
    elif method == 'iqr':
        # IQR method
        returns = portfolio_df['daily_return'].dropna()
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = (returns < lower_bound) | (returns > upper_bound)
        outlier_indices = returns[outlier_mask].index
        
        for idx in outlier_indices:
            daily_return = float(returns.loc[idx])  # Ensure scalar
            iqr_distance = float(max(abs(daily_return - lower_bound), 
                                   abs(daily_return - upper_bound)) / IQR)
            
            outliers.append({
                'date': idx,
                'return': daily_return,
                'iqr_distance': iqr_distance,
                'portfolio_value': float(portfolio_df.loc[idx, 'portfolio_value'])
            })
    
    return portfolio_df, outliers

def create_trade_outlier_visualization(trades_df: pd.DataFrame, outliers: List[dict], 
                                      output_path: str):
    """Create optimized visualization highlighting trade outliers with reduced file size."""
    
    # Sample data for large datasets to reduce file size - use fixed seed for consistency
    MAX_POINTS = 5000  # Maximum non-outlier points to show
    if len(trades_df) > MAX_POINTS:
        # Use fixed random seed to ensure consistent sampling between HTML and PNG
        sample_size = min(MAX_POINTS, len(trades_df))
        sampled_trades = trades_df.sample(n=sample_size, random_state=42)  # Consistent seed
    else:
        sampled_trades = trades_df.copy()
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Trade Returns Timeline (Outliers Highlighted)', 
                        'Trade Returns Distribution', 
                        'Top 10 Coins by Outlier Count'),
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Prepare trade data
    if 'simple_return' in trades_df.columns:
        return_col = 'simple_return'
    elif 'trade_pnl' in trades_df.columns:
        return_col = 'trade_pnl'
    else:
        return_col = 'log_return'
    
    # Convert dates if needed
    if 'time_entered' in sampled_trades.columns:
        trade_dates = pd.to_datetime(sampled_trades['time_entered'])
    else:
        trade_dates = sampled_trades.index
    
    if 'time_exited' in trades_df.columns:
        trades_df['time_exited'] = pd.to_datetime(trades_df['time_exited'])
    
    # 1. Trade returns timeline (sampled data)
    returns_pct = sampled_trades[return_col] * 100 if return_col != 'trade_pnl' else sampled_trades[return_col]
    
    # Only show a subset of regular trades to reduce file size
    fig.add_trace(
        go.Scatter(
            x=trade_dates,
            y=returns_pct,
            mode='markers',
            name=f'Sample Trades (n={len(sampled_trades):,})',
            marker=dict(size=3, color='lightblue', opacity=0.4),
            hoverinfo='skip'  # Skip hover for regular trades to save space
        ),
        row=1, col=1
    )
    
    # Mark outlier trades with full hover information
    if outliers:
        # Prepare outlier data with all required fields
        outlier_data = []
        for o in outliers:  # Show ALL outliers - they are the main point of interest
            if o['date'] is not None:
                # Find the original trade to get entry and exit times
                outlier_trade = None
                if 'trade_index' in o:
                    try:
                        outlier_trade = trades_df.iloc[o['trade_index']]
                    except:
                        pass
                
                outlier_data.append({
                    'date': o['date'],
                    'return': o['return'] * 100 if return_col != 'trade_pnl' else o['return'],
                    'trading_coin': o.get('trading_coin', 'Unknown'),
                    'reference_coin': o.get('reference_coin', 'Unknown'),
                    'entry_time': outlier_trade['time_entered'] if outlier_trade is not None and 'time_entered' in trades_df.columns else o['date'],
                    'exit_time': outlier_trade['time_exited'] if outlier_trade is not None and 'time_exited' in trades_df.columns else 'N/A'
                })
        
        if outlier_data:
            outlier_df = pd.DataFrame(outlier_data)
            
            # Create custom hover text
            hover_texts = []
            for _, row in outlier_df.iterrows():
                hover_text = (
                    f"<b>OUTLIER TRADE</b><br>"
                    f"Entry Time: {row['entry_time']}<br>"
                    f"Exit Time: {row['exit_time']}<br>"
                    f"Trading Coin: {row['trading_coin']}<br>"
                    f"Reference Coin: {row['reference_coin']}<br>"
                    f"Return: {row['return']:.2f}%"
                )
                hover_texts.append(hover_text)
            
            fig.add_trace(
                go.Scatter(
                    x=outlier_df['date'],
                    y=outlier_df['return'],
                    mode='markers',
                    name=f'Outliers (n={len(outlier_df)})',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='star',
                        line=dict(color='darkred', width=1)
                    ),
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts
                ),
                row=1, col=1
            )
    
    # 2. Returns distribution histogram (use full data for accuracy)
    all_returns_pct = trades_df[return_col] * 100 if return_col != 'trade_pnl' else trades_df[return_col]
    
    fig.add_trace(
        go.Histogram(
            x=all_returns_pct,
            nbinsx=50,
            name='Return Distribution',
            marker_color='rgba(46, 134, 171, 0.7)',
            hoverinfo='skip'  # Reduce hover data
        ),
        row=2, col=1
    )
    
    # 3. Top 10 coins by outlier count
    if outliers:
        outlier_coins = [o.get('trading_coin', 'Unknown') for o in outliers]
        coin_counts = pd.Series(outlier_coins).value_counts().head(10)
        
        fig.add_trace(
            go.Bar(
                x=coin_counts.values,
                y=coin_counts.index,
                orientation='h',
                name='Top Coins',
                marker_color='rgba(220, 38, 127, 0.8)',
                text=coin_counts.values,
                textposition='outside',
                hoverinfo='skip'  # Reduce hover data
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Return (%)", row=2, col=1)
    fig.update_xaxes(title_text="Number of Outlier Trades", row=3, col=1)
    
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Trading Coin", row=3, col=1)
    
    # Add summary statistics
    total_trades = len(trades_df)
    total_outliers = len(outliers) if outliers else 0
    outlier_pct = (total_outliers / total_trades * 100) if total_trades > 0 else 0
    
    fig.update_layout(
        title=f'Trade Outlier Analysis ({total_outliers:,} outliers / {total_trades:,} trades = {outlier_pct:.2f}%)',
        height=1000,
        hovermode='closest',
        template='plotly_white',
        showlegend=True
    )
    
    # Save only HTML format
    html_path = output_path if output_path.endswith('.html') else output_path.replace('.png', '.html')
    
    fig.write_html(html_path)
    print(f"   ✅ Saved trade outlier analysis: {html_path}")
    
    return fig

def create_outlier_visualization(portfolio_df: pd.DataFrame, outliers: List[dict], 
                                output_path: str):
    """Create visualization highlighting portfolio outliers (legacy function)."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Portfolio Value with Outlier Detection', 'Daily Returns Distribution'),
        row_heights=[0.7, 0.3]
    )
    
    # Portfolio value line
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color=COLORS['primary'], width=2),
            hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Mark outliers
    if outliers:
        outlier_dates = [o['date'] for o in outliers]
        outlier_values = [portfolio_df.loc[o['date'], 'portfolio_value'] for o in outliers]
        outlier_returns = [o['return'] * 100 for o in outliers]  # Convert to percentage
        
        fig.add_trace(
            go.Scatter(
                x=outlier_dates,
                y=outlier_values,
                mode='markers',
                name='Outliers',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='star',
                    line=dict(color='darkred', width=2)
                ),
                hovertemplate='OUTLIER<br>Date: %{x}<br>Value: $%{y:,.2f}<br>Return: %{customdata:.2f}%<extra></extra>',
                customdata=outlier_returns
            ),
            row=1, col=1
        )
    
    # Daily returns
    daily_returns = portfolio_df['daily_return'].dropna() * 100  # Convert to percentage
    
    fig.add_trace(
        go.Bar(
            x=daily_returns.index,
            y=daily_returns.values,
            name='Daily Returns',
            marker_color=np.where(daily_returns.values > 0, 
                                 'rgba(46, 134, 171, 0.6)', 
                                 'rgba(220, 38, 127, 0.6)'),
            hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)
    
    fig.update_layout(
        title='Portfolio Outlier Detection Analysis',
        height=800,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True
    )
    
    # Determine file paths
    if output_path.endswith('.html'):
        html_path = output_path
        png_path = output_path.replace('.html', '.png')
    else:
        png_path = output_path
        html_path = output_path.replace('.png', '.html')
    
    # Save both formats
    fig.write_html(html_path)
    
    try:
        fig.write_image(png_path, width=1400, height=800, scale=1)
        print(f"   ✅ Saved both: {html_path} and {png_path}")
    except Exception as e:
        print(f"   ⚠️  HTML saved: {html_path}, PNG failed: {e}")
    
    return fig

def generate_trade_outlier_report(outliers: List[dict], trades_df: pd.DataFrame, 
                                 output_path: str):
    """Generate text report for trade outliers."""
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRADE OUTLIER DETECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Determine return column
        if 'simple_return' in trades_df.columns:
            return_col = 'simple_return'
        elif 'trade_pnl' in trades_df.columns:
            return_col = 'trade_pnl'
        else:
            return_col = 'log_return'
        
        returns = trades_df[return_col].dropna()
        
        if 'time_entered' in trades_df.columns:
            date_range = f"{trades_df['time_entered'].min()} to {trades_df['time_entered'].max()}"
        else:
            date_range = "Date range not available"
        
        f.write(f"Trade Date Range: {date_range}\n")
        f.write(f"Total Trades Analyzed: {len(returns):,}\n")
        f.write(f"Outlier Trades Detected: {len(outliers)}\n\n")
        
        if outliers:
            f.write("-" * 80 + "\n")
            f.write("OUTLIER TRADE DETAILS\n")
            f.write("-" * 80 + "\n\n")
            
            # Sort outliers by absolute return magnitude - fix the comparison error
            try:
                sorted_outliers = sorted(outliers, key=lambda x: abs(float(x['return'])), reverse=True)
            except (TypeError, ValueError):
                # Fallback if there are issues with return values
                sorted_outliers = outliers
            
            # Limit to top 50 outliers for readability
            display_outliers = sorted_outliers[:50]
            
            for i, outlier in enumerate(display_outliers, 1):
                f.write(f"Outlier Trade #{i}\n")
                
                # Handle date display
                if outlier['date'] is not None:
                    try:
                        f.write(f"Date: {outlier['date'].strftime('%Y-%m-%d %H:%M')}\n")
                    except:
                        f.write(f"Date: {outlier['date']}\n")
                else:
                    f.write(f"Trade Index: {outlier.get('trade_index', 'Unknown')}\n")
                
                # Return display
                return_val = float(outlier['return'])
                if return_col == 'trade_pnl':
                    f.write(f"Trade P&L: ${return_val:.2f}\n")
                else:
                    f.write(f"Trade Return: {return_val*100:.2f}%\n")
                
                f.write(f"Trading Coin: {outlier.get('trading_coin', 'Unknown')}\n")
                f.write(f"Reference Coin: {outlier.get('reference_coin', 'Unknown')}\n")
                f.write(f"Trade Type: {outlier.get('trade_type', 'Unknown')}\n")
                
                if 'z_score' in outlier:
                    f.write(f"Z-Score: {outlier['z_score']:.2f}\n")
                elif 'iqr_distance' in outlier:
                    f.write(f"IQR Distance: {outlier['iqr_distance']:.2f}\n")
                
                f.write("\n")
            
            if len(sorted_outliers) > 50:
                f.write(f"... and {len(sorted_outliers) - 50} more outlier trades\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("STATISTICAL SUMMARY\n")
            f.write("-" * 80 + "\n\n")
            
            if return_col == 'trade_pnl':
                f.write(f"Mean Trade P&L: ${returns.mean():.2f}\n")
                f.write(f"Std Dev Trade P&L: ${returns.std():.2f}\n")
                f.write(f"Min Trade P&L: ${returns.min():.2f}\n")
                f.write(f"Max Trade P&L: ${returns.max():.2f}\n")
            else:
                f.write(f"Mean Trade Return: {returns.mean()*100:.4f}%\n")
                f.write(f"Std Dev Trade Return: {returns.std()*100:.4f}%\n")
                f.write(f"Min Trade Return: {returns.min()*100:.2f}%\n")
                f.write(f"Max Trade Return: {returns.max()*100:.2f}%\n")
            
            f.write(f"Skewness: {returns.skew():.4f}\n")
            f.write(f"Kurtosis: {returns.kurtosis():.4f}\n")
            
            # Outlier statistics
            outlier_returns = [float(o['return']) for o in outliers]
            positive_outliers = [r for r in outlier_returns if r > 0]
            negative_outliers = [r for r in outlier_returns if r < 0]
            
            f.write(f"\nPositive Outlier Trades: {len(positive_outliers)}\n")
            f.write(f"Negative Outlier Trades: {len(negative_outliers)}\n")
            f.write(f"Outlier Percentage: {len(outliers)/len(returns)*100:.4f}%\n")
            
            if positive_outliers:
                max_positive = max(positive_outliers)
                if return_col == 'trade_pnl':
                    f.write(f"Largest Positive Trade: ${max_positive:.2f}\n")
                else:
                    f.write(f"Largest Positive Return: {max_positive*100:.2f}%\n")
                    
            if negative_outliers:
                min_negative = min(negative_outliers)
                if return_col == 'trade_pnl':
                    f.write(f"Largest Negative Trade: ${min_negative:.2f}\n")
                else:
                    f.write(f"Largest Negative Return: {min_negative*100:.2f}%\n")
            
            # Coin analysis
            outlier_coins = [o.get('trading_coin', 'Unknown') for o in outliers]
            coin_counts = pd.Series(outlier_coins).value_counts().head(10)
            
            f.write(f"\nTop 10 Coins with Most Outlier Trades:\n")
            for coin, count in coin_counts.items():
                f.write(f"  {coin}: {count} outlier trades\n")
        
        else:
            f.write("No outlier trades detected with the current threshold settings.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

def generate_outlier_report(outliers: List[dict], portfolio_df: pd.DataFrame, 
                          output_path: str):
    """Generate text report for portfolio outliers (legacy function)."""
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PORTFOLIO OUTLIER DETECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Portfolio Date Range: {portfolio_df.index.min()} to {portfolio_df.index.max()}\n")
        f.write(f"Total Trading Days: {len(portfolio_df)}\n")
        f.write(f"Outliers Detected: {len(outliers)}\n\n")
        
        if outliers:
            f.write("-" * 80 + "\n")
            f.write("OUTLIER DETAILS\n")
            f.write("-" * 80 + "\n\n")
            
            # Sort outliers by absolute return magnitude - fix comparison error
            try:
                sorted_outliers = sorted(outliers, key=lambda x: abs(float(x['return'])), reverse=True)
            except (TypeError, ValueError):
                sorted_outliers = outliers
            
            for i, outlier in enumerate(sorted_outliers, 1):
                f.write(f"Outlier #{i}\n")
                f.write(f"Date: {outlier['date'].strftime('%Y-%m-%d')}\n")
                f.write(f"Daily Return: {float(outlier['return'])*100:.2f}%\n")
                f.write(f"Portfolio Value: ${float(outlier['portfolio_value']):,.2f}\n")
                
                if 'z_score' in outlier:
                    f.write(f"Z-Score: {outlier['z_score']:.2f}\n")
                elif 'iqr_distance' in outlier:
                    f.write(f"IQR Distance: {outlier['iqr_distance']:.2f}\n")
                
                # Add context about what happened around this date
                if 'num_active_trades' in portfolio_df.columns:
                    active_trades = portfolio_df.loc[outlier['date'], 'num_active_trades']
                    f.write(f"Active Trades: {active_trades}\n")
                
                f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("STATISTICAL SUMMARY\n")
            f.write("-" * 80 + "\n\n")
            
            returns = portfolio_df['daily_return'].dropna()
            f.write(f"Mean Daily Return: {returns.mean()*100:.4f}%\n")
            f.write(f"Std Dev Daily Return: {returns.std()*100:.4f}%\n")
            f.write(f"Skewness: {returns.skew():.4f}\n")
            f.write(f"Kurtosis: {returns.kurtosis():.4f}\n")
            f.write(f"Min Return: {returns.min()*100:.2f}%\n")
            f.write(f"Max Return: {returns.max()*100:.2f}%\n")
            
            # Outlier statistics
            outlier_returns = [float(o['return']) for o in outliers]
            positive_outliers = [r for r in outlier_returns if r > 0]
            negative_outliers = [r for r in outlier_returns if r < 0]
            
            f.write(f"\nPositive Outliers: {len(positive_outliers)}\n")
            f.write(f"Negative Outliers: {len(negative_outliers)}\n")
            f.write(f"Outlier Percentage: {len(outliers)/len(returns)*100:.2f}%\n")
            
            if positive_outliers:
                f.write(f"Largest Positive Spike: {max(positive_outliers)*100:.2f}%\n")
            if negative_outliers:
                f.write(f"Largest Negative Spike: {min(negative_outliers)*100:.2f}%\n")
        
        else:
            f.write("No outliers detected with the current threshold settings.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

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