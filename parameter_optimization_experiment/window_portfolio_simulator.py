#!/usr/bin/env python3
"""
Window Portfolio Simulator for Parameter Optimization Experiment

Simulates portfolio trading with corrected budget allocation logic:
- allocation_per_trade = 1.1 * current_portfolio_value / N_selected_pairs
- Max concurrent trades ≈ 90% of N_pairs
- Dynamic rebalancing after each trade completion
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
import sys

# Import core functions
from core import (
    cmma, threshold_revert_signal, load_csv_with_cache, align_frames,
    LOOKBACK, ATR_LOOKBACK, THRESHOLD
)

# Add parent directory to path
sys.path.append('..')
from trades_from_signal import get_trades_from_signal

class WindowPortfolioSimulator:
    """Portfolio simulator with corrected budget allocation logic."""
    
    def __init__(self, initial_capital: float = 1000.0, n_pairs: int = 1):
        """
        Initialize portfolio simulator.
        
        Args:
            initial_capital: Starting capital
            n_pairs: Number of selected pairs for dynamic allocation
        """
        self.initial_capital = initial_capital
        self.n_pairs = n_pairs
        
        # Portfolio state
        self.available_budget = initial_capital
        self.allocated_budget = 0.0
        self.total_portfolio_value = initial_capital
        
        # Tracking
        self.active_trades = {}  # trade_id -> trade_info
        self.completed_trades = []
        self.portfolio_timeline = []  # Track portfolio value over time
        
        # Calculate max concurrent trades: N_pairs / 1.1
        self.max_trades = max(1, int(self.n_pairs / 1.1)) if self.n_pairs > 0 else 1
        
        print(f"🏦 Portfolio Simulator Initialized:")
        print(f"   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Selected Pairs: {self.n_pairs}")
        print(f"   Max Concurrent Trades: {self.max_trades} (~{self.max_trades/max(self.n_pairs,1)*100:.0f}% of pairs)")
        print(f"   Initial Trade Amount: ${initial_capital / self.max_trades:,.2f}")
    
    def get_current_trade_amount(self) -> float:
        """Calculate current trade amount based on portfolio value (dynamic rebalancing)."""
        current_portfolio_value = self.available_budget + self.allocated_budget
        return current_portfolio_value / self.max_trades if self.max_trades > 0 else 0.0
    
    def can_allocate_trade(self, trade_amount: float) -> bool:
        """Check if we can allocate money for a new trade."""
        return (self.available_budget >= trade_amount) and (len(self.active_trades) < self.max_trades)
    
    def open_trade(self, trade_info: Dict, trade_amount: float) -> str:
        """Open a new trade and allocate budget."""
        
        trade_id = trade_info['trade_id']
        
        # Allocate budget
        self.available_budget -= trade_amount
        self.allocated_budget += trade_amount
        
        # Store trade info
        self.active_trades[trade_id] = {
            'entry_time': trade_info['entry_time'],
            'exit_time': trade_info['exit_time'],
            'trade_amount': trade_amount,
            'trade_return': trade_info['trade_return'],  # Simple return
            'trading_coin': trade_info['trading_coin'],
            'reference_coin': trade_info['reference_coin'],
            'strategy_type': trade_info['strategy_type'],
            'trade_type': trade_info['trade_type']
        }
        
        return trade_id
    
    def close_trade(self, trade_id: str, current_time: pd.Timestamp) -> float:
        """Close a trade and update portfolio."""
        
        if trade_id not in self.active_trades:
            return 0.0
        
        trade_info = self.active_trades[trade_id]
        trade_amount = trade_info['trade_amount']
        trade_return = trade_info['trade_return']
        
        # Calculate P&L using simple returns
        trade_pnl = trade_amount * trade_return
        final_amount = trade_amount + trade_pnl
        
        # Return capital to available budget
        self.allocated_budget -= trade_amount
        self.available_budget += final_amount
        
        # Record completed trade
        self.completed_trades.append({
            'trade_id': trade_id,
            'entry_time': trade_info['entry_time'],
            'exit_time': current_time,
            'trade_amount': trade_amount,
            'trade_pnl': trade_pnl,
            'trade_return': trade_return,
            'trading_coin': trade_info['trading_coin'],
            'reference_coin': trade_info['reference_coin'],
            'strategy_type': trade_info['strategy_type'],
            'trade_type': trade_info['trade_type']
        })
        
        # Remove from active trades
        del self.active_trades[trade_id]
        
        return trade_pnl
    
    def update_tracking(self, current_time: pd.Timestamp):
        """Update timeline tracking."""
        
        self.total_portfolio_value = self.available_budget + self.allocated_budget
        
        self.portfolio_timeline.append({
            'timestamp': current_time,
            'portfolio_value': self.total_portfolio_value,
            'available_budget': self.available_budget,
            'allocated_budget': self.allocated_budget,
            'num_active_trades': len(self.active_trades),
            'current_trade_amount': self.get_current_trade_amount()
        })
    
    def load_oos_trades_for_pairs(self, selected_pairs_df: pd.DataFrame) -> List[Dict]:
        """Load OOS trades from pre-saved JSON files for selected pairs."""
        
        all_oos_trades = []
        trade_id_counter = 0
        
        print(f"📂 Loading OOS trades for {len(selected_pairs_df)} selected pairs...")
        
        for _, pair in tqdm(selected_pairs_df.iterrows(), total=len(selected_pairs_df), desc="Loading OOS trades"):
            ref_coin = pair['reference_coin']
            trading_coin = pair['trading_coin']
            strategy_type = pair['trading_type']
            
            # Load trades from CSV file (much faster than JSON)
            csv_file = os.path.join("out_of_sample/trades", 
                                  f"{ref_coin}_{trading_coin}", 
                                  f"{strategy_type}_trades_fast.csv")
            
            # Fallback to JSON if CSV doesn't exist
            json_file = os.path.join("out_of_sample/trades", 
                                   f"{ref_coin}_{trading_coin}", 
                                   f"{strategy_type}_trades.json")
            
            trades_file = csv_file if os.path.exists(csv_file) else json_file
            
            if os.path.exists(trades_file):
                try:
                    if trades_file.endswith('.csv'):
                        # Fast CSV loading with vectorized operations
                        df = pd.read_csv(trades_file)
                        df = df.dropna(subset=['time_entered', 'time_exited', 'log_return'])
                        
                        if len(df) > 0:
                            # Vectorized datetime conversion
                            df['entry_time'] = pd.to_datetime(df['time_entered'])
                            df['exit_time'] = pd.to_datetime(df['time_exited'])
                            
                            # Vectorized return conversion
                            df['trade_return'] = np.exp(df['log_return']) - 1
                            
                            # Create trade IDs vectorized
                            df['trade_id'] = [f"{trading_coin}_{ref_coin}_{trade_id_counter + i:06d}" 
                                            for i in range(len(df))]
                            trade_id_counter += len(df)
                            
                            # Add metadata columns
                            df['trading_coin'] = trading_coin
                            df['reference_coin'] = ref_coin
                            df['strategy_type'] = strategy_type
                            
                            # Convert DataFrame to list of dicts (vectorized)
                            trade_dicts = df[['entry_time', 'exit_time', 'trade_return', 'trading_coin', 
                                            'reference_coin', 'strategy_type', 'trade_type', 'trade_id']].to_dict('records')
                            all_oos_trades.extend(trade_dicts)
                    else:
                        # Fallback JSON loading (slower)
                        with open(trades_file, 'r') as f:
                            trades_data = json.load(f)
                        
                        for trade in trades_data:
                            if trade['time_entered'] and trade['time_exited'] and trade['log_return'] is not None:
                                simple_return = np.exp(trade['log_return']) - 1
                                trade_id = f"{trading_coin}_{ref_coin}_{trade_id_counter:06d}"
                                trade_id_counter += 1
                                
                                all_oos_trades.append({
                                    'entry_time': pd.to_datetime(trade['time_entered']),
                                    'exit_time': pd.to_datetime(trade['time_exited']),
                                    'trade_return': simple_return,
                                    'trading_coin': trading_coin,
                                    'reference_coin': ref_coin,
                                    'strategy_type': strategy_type,
                                    'trade_type': trade['trade_type'],
                                    'trade_id': trade_id
                                })
                            
                except Exception as e:
                    print(f"⚠️  Could not load trades for {ref_coin}_{trading_coin}: {e}")
                    continue
        
        print(f"✅ Loaded {len(all_oos_trades):,} OOS trades")
        return all_oos_trades
    
    def simulate_portfolio(self, selected_pairs_df: pd.DataFrame) -> Dict:
        """Run complete portfolio simulation on OOS data."""
        
        # Load all OOS trades for selected pairs
        all_oos_trades = self.load_oos_trades_for_pairs(selected_pairs_df)
        
        if not all_oos_trades:
            print("❌ No OOS trades loaded!")
            return {}
        
        print(f"📊 Loaded {len(all_oos_trades):,} OOS trades")
        
        # Create chronological events
        events = []
        
        for trade in all_oos_trades:
            # Entry event
            events.append({
                'timestamp': trade['entry_time'],
                'type': 'entry',
                'trade': trade
            })
            
            # Exit event
            events.append({
                'timestamp': trade['exit_time'],
                'type': 'exit',
                'trade_id': trade['trade_id']
            })
        
        # Sort events chronologically
        events.sort(key=lambda x: x['timestamp'])
        
        print(f"🔄 Processing {len(events)} chronological events...")
        
        # Process events
        trades_opened = 0
        trades_closed = 0
        trades_rejected = 0
        
        for event in tqdm(events, desc="Processing events"):
            current_time = event['timestamp']
            
            if event['type'] == 'exit':
                trade_id = event['trade_id']
                if trade_id in self.active_trades:
                    self.close_trade(trade_id, current_time)
                    trades_closed += 1
            
            elif event['type'] == 'entry':
                trade = event['trade']
                
                # Calculate current trade amount (dynamic rebalancing)
                current_trade_amount = self.get_current_trade_amount()
                
                if self.can_allocate_trade(current_trade_amount) and current_trade_amount > 0:
                    self.open_trade(trade, current_trade_amount)
                    trades_opened += 1
                else:
                    trades_rejected += 1
            
            # Update tracking
            self.update_tracking(current_time)
        
        # Calculate final metrics
        self.total_portfolio_value = self.available_budget + self.allocated_budget
        final_return = (self.total_portfolio_value / self.initial_capital) - 1
        
        print(f"\n📊 PORTFOLIO SIMULATION RESULTS:")
        print(f"   Trades Opened: {trades_opened:,}")
        print(f"   Trades Closed: {trades_closed:,}")
        print(f"   Trades Rejected: {trades_rejected:,}")
        print(f"   Still Active: {len(self.active_trades)}")
        print(f"   Final Portfolio Value: ${self.total_portfolio_value:,.2f}")
        print(f"   Total Return: {final_return:.2%}")
        
        return self.calculate_portfolio_metrics()
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate portfolio performance metrics."""
        
        if not self.completed_trades:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'num_trades': 0
            }
        
        # Portfolio value timeline
        timeline_df = pd.DataFrame(self.portfolio_timeline)
        timeline_df = timeline_df.set_index('timestamp').sort_index()
        
        # Calculate proper daily returns (resample to daily intervals)
        portfolio_values = timeline_df['portfolio_value']
        daily_portfolio = portfolio_values.resample('D').last()  # End-of-day values
        daily_returns = daily_portfolio.pct_change().dropna()   # Actual daily returns
        
        # Calculate metrics
        total_return = (self.total_portfolio_value / self.initial_capital) - 1
        
        # Sharpe ratio (daily returns annualized)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown (use same daily values for consistency)
        cumulative_values = daily_portfolio / self.initial_capital
        peak = cumulative_values.cummax()
        drawdown = (cumulative_values - peak) / peak
        max_drawdown = float(drawdown.min())
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(365) if len(daily_returns) > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'num_trades': len(self.completed_trades),
            'initial_capital': self.initial_capital,
            'final_value': self.total_portfolio_value
        }
    
    def create_visualizations(self, output_dir: str):
        """Create portfolio simulation visualizations."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.portfolio_timeline:
            print("❌ No timeline data for visualizations")
            return
        
        # Convert timeline data to DataFrames
        portfolio_df = pd.DataFrame(self.portfolio_timeline).set_index('timestamp').sort_index()
        
        # 1. Portfolio Equity Curve
        plt.figure(figsize=(15, 8))
        plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], 'b-', linewidth=2, label='Portfolio Value')
        plt.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'portfolio_equity_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Budget Allocation Timeline
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_df.index, portfolio_df['available_budget'], 'g-', linewidth=2, label='Available Budget')
        plt.plot(portfolio_df.index, portfolio_df['allocated_budget'], 'r-', linewidth=2, label='Allocated Budget')
        plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], 'b-', linewidth=2, label='Total Portfolio Value')
        plt.title('Budget Allocation Over Time')
        plt.xlabel('Date')
        plt.ylabel('Amount ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(portfolio_df.index, portfolio_df['current_trade_amount'], 'purple', linewidth=2, label='Current Trade Amount')
        plt.title('Dynamic Trade Amount (Rebalancing)')
        plt.xlabel('Date')
        plt.ylabel('Trade Amount ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'budget_allocation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Active Trades Timeline
        plt.figure(figsize=(15, 8))
        plt.plot(portfolio_df.index, portfolio_df['num_active_trades'], 'orange', linewidth=2, label='Active Trades')
        plt.axhline(y=self.max_trades, color='red', linestyle='--', alpha=0.7, 
                   label=f'Max Concurrent ({self.max_trades})')
        plt.title('Active Trades Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Active Trades')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'active_trades_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Drawdown Curve
        plt.figure(figsize=(15, 8))
        cumulative_values = portfolio_df['portfolio_value'] / self.initial_capital
        peak = cumulative_values.cummax()
        drawdown = (cumulative_values - peak) / peak
        
        plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        plt.plot(drawdown.index, drawdown, 'r-', linewidth=2)
        plt.title('Portfolio Drawdown Curve')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'drawdown_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to: {output_dir}/")
    
    def save_metrics(self, output_file: str):
        """Save portfolio metrics to JSON file."""
        
        metrics = self.calculate_portfolio_metrics()
        
        # Add additional details
        metrics.update({
            'selected_pairs': self.n_pairs,
            'max_concurrent_trades': self.max_trades,
            'simulation_date': datetime.now().isoformat(),
            'completed_trades_count': len(self.completed_trades)
        })
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Portfolio metrics saved to: {output_file}")

def main():
    """Test the portfolio simulator."""
    
    # Test with sample configuration
    simulator = WindowPortfolioSimulator(initial_capital=1000.0, n_pairs=10)
    
    print("✅ Portfolio simulator ready for testing")

if __name__ == "__main__":
    main()