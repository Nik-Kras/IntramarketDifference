#!/usr/bin/env python3
"""
Portfolio Simulation Script

Simulates chronological trading of all pairs from OOS experiments with proper budget management.
Tracks available budget and prevents over-allocation of capital.

Usage:
    python portfolio_simulation.py --initial_budget 1000 --allocation_fraction 0.01
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict


class PortfolioSimulator:
    """Simulates portfolio trading with proper budget allocation management."""
    
    def __init__(self, initial_budget: float = 1000.0, allocation_fraction: float = 0.01):
        """
        Initialize portfolio simulator.
        
        Args:
            initial_budget: Starting capital in dollars
            allocation_fraction: Fraction of available budget to allocate per trade (0.01 = 1%)
        """
        self.initial_budget = initial_budget
        self.allocation_fraction = allocation_fraction
        
        # Portfolio state
        self.available_budget = initial_budget
        self.allocated_budget = 0.0
        self.total_portfolio_value = initial_budget
        
        # Tracking
        self.active_trades = {}  # trade_id -> trade_info
        self.completed_trades = []
        self.equity_curve = []
        self.budget_tracking = []
        
        max_trades = int(1 / allocation_fraction)
        print(f"🏦 Portfolio Simulator Initialized:")
        print(f"   Initial Budget: ${initial_budget:,.2f}")
        print(f"   Allocation Fraction: {allocation_fraction:.1%}")
        print(f"   Maximum Concurrent Trades: {max_trades}")
        print(f"   Initial Trade Size: ${initial_budget / max_trades:,.2f}")
        
    def load_trades_from_directories(self, year_2023_dir: str, year_2024_dir: str) -> pd.DataFrame:
        """Load and combine all trades from both year directories."""
        
        all_trades = []
        
        for year, year_dir in [(2023, year_2023_dir), (2024, year_2024_dir)]:
            if not os.path.exists(year_dir):
                print(f"⚠️  Directory not found: {year_dir}")
                continue
                
            trade_files = [f for f in os.listdir(year_dir) if f.endswith('.json')]
            print(f"📂 Loading {len(trade_files)} trade files from {year}")
            
            for trade_file in trade_files:
                file_path = os.path.join(year_dir, trade_file)
                try:
                    with open(file_path, 'r') as f:
                        trades_data = json.load(f)
                    
                    # Extract pair info from filename
                    pair_info = trade_file.replace('.json', '')
                    
                    # Add metadata to each trade
                    for i, trade in enumerate(trades_data):
                        trade['source_pair'] = pair_info
                        trade['year'] = year
                        trade['trade_id'] = f"{pair_info}_{year}_{i}"
                        
                    all_trades.extend(trades_data)
                    
                except Exception as e:
                    print(f"⚠️  Could not load {trade_file}: {e}")
        
        if not all_trades:
            raise ValueError("No trades found in the specified directories!")
        
        # Convert to DataFrame and sort chronologically
        trades_df = pd.DataFrame(all_trades)
        trades_df['time_entered'] = pd.to_datetime(trades_df['time_entered'])
        trades_df['time_exited'] = pd.to_datetime(trades_df['time_exited'])
        trades_df = trades_df.sort_values('time_entered').reset_index(drop=True)
        
        print(f"✅ Loaded {len(trades_df)} total trades")
        print(f"📅 Period: {trades_df['time_entered'].min()} to {trades_df['time_exited'].max()}")
        
        return trades_df
    
    def can_allocate_trade(self, trade_amount: float) -> bool:
        """Check if we have enough available budget for a new trade."""
        # Check both: sufficient budget AND not exceeding max trades
        max_trades = int(1 / self.allocation_fraction)
        return (self.available_budget >= trade_amount) and (len(self.active_trades) < max_trades)
    
    def open_trade(self, trade: Dict, trade_amount: float) -> None:
        """Open a new trade by allocating budget."""
        
        trade_id = trade['trade_id']
        
        # Allocate budget
        self.available_budget -= trade_amount
        self.allocated_budget += trade_amount
        
        # Track active trade
        self.active_trades[trade_id] = {
            'trade_amount': trade_amount,
            'entry_time': trade['time_entered'],
            'exit_time': trade['time_exited'],
            'pcnt_return': trade['pcnt_return'],
            'source_pair': trade['source_pair'],
            'trade_type': trade['trade_type']
        }
        
    def close_trade(self, trade_id: str, current_time: pd.Timestamp) -> float:
        """Close an active trade and return the profit/loss."""
        
        if trade_id not in self.active_trades:
            return 0.0
        
        trade_info = self.active_trades[trade_id]
        trade_amount = trade_info['trade_amount']
        pcnt_return = trade_info['pcnt_return']
        
        # Calculate profit/loss
        trade_pnl = trade_amount * (pcnt_return / 100)
        final_amount = trade_amount + trade_pnl
        
        # Return capital to available budget
        self.available_budget += final_amount
        self.allocated_budget -= trade_amount
        
        # Track completed trade
        completed_trade = trade_info.copy()
        completed_trade.update({
            'close_time': current_time,
            'trade_pnl': trade_pnl,
            'final_amount': final_amount
        })
        self.completed_trades.append(completed_trade)
        
        # Remove from active trades
        del self.active_trades[trade_id]
        
        return trade_pnl
    
    def update_portfolio_value(self, current_time: pd.Timestamp) -> None:
        """Update total portfolio value and record equity point."""
        
        # Portfolio value = available budget + allocated budget (no mark-to-market)
        self.total_portfolio_value = self.available_budget + self.allocated_budget
        
        # Record equity curve point
        self.equity_curve.append({
            'timestamp': current_time,
            'portfolio_value': self.total_portfolio_value,
            'available_budget': self.available_budget,
            'allocated_budget': self.allocated_budget,
            'num_active_trades': len(self.active_trades)
        })
    
    def simulate_portfolio(self, trades_df: pd.DataFrame) -> Dict:
        """Run the main portfolio simulation."""
        
        print(f"\n🚀 Starting Portfolio Simulation")
        print(f"📊 Processing {len(trades_df)} trades chronologically...")
        
        # Create timeline of all events (entries and exits)
        events = []
        
        # Add entry events
        for _, trade in trades_df.iterrows():
            events.append({
                'timestamp': trade['time_entered'],
                'type': 'entry',
                'trade': trade.to_dict()
            })
        
        # Add exit events
        for _, trade in trades_df.iterrows():
            events.append({
                'timestamp': trade['time_exited'],
                'type': 'exit',
                'trade_id': trade['trade_id']
            })
        
        # Sort events chronologically
        events.sort(key=lambda x: x['timestamp'])
        
        # Process events
        trades_opened = 0
        trades_closed = 0
        trades_rejected = 0
        max_active_trades = 0
        
        for event in events:
            current_time = event['timestamp']
            
            if event['type'] == 'exit':
                # Close trade
                trade_pnl = self.close_trade(event['trade_id'], current_time)
                if trade_pnl != 0:  # Only count if trade was actually opened
                    trades_closed += 1
                    
            elif event['type'] == 'entry':
                # Try to open new trade
                trade = event['trade']
                # CORRECT: Calculate trade amount based on current total portfolio value
                # divided by maximum number of allowed trades
                max_trades = int(1 / self.allocation_fraction)
                current_portfolio_value = self.available_budget + self.allocated_budget
                trade_amount = current_portfolio_value / max_trades
                
                if self.can_allocate_trade(trade_amount) and trade_amount > 0:
                    self.open_trade(trade, trade_amount)
                    trades_opened += 1
                else:
                    trades_rejected += 1
            
            # Track maximum active trades
            max_active_trades = max(max_active_trades, len(self.active_trades))
            
            # Update portfolio tracking
            self.update_portfolio_value(current_time)
        
        # Close any remaining active trades at the end
        remaining_trades = len(self.active_trades)
        if remaining_trades > 0:
            print(f"⚠️  {remaining_trades} trades still active at simulation end")
        
        # Calculate final metrics
        final_pnl = self.total_portfolio_value - self.initial_budget
        total_return = (self.total_portfolio_value / self.initial_budget) - 1
        
        # Calculate metrics from completed trades
        metrics = self.calculate_portfolio_metrics()
        
        print(f"\n📊 SIMULATION RESULTS:")
        print(f"   Trades Opened: {trades_opened}")
        print(f"   Trades Closed: {trades_closed}")  
        print(f"   Trades Rejected (insufficient budget): {trades_rejected}")
        print(f"   Maximum Active Trades: {max_active_trades}")
        print(f"   Theoretical Max Active (1/allocation_fraction): {int(1/self.allocation_fraction)}")
        print(f"   Final Portfolio Value: ${self.total_portfolio_value:,.2f}")
        print(f"   Total P&L: ${final_pnl:,.2f}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Available Budget at End: ${self.available_budget:,.2f}")
        print(f"   Still Allocated: ${self.allocated_budget:,.2f}")
        
        return metrics
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate portfolio performance metrics."""
        
        if not self.completed_trades:
            return {
                'total_return': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'num_trades': 0
            }
        
        # Extract returns from completed trades
        trade_returns = [t['trade_pnl'] / self.initial_budget for t in self.completed_trades]
        
        # Total return
        total_return = (self.total_portfolio_value / self.initial_budget) - 1
        
        # Profit factor
        gains = sum([r for r in trade_returns if r > 0])
        losses = sum([r for r in trade_returns if r < 0])
        profit_factor = gains / abs(losses) if losses < 0 else float('inf') if gains > 0 else 0
        
        # Max drawdown from equity curve
        if len(self.equity_curve) > 1:
            equity_values = [point['portfolio_value'] for point in self.equity_curve]
            peak = np.maximum.accumulate(equity_values)
            drawdown = (equity_values - peak) / peak
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0
        
        # Sharpe ratio
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return {
            'initial_budget': self.initial_budget,
            'final_portfolio_value': self.total_portfolio_value,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(self.completed_trades),
            'allocation_fraction': self.allocation_fraction
        }
    
    def create_visualizations(self, output_dir: str) -> None:
        """Create portfolio visualization charts."""
        
        if not self.equity_curve:
            print("⚠️  No equity curve data to visualize")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df = equity_df.set_index('timestamp').sort_index()
        
        # 1. Portfolio Equity Curve
        plt.figure(figsize=(15, 8))
        plt.plot(equity_df.index, equity_df['portfolio_value'], linewidth=1.5, color='blue', label='Portfolio Value')
        plt.axhline(y=self.initial_budget, color='red', linestyle='--', alpha=0.7, label='Initial Budget')
        plt.title('Portfolio Equity Curve (Out-of-Sample)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'portfolio_equity_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Budget Allocation Over Time
        plt.figure(figsize=(15, 8))
        plt.fill_between(equity_df.index, 0, equity_df['available_budget'], alpha=0.7, color='green', label='Available Budget')
        plt.fill_between(equity_df.index, equity_df['available_budget'], equity_df['portfolio_value'], alpha=0.7, color='orange', label='Allocated Budget')
        plt.title('Budget Allocation Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Budget ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'budget_allocation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Active Trades Over Time
        plt.figure(figsize=(15, 8))
        plt.plot(equity_df.index, equity_df['num_active_trades'], linewidth=1.5, color='purple')
        plt.fill_between(equity_df.index, equity_df['num_active_trades'], alpha=0.3, color='purple')
        plt.title('Number of Active Trades Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Active Trades', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'active_trades_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Drawdown Curve
        equity_values = equity_df['portfolio_value'].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak
        
        plt.figure(figsize=(15, 8))
        plt.fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='red')
        plt.plot(equity_df.index, drawdown, color='red', linewidth=1)
        plt.title('Portfolio Drawdown Curve', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'drawdown_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {output_dir}")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Portfolio Simulation for OOS Trading Results')
    parser.add_argument('--initial_budget', type=float, default=1000.0, 
                       help='Initial portfolio budget in dollars (default: 1000)')
    parser.add_argument('--allocation_fraction', type=float, default=0.01,
                       help='Fraction of available budget to allocate per trade (default: 0.01)')
    parser.add_argument('--output_dir', type=str, default='oos_experiments/portfolio_simulation',
                       help='Output directory for results (default: oos_experiments/portfolio_simulation)')
    
    args = parser.parse_args()
    
    print("🚀 Portfolio Simulation for OOS Trading Results")
    print("=" * 60)
    
    # Define trade directories
    year_2023_dir = "oos_experiments/trades/year_2023"
    year_2024_dir = "oos_experiments/trades/year_2024"
    
    # Initialize simulator
    simulator = PortfolioSimulator(
        initial_budget=args.initial_budget,
        allocation_fraction=args.allocation_fraction
    )
    
    try:
        # Load trades
        trades_df = simulator.load_trades_from_directories(year_2023_dir, year_2024_dir)
        
        # Run simulation
        metrics = simulator.simulate_portfolio(trades_df)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(args.output_dir, 'portfolio_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualizations
        simulator.create_visualizations(args.output_dir)
        
        print(f"\n✅ Portfolio simulation completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        raise


if __name__ == "__main__":
    main()