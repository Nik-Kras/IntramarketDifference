#!/usr/bin/env python3
"""
Portfolio Simulation for 2024 Out-of-Sample Data

Simulates trading a portfolio of all validated pairs using proper budget allocation.
Includes visualizations for allocated budget and active trades.

Usage:
    python simulate_portfolio_2024.py --initial_budget 10000 --allocation_fraction 0.01
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from typing import Dict
import glob
import sys

# -----------------------
# Configuration
# -----------------------
OUT_OF_SAMPLE_DATA_DIR = "data/out_of_sample"
VALIDATION_RESULTS_FILE = "results/out_of_sample/oos_validation_results.csv"
TRADES_DIR = "results/out_of_sample/trades"
PORTFOLIO_DIR = "portfolio_simulation"

# Algorithm parameters
LOOKBACK = 24
ATR_LOOKBACK = 168
THRESHOLD = 0.25


class PortfolioSimulator:
    """Simulates portfolio trading with proper budget allocation management."""
    
    def __init__(self, initial_budget: float = 1000.0, allocation_fraction: float = 0.01):
        self.initial_budget = initial_budget
        self.allocation_fraction = allocation_fraction
        
        # Portfolio state
        self.available_budget = initial_budget
        self.allocated_budget = 0.0
        self.total_portfolio_value = initial_budget
        
        # Tracking
        self.active_trades = {}
        self.completed_trades = []
        self.equity_curve = []
        
        max_trades = int(1 / allocation_fraction)
        print(f"🏦 Portfolio Simulator Initialized:")
        print(f"   Initial Budget: ${initial_budget:,.2f}")
        print(f"   Allocation Fraction: {allocation_fraction:.1%}")
        print(f"   Maximum Concurrent Trades: {max_trades}")
        print(f"   Initial Trade Size: ${initial_budget / max_trades:,.2f}")
    
    def load_all_saved_trades(self) -> list:
        """Load all saved trade JSON files from the trades directory."""
        
        all_trades = []
        trade_id_counter = 0  # Global counter to ensure unique trade IDs
        
        # Find all trade JSON files
        trade_files = glob.glob(os.path.join(TRADES_DIR, "**/*.json"), recursive=True)
        
        print(f"📂 Found {len(trade_files)} trade files to load...")
        
        for trade_file in tqdm(trade_files, desc="Loading trades"):
            try:
                with open(trade_file, 'r') as f:
                    trades_data = json.load(f)
                
                # Extract trading info from filename
                filename = os.path.basename(trade_file)
                # Expected format: {TRADING_COIN}_{REF_COIN}_{trading_type}_trades.json
                parts = filename.replace('_trades.json', '').split('_')
                if len(parts) >= 3:
                    trading_coin = parts[0]
                    ref_coin = parts[1]
                    trading_type = '_'.join(parts[2:])  # Handle multi-word trading types
                    
                    # Process each trade in the file
                    for trade_idx, trade in enumerate(trades_data):
                        if trade['time_entered'] and trade['time_exited'] and trade['log_return'] is not None:
                            # Create globally unique trade ID
                            unique_trade_id = f"{trading_coin}_{ref_coin}_{trade_id_counter:06d}"
                            trade_id_counter += 1
                            
                            all_trades.append({
                                'entry_time': pd.to_datetime(trade['time_entered']),
                                'exit_time': pd.to_datetime(trade['time_exited']),
                                'log_return': trade['log_return'],
                                'trade_type': trade['trade_type'],
                                'trading_coin': trading_coin,
                                'ref_coin': ref_coin,
                                'trading_strategy': trading_type,
                                'pair_id': f"{trading_coin}_{ref_coin}",
                                'trade_id': unique_trade_id
                            })
                            
            except Exception as e:
                print(f"⚠️  Could not load {trade_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(all_trades)} individual trades from {len(trade_files)} files")
        return all_trades
    
    # Removed generate_trades_for_pair - now loading from saved JSON files
    
    def can_allocate_trade(self, trade_amount: float) -> bool:
        """Check if we can allocate a new trade."""
        max_trades = int(1 / self.allocation_fraction)
        return (self.available_budget >= trade_amount) and (len(self.active_trades) < max_trades)
    
    def open_trade(self, trade_info: Dict, trade_amount: float) -> None:
        """Open a new trade."""
        trade_id = trade_info['trade_id']
        
        # Allocate budget
        self.available_budget -= trade_amount
        self.allocated_budget += trade_amount
        
        # Track active trade
        self.active_trades[trade_id] = {
            'trade_amount': trade_amount,
            'entry_time': trade_info['entry_time'],
            'exit_time': trade_info['exit_time'],
            'trade_return': trade_info['trade_return'],
            'pair_id': trade_info['pair_id'],
            'trade_type': trade_info['trade_type']
        }
    
    def close_trade(self, trade_id: str, current_time: pd.Timestamp) -> float:
        """Close an active trade."""
        if trade_id not in self.active_trades:
            return 0.0
        
        trade_info = self.active_trades[trade_id]
        trade_amount = trade_info['trade_amount']
        trade_return = trade_info['trade_return']
        
        # Calculate P&L using simple returns
        trade_pnl = trade_amount * trade_return
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
    
    def update_portfolio_tracking(self, current_time: pd.Timestamp) -> None:
        """Update portfolio tracking."""
        self.total_portfolio_value = self.available_budget + self.allocated_budget
        
        self.equity_curve.append({
            'timestamp': current_time,
            'portfolio_value': self.total_portfolio_value,
            'available_budget': self.available_budget,
            'allocated_budget': self.allocated_budget,
            'num_active_trades': len(self.active_trades)
        })
    
    def simulate_portfolio(self) -> Dict:
        """Run portfolio simulation using all saved trades."""
        
        print(f"\n🚀 Starting Portfolio Simulation")
        
        # Load all saved trades
        all_trades_data = self.load_all_saved_trades()
        
        if not all_trades_data:
            print("❌ No trades loaded!")
            return {}
        
        # Create timeline of events
        events = []
        
        for trade in all_trades_data:
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
        
        # Process events
        trades_opened = 0
        trades_closed = 0
        trades_rejected = 0
        trades_not_found = 0  # Track trades that couldn't be closed
        max_active_trades = 0
        
        print(f"📊 Processing {len(events)} events...")
        
        for i, event in enumerate(events):
            current_time = event['timestamp']
            
            if event['type'] == 'exit':
                trade_id = event['trade_id']
                if trade_id in self.active_trades:
                    trade_pnl = self.close_trade(trade_id, current_time)
                    trades_closed += 1
                else:
                    trades_not_found += 1
                    # Debug: log some failed closures
                    if trades_not_found <= 5:
                        print(f"⚠️  Trade not found for closure: {trade_id} at {current_time}")
            
            elif event['type'] == 'entry':
                trade = event['trade']
                
                # Calculate trade amount based on current portfolio value
                max_trades = int(1 / self.allocation_fraction)
                current_portfolio_value = self.available_budget + self.allocated_budget
                trade_amount = current_portfolio_value / max_trades
                
                if self.can_allocate_trade(trade_amount) and trade_amount > 0:
                    # Convert log return to simple return for simulation
                    # log_return to simple return: exp(log_return) - 1
                    simple_return = np.exp(trade['log_return']) - 1
                    trade_with_simple_return = trade.copy()
                    trade_with_simple_return['trade_return'] = simple_return
                    
                    self.open_trade(trade_with_simple_return, trade_amount)
                    trades_opened += 1
                else:
                    trades_rejected += 1
            
            # Update tracking
            max_active_trades = max(max_active_trades, len(self.active_trades))
            self.update_portfolio_tracking(current_time)
            
            # Progress tracking
            if i % 10000 == 0:
                print(f"   Processed {i:,} events. Active trades: {len(self.active_trades)}")
        
        # Final active trades check
        if len(self.active_trades) > 0:
            print(f"⚠️  {len(self.active_trades)} trades never closed:")
            for trade_id, trade_info in list(self.active_trades.items())[:5]:  # Show first 5
                print(f"     {trade_id}: opened {trade_info['entry_time']}")
        
        # Calculate final metrics
        final_pnl = self.total_portfolio_value - self.initial_budget
        total_return = (self.total_portfolio_value / self.initial_budget) - 1
        
        print(f"\n📊 PORTFOLIO SIMULATION RESULTS:")
        print(f"   Trades Opened: {trades_opened}")
        print(f"   Trades Closed: {trades_closed}")
        print(f"   Trades Rejected: {trades_rejected}")
        print(f"   Trades Not Found (couldn't close): {trades_not_found}")
        print(f"   Still Active: {len(self.active_trades)}")
        print(f"   Maximum Active Trades: {max_active_trades}")
        print(f"   Theoretical Max: {int(1/self.allocation_fraction)}")
        print(f"   Final Portfolio Value: ${self.total_portfolio_value:,.2f}")
        print(f"   Total P&L: ${final_pnl:,.2f}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"\n🔍 DEBUG INFO:")
        print(f"   Trade Match Rate: {trades_closed / (trades_closed + trades_not_found) * 100:.1f}%" if (trades_closed + trades_not_found) > 0 else "   No trades to match")
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate portfolio performance metrics."""
        
        if not self.completed_trades:
            return {}
        
        # Extract returns
        trade_returns = [t['trade_pnl'] / self.initial_budget for t in self.completed_trades]
        
        # Total return
        total_return = (self.total_portfolio_value / self.initial_budget) - 1
        
        # Profit factor
        gains = sum([r for r in trade_returns if r > 0])
        losses = sum([r for r in trade_returns if r < 0])
        profit_factor = gains / abs(losses) if losses < 0 else float('inf') if gains > 0 else 0
        
        # Max drawdown
        if len(self.equity_curve) > 1:
            equity_values = [p['portfolio_value'] for p in self.equity_curve]
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
        """Create portfolio visualizations."""
        
        if not self.equity_curve:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df = equity_df.set_index('timestamp').sort_index()
        
        # 1. Portfolio Equity Curve
        plt.figure(figsize=(15, 8))
        plt.plot(equity_df.index, equity_df['portfolio_value'], linewidth=1.5, color='blue', label='Portfolio Value')
        plt.axhline(y=self.initial_budget, color='red', linestyle='--', alpha=0.7, label='Initial Budget')
        plt.title('Portfolio Equity Curve - 2024 Out-of-Sample', fontsize=16)
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
        plt.fill_between(equity_df.index, 0, equity_df['available_budget'], 
                        alpha=0.7, color='green', label='Available Budget')
        plt.fill_between(equity_df.index, equity_df['available_budget'], equity_df['portfolio_value'], 
                        alpha=0.7, color='orange', label='Allocated Budget')
        plt.title('Budget Allocation Over Time - 2024 Out-of-Sample', fontsize=16)
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
        plt.axhline(y=int(1/self.allocation_fraction), color='red', linestyle='--', 
                   alpha=0.7, label=f'Max Trades ({int(1/self.allocation_fraction)})')
        plt.title('Number of Active Trades Over Time - 2024 Out-of-Sample', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Active Trades', fontsize=12)
        plt.legend()
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
        plt.title('Portfolio Drawdown Curve - 2024 Out-of-Sample', fontsize=16)
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
    
    parser = argparse.ArgumentParser(description='Portfolio Simulation for 2024 OOS Data')
    parser.add_argument('--initial_budget', type=float, default=1000.0,
                       help='Initial portfolio budget (default: 1000)')
    parser.add_argument('--allocation_fraction', type=float, default=0.01,
                       help='Allocation fraction per trade (default: 0.01)')
    
    args = parser.parse_args()
    
    print("🚀 Portfolio Simulation for 2024 Out-of-Sample Data")
    print("=" * 55)
    
    # Create output directory
    os.makedirs(PORTFOLIO_DIR, exist_ok=True)
    
    # Check that trades directory exists
    if not os.path.exists(TRADES_DIR):
        print(f"❌ Trades directory not found: {TRADES_DIR}")
        print("Run run_out_of_sample_validation.py first to generate trades.")
        return
    
    trade_files = glob.glob(os.path.join(TRADES_DIR, "**/*.json"), recursive=True)
    if not trade_files:
        print(f"❌ No trade files found in: {TRADES_DIR}")
        print("Run run_out_of_sample_validation.py first to generate trades.")
        return
    
    print(f"📊 Found {len(trade_files)} trade files to simulate")
    
    # Initialize simulator
    simulator = PortfolioSimulator(
        initial_budget=args.initial_budget,
        allocation_fraction=args.allocation_fraction
    )
    
    try:
        # Run simulation
        metrics = simulator.simulate_portfolio()
        
        # Save metrics
        with open(os.path.join(PORTFOLIO_DIR, 'portfolio_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualizations
        simulator.create_visualizations(PORTFOLIO_DIR)
        
        print(f"\n✅ Portfolio simulation completed!")
        print(f"📁 Results saved to: {PORTFOLIO_DIR}")
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        raise


if __name__ == "__main__":
    main()