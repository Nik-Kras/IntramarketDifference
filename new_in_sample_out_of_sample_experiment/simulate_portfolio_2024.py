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
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from typing import Dict
import sys

# Add parent directory to path
sys.path.append('..')
from trades_from_signal import get_trades_from_signal

# -----------------------
# Configuration
# -----------------------
OUT_OF_SAMPLE_DATA_DIR = "data/out_of_sample"
VALIDATION_RESULTS_FILE = "results/out_of_sample/oos_validation_results.csv"
PORTFOLIO_DIR = "portfolio_simulation"

# Algorithm parameters
LOOKBACK = 24
ATR_LOOKBACK = 168
THRESHOLD = 0.25


class PortfolioSimulator:
    """Simulates portfolio trading with proper budget allocation management."""
    
    def __init__(self, initial_budget: float = 10000.0, allocation_fraction: float = 0.01):
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
    
    def load_oos_data(self, coin: str) -> pd.DataFrame:
        """Load out-of-sample data for a coin."""
        filepath = os.path.join(OUT_OF_SAMPLE_DATA_DIR, f"{coin}_oos.csv")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            return df.sort_index().dropna()
        except Exception:
            return None
    
    def generate_trades_for_pair(self, ref_coin: str, trading_coin: str) -> pd.DataFrame:
        """Generate all trades for a specific pair."""
        
        # Load data
        ref_df = self.load_oos_data(ref_coin)
        traded_df = self.load_oos_data(trading_coin)
        
        if ref_df is None or traded_df is None:
            return pd.DataFrame()
        
        # Align data
        ix = ref_df.index.intersection(traded_df.index)
        ref_df = ref_df.loc[ix]
        traded_df = traded_df.loc[ix]
        
        if len(ref_df) < 100:
            return pd.DataFrame()
        
        try:
            # Generate signals using same algorithm
            traded_df = traded_df.copy()
            traded_df["diff"] = np.log(traded_df["close"]).diff()
            traded_df["next_return"] = traded_df["diff"].shift(-1)
            
            # CMMA indicators
            ref_atr = ta.atr(ref_df["high"], ref_df["low"], ref_df["close"], ATR_LOOKBACK)
            ref_ma = ref_df["close"].rolling(LOOKBACK).mean()
            ref_cmma = (ref_df["close"] - ref_ma) / (ref_atr * LOOKBACK ** 0.5)
            
            trd_atr = ta.atr(traded_df["high"], traded_df["low"], traded_df["close"], ATR_LOOKBACK)
            trd_ma = traded_df["close"].rolling(LOOKBACK).mean()
            trd_cmma = (traded_df["close"] - trd_ma) / (trd_atr * LOOKBACK ** 0.5)
            
            intermarket_diff = trd_cmma - ref_cmma
            
            # Generate signals
            signal = np.zeros(len(intermarket_diff))
            position = 0
            values = intermarket_diff.values
            for i in range(len(values)):
                v = values[i]
                if not np.isnan(v):
                    if v > THRESHOLD:
                        position = 1
                    if v < -THRESHOLD:
                        position = -1
                    if position == 1 and v <= 0:
                        position = 0
                    if position == -1 and v >= 0:
                        position = 0
                signal[i] = position
            
            traded_df["sig"] = signal
            
            # Get trades
            long_trades, short_trades, all_trades = get_trades_from_signal(traded_df, signal)
            
            # Add pair information
            all_trades['ref_coin'] = ref_coin
            all_trades['trading_coin'] = trading_coin
            all_trades['pair_id'] = f"{ref_coin}_{trading_coin}"
            
            return all_trades
            
        except Exception:
            return pd.DataFrame()
    
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
    
    def simulate_portfolio(self, validated_pairs: pd.DataFrame) -> Dict:
        """Run portfolio simulation on validated pairs."""
        
        print(f"\n🚀 Starting Portfolio Simulation")
        print(f"📊 Processing {len(validated_pairs)} validated pairs...")
        
        # Generate all trades from all pairs
        all_trades_data = []
        
        for _, pair in tqdm(validated_pairs.iterrows(), total=len(validated_pairs), desc="Generating trades"):
            ref_coin = pair['reference_coin']
            trading_coin = pair['trading_coin']
            
            pair_trades = self.generate_trades_for_pair(ref_coin, trading_coin)
            
            if len(pair_trades) > 0:
                for entry_time, trade in pair_trades.iterrows():
                    all_trades_data.append({
                        'entry_time': entry_time,
                        'exit_time': trade['exit_time'],
                        'trade_return': trade['return'],  # Use simple return
                        'trade_type': 'long' if trade['type'] == 1 else 'short',
                        'pair_id': trade['pair_id'],
                        'trade_id': f"{trade['pair_id']}_{entry_time.strftime('%Y%m%d_%H%M%S')}"
                    })
        
        if not all_trades_data:
            print("❌ No trades generated!")
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
        max_active_trades = 0
        
        for event in events:
            current_time = event['timestamp']
            
            if event['type'] == 'exit':
                trade_pnl = self.close_trade(event['trade_id'], current_time)
                if trade_pnl != 0:
                    trades_closed += 1
            
            elif event['type'] == 'entry':
                trade = event['trade']
                
                # Calculate trade amount
                max_trades = int(1 / self.allocation_fraction)
                current_portfolio_value = self.available_budget + self.allocated_budget
                trade_amount = current_portfolio_value / max_trades
                
                if self.can_allocate_trade(trade_amount) and trade_amount > 0:
                    self.open_trade(trade, trade_amount)
                    trades_opened += 1
                else:
                    trades_rejected += 1
            
            # Update tracking
            max_active_trades = max(max_active_trades, len(self.active_trades))
            self.update_portfolio_tracking(current_time)
        
        # Calculate final metrics
        final_pnl = self.total_portfolio_value - self.initial_budget
        total_return = (self.total_portfolio_value / self.initial_budget) - 1
        
        print(f"\n📊 PORTFOLIO SIMULATION RESULTS:")
        print(f"   Trades Opened: {trades_opened}")
        print(f"   Trades Closed: {trades_closed}")
        print(f"   Trades Rejected: {trades_rejected}")
        print(f"   Maximum Active Trades: {max_active_trades}")
        print(f"   Theoretical Max: {int(1/self.allocation_fraction)}")
        print(f"   Final Portfolio Value: ${self.total_portfolio_value:,.2f}")
        print(f"   Total P&L: ${final_pnl:,.2f}")
        print(f"   Total Return: {total_return:.2%}")
        
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
    parser.add_argument('--initial_budget', type=float, default=10000.0,
                       help='Initial portfolio budget (default: 10000)')
    parser.add_argument('--allocation_fraction', type=float, default=0.01,
                       help='Allocation fraction per trade (default: 0.01)')
    
    args = parser.parse_args()
    
    print("🚀 Portfolio Simulation for 2024 Out-of-Sample Data")
    print("=" * 55)
    
    # Create output directory
    os.makedirs(PORTFOLIO_DIR, exist_ok=True)
    
    # Load validated pairs
    if not os.path.exists(VALIDATION_RESULTS_FILE):
        print(f"❌ Validation results not found: {VALIDATION_RESULTS_FILE}")
        print("Run run_out_of_sample_validation.py first.")
        return
    
    validated_pairs = pd.read_csv(VALIDATION_RESULTS_FILE)
    print(f"📊 Loaded {len(validated_pairs)} validated pairs")
    
    # Initialize simulator
    simulator = PortfolioSimulator(
        initial_budget=args.initial_budget,
        allocation_fraction=args.allocation_fraction
    )
    
    try:
        # Run simulation
        metrics = simulator.simulate_portfolio(validated_pairs)
        
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