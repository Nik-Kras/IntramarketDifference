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
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
# Import core functions
# Core imports not used in this file

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
        trade_log_return = trade_info['trade_return']  # This is actually a log return
        
        # Convert log return to simple return for P&L calculation
        # Simple return = exp(log_return) - 1
        trade_simple_return = np.exp(trade_log_return) - 1
        trade_pnl = trade_amount * trade_simple_return
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
            'trade_return': trade_simple_return,
            'trading_coin': trade_info['trading_coin'],
            'reference_coin': trade_info['reference_coin'],
            'strategy_type': trade_info['strategy_type'],
            'trade_type': trade_info['trade_type']
        })
        
        # Remove from active trades
        del self.active_trades[trade_id]
        
        return trade_pnl
    
    def update_tracking(self, current_time: pd.Timestamp) -> None:
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
                            
                            # Convert DataFrame to list of dicts with correct column names
                            df_for_export = df.copy()
                            df_for_export['time_entered'] = df['entry_time'] 
                            df_for_export['time_exited'] = df['exit_time']
                            df_for_export['log_return'] = df['log_return']  # Keep original log_return for event creation
                            
                            trade_dicts = df_for_export[['time_entered', 'time_exited', 'log_return', 'trade_return', 'trading_coin', 
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
    
    def _convert_trades_to_standard_format(self, trades_data: List[Dict]) -> List[Dict]:
        """Convert custom trades data format to standard format."""
        
        all_trades = []
        
        for pair_data in trades_data:
            oos_trades = pair_data['oos_trades']
            
            for trade in oos_trades:
                all_trades.append({
                    'reference_coin': pair_data['reference_coin'],
                    'trading_coin': pair_data['trading_coin'],
                    'strategy_type': pair_data['trading_type'],
                    'time_entered': trade['time_entered'],
                    'time_exited': trade['time_exited'],
                    'log_return': trade['log_return'],
                    'trade_type': trade.get('trade_type', 'unknown'),
                    'trade_id': f"{pair_data['reference_coin']}_{pair_data['trading_coin']}_{trade['time_entered']}"
                })
        
        return all_trades
    
    def _create_chronological_events(self, all_trades: List[Dict]) -> List[Dict]:
        """Create chronological timeline events from trades."""
        
        events = []
        
        for trade in all_trades:
            # Entry event
            events.append({
                'timestamp': pd.to_datetime(trade['time_entered']),
                'type': 'entry',
                'trade': {
                    'trade_id': trade['trade_id'],
                    'entry_time': pd.to_datetime(trade['time_entered']),
                    'exit_time': pd.to_datetime(trade['time_exited']),
                    'trade_return': trade['log_return'],
                    'trading_coin': trade['trading_coin'],
                    'reference_coin': trade['reference_coin'],
                    'strategy_type': trade['strategy_type'],
                    'trade_type': trade['trade_type']
                }
            })
            
            # Exit event
            events.append({
                'timestamp': pd.to_datetime(trade['time_exited']),
                'type': 'exit',
                'trade_id': trade['trade_id']
            })
        
        # Sort events chronologically
        events.sort(key=lambda x: x['timestamp'])
        
        return events
    
    def _process_timeline_events(self, events: List[Dict]) -> Dict:
        """Process chronological events and simulate trading."""
        
        print(f"🔄 Processing {len(events)} chronological events...")
        
        trades_opened = 0
        trades_closed = 0
        trades_rejected = 0
        
        for event in tqdm(events, desc="Processing events"):
            current_time = event['timestamp']
            
            # Update portfolio timeline
            current_portfolio_value = self.available_budget + self.allocated_budget
            self.portfolio_timeline.append({
                'timestamp': current_time,
                'portfolio_value': current_portfolio_value,
                'available_budget': self.available_budget,
                'allocated_budget': self.allocated_budget,
                'active_trades_count': len(self.active_trades),
                'current_trade_amount': self.get_current_trade_amount()
            })
            
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
        
        # Close any remaining active trades
        final_time = events[-1]['timestamp'] if events else pd.Timestamp.now()
        remaining_trades = list(self.active_trades.keys())
        for trade_id in remaining_trades:
            self.close_trade(trade_id, final_time)
            trades_closed += 1
        
        # Calculate and return portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics()
        
        # Print summary
        print(f"📊 Portfolio Simulation Summary:")
        print(f"   Trades Opened: {trades_opened:,}")
        print(f"   Trades Closed: {trades_closed:,}")
        print(f"   Trades Rejected: {trades_rejected:,}")
        
        return portfolio_metrics

    def simulate_portfolio_with_trades(self, trades_data: List[Dict], 
                                      oos_start: str, oos_end: str) -> Dict:
        """
        Simulate portfolio using pre-loaded trades data for custom time periods.
        
        Args:
            trades_data: List of pair dictionaries with 'oos_trades' key
            oos_start: OOS period start date
            oos_end: OOS period end date
        
        Returns:
            Portfolio performance metrics dictionary
        """
        
        print(f"💰 Simulating portfolio with {len(trades_data)} pairs")
        print(f"   OOS Period: {oos_start} to {oos_end}")
        
        # Convert to standard format
        all_oos_trades = self._convert_trades_to_standard_format(trades_data)
        
        if not all_oos_trades:
            print("❌ No OOS trades to simulate!")
            return {}
        
        print(f"📊 Processing {len(all_oos_trades):,} OOS trades")
        
        # Create events and process - now properly returns metrics
        events = self._create_chronological_events(all_oos_trades)
        portfolio_metrics = self._process_timeline_events(events)
        
        return portfolio_metrics

    def simulate_portfolio(self, selected_pairs_df: pd.DataFrame) -> Dict:
        """
        Run complete portfolio simulation on OOS data.
        
        Args:
            selected_pairs_df: DataFrame of selected pairs
            
        Returns:
            Portfolio performance metrics dictionary
        """
        
        # Load all OOS trades for selected pairs
        all_oos_trades = self.load_oos_trades_for_pairs(selected_pairs_df)
        
        if not all_oos_trades:
            print("❌ No OOS trades loaded!")
            return {}
        
        print(f"📊 Loaded {len(all_oos_trades):,} OOS trades")
        
        # Create events and process - now properly returns metrics
        events = self._create_chronological_events(all_oos_trades)
        portfolio_metrics = self._process_timeline_events(events)
        
        return portfolio_metrics
    
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
        
        # Calculate metrics - use final portfolio value from timeline
        final_portfolio_value = portfolio_values.iloc[-1] if len(portfolio_values) > 0 else self.initial_capital
        total_return = (final_portfolio_value / self.initial_capital) - 1
        
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
            'final_value': final_portfolio_value
        }
    
    def create_visualizations(self, output_dir: str) -> None:
        """Create portfolio simulation visualizations using shared utilities."""
        
        from visualization_utils import (
            create_portfolio_equity_curve,
            create_budget_allocation_timeline,
            create_active_trades_timeline,
            create_drawdown_curve
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.portfolio_timeline:
            print("❌ No timeline data for visualizations")
            return
        
        # Convert timeline data to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_timeline)
        
        # Ensure timestamp column exists and set as index
        if 'timestamp' in portfolio_df.columns:
            portfolio_df = portfolio_df.set_index('timestamp').sort_index()
        elif 'time' in portfolio_df.columns:
            portfolio_df = portfolio_df.set_index('time').sort_index()
        else:
            print("❌ No timestamp column found in portfolio timeline")
            return
        
        # Create visualizations using shared functions
        create_portfolio_equity_curve(
            portfolio_df, self.initial_capital,
            os.path.join(output_dir, 'portfolio_equity_curve.png')
        )
        
        create_budget_allocation_timeline(
            portfolio_df,
            os.path.join(output_dir, 'budget_allocation.png')
        )
        
        create_active_trades_timeline(
            portfolio_df, self.max_trades,
            os.path.join(output_dir, 'active_trades_timeline.png')
        )
        
        create_drawdown_curve(
            portfolio_df, self.initial_capital,
            os.path.join(output_dir, 'drawdown_curve.png')
        )
        
        print(f"✅ Visualizations saved to: {output_dir}/")
    
    def save_metrics(self, output_file: str) -> None:
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

def main() -> None:
    """Test the portfolio simulator."""
    
    # Test with sample configuration
    simulator = WindowPortfolioSimulator(initial_capital=1000.0, n_pairs=10)
    
    print("✅ Portfolio simulator ready for testing")

if __name__ == "__main__":
    main()