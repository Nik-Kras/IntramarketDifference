#!/usr/bin/env python3
"""
Rolling Period Manager

Handles time period splits and data range management for rolling window validation experiments.
Provides utilities for creating IS/OOS combinations and managing temporal data separation.
"""

import pandas as pd
from typing import List, Tuple, Dict, Any
import os

class RollingPeriodManager:
    """Manages rolling time periods for robust window validation."""
    
    # Default window lengths for experiments
    DEFAULT_WINDOW_LENGTHS = [12, 15, 18, 21, 24]
    
    def __init__(self, 
                 start_year: int = 2018, 
                 end_year: int = 2024,
                 is_length_months: int = 24,
                 oos_length_months: int = 12):
        """
        Initialize rolling period manager.
        
        Args:
            start_year: First year of available data
            end_year: Last year of available data  
            is_length_months: Length of In-Sample period in months
            oos_length_months: Length of Out-of-Sample period in months
            
        Raises:
            ValueError: If parameters are invalid
        """
        if start_year >= end_year:
            raise ValueError(f"start_year ({start_year}) must be < end_year ({end_year})")
        if end_year - start_year < 3:
            raise ValueError(f"Need at least 3 years of data, got {end_year - start_year}")
        if is_length_months <= 0 or oos_length_months <= 0:
            raise ValueError("IS and OOS lengths must be positive")
            
        self.start_year = start_year
        self.end_year = end_year
        self.is_length_months = is_length_months
        self.oos_length_months = oos_length_months
        
        self.periods = self._generate_rolling_periods()
    
    def _generate_rolling_periods(self) -> List[Dict[str, str]]:
        """Generate rolling IS/OOS period combinations with exact 2-year IS and 1-year OOS."""
        
        # Generate periods dynamically based on available years
        # Each period needs: IS (2 years) + OOS (1 year) = 3 years total
        max_start_year = self.end_year - 3 + 1  # Include the final period ending at end_year
        return [self._create_period(year) for year in range(self.start_year, max_start_year)]
    
    def _create_period(self, is_start_year: int) -> Dict[str, str]:
        """Generate a single period definition.
        
        Args:
            is_start_year: Starting year for the In-Sample period
            
        Returns:
            Dictionary with period dates and metadata
        """
        is_end_year = is_start_year + 2
        oos_end_year = is_end_year + 1
        
        return {
            'is_start': f'{is_start_year}-01-01',
            'is_end': f'{is_end_year}-01-01',
            'oos_start': f'{is_end_year}-01-01',
            'oos_end': f'{oos_end_year}-01-01',
            'period_name': f'{is_start_year}-{is_end_year}_IS_{is_end_year}-{oos_end_year}_OOS',
            'is_years': f'{is_start_year}-{is_end_year}',
            'oos_years': f'{is_end_year}-{oos_end_year}'
        }
    
    def get_periods(self) -> List[Dict[str, str]]:
        """Get all generated rolling periods."""
        return self.periods
    
    def get_period_by_name(self, period_name: str) -> Dict[str, str]:
        """Get specific period by name.
        
        Args:
            period_name: Name of the period to retrieve
            
        Returns:
            Period dictionary
            
        Raises:
            ValueError: If period_name not found
        """
        for period in self.periods:
            if period['period_name'] == period_name:
                return period
        raise ValueError(f"Period {period_name} not found")
    
    def filter_data_by_period(self, df: pd.DataFrame, period: Dict[str, str], 
                             data_type: str = 'is') -> pd.DataFrame:
        """
        Filter dataframe by period dates.
        
        Args:
            df: DataFrame with datetime index
            period: Period dictionary from get_periods()
            data_type: 'is' for In-Sample, 'oos' for Out-of-Sample
            
        Returns:
            Filtered DataFrame
            
        Raises:
            ValueError: If data_type is invalid or DataFrame lacks datetime info
        """
        if data_type not in ['is', 'oos']:
            raise ValueError(f"data_type must be 'is' or 'oos', got '{data_type}'")
        if data_type == 'is':
            start_date = period['is_start']
            end_date = period['is_end']
        elif data_type == 'oos':
            start_date = period['oos_start'] 
            end_date = period['oos_end']
        else:
            raise ValueError("data_type must be 'is' or 'oos'")
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            elif 'open time' in df.columns:
                df = df.set_index('open time')
            else:
                raise ValueError("DataFrame must have datetime index or 'date'/'open time' column")
        
        # Convert to datetime if needed
        df.index = pd.to_datetime(df.index)
        
        # Filter by date range
        mask = (df.index >= start_date) & (df.index < end_date)
        return df[mask].copy()
    
    def get_window_data_range(self, period: Dict[str, str], window_months: int) -> Tuple[str, str]:
        """
        Get the actual data range for a specific window within IS period.
        
        The window extends backwards from the IS end date by window_months.
        If this would go before the IS start date, it's clipped to IS start.
        
        Args:
            period: Period dictionary from get_periods()
            window_months: Window length in months (must be > 0 and <= IS period length)
            
        Returns:
            Tuple of (start_date, end_date) for data loading
            
        Raises:
            ValueError: If window_months is invalid
        """
        if window_months <= 0:
            raise ValueError(f"window_months must be positive, got {window_months}")
        if window_months > self.is_length_months:
            raise ValueError(f"window_months ({window_months}) cannot exceed IS period length ({self.is_length_months})")
            
        is_start = pd.to_datetime(period['is_start'])
        is_end = pd.to_datetime(period['is_end'])
        
        # Calculate window start (go back from IS_END)
        window_start = is_end - pd.DateOffset(months=window_months)
        
        # Make sure we don't go before overall IS start
        actual_start = max(window_start, is_start)
        
        return actual_start.strftime('%Y-%m-%d'), is_end.strftime('%Y-%m-%d')
    
    def create_period_directory_structure(self, base_dir: str = "results/rolling_experiments") -> str:
        """Create directory structure for rolling experiments.
        
        Returns:
            str: The base directory path that was created
        """
        
        os.makedirs(base_dir, exist_ok=True)
        
        for period in self.periods:
            period_dir = os.path.join(base_dir, period['period_name'])
            os.makedirs(period_dir, exist_ok=True)
            
            # Create subdirectories for each window length
            for window_months in self.DEFAULT_WINDOW_LENGTHS:
                window_dir = os.path.join(period_dir, f"{window_months}mo")
                os.makedirs(window_dir, exist_ok=True)
        
        # Create analysis directory
        analysis_dir = os.path.join(base_dir, "rolling_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        print(f"📁 Created directory structure for {len(self.periods)} periods")
        return base_dir
    
    def get_market_regime_info(self, period: Dict[str, str]) -> Dict[str, Any]:
        """Get market regime characteristics for a period (placeholder for future enhancement).
        
        Args:
            period: Period dictionary from get_periods()
            
        Returns:
            Dictionary with market regime information
        """
        
        # This could be enhanced with actual market data analysis
        return {
            'period_name': period['period_name'],
            'is_start': period['is_start'],
            'oos_start': period['oos_start'],
            'market_phase': 'unknown',  # Could analyze crypto market trends
            'volatility_regime': 'unknown'  # Could calculate volatility metrics
        }
    
    def print_experiment_overview(self) -> None:
        """Print overview of all rolling periods."""
        
        print("🔄 Rolling Window Validation Experiment Overview")
        print("=" * 60)
        print(f"Total Periods: {len(self.periods)}")
        print(f"IS Length: {self.is_length_months} months")
        print(f"OOS Length: {self.oos_length_months} months")
        print(f"Window Lengths to Test: {self.DEFAULT_WINDOW_LENGTHS} months")
        print("")
        print("Period Schedule:")
        print("-" * 60)
        
        for i, period in enumerate(self.periods, 1):
            print(f"{i}. {period['period_name']}")
            print(f"   IS:  {period['is_start']} to {period['is_end']}")
            print(f"   OOS: {period['oos_start']} to {period['oos_end']}")
            print("")
        
        total_experiments = len(self.periods) * 5  # 5 window lengths
        print(f"Total Window Experiments: {total_experiments}")
        print(f"Estimated Completion Time: {total_experiments * 7:.0f} minutes (assuming 5-10 minutes per window)")

def main() -> None:
    """Demo the rolling period manager."""
    
    # Create manager with custom parameters
    manager = RollingPeriodManager(
        start_year=2018,
        end_year=2024,
        is_length_months=24,  # 2 years IS
        oos_length_months=12  # 1 year OOS
    )
    
    # Print overview
    manager.print_experiment_overview()
    
    # Create directory structure
    created_dir = manager.create_period_directory_structure()
    print(f"📁 Directory structure created at: {created_dir}")
    
    # Demo period filtering
    print("\n🔬 Demo: Period Data Filtering")
    print("-" * 40)
    
    # Create sample data
    date_range = pd.date_range('2018-01-01', '2024-12-31', freq='D')
    sample_df = pd.DataFrame({
        'close': range(len(date_range)),
        'volume': range(len(date_range))
    }, index=date_range)
    
    # Test filtering for first period
    first_period = manager.get_periods()[0]
    print(f"Testing period: {first_period['period_name']}")
    
    is_data = manager.filter_data_by_period(sample_df, first_period, 'is')
    oos_data = manager.filter_data_by_period(sample_df, first_period, 'oos')
    
    print(f"IS Data: {len(is_data)} days ({is_data.index[0]} to {is_data.index[-1]})")
    print(f"OOS Data: {len(oos_data)} days ({oos_data.index[0]} to {oos_data.index[-1]})")
    
    # Test window data range
    window_start, window_end = manager.get_window_data_range(first_period, 18)
    print(f"18-month window range: {window_start} to {window_end}")

if __name__ == "__main__":
    main()