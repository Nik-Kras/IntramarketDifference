import pandas as pd
import numpy as np

def get_trades_from_signal(data: pd.DataFrame, signal: np.array):
    # Gets trade entry and exit times from a signal
    # that has values of -1, 0, 1. Denoting short,flat,and long.
    # No position sizing.
    
    # TODO: PERFORMANCE OPTIMIZATION - CRITICAL BOTTLENECK
    # PERFORMANCE: This function takes ~6,750μs per pair (42% of total processing time)
    # ISSUE: O(n) Python loop through 13,000+ signal values per pair
    # 
    # OPTIMIZATION: Replace with vectorized NumPy operations
    # 1. Use np.diff(signal) to detect signal changes in O(1) operation
    # 2. Use boolean masking: entry_points = (signal != 0) & (np.diff(signal, prepend=0) != 0)
    # 3. Vectorize price/timestamp extraction using fancy indexing
    # 4. Build trade arrays directly without Python loops
    # 
    # TESTING: Create comprehensive test cases before optimization
    # - Test known signal sequences: [0,1,0,-1,0] → expect 1 long + 1 short trade
    # - Compare output DataFrames (long_trades, short_trades, all_trades) for identical results
    # - Test edge cases: no trades, single trade, multiple alternating trades
    # 
    # EXPECTED: 5-10x faster (from 6,750μs to 675-1,350μs per pair)

    long_trades = []
    short_trades = []

    close_arr = data['close'].to_numpy()
    last_sig = 0.0
    open_trade = None
    idx = data.index
    for i in range(len(data)):
        if signal[i] == 1.0 and last_sig != 1.0: # Long entry
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                short_trades.append(open_trade)

            open_trade = [idx[i], close_arr[i], -1, np.nan]
        if signal[i] == -1.0  and last_sig != -1.0: # Short entry
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                long_trades.append(open_trade)

            open_trade = [idx[i], close_arr[i], -1, np.nan]
        
        if signal[i] >= 0.0 and last_sig == -1.0: # Short exit
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                short_trades.append(open_trade)
                open_trade = None

        if signal[i] <= 0.0  and last_sig == 1.0: # Long exit
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                long_trades.append(open_trade)
                open_trade = None

        last_sig = signal[i]

    long_trades = pd.DataFrame(long_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])
    short_trades = pd.DataFrame(short_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])

    long_trades['return'] = (long_trades['exit_price'] - long_trades['entry_price']) / long_trades['entry_price']
    short_trades['return'] = -1 * (short_trades['exit_price'] - short_trades['entry_price']) / short_trades['entry_price']
    long_trades = long_trades.set_index('entry_time')
    short_trades = short_trades.set_index('entry_time')
    
    long_trades['type'] = 1
    short_trades['type'] = -1
    all_trades = pd.concat([long_trades, short_trades])
    all_trades = all_trades.sort_index()
    
    return long_trades, short_trades, all_trades
