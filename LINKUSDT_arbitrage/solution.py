# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:20:46 2025

@author: Linus.Ong
"""
import os
import numpy as np
import pandas as pd

def get_data(directory = r"C:\Users\linus.ong\Downloads\python_testing1"):
    
    bar_data_BNC_path = os.path.join(directory, "bar_bnc.csv")
    bar_data_HB_path = os.path.join(directory, "bar_hb.csv")
    
    orderbook_BNC_path = os.path.join(directory, "orderbook_bnc.csv")
    orderbook_HB_path = os.path.join(directory, "orderbook_hb.csv")
    
    bar_data_BNC_df = pd.read_csv(bar_data_BNC_path)
    bar_data_HB_df = pd.read_csv(bar_data_HB_path)
    
    orderbook_BNC_df = pd.read_csv(orderbook_BNC_path)
    orderbook_HB_df = pd.read_csv(orderbook_HB_path)
    
    return bar_data_BNC_df, orderbook_BNC_df, bar_data_HB_df, orderbook_HB_df

def get_all_timestamps_between_two_timestamps(start_timestamp, end_timestamp, freq = "S"):
    
    range_of_timestamps = pd.date_range(start_timestamp, end_timestamp, freq = freq)
    timestamp_df = range_of_timestamps.to_frame()
    timestamp_df = timestamp_df.drop(timestamp_df.columns[0], axis = 1)
    
    return timestamp_df

def calculate_market_depth_compared_to_volume_required(x, total_volume_required, total_volume_col_name, ask_vol, ask_price):
    
    if total_volume_required >= x[total_volume_col_name]:
        
        return 0
    
    else:
        
        return x[ask_vol]/x[total_volume_col_name]*x[ask_price]

def clean_data():
    
    bar_data_BNC_df, orderbook_BNC_df, bar_data_HB_df, orderbook_HB_df = get_data()
    
    orderbook_BNC_df['timestamp'] = pd.to_datetime(orderbook_BNC_df['timestamp'])
    orderbook_BNC_df = orderbook_BNC_df.set_index('timestamp').drop('symbol', axis=1) 
    orderbook_BNC_df.columns = ["BNC_" + col for col in orderbook_BNC_df.columns]
    
    orderbook_HB_df['timestamp'] = pd.to_datetime(orderbook_HB_df['timestamp'])
    orderbook_HB_df = orderbook_HB_df.set_index('timestamp').drop('symbol', axis=1) 
    orderbook_HB_df.columns = ["HB_" + col for col in orderbook_HB_df.columns]
    
    start_timestamp, end_timestamp = bar_data_BNC_df['timestamp'].values[0], bar_data_BNC_df['timestamp'].values[-1]
    timestamp_df = get_all_timestamps_between_two_timestamps(start_timestamp, end_timestamp, freq = "S")
    
    merged_df = pd.merge(timestamp_df, orderbook_BNC_df, left_index=True, right_index=True, how='left')
    merged_df = pd.merge(merged_df, orderbook_HB_df, left_index=True, right_index=True, how='left')
    
    merged_df['BNC-HB_buy_vol_1'] = merged_df[['BNC_a1', 'HB_b1', 'BNC_av1', 'HB_bv1']].apply(lambda x: (x['BNC_a1'] - x['HB_b1'], min(x['BNC_av1'], x['HB_bv1'])), axis = 1)
    merged_df['BNC-HB_sell_vol_1'] = merged_df[['BNC_b1', 'HB_a1', 'BNC_bv1', 'HB_av1']].apply(lambda x: (x['BNC_b1'] - x['HB_a1'], min(x['BNC_bv1'], x['HB_av1'])), axis = 1)
    
    # data_df = create_market_depth(merged_df, number_of_depths = 10)
    
    return merged_df

def get_signal(x, cost = 0.002):
    
    if x['BNC-HB_sell_vol_1'][0] > cost:
        
        return -1
    
    elif x['BNC-HB_buy_vol_1'][0] < -cost:
        
        return 1
    
    else:
        
        return 0
    
def get_best_bid_ask_and_executed_volume(x, max_depth_to_check = 10, cost = 0.002):
    
    if x['signal'] == 1:
        
        # check for buy BNC and sell HB spread
        BNC_prefix = 'BNC_a'
        BNC_asks = [BNC_prefix + str(x) for x in list(range(1, max_depth_to_check +1))]
        HB_prefix = 'HB_b'
        HB_bids = [HB_prefix + str(x) for x in list(range(1, max_depth_to_check +1))]
        
        
        combinations_that_work = []
        total_bid_vol = 0
        total_ask_vol = 0
        for ask_price in BNC_asks:
            
            total_ask_vol += x[ask_price.replace("_a", "_av")]
            
            for bid_price in HB_bids:
                
                total_bid_vol += x[bid_price.replace("_b", "_bv")]
                
                spread = x[ask_price] - x[bid_price]
                if spread < -cost:
                    
                    vol_executed = min(total_bid_vol, total_ask_vol)
                    if combinations_that_work == []:
                        combinations_that_work = [ask_price, bid_price, round(spread, 6), vol_executed]
                    elif vol_executed > combinations_that_work[-1]:
                        combinations_that_work = [ask_price, bid_price, round(spread, 6), vol_executed]
                        
            total_bid_vol = 0
                    
    elif x['signal'] == -1:
        
        # check for sell BNC and buy HB spread
        BNC_prefix = 'BNC_b'
        BNC_asks = [BNC_prefix + str(x) for x in list(range(1, max_depth_to_check +1))]
        HB_prefix = 'HB_a'
        HB_bids = [HB_prefix + str(x) for x in list(range(1, max_depth_to_check +1))]
        
        combinations_that_work = []
        total_bid_vol = 0
        total_ask_vol = 0
        for bid_price in BNC_asks:
            
            total_bid_vol += x[bid_price.replace("_b", "_bv")]
            
            for ask_price in HB_bids:
                
                total_ask_vol += x[ask_price.replace("_a", "_av")]
                
                spread = x[bid_price] - x[ask_price]
                if spread > cost:

                    vol_executed = min(total_bid_vol, total_ask_vol)
                    if combinations_that_work == []:
                        combinations_that_work = [ask_price, bid_price, round(spread, 6), vol_executed]
                    elif vol_executed > combinations_that_work[-1]:
                        combinations_that_work = [ask_price, bid_price, round(spread, 6), vol_executed]
                        
            total_ask_vol = 0
        
    else:
        
          
        combinations_that_work = ""
      
    return combinations_that_work

def backtest(df, unwind_at):
    
    exit_buy = unwind_at
    exit_sell = -unwind_at
    
    df['traded_signal'] = 0
    df['accumulated_position'] = 0
    df['average_traded_price'] = 0
    df['exited_price'] = 0
    
    position_side = 0
    accumulated_position = 0
    entry_average_price = 0

    
    for i in range(0, len(df)):
        
        row = df.iloc[i]
        signal = row['signal']
        
        if signal != 0 and position_side == 0: # enter
        
            position_side = signal
            entry_average_price = row['entry_spread']
            accumulated_position = row['volume_executed']
            
            df.at[row.name, 'traded_signal'] = position_side
            df.at[row.name, 'accumulated_position'] = accumulated_position
            df.at[row.name, 'average_traded_price'] = entry_average_price
            
            
        elif position_side == signal and signal != 0: # accumulate position on same side
            
            first_partprice = row['volume_executed']/(accumulated_position + row['volume_executed']) *  row['entry_spread']
            second_partprice = accumulated_position/(accumulated_position + row['volume_executed']) *  entry_average_price
            entry_average_price = first_partprice + second_partprice
            
            accumulated_position += row['volume_executed']
            
            df.at[row.name, 'traded_signal'] = position_side
            df.at[row.name, 'accumulated_position'] = accumulated_position
            df.at[row.name, 'average_traded_price'] = entry_average_price
            
        elif position_side != 0 and (signal == 0 or signal == position_side * -1):
            
            # check for unwind
            
            if position_side == -1: # check for buy spread to exit
                
                buy_spread = row['BNC-HB_buy_vol_1'][0]
                
                if buy_spread <= exit_sell:
                    
                    buy_spread_volume = row['BNC-HB_buy_vol_1'][1]
                    if buy_spread_volume >= accumulated_position: # unwind all
                        
                        df.at[row.name, 'traded_signal'] = position_side
                        df.at[row.name, 'accumulated_position'] = accumulated_position
                        df.at[row.name, 'average_traded_price'] = entry_average_price
                        df.at[row.name, 'exited_price'] = buy_spread
                        
                        accumulated_position = 0
                        position_side = 0
                        entry_average_price = 0
                        
                    else: # partially unwind
                        
                        accumulated_position -= buy_spread_volume
                        
                        df.at[row.name, 'traded_signal'] = position_side
                        df.at[row.name, 'accumulated_position'] = accumulated_position
                        df.at[row.name, 'average_traded_price'] = entry_average_price
                        df.at[row.name, 'exited_price'] = buy_spread
                        
                else:
                    
                    df.at[row.name, 'traded_signal'] = position_side
                    df.at[row.name, 'accumulated_position'] = accumulated_position
                    df.at[row.name, 'average_traded_price'] = entry_average_price
            
            elif position_side == 1: # check for sell spread to exit
            
                sell_spread = row['BNC-HB_sell_vol_1'][0]
                
                if sell_spread >= exit_buy:
                    
                    sell_spread_volume = row['BNC-HB_sell_vol_1'][1]
                    if sell_spread_volume >= accumulated_position: # unwind all

                        df.at[row.name, 'traded_signal'] = position_side
                        df.at[row.name, 'accumulated_position'] = accumulated_position
                        df.at[row.name, 'average_traded_price'] = entry_average_price
                        df.at[row.name, 'exited_price'] = sell_spread
                        
                        accumulated_position = 0
                        position_side = 0
                        entry_average_price = 0
                        
                    else: # partially unwind
                    
                        accumulated_position -= sell_spread_volume
                        
                        df.at[row.name, 'traded_signal'] = position_side
                        df.at[row.name, 'accumulated_position'] = accumulated_position
                        df.at[row.name, 'average_traded_price'] = entry_average_price
                        df.at[row.name, 'exited_price'] = sell_spread
                        
                else:
                    
                    df.at[row.name, 'traded_signal'] = position_side
                    df.at[row.name, 'accumulated_position'] = accumulated_position
                    df.at[row.name, 'average_traded_price'] = entry_average_price
                
            
        else:
            
            df.at[row.name, 'traded_signal'] = position_side
            df.at[row.name, 'accumulated_position'] = accumulated_position
            df.at[row.name, 'average_traded_price'] = entry_average_price
            
    return df

def get_tradeable_prices(x):
    
    if x['traded_signal'] == 1:
        
        return x['BNC-HB_sell_vol_1'][0]
    
    elif x['traded_signal'] == -1:
        
        return x['BNC-HB_buy_vol_1'][0]
    
    else:
        
        return 0
    
def get_absolute_USD_returns(x):
    
    if x['traded_signal'] == -1:
        
        return x['accumulated_position'] * (x['average_traded_price'] - x['tradeable_prices'])
    
    elif x['traded_signal'] == 1:
        
        return x['accumulated_position'] * (x['tradeable_prices'] - x['average_traded_price'])
    
    else:
        
        return 0

def calculate_risk_profits(backtest_df, margin = 0.05):
    
    # backtest_df['tradeable_prices'] = backtest_df[['signal', 'entry_spread', 'traded_signal', 'BNC-HB_buy_vol_1', 'BNC-HB_sell_vol_1']].apply(lambda x: get_tradeable_prices(x), axis = 1)
    # backtest_df['abs_USD_returns'] = backtest_df.apply(lambda x: get_absolute_USD_returns(x), axis = 1)
    # backtest_df['percentage_returns'] = backtest_df['abs_USD_returns'].apply(lambda x: x/(27*margin))
    # backtest_df['cumulative_abs_USD_returns'] = backtest_df['abs_USD_returns'].cumsum()
    
    backtest_df['realized_absolute_returns'] = 0
    
    for i in range(1, len(backtest_df)):
        
        prev_row = backtest_df.iloc[i-1]
        row = backtest_df.iloc[i]
        
        if row['exited_price'] != 0:
            
            if row['traded_signal'] == -1:
            
                qty_realized = abs(prev_row['accumulated_position'] - row['accumulated_position'])
                spread_captured = row['average_traded_price'] - row['exited_price']
                backtest_df.at[row.name, 'realized_absolute_returns'] = qty_realized * spread_captured
                
            elif row['traded_signal'] == 1:
                
                qty_realized = abs(prev_row['accumulated_position'] - row['accumulated_position'])
                spread_captured = row['exited_price'] - row['average_traded_price']
                backtest_df.at[row.name, 'realized_absolute_returns'] = qty_realized * spread_captured
                
    backtest_df['cumulative_realized_absolute_returns'] = backtest_df['realized_absolute_returns'].cumsum()
    
    required_df = backtest_df[['cumulative_realized_absolute_returns']].resample('T').last()
    # backtest_df['percentage_returns'] = backtest_df['realized_absolute_returns'].apply(lambda x: x/(27*margin))
    # backtest_df['cumulative_returns'] = backtest_df['percentage_returns'].cumsum()
    
    return required_df, required_df['cumulative_realized_absolute_returns'].values[-1]      
            
def main(margin = 0.02):
    
    df = clean_data()
    
    df['signal'] = df[['BNC-HB_sell_vol_1', 'BNC-HB_buy_vol_1']].apply(lambda x: get_signal(x), axis = 1)
    df['best_bid_ask_vol'] = df.apply(lambda x: get_best_bid_ask_and_executed_volume(x, max_depth_to_check = 10, cost = 0.002), axis = 1)
    df['entry_spread'] = df['best_bid_ask_vol'].apply(lambda x: '' if x == '' else x[2])
    df['volume_executed'] = df['best_bid_ask_vol'].apply(lambda x: '' if x == '' else x[3])
    
    # optimize a parameter on where to exit
    
    
    unwinds = list(np.arange(0, 0.022, 0.002))

    record = []
    best_level_to_unwind = 0
    best_PNL = 0
    for unwind_level in unwinds:

        backtest_df = backtest(df, unwind_level)
        _, cumulative_PNL = calculate_risk_profits(backtest_df, margin = 0.05)
        record.append((unwind_level, cumulative_PNL))
        
        if cumulative_PNL > best_PNL:
            
            best_level_to_unwind = unwind_level
            best_PNL = cumulative_PNL
            
    """
    Best level to unwind at is 0
    """
            
    
    
    
    
    
        
    

    
    
    

    
    
    


