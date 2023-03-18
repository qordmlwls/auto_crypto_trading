from os.path import join
from typing import List, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.component.binance.constraint import TIME_WINDOW, COLUMNS, COLUMN_LIMIT
from src.module.utills.data import parallelize_list_to_df, get_ma

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


ORDER_BOOK_RANK_SIZE = 100


def process_func(data_list: np.ndarray) -> pd.DataFrame:
    data_list = data_list.tolist()
    df_list = []
    for idx, data in tqdm(enumerate(data_list)):
        # 거래량 기준 정렬    
        bids = sorted(data['order_book']['bids'], key=lambda x: x[1], reverse=True)[:COLUMN_LIMIT]
        asks = sorted(data['order_book']['asks'], key=lambda x: x[1], reverse=True)[:COLUMN_LIMIT]
        
        price = pd.DataFrame(zip([data['ticker']['open']], [data['ticker']['high']], [data['ticker']['low']],
                                    [data['ticker']['close']], [data['ticker']['baseVolume']], [data['ticker']['datetime']]), columns=['open', 'high', 'low', 'close', 'volume', 'datetime'])
        for i in range(COLUMN_LIMIT):
            price[f'bid_{i}'] = bids[i][0]
            price[f'bid_volume_{i}'] = bids[i][1]
            price[f'ask_{i}'] = asks[i][0]
            price[f'ask_volume_{i}'] = asks[i][1]
        if idx == 0:
            pass
        
        # 차이가 1분인 데이터
        else: 
            diff = abs(int(data['ticker']['datetime'].split(':')[1]) - int(data_list[idx - 1]['ticker']['datetime'].split(':')[1]))
            if diff == 1 or diff == 59:     
                pass
            
            else:
                # 결측치 채우기 위해 1분 단위로 데이터 채워넣기 backfill
                for i in range(diff - 1):
                    df_list.append(price)
                    # if i == 10: # 10분 까지만 backfill
                    #     break
            
        # 현 시점 데이터
        df_list.append(price)
    df = pd.concat(df_list, axis=0)
    return df

def get_weighted_average_price(data: pd.DataFrame) -> pd.DataFrame:
    # data['weighted_average_price'] = (data['bid_0'] * data['bid_volume_0'] + data['ask_0'] * data['ask_volume_0']) / (data['bid_volume_0'] + data['ask_volume_0'])
    data['weighted_average_price1'] = 0.0
    data['weighted_average_price2'] = 0.0
    data['volume_imbalance'] = 0.0
    for idx in tqdm(range(len(data))):
        highest_bid_idx = np.argmax([data.loc[idx, f'bid_{i}'] for i in range(COLUMN_LIMIT)])
        lowest_ask_idx = np.argmin([data.loc[idx, f'ask_{i}'] for i in range(COLUMN_LIMIT)])
        second_highest_bid_idx = np.argsort([data.loc[idx, f'bid_{i}'] for i in range(COLUMN_LIMIT)])[::-1][1]
        second_lowest_ask_idx = np.argsort([data.loc[idx, f'ask_{i}'] for i in range(COLUMN_LIMIT)])[1]
        data.loc[idx, 'weighted_average_price1'] = (data.loc[idx, f'bid_{highest_bid_idx}'] * data.loc[idx, f'bid_volume_{highest_bid_idx}'] + data.loc[idx, f'ask_{lowest_ask_idx}'] * data.loc[idx, f'ask_volume_{lowest_ask_idx}']) / (data.loc[idx, f'bid_volume_{highest_bid_idx}'] + data.loc[idx, f'ask_volume_{lowest_ask_idx}'])
        data.loc[idx, 'weighted_average_price2'] = (data.loc[idx, f'bid_{second_highest_bid_idx}'] * data.loc[idx, f'bid_volume_{second_highest_bid_idx}'] + data.loc[idx, f'ask_{second_lowest_ask_idx}'] * data.loc[idx, f'ask_volume_{second_lowest_ask_idx}']) / (data.loc[idx, f'bid_volume_{second_highest_bid_idx}'] + data.loc[idx, f'ask_volume_{second_lowest_ask_idx}'])
        data.loc[idx, 'volume_imbalance'] = abs((data.loc[idx, f'ask_volume_{lowest_ask_idx}'] + data.loc[idx, f'ask_volume_{second_lowest_ask_idx}']) - (data.loc[idx, f'bid_volume_{highest_bid_idx}'] + data.loc[idx, f'bid_volume_{second_highest_bid_idx}']))
        
    return data
        
def preprocess(data_list: List, config: Dict) -> pd.DataFrame:

    # df_list = []
    # for idx, data in tqdm(enumerate(data_list)):
    #     # 거래량 기준 정렬    
    #     bids = sorted(data['order_book']['bids'], key=lambda x: x[1], reverse=True)[:COLUMN_LIMIT]
    #     asks = sorted(data['order_book']['asks'], key=lambda x: x[1], reverse=True)[:COLUMN_LIMIT]
        
    #     price = pd.DataFrame(zip([data['ticker']['open']], [data['ticker']['high']], [data['ticker']['low']],
    #                                 [data['ticker']['close']], [data['ticker']['baseVolume']]), columns=['open', 'high', 'low', 'close', 'volume'])
    #     for i in range(COLUMN_LIMIT):
    #         price[f'bid_{i}'] = bids[i][0]
    #         price[f'bid_volume_{i}'] = bids[i][1]
    #         price[f'ask_{i}'] = asks[i][0]
    #         price[f'ask_volume_{i}'] = asks[i][1]
    #     if idx == 0:
    #         pass
        
    #     # 차이가 1분인 데이터
    #     else: 
    #         diff = abs(int(data['ticker']['datetime'].split(':')[1]) - int(data_list[idx - 1]['ticker']['datetime'].split(':')[1]))
    #         if diff == 1 or diff == 59:     
    #             pass
            
    #         else:
    #             # 결측치 채우기 위해 1분 단위로 데이터 채워넣기 backfill
    #             for i in range(diff - 1):
    #                 df_list.append(price)
            
    #     # 현 시점 데이터
    #     df_list.append(price)
    # df = pd.concat(df_list)
    # truncate df for time series
    df = parallelize_list_to_df(data_list, process_func)
    df.sort_values(by='datetime', inplace=True)
    
    df.drop('datetime', axis=1, inplace=True)
    # columns = ['open', 'high', 'low', 'close', 'volume', f"ma_{config['moving_average_window']}", "ma_25"] + [f'bid_{i}' for i in range(COLUMN_LIMIT)] \
    #             + [f'ask_{i}' for i in range(COLUMN_LIMIT)] + [f'bid_volume_{i}' for i in range(COLUMN_LIMIT)] \
    #             + [f'ask_volume_{i}' for i in range(COLUMN_LIMIT)]
    
    if len(set(['open', 'high', 'low', 'close', 'volume']) & set(df.columns)) == 5:
        df[['open_diff', 'high_diff', 'low_diff', 'close_diff', 'volume_diff']] = df[['open', 'high', 'low', 'close', 'volume']].diff()
        df = df.iloc[1:]
        
    df = get_ma(df, config['moving_average_window'])
    # df = get_ma(df, 25)[COLUMNS]
    df = get_ma(df, 25)
    df.reset_index(drop=True, inplace=True)
    df = df.iloc[config['moving_average_window'] - 1:]
    df = df.iloc[-config['time_minute_limit']:]
    df.reset_index(drop=True, inplace=True)
    df = get_weighted_average_price(df)
    if len(df) % TIME_WINDOW != 0:
        df = df.iloc[(len(df) % TIME_WINDOW):]
    return df
    