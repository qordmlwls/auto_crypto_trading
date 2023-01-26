import os
from os.path import join
import json
from typing import List, NoReturn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from sklearn.externals import joblib

# from src.component.binance.constraint import BINANCE_API_KEY, BINANCE_SECRET_KEY
# from src.component.binance.binance import Binance


# ROOT_DIR = os.environ.get('PYTHONPATH', '')
# DATA_DIR = join(ROOT_DIR, 'data')
# TARGET_COIN_TICKER = 'BTC/USDT'
# TARGET_COIN_SYMBOL = 'BTCUSDT'
ORDER_BOOK_RANK_SIZE = 100


# class Preprocess:
#     def __init__(self):
#         self.data_list = []
#         # self.binance = Binance(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    
#     def load_data(self, file_path) -> NoReturn:
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#         self.data_list.append(data)
        
    # def collect_ohlcv(self, start_time, end_time):
    #     time = start_time
    #     while True:
    #         tohlcv = self.binance.binance.fetch_ohlcv(
    #             symbol=TARGET_COIN_SYMBOL,
    #             timeframe="1m",
    #             params={'startTime':time},
    #             limit=1500
    #         )
    #         time = tohlcv[-1][0]
        
def preprocess(data_list: List, output_dir: str) -> pd.DataFrame:
    # file_list = os.listdir(DATA_DIR)
    # # min_list = []
    # for file in file_list:
    #     if 'data_' in file:
    #         self.load_data(join(DATA_DIR, file))
        # min_list.append(int(file.split('_')[2].split('.')[0]))
    # min_list.sort()
    # start_time = min_list[0]  # 제일 처음 시간 (오늘, 어제 있으면 어제)
    # end_time = min_list[-1]
    df_list = []
    for data in data_list:
        # 거래량 기준 정렬
        bids = sorted(data['order_book']['bids'], key=lambda x: x[1], reverse=True)[:ORDER_BOOK_RANK_SIZE]
        asks = sorted(data['order_book']['asks'], key=lambda x: x[1], reverse=True)[:ORDER_BOOK_RANK_SIZE]
        
        price = pd.DataFrame(zip([data['ticker']['open']], [data['ticker']['high']], [data['ticker']['low']],
                                    [data['ticker']['close']], [data['ticker']['baseVolume']]), columns=['open', 'high', 'low', 'close', 'volume'])
        for i in range(ORDER_BOOK_RANK_SIZE):
            price[f'bid_{i}'] = bids[i][0]
            price[f'bid_volume_{i}'] = bids[i][1]
            price[f'ask_{i}'] = asks[i][0]
            price[f'ask_volume_{i}'] = asks[i][1]
        df_list.append(price)
    df = pd.concat(df_list)
    # target = df[['close']]
    # scaler_x = MinMaxScaler()
    # scaler_x.fit(df)
    # scaled_df = pd.DataFrame(scaler_x.transform(df), columns=df.columns)
    
    # scaler_y = MinMaxScaler()
    # scaler_y.fit(target)
    # target_scaled = pd.DataFrame(scaler_y.transform(target), columns=target.columns)
    
    # # save scaler
    # file_name_x = join(output_dir, 'scaler_x.pkl')
    # file_name_y = join(output_dir, 'scaler_y.pkl')
    # joblib.dump(scaler_x, file_name_x)
    # joblib.dump(scaler_y, file_name_y)
    return df
    