import os
from typing import Dict, List
from torch import Tensor

import numpy as np 
import pandas as pd
from multiprocessing import Pool
import torch

from sklearn.preprocessing import MinMaxScaler

def rolling_window(data: np.array, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def prepare_batch(batch: List[Dict]) -> Dict[str, Tensor]:
    # x: (batch_size, frame_size, feature_size) 
    # y: (batch_size, frame_size)
    tensor_list_x = []
    tensor_list_y = []
    for i in range(len(batch)):
        tensor_list_x.append(torch.tensor(batch[i]['x'].values))
        tensor_list_y.append(torch.tensor(batch[i]['y'].values))
    
    # df = pd.concat(df_list_x)
    # target = pd.concat(df_list_y)
    # idx_array = np.array([i for i in range(len(df))][:-1])  # 마지막 데이터는 target이 없으므로 제외
    # rolling_tensor = torch.tensor(rolling_window(idx_array, frame_size))
    
    # embedding = torch.tensor(df.values)
    # time_embedding = embedding[rolling_tensor.long()]
    time_embedding = torch.stack(tensor_list_x, dim=0)
    target = torch.stack(tensor_list_y, dim=0).squeeze(2)
    # target = torch.tensor(rolling_window(target['close'].values[1:], frame_size))  # 첫번째 target은 데이터가 없으므로 제외
    return {
        'data': time_embedding,
        'target': target
    }


def parallelize_list_to_df(data_list: List, func):
    num_cores = os.cpu_count() -1
    if num_cores < 8:
        num_cores = 1
    if len(data_list) < num_cores:
        num_cores = len(data_list)
        # for avoiding os.fork() cannot allocate memory error
        num_cores = int(num_cores / 5)
        # num_cores = 1
    data_split = np.array_split(data_list, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, data_split))
    return df

def get_ma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
    # df[f'ma_{window}'] = df[f'ma_{window}'].shift(1)
    df[f'ma_{window}'] = df[f'ma_{window}'].fillna(0)
    return df

def robust_scaling(df: pd.DataFrame, maximum: int, minimum: int) -> pd.DataFrame:
    for column in df.columns:
        scaler = MinMaxScaler(feature_range=(minimum, maximum))
        df.loc[:, column] = df.loc[:, column].fillna(0).astype(float, errors='ignore')
        trd = df[column].describe()['75%']
        first = df[column].describe()['25%']
        tmp_df = df.loc[((df[column] < trd) & (df[column] > first)), [column]]
        scaler.fit(tmp_df)

        high_df = df.loc[df[column] >= trd, [column]]
        low_df = df.loc[df[column] <= first, [column]]

        df.loc[tmp_df.index, [column]] = scaler.fit_transform(df.loc[tmp_df.index, [column]])
        df.loc[high_df.index, [column]] = maximum
        df.loc[low_df.index, [column]] = minimum
    return df

def make_robust(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        trd = df[column].describe()['75%']
        first = df[column].describe()['25%']
        maximum = df[column].describe()['max']
        minimum = df[column].describe()['min']
        df.loc[df[column] >= trd, [column]] = maximum
        df.loc[df[column] <= first, [column]] = minimum
    return df

if __name__ == '__main__':
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # for test
    # import pandas as pd
    # import json
    # from src.component.preprocess.preprocess import preprocess
    # import torch
    # with open('/Users/euijinbaek/auto_crypto_trading/order_books/data_202301231657.json', 'r') as f:
    #     data = json.load(f)
    # with open('/Users/euijinbaek/auto_crypto_trading/order_books/data_202301232240.json', 'r') as f:
    #     data2 = json.load(f)
    # with open('/Users/euijinbaek/auto_crypto_trading/order_books/data_202301232253.json', 'r') as f:
    #     data3 = json.load(f)
    # data_list = [data, data2, data3]
    # df = preprocess(data_list)
    # idx_array = np.array([i for i in range(len(df))][:-1])
    # WINDOW_SIZE = 30
    # rolling_tensor = torch.tensor(rolling_window(idx_array, WINDOW_SIZE))
    
    # embedding = torch.tensor(df.values)
    # time_embedding = embedding[rolling_tensor.long()]
    # target = rolling_window(df['close'].values[1:], WINDOW_SIZE)
    
    print(rolling_window(data, 3))
    print(rolling_window(data, 4))
    print(rolling_window(data, 5))
    print(rolling_window(data, 6))
    print(rolling_window(data, 7))
    print(rolling_window(data, 8))
    print(rolling_window(data, 9))
    print(rolling_window(data, 10))
