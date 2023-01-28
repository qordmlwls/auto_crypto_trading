from typing import Dict, List
from torch import Tensor

import numpy as np 
import pandas as pd
import torch


def rolling_window(data: np.array, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def prepare_batch(batch: List[Dict], frame_size: int) -> Dict[str, Tensor]:
    # x: (batch_size, frame_size, feature_size) 
    # y: (batch_size, frame_size)
    df_list_x = []
    df_list_y = []
    for i in range(len(batch)):
        df_list_x.append(batch[i]['x'])
        df_list_y.append(batch[i]['y'])
    df = pd.concat(df_list_x, axis=1).transpose()
    target = pd.concat(df_list_y, axis=1).transpose()
    idx_array = np.array([i for i in range(len(df))][:-1])  # 마지막 데이터는 target이 없으므로 제외
    rolling_tensor = torch.tensor(rolling_window(idx_array, frame_size))
    
    embedding = torch.tensor(df.values)
    time_embedding = embedding[rolling_tensor.long()]
    target = torch.tensor(rolling_window(target['close'].values[1:], frame_size))  # 첫번째 target은 데이터가 없으므로 제외
    return {
        'data': time_embedding,
        'target': target
    }
    

if __name__ == '__main__':
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # for test
    import pandas as pd
    import json
    from src.component.preprocess.preprocess import preprocess
    import torch
    with open('/Users/euijinbaek/auto_crypto_trading/order_books/data_202301231657.json', 'r') as f:
        data = json.load(f)
    with open('/Users/euijinbaek/auto_crypto_trading/order_books/data_202301232240.json', 'r') as f:
        data2 = json.load(f)
    with open('/Users/euijinbaek/auto_crypto_trading/order_books/data_202301232253.json', 'r') as f:
        data3 = json.load(f)
    data_list = [data, data2, data3]
    df = preprocess(data_list)
    idx_array = np.array([i for i in range(len(df))][:-1])
    WINDOW_SIZE = 30
    rolling_tensor = torch.tensor(rolling_window(idx_array, WINDOW_SIZE))
    
    embedding = torch.tensor(df.values)
    time_embedding = embedding[rolling_tensor.long()]
    target = rolling_window(df['close'].values[1:], WINDOW_SIZE)
    
    print(rolling_window(data, 3))
    print(rolling_window(data, 4))
    print(rolling_window(data, 5))
    print(rolling_window(data, 6))
    print(rolling_window(data, 7))
    print(rolling_window(data, 8))
    print(rolling_window(data, 9))
    print(rolling_window(data, 10))
    