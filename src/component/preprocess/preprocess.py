from os.path import join
from typing import List
import pandas as pd
from tqdm import tqdm


ORDER_BOOK_RANK_SIZE = 100

        
def preprocess(data_list: List) -> pd.DataFrame:

    df_list = []
    for data in tqdm(data_list):
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
    return df
    