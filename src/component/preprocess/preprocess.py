from os.path import join
from typing import List
import pandas as pd
from tqdm import tqdm


ORDER_BOOK_RANK_SIZE = 100

        
def preprocess(data_list: List) -> pd.DataFrame:

    df_list = []
    for idx, data in tqdm(enumerate(data_list)):
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
            
        # 현 시점 데이터
        df_list.append(price)
    df = pd.concat(df_list)
    return df
    