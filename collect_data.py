import os
import json
from datetime import datetime

from src.component.binance.constraint import BINANCE_API_KEY, BINANCE_SECRET_KEY
from src.component.binance.binance import Binance
from src.module.db.redis import Redis
from src.module.db.s3 import S3
from src.component.preprocess.preprocess import ORDER_BOOK_RANK_SIZE

# 해당 모듈은 매 1분마다 실행되어야 한다.

TARGET_COIN_TICKER = 'BTC/USDT'
TARGET_COIN_SYMBOL = 'BTCUSDT'
DATA_DIR = 'order_books'
TIME_WINDOW = 30
BUKET_NAME = 'autocryptotrading'


def main():
    s3 = S3(BUKET_NAME)
    redis = Redis(host='localhost', port=6379, db=0)
    binance = Binance(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    order_book = binance.binance.fetch_order_book(TARGET_COIN_SYMBOL)
    ticker = binance.binance.fetch_ticker(TARGET_COIN_TICKER)

    now = datetime.now().strftime('%Y%m%d%H%M')
    data = {
        "order_book": order_book,
        "ticker": ticker
    }
    # @TODO: Redis, s3에 저장
    with open(os.path.join(DATA_DIR, f'data_{now}.json'), 'w') as f:
        json.dump(data, f)

    s3.upload_file(os.path.join(DATA_DIR, f'data_{now}.json'), f'data/data_{now}.json')

    os.remove(os.path.join(DATA_DIR, f'data_{now}.json'))

    if redis.size() > TIME_WINDOW:
        redis.delete(redis.keys()[0])

    data_redis = {
        'open': ticker['open'],
        'high': ticker['high'],
        'low': ticker['low'],
        'close': ticker['close'],
        'volume': ticker['baseVolume'],
    }
    bids = sorted(order_book['bids'], key=lambda x: x[1], reverse=True)[:ORDER_BOOK_RANK_SIZE]
    asks = sorted(order_book['asks'], key=lambda x: x[1], reverse=True)[:ORDER_BOOK_RANK_SIZE]
    data_redis.update({
        f'bid_{i}': bid[0] for i, bid in enumerate(bids)
    })
    data_redis.update({
        f'bid_volume_{i}': bid[1] for i, bid in enumerate(bids)
    })
    data_redis.update({
        f'ask_{i}': ask[0] for i, ask in enumerate(asks)
    })
    data_redis.update({
        f'ask_volume_{i}': ask[1] for i, ask in enumerate(asks)
    })
    redis.set(f'data_{now}', json.dumps(data_redis))
    print(redis.get(f'data_{now}'))


if __name__ == '__main__':
    main()
    