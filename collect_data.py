import os
import json
from datetime import datetime

from src.component.binance.constraint import BINANCE_API_KEY, BINANCE_SECRET_KEY
from src.component.binance.binance import Binance
from src.module.db.redis import Redis
from src.module.db.s3 import S3

# 해당 모듈은 매 1분마다 실행되어야 한다.

TARGET_COIN_TICKER = 'BTC/USDT'
TARGET_COIN_SYMBOL = 'BTCUSDT'
DATA_DIR = 'order_books'
TIME_WINDOW = 30
BUKET_NAME = 'data'


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

s3.upload_file(os.path.join(DATA_DIR, f'data_{now}.json'), f'data_{now}.json')

os.remove(os.path.join(DATA_DIR, f'data_{now}.json'))

if redis.size() > TIME_WINDOW:
    redis.delete(redis.keys()[0])

redis.set(f'data_{now}', json.dumps(data))
print(redis.get(f'data_{now}'))

