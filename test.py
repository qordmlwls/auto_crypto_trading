import os
import json
import pprint
import ccxt
from datetime import datetime, timedelta

from src.component.binance.constraint import BINANCE_API_KEY, BINANCE_SECRET_KEY
from src.module.db.redis.redis import Redis

TARGET_COIN_TICKER = 'BTC/USDT'
TARGET_COIN_SYMBOL = 'BTCUSDT'
ROOT_DIR = os.environ.get('PYTHONPATH', '')
DATA_DIR = os.path.join(ROOT_DIR, 'order_books')

print(os.path.dirname(os.path.abspath(__file__)))
binance = ccxt.binance(config={
    'apikey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET_KEY,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})

btc = binance.fetch_ticker(TARGET_COIN_TICKER)
pprint.pprint(btc)
# tohlcv = binance.fetch_ohlcv(
#     symbol=TARGET_COIN_SYMBOL,
#     timeframe="1m",
#     params={'startTime':datetime.now() - timedelta(days=30)},
#     limit=1500
# )
# pprint.pprint(tohlcv)

# order_book = binance.fetch_order_book(TARGET_COIN_SYMBOL,
#                                       )

# # pprint.pprint(order_book)
# redis = Redis(host='localhost', port=6379, db=0)

# with open(os.path.join(DATA_DIR, 'data_202301231657.json'), 'r') as f:
#     order_book = json.load(f)
    
# with open(os.path.join(DATA_DIR, 'data_202301232240.json'), 'r') as f:
#     order_book2 = json.load(f)
# top_5_bid = sorted(order_book['bids'], key=lambda x: x[0], reverse=True)[:5]
# print('end')
