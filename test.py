import pprint
import ccxt
from datetime import datetime

from constraint import BINANCE_API_KEY, BINANCE_SECRET_KEY

TARGET_COIN_TICKER = 'BTC/USDT'
TARGET_COIN_SYMBOL = 'BTCUSDT'

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
tohlcv = binance.fetch_ohlcv(
    symbol=TARGET_COIN_SYMBOL,
    timeframe="1m",
    params={'startTime':datetime.now()},
    limit=1500
)
pprint.pprint(tohlcv)

order_book = binance.fetch_order_book(TARGET_COIN_SYMBOL,
                                      )

pprint.pprint(order_book)
