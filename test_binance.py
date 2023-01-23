import pprint
from binance.client import Client
from constraint import BINANCE_API_KEY, BINANCE_SECRET_KEY


TARGET_COIN_TICKER = 'BTC/USDT'
TARGET_COIN_SYMBOL = 'BTCUSDT'

client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
ticker = client.get_ticker(symbol=TARGET_COIN_SYMBOL)
ticker2 = client.get_symbol_ticker(symbol=TARGET_COIN_SYMBOL)
pprint.pprint(ticker)
pprint.pprint(ticker2)
