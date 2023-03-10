import os


BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', '')


TARGET_COIN_TICKER = 'BTC/USDT'
TARGET_COIN_SYMBOL = 'BTCUSDT'

TIME_WINDOW = 15
MOVING_AVERAGE_WINDOW = 100
COLUMN_LIMIT = 100

COLUMNS = ['open', 'high', 'low', 'close', 'volume', f"ma_{MOVING_AVERAGE_WINDOW}", "datetime"] + [f'bid_{i}' for i in range(COLUMN_LIMIT)] \
            + [f'ask_{i}' for i in range(COLUMN_LIMIT)] + [f'bid_volume_{i}' for i in range(COLUMN_LIMIT)] \
            + [f'ask_volume_{i}' for i in range(COLUMN_LIMIT)]

LEVERAGE = 3  # 레버리지 많이 하면 세금 많이 내야함
#아래는 타겟 수익율로 마음껏 조절하세요
#타겟 레이트 0.001 
TARGET_RATE = 0.001
#타겟 수익율 0.1%
TARGET_REVENUE_RATE = TARGET_RATE * 100.0

#스탑로스 비율설정 0.5는 원금의 마이너스 50%를 의미한다. 0.1은 마이너스 10%
STOP_LOSS_RATE = 0.5
# 손절 마이너스 수익률
DANGER_RATE = -5.0

# 거래량
TRADE_RATE = 5.0

# 예상 변동성, 해당 수치를 넘어가면 매수/매도한다.
FUTURE_CHANGES_DIR = '/home/ubuntu/auto_crypto_trading/future_changes'
PLUS_FUTURE_PRICE_RATE = 0.24
MINUS_FUTURE_PRICE_RATE = -0.14
MINIMUM_FUTURE_PRICE_RATE = 0.11
FUTURE_CHANGE_MULTIPLIER = 0.30  # 진입 비율
SWITCHING_CHANGE_MULTIPLIER = 0.90  # 스위칭하는 비율. 진입 비율보다 더 빡세게 잡는다.
STONG_SWITCHING_RATE = 0.
FUTURE_MAX_LEN = 120  # 2시간
FUTURE_MIN_LEN = 60  # 1시간
# Real Revenue
STOP_PROFIT_RATE = 0.15
# Revenue
STOP_REVENUE_PROFIT_RATE = 0.1
PROFIT_AMOUNT_MULTIPLIER = 3
TAKE_PROFIT_MULTIPLIER = 2
# Exit
EXIT_PRICE_CHANGE = 80

# For Scaling
CURRENT_VARIANCE = 0.05

# 손절
LOSS_CRITERIA_RATE = 0.2

# 재학습
RETRAIN_SIGNAL_RATE = 0.80

# loss type
LOSS_TYPE = 'huber'

