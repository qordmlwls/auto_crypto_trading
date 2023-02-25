import os


BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', '')


TARGET_COIN_TICKER = 'BTC/USDT'
TARGET_COIN_SYMBOL = 'BTCUSDT'

TIME_WINDOW = 15

LEVERAGE = 10
#아래는 타겟 수익율로 마음껏 조절하세요
#타겟 레이트 0.001 
TARGET_RATE = 0.001
#타겟 수익율 0.1%
TARGET_REVENUE_RATE = TARGET_RATE * 100.0

#스탑로스 비율설정 0.5는 원금의 마이너스 50%를 의미한다. 0.1은 마이너스 10%
STOP_LOSS_RATE = 0.5
# 손절 마이너스 수익률
DANGER_RATE = -5.0

# 예상 변동성, 해당 수치를 넘어가면 매수/매도한다.
FUTURE_CHANGES_DIR = '/home/ubuntu/auto_crypto_trading/future_changes'
PLUS_FUTURE_PRICE_RATE = 0.15
MINUS_FUTURE_PRICE_RATE = -0.10
FUTURE_CHANGE_MULTIPLIER = 0.95
FUTURE_MAX_LEN = 120  # 2시간
FUTURE_MIN_LEN = 60  # 1시간
# Real Revenue
STOP_PROFIT_RATE = 0.15
# Revenue
STOP_REVENUE_PROFIT_RATE = 0.08
PROFIT_AMOUNT_MULTIPLIER = 2

# For Scaling
CURRENT_VARIANCE = 0.05
