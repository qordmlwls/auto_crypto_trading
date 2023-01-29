import time
import ccxt
from typing import Dict


class Binance:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

        self.binance = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            },
        })  # binance 객체 생성
    
    def set_leverage(self, ticker, leverage):
        time.sleep(0.1)
        try:
            self.binance.fapiPrivate_post_leverage({
                'symbol': ticker,
                'leverage': leverage
            })
        except Exception as e:
            print('---', e)
            
    def position_check(self, ticker) -> Dict:
        balance = self.binance.fetch_balance(params={"type": "future"})
        print(balance['USDT'])
        print("Total Balance: ", balance['USDT']['total'])  # 총 금액
        print('Free Balance: ', balance['USDT']['free'])  # 사용 가능한 금액
        for position in balance['info']['positions']:
            if position['symbol'] == ticker:
                amount = position['positionAmt']
                isolated = position['isolated']
            break
        if not isolated:
            try:
                self.binance.fapiPrivate_post_positionmargin({
                    'symbol': ticker, 'marginType': 'ISOLATED'
                    })
            except Exception as e:
                print('---', e)
                
        return {
            'amount': amount,
            'isolated': isolated,
            'total': balance['USDT']['total']
        }
        

def set_stop_loss_price(binance, ticker, stop_price, rest=True):
    if rest:
        time.sleep(0.1)

    # 주문 정보를 읽어온다.
    orders = binance.fetch_orders(ticker)

    stop_loss_ok = False
    for order in orders:

        if order['status'] == "open" and order['type'] == 'stop_market':
            # print(order)
            stop_loss_ok = True
            break

    # 스탑로스 주문이 없다면 주문을 건다!
    if not stop_loss_ok:

        if rest:
            time.sleep(10.0)

        # 잔고 데이타를 가지고 온다.
        balance = binance.fetch_balance(params={"type": "future"})

        if rest:
            time.sleep(0.1)

        amt = 0
        entry_price = 0

        # 평균 매입단가와 수량을 가지고 온다.
        for posi in balance['info']['positions']:
            if posi['symbol'] == ticker.replace("/", ""):
                entry_price = float(posi['entryPrice'])
                amt = float(posi['positionAmt'])

        # 롱일땐 숏을 잡아야 되고
        side = "sell"
        # 숏일땐 롱을 잡아야 한다.
        if amt < 0:
            side = "buy"

        params = {
            'stopPrice': stop_price,
            'closePosition': True
        }

        print("side:", side, "   stopPrice:", stop_price, "   entryPrice:", entry_price)
        # 스탑 로스 주문을 걸어 놓는다.
        print(binance.create_order(ticker, 'STOP_MARKET', side, abs(amt), stop_price, params))

        print("####STOPLOSS SETTING DONE ######################")
