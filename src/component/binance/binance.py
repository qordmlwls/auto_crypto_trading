import time
import ccxt
from typing import Dict

from src.component.binance.constraint import LEVERAGE


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
                amount = float(position['positionAmt'])
                isolated = position['isolated']
                entry_price = float(position['entryPrice'])
                unrealized_pnl = float(position['unrealizedProfit'])
                
                if not isolated:
                    try:
                        response = self.binance.fapiPrivate_post_margintype({
                            'symbol': ticker, 'marginType': 'ISOLATED'
                            })
                        if response['msg'] == 'success':
                            isolated = True
                    except Exception as e:
                        print('---', e)
                break
                
        return {
            'amount': amount,
            'isolated': isolated,
            'total': balance['USDT']['total'],
            'free': balance['USDT']['free'],
            'entry_price': entry_price,
            'unrealized_pnl': unrealized_pnl
        }
    
    # 구매할 수량을 계산한다. 첫번째: 돈(USDT), 두번째:코인 가격, 세번째: 비율 1.0이면 100%, 0.5면 50%
    def get_amout(self, usdt, coion_price, rate):
        target = usdt * rate
        amount = target / coion_price
        return amount
    
    def get_minimum_amout(self, ticker):
        limit_values = self.binance.markets[ticker]['limits']
        
        min_amount = limit_values['amount']['min']
        min_cost = limit_values['cost']['min']
        min_price = limit_values['price']['min']
        
        coin_info = self.binance.fetch_ticker(ticker)
        coin_price = coin_info['last']
        
        # get mininum unit price to be able to order
        if min_price < coin_price:
            min_price = coin_price

        # order cost = price * amount
        min_order_cost = min_price * min_amount

        num_min_amount = 1

        if min_cost is not None and min_order_cost < min_cost:
            # if order cost is smaller than min cost
            # increase the order cost bigger than min cost
            # by the multiple number of minimum amount
            while min_order_cost < min_cost:
                num_min_amount = num_min_amount + 1
                min_order_cost = min_price * (num_min_amount * min_amount)

        return num_min_amount * min_amount
    
    def get_now_price(self, ticker):
        coin_info = self.binance.fetch_ticker(ticker)
        coin_price = coin_info['last']
        return coin_price
    
    # 지정가로 매수한다. 첫번째: 코인 티커, 두번째: 포지션, 세번째: 매수 수량, 네번째: 매수 가격
    def create_order(self, ticker, side, amount, price):
        time.sleep(0.1)
        try:
            self.binance.create_order(ticker, 'limit', side, amount, price)
        except Exception as e:
            print('---', e)
    
    def create_market_order(self, ticker, side, amount):
        time.sleep(0.1)
        try:
            self.binance.create_order(ticker, 'market', side, amount)
        except Exception as e:
            print('---', e)
    
    def cancel_all_orders(self, ticker):
        time.sleep(0.1)
        try:
            self.binance.cancel_all_orders(ticker)
        except Exception as e:
            print('---', e)
    
    def cancel_order(self, ticker, order_id):
        time.sleep(0.1)
        try:
            self.binance.cancel_order(order_id, ticker)
        except Exception as e:
            print('---', e)

    #스탑로스를 걸어놓는다. 해당 가격에 해당되면 바로 손절한다. 첫번째: 바이낸스 객체, 두번째: 코인 티커, 세번째: 손절 수익율 (1.0:마이너스100% 청산, 0.9:마이너스 90%, 0.5: 마이너스 50%)
    #네번째 웹훅 알림에서 사용할때는 마지막 파라미터를 False로 넘겨서 사용한다. 트레이딩뷰 웹훅 강의 참조..
    def set_stop_loss(self, ticker, cut_rate, rest=True):
        
        if rest:  # cancel중일 수 있으므로 10초간 대기한다.
            time.sleep(10)
             
        # 주문 정보를 읽어온다.
        orders = self.binance.fetch_orders(ticker)
        
        stop_loss_ok = False
        stop_order_list = []
        for order in orders:
            
            if order['status'] == "open" and order['type'] == 'stop_market':
                print(order)
                stop_order_list.append(order)
                stop_loss_ok = True
                # break
        target_symbol = ticker.replace("/", "")
        position = self.position_check(target_symbol)
        if len(stop_order_list) > 0:
            order = stop_order_list[0]
        # 스탑로스가 없거나 반대로 걸려있으면 스탑로스를 건다.
        if (not stop_loss_ok and position['amount'] !=0) \
            or (stop_loss_ok and order['side'] == 'buy' and position['amount'] > 0) \
            or (stop_loss_ok and order['side'] == 'sell' and position['amount'] < 0):
            
            # if rest:
            #     time.sleep(10.0)
            target_symbol = ticker.replace("/", "")
            position = self.position_check(target_symbol)
            if (stop_loss_ok and order['side'] == 'buy' and position['amount'] > 0) or (stop_loss_ok and order['side'] == 'sell' and position['amount'] < 0):
                self.binance.cancel_order(order['id'], ticker)
            #롱일땐 숏을 잡아야 되고
            side = "sell"
            #숏일땐 롱을 잡아야 한다.
            if position['amount'] < 0:
                side = "buy"
            
            danger_rate = ((100.0 / LEVERAGE) * cut_rate) * 1.0
            
            #롱일 경우의 손절 가격을 정한다.
            stop_price = position['entry_price'] * (1.0 - danger_rate * 0.01)
            
            #숏일 경우의 손절 가격을 정한다.
            if position['amount'] < 0:
                stop_price = position['entry_price'] * (1.0 + danger_rate * 0.01)
            
            params = {
                'stopPrice': stop_price,
                'closePosition': True
            }
            
            print('side: ', side, 'stop_price: ', stop_price, 'entry_price: ', position['entry_price'], 'danger_rate: ', danger_rate)
            #스탑 로스 주문을 걸어 놓는다.
            print(self.binance.create_order(ticker, 'STOP_MARKET', side, abs(position['amount']), stop_price, params))
            
            print("####STOPLOSS SETTING DONE ######################")
        # 포지션 없다면 스탑로스를 취소한다.
        elif stop_loss_ok and position['amount'] == 0:
            for order in stop_order_list:
                self.binance.cancel_order(order['id'], ticker)
            print("####STOPLOSS CANCEL DONE ######################")
        else:
            return
    
    def cancel_failed_order(self, ticker):
        time.sleep(0.1)
        orders = self.binance.fetch_orders(ticker)
        for order in orders:
            if order['status'] == "open" and order['type'] == 'limit':
                self.binance.cancel_order(order['id'], ticker)
                
            
    # def set_stop_loss_price(self, ticker, stop_price, rest=True):
    #     if rest:
    #         time.sleep(0.1)

    #     # 주문 정보를 읽어온다.
    #     orders = self.binance.fetch_orders(ticker)

    #     stop_loss_ok = False
    #     for order in orders:

    #         if order['status'] == "open" and order['type'] == 'stop_market':
    #             # print(order)
    #             stop_loss_ok = True
    #             break

    #     # 스탑로스 주문이 없다면 주문을 건다!
    #     if not stop_loss_ok:

    #         if rest:
    #             time.sleep(10.0)

    #         # 잔고 데이타를 가지고 온다.
    #         balance = self.binance.fetch_balance(params={"type": "future"})

    #         if rest:
    #             time.sleep(0.1)

            
    #         position = self.position_check(ticker)
    #         # amt = 0
    #         # entry_price = 0

    #         # # 평균 매입단가와 수량을 가지고 온다.
    #         # for posi in balance['info']['positions']:
    #         #     if posi['symbol'] == ticker.replace("/", ""):
    #         #         entry_price = float(posi['entryPrice'])
    #         #         amt = float(posi['positionAmt'])

    #         # 롱일땐 숏을 잡아야 되고
    #         side = "sell"
    #         # 숏일땐 롱을 잡아야 한다.
    #         if position['amount'] < 0:
    #             side = "buy"

    #         params = {
    #             'stopPrice': stop_price,
    #             'closePosition': True
    #         }

    #         print("side:", side, "   stopPrice:", stop_price, "   entryPrice:", entry_price)
    #         # 스탑 로스 주문을 걸어 놓는다.
    #         print(binance.create_order(ticker, 'STOP_MARKET', side, abs(amt), stop_price, params))

    #         print("####STOPLOSS SETTING DONE ######################")
