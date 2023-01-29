import os
import json
from typing import List

import boto3
import numpy as np
import time

from src.component.binance.constraint import (
    TIME_WINDOW, BINANCE_API_KEY, BINANCE_SECRET_KEY, 
    TARGET_COIN_SYMBOL, TARGET_COIN_TICKER,
    LEVERAGE, TARGET_RATE, TARGET_REVENUE_RATE, STOP_LOSS_RATE, DANGER_RATE
)
from src.component.binance.binance import Binance
from src.module.db.redis import Redis


def chek_futre_price(current_price, future_price_list: List):
    chages = [abs(future_price - current_price) for future_price in future_price_list]
    max_index = np.argmax(chages)
    return {
        'max_chage': chages[max_index],
        'max_index': max_index,
    }


def main():
    client = boto3.client('sagemaker-runtime')
    redis = Redis(host='localhost', port=6379, db=0)
    binance = Binance(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    binance.set_leverage(TARGET_COIN_SYMBOL, LEVERAGE)
    position = binance.position_check(TARGET_COIN_SYMBOL)
    #시장가 taker 0.04, 지정가 maker 0.02 -> 시장가가 수수료가 더 비싸다.

    #시장가 숏 포지션 잡기 
    #print(binance.create_market_sell_order(Target_Coin_Ticker, 0.002))
    #print(binance.create_order(Target_Coin_Ticker, 'market', 'sell', 0.002, None))
    
    if redis.size() == TIME_WINDOW:
        
        data_list = redis.all()
        data_list.sort(key=lambda x: x['ticker']['timestamp'])
        request_body = json.dumps({
            "data_list": data_list
        })
        res = client.invoke_endpoint(EndpointName='Autotrading-Endpoint',
                                    ContentType='application/json',
                                    Accept='application/json',
                                    Body=request_body)
        # next 30분 각각의 예측값을 받아온다. 길이 30
        res_data = json.loads(res['Body'].read().decode('utf-8')) 

    current_price = data_list[0]['ticker']['close']
    # 레버리지에 따를 최대 매수 가능 수량
    max_amount = round(binance.get_amout(position['total'], current_price, 0.5), 3) * LEVERAGE
    
    one_percent_amount = round(max_amount * 0.01, 3)
    
    #첫 매수 비중을 구한다.. 여기서는 5%! 
    # * 20.0 을 하면 20%가 되겠죠? 마음껏 조절하세요!
    first_amount = one_percent_amount * 5.0

    # 비트코인의 경우 0.001로 그보다 작은 수량으로 주문을 넣으면 오류가 납니다.!
    #최소 주문 수량을 가져온다 
    # 해당 함수는 수강생 분이 만드신 걸로 아래 링크를 참고하세요!
    # https://blog.naver.com/zhanggo2/222722244744
    minimun_amount = binance.get_minimum_amout(TARGET_COIN_SYMBOL)
    
    #minimun_amount 안에는 최소 주문수량이 들어가 있습니다. 비트코인이니깐 0.001보다 작다면 0.001개로 셋팅해줍니다.
    if first_amount < minimun_amount:
        first_amount = minimun_amount
    
    #음수를 제거한 절대값 수량 ex -0.1 -> 0.1 로 바꿔준다.
    abs_amt = abs(position['amount'])

    if res_data:
        futre_change = chek_futre_price(current_price, res_data)
    else:
        pass
    
    #0이면 포지션 잡기전
    if abs_amt == 0 and res_data:
        
        if futre_change['change'] > 0.1:
            print("------------------------------------------------------")
            print("Buy", first_amount, TARGET_COIN_SYMBOL)
            print("------------------------------------------------------")
            #매수 주문을 넣는다.
            binance.create_order(TARGET_COIN_SYMBOL, first_amount, 'buy', current_price)
            binance.set_stop_loss(TARGET_COIN_SYMBOL, STOP_LOSS_RATE)
        elif futre_change['change'] < -0.1:
            print("------------------------------------------------------")
            print("Sell", first_amount, TARGET_COIN_SYMBOL)
            print("------------------------------------------------------")
            #매도 주문을 넣는다.
            binance.create_order(TARGET_COIN_SYMBOL, first_amount, 'sell', current_price)
            binance.set_stop_loss(TARGET_COIN_SYMBOL, STOP_LOSS_RATE)

        
    #0이 아니라면 포지션 잡은 상태
    else:
        print("------------------------------------------------------")
        #현재까지 구매한 퍼센트! 즉 비중!! 현재 보유 수량을 1%의 수량으로 나누면 된다.
        buy_percent = abs_amt / one_percent_amount
        print("Buy Percent : ", buy_percent)    

        #수익율을 구한다!
        revenue_rate = (current_price - position['entry_price']) / position['entry_price'] * 100.0
        #단 숏 포지션일 경우 수익이 나면 마이너스로 표시 되고 손실이 나면 플러스가 표시 되므로 -1을 곱하여 바꿔준다.
        if position['amount'] < 0:
            revenue_rate = revenue_rate * -1.0
        #레버리지를 곱한 실제 수익율
        leverage_revenu_rate = revenue_rate * LEVERAGE  
        
        print("Revenue Rate : ", revenue_rate,", Real Revenue Rate : ", leverage_revenu_rate)
        
        #레버리지를 곱한 실제 손절 할 마이너스 수익율
        #레버리지를 곱하고 난 여기가 실제 내 원금 대비 실제 손실율입니다!
        leverage_danger_rate = DANGER_RATE * LEVERAGE
        # @TODO: 이미 잡은 포지션에 따른 매수매도 로직
        #@TODO: 목표 future_change, 추가매수, 첫구매 비율 공통상수로 빼기
        amount = one_percent_amount * 5.0
        if amount < minimun_amount:
            amount = minimun_amount
        print("Danger Rate : ", DANGER_RATE,", Real Danger Rate : ", leverage_danger_rate)    
        if leverage_revenu_rate > 0.5:
            if position['amount'] > 0:
                # 5% 매도
                print('------------------------------------------------------')
                print('이익 0.5% 이상이므로 5% 매도')
                binance.create_order(TARGET_COIN_SYMBOL, amount, 'sell', current_price)
                position['amount'] = position['amount'] - amount
                binance.set_stop_loss(TARGET_COIN_SYMBOL, STOP_LOSS_RATE)
            elif position['amount'] < 0:
                print('------------------------------------------------------')
                print('이익 0.5% 이상이므로 5% 매수')
                binance.create_order(TARGET_COIN_SYMBOL, amount, 'buy', current_price)
                position['amount'] = position['amount'] + amount
                binance.set_stop_loss(TARGET_COIN_SYMBOL, STOP_LOSS_RATE)

        # 숏 포지션일 경우
        if position['amount'] < 0 and res_data:
            if futre_change['change'] < - 0.1:
                # 5% 추가 매도
                print("------------------------------------------------------")
                print("Sell", amount, TARGET_COIN_SYMBOL)
                binance.create_order(TARGET_COIN_SYMBOL, amount, 'sell', current_price)
                binance.set_stop_loss(TARGET_COIN_SYMBOL, STOP_LOSS_RATE)
                print("------------------------------------------------------")
            elif futre_change['change'] > 0.1:
                # 포지션 종료, 5% 추가 매수
                print("------------------------------------------------------")
                print("Buy", amount, TARGET_COIN_SYMBOL)
                binance.create_order(TARGET_COIN_SYMBOL, amount + abs_amt, 'buy', current_price)
                print("------------------------------------------------------")
                binance.set_stop_loss(TARGET_COIN_SYMBOL, STOP_LOSS_RATE)
                
            #내 보유 수량의 절반을 손절한다 단!! 매수 비중이 90% 이상이면서 내 수익율이 손절 마이너스 수익율보다 작을 때
            if revenue_rate <= DANGER_RATE and buy_percent >= 90.0:
                
                #주문 취소후
                binance.cancel_all_orders(TARGET_COIN_SYMBOL)
                time.sleep(0.1)
                '''
                #클래스에선 수수료 절감 차원에서 지정가로 잡았지만 단점은 100% 포지션이 종료되거나 잡힌다는 보장이 없다는 점입니다.
                                    
                #해당 코인 가격을 가져온다.
                coin_price = myBinance.GetCoinNowPrice(binanceX, Target_Coin_Ticker)

                #숏 포지션을 잡는다
                print(binanceX.create_limit_sell_order(Target_Coin_Ticker, abs_amt / 2.0, coin_price))
                '''
                #따라서 여기서는 시장가로 잡습니다 <- 이렇게 하는걸 권장드려요!
                #숏 포지션을 잡는다
                #print(binanceX.create_market_sell_order(Target_Coin_Ticker, abs_amt / 2.0))
                print(binance.create_market_order(TARGET_COIN_SYMBOL, 'buy', abs_amt / 2.0))

                #스탑 로스 설정을 건다.
                binance.set_stop_loss(TARGET_COIN_SYMBOL, STOP_LOSS_RATE)
        
        # 롱 포지션일 경우
        elif position['amount'] > 0 and res_data:
            if futre_change['change'] > 0.1:
                # 5% 추가 매수
                print("------------------------------------------------------")
                print("Buy", amount, TARGET_COIN_SYMBOL)
                binance.create_order(TARGET_COIN_SYMBOL, amount, 'buy', current_price)
                print("------------------------------------------------------")
                binance.set_stop_loss(TARGET_COIN_SYMBOL, STOP_LOSS_RATE)
            elif futre_change['change'] < -0.1:
                # 포지션 종료, 5% 추가 매도
                print("------------------------------------------------------")
                print("Sell", amount, TARGET_COIN_SYMBOL)
                binance.create_order(TARGET_COIN_SYMBOL, amount + abs_amt, 'sell', current_price)
                print("------------------------------------------------------")
                binance.set_stop_loss(TARGET_COIN_SYMBOL, STOP_LOSS_RATE)

            #내 보유 수량의 절반을 손절한다 단!! 매수 비중이 90% 이상이면서 내 수익율이 손절 마이너스 수익율보다 작을 때
            if revenue_rate <= DANGER_RATE and buy_percent >= 90.0:
                
                #주문 취소후
                binance.cancel_all_orders(TARGET_COIN_SYMBOL)
                time.sleep(0.1)
                '''
                #클래스에선 수수료 절감 차원에서 지정가로 잡았지만 단점은 100% 포지션이 종료되거나 잡힌다는 보장이 없다는 점입니다.
                                    
                #해당 코인 가격을 가져온다.
                coin_price = myBinance.GetCoinNowPrice(binanceX, Target_Coin_Ticker)

                #숏 포지션을 잡는다
                print(binanceX.create_limit_sell_order(Target_Coin_Ticker, abs_amt / 2.0, coin_price))
                '''
                #따라서 여기서는 시장가로 잡습니다 <- 이렇게 하는걸 권장드려요!
                #숏 포지션을 잡는다
                #print(binanceX.create_market_sell_order(Target_Coin_Ticker, abs_amt / 2.0))
                print(binance.create_market_order(TARGET_COIN_SYMBOL, 'sell', abs_amt / 2.0))

                #스탑 로스 설정을 건다.
                binance.set_stop_loss(TARGET_COIN_SYMBOL, STOP_LOSS_RATE)
            
    
if __name__ == '__main__':
    main()