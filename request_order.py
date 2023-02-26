import os
import json
from typing import List, Dict, Tuple

import boto3
import numpy as np
import time

from src.component.binance.constraint import (
    TIME_WINDOW, BINANCE_API_KEY, BINANCE_SECRET_KEY, 
    TARGET_COIN_SYMBOL, TARGET_COIN_TICKER,
    LEVERAGE, TARGET_RATE, TARGET_REVENUE_RATE, STOP_LOSS_RATE, DANGER_RATE, PLUS_FUTURE_PRICE_RATE, MINUS_FUTURE_PRICE_RATE,
    STOP_PROFIT_RATE, PROFIT_AMOUNT_MULTIPLIER, STOP_REVENUE_PROFIT_RATE, CURRENT_VARIANCE, FUTURE_CHANGES_DIR, FUTURE_CHANGE_MULTIPLIER, 
    FUTURE_MAX_LEN, FUTURE_MIN_LEN, TRADE_RATE
)
from src.component.binance.binance import Binance
from src.module.db.redis.redis import Redis


def chek_futre_price(res_data: List, future_changes: Dict) -> Tuple[Dict, Dict]:
    pos_cnt = len([change for change in res_data if change > 0])
    neg_cnt = len([change for change in res_data if change < 0])
    # pos, neg중 더 많은 쪽의 change를 계산하고, max_index를 반환한다.
    if pos_cnt > neg_cnt:
        max_index = np.argmax(res_data)
        if not "plus_future_changes" in future_changes.keys():
            future_changes["plus_future_changes"] = [res_data[max_index] * 100]
        else:
            future_changes["plus_future_changes"].append(res_data[max_index] * 100)
            future_changes["plus_future_changes"] = future_changes["plus_future_changes"][-FUTURE_MAX_LEN:]
    else:
        max_index = np.argmin(res_data)
        if not "minus_future_changes" in future_changes.keys():
            future_changes["minus_future_changes"] = [res_data[max_index] * 100]
        else:
            future_changes["minus_future_changes"].append(res_data[max_index] * 100)
            future_changes["minus_future_changes"] = future_changes["minus_future_changes"][-FUTURE_MAX_LEN:]
    # chages = [abs(future_price - current_price) for future_price in future_price_list]
    # max_index = np.argmax(chages)    
    future_change = {"max_chage": res_data[max_index] * 100, "max_index": max_index}
    return future_change, future_changes


def main():
    client = boto3.client("sagemaker-runtime")
    redis = Redis(host="localhost", port=6379, db=0)
    binance = Binance(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    binance.set_leverage(TARGET_COIN_SYMBOL, LEVERAGE)
    position = binance.position_check(TARGET_COIN_SYMBOL)
    future_changes = json.load(open(os.path.join(FUTURE_CHANGES_DIR, "future_changes.json"), "r"))
    #시장가 taker 0.04, 지정가 maker 0.02 -> 시장가가 수수료가 더 비싸다.

    #시장가 숏 포지션 잡기 
    #print(binance.create_market_sell_order(Target_Coin_Ticker, 0.002))
    #print(binance.create_order(Target_Coin_Ticker, "market", "sell", 0.002, None))
    
    # 수집 시간
    time.sleep(0.1)
    if redis.size() == TIME_WINDOW:
        
        keys = list(redis.keys())
        keys.sort()
        data_list = redis.get_many(keys)
        request_body = json.dumps({
            "data_list": data_list
        })
        res = client.invoke_endpoint(EndpointName="Autotrading-Endpoint",
                                    ContentType="application/json",
                                    Accept="application/json",
                                    Body=request_body)
        # next 30분 각각의 예측값을 받아온다. 길이 30
        res_data = json.loads(res["Body"].read().decode("utf-8"))["prediction"]
        # res_data = [24000 for _ in range(30)]
        
        
    current_price = data_list[-1]["close"]
    # scale이 안 맞으므로 맞춰줌
    # diff = current_price - res_data[0]
    # res_data = [price + diff for price in res_data]
    # 레버리지에 따를 최대 매수 가능 수량
    
    # 예측값과 현재값의 scale을 맞춰준다.
    # current_diff = [abs(current_price - data_list[i]["close"]) / current_price * 100 for i in range(0, len(data_list) - 1)]
    # variance = np.mean(current_diff)
    variance = CURRENT_VARIANCE
    future_diff = [abs(res_data[i]) * 100 for i in range(0, len(res_data))]
    future_variance = np.mean(future_diff)
    # scaling
    res_data = [res_data[i] * (1 + variance / future_variance) for i in range(0, len(res_data))]
    
    max_amount = round(binance.get_amout(position["total"], current_price, 0.5), 3) * LEVERAGE
    
    one_percent_amount = round(max_amount * 0.01, 3)
    
    #첫 매수 비중을 구한다.. 여기서는 5%! 
    # * 20.0 을 하면 20%가 되겠죠? 마음껏 조절하세요!
    first_amount = one_percent_amount * TRADE_RATE

    # 비트코인의 경우 0.001로 그보다 작은 수량으로 주문을 넣으면 오류가 납니다.!
    #최소 주문 수량을 가져온다 
    # 해당 함수는 수강생 분이 만드신 걸로 아래 링크를 참고하세요!
    # https://blog.naver.com/zhanggo2/222722244744
    minimun_amount = binance.get_minimum_amout(TARGET_COIN_TICKER)
    
    #minimun_amount 안에는 최소 주문수량이 들어가 있습니다. 비트코인이니깐 0.001보다 작다면 0.001개로 셋팅해줍니다.
    if first_amount < minimun_amount:
        first_amount = minimun_amount
    
    #음수를 제거한 절대값 수량 ex -0.1 -> 0.1 로 바꿔준다.
    abs_amt = abs(position["amount"])

    if res_data:
        global PLUS_FUTURE_PRICE_RATE
        global MINUS_FUTURE_PRICE_RATE
        futre_change, future_changes = chek_futre_price(res_data, future_changes)
        # volatility
        # max_index = np.argmax([abs(change) for change in res_data])
        # futre_change = {"max_chage": res_data[max_index] * 100, "max_index": max_index}
        # scaling 할경우
        # futre_change = {"max_chage": res_data[max_index], "max_index": max_index}
        if 'plus_future_changes' in future_changes.keys():
            if len(future_changes["plus_future_changes"]) >= FUTURE_MIN_LEN:
                # 상위권을 cut_change로 정한다.
                plus_chages = future_changes["plus_future_changes"].copy()
                plus_chages.sort(reverse=False)
                # future_changes["plus_future_changes"].sort(reverse=False)
                # PLUS_FUTURE_PRICE_RATE = future_changes["plus_future_changes"][int(len(future_changes["plus_future_changes"]) * FUTURE_CHANGE_MULTIPLIER)]
                PLUS_FUTURE_PRICE_RATE = plus_chages[int(len(plus_chages) * FUTURE_CHANGE_MULTIPLIER)]
        if 'minus_future_changes' in future_changes.keys():
            if len(future_changes["minus_future_changes"]) >= FUTURE_MIN_LEN:
                minus_chages = future_changes["minus_future_changes"].copy()
                minus_chages.sort(reverse=True)
                # future_changes["minus_future_changes"].sort(reverse=True)
                # MINUS_FUTURE_PRICE_RATE = future_changes["minus_future_changes"][int(len(future_changes["minus_future_changes"]) * FUTURE_CHANGE_MULTIPLIER)]
                MINUS_FUTURE_PRICE_RATE = minus_chages[int(len(minus_chages) * FUTURE_CHANGE_MULTIPLIER)]
        with open(os.path.join(FUTURE_CHANGES_DIR, "future_changes.json"), "w") as f:
            json.dump(future_changes, f)
        print("------------------------------------------------------")
        print("future price change", futre_change["max_chage"], "%")
        print("plus future price change", PLUS_FUTURE_PRICE_RATE, "%")
        print("minus future price change", MINUS_FUTURE_PRICE_RATE, "%")
    else:
        pass
        
    #0이면 포지션 잡기전
    if abs_amt == 0 and res_data:
        
        if futre_change["max_chage"] > PLUS_FUTURE_PRICE_RATE:
            print("------------------------------------------------------")
            print("Buy", first_amount, TARGET_COIN_TICKER)
            print("------------------------------------------------------")
            #매수 주문을 넣는다.
            # binance.create_order(TARGET_COIN_TICKER, "buy", first_amount, current_price)
            binance.create_market_order(TARGET_COIN_TICKER, "buy", first_amount)
            binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
        elif futre_change["max_chage"] < MINUS_FUTURE_PRICE_RATE:
            print("------------------------------------------------------")
            print("Sell", first_amount, TARGET_COIN_TICKER)
            print("------------------------------------------------------")
            #매도 주문을 넣는다.
            # binance.create_order(TARGET_COIN_TICKER, "sell", first_amount, current_price)
            binance.create_market_order(TARGET_COIN_TICKER, "sell", first_amount)
            binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
        else:
            # 혹시 스탑로스 걸려있음 제거한다.
            binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)

        
    #0이 아니라면 포지션 잡은 상태
    else:
        print("------------------------------------------------------")
        #현재까지 구매한 퍼센트! 즉 비중!! 현재 보유 수량을 1%의 수량으로 나누면 된다.
        buy_percent = abs_amt / (max_amount * 0.01)
        print("Buy Percent : ", buy_percent)    

        #수익율을 구한다!
        revenue_rate = (current_price - position["entry_price"]) / position["entry_price"] * 100.0
        #단 숏 포지션일 경우 수익이 나면 마이너스로 표시 되고 손실이 나면 플러스가 표시 되므로 -1을 곱하여 바꿔준다.
        if position["amount"] < 0:
            revenue_rate = revenue_rate * -1.0
        #레버리지를 곱한 실제 수익율 - 원래 수량보다 레버리지률을 곱한만큼 더 산 것이므로 수익율도 레버리지 곱한 것이 된다.
        leverage_revenu_rate = revenue_rate * LEVERAGE  
        
        print("Revenue Rate : ", revenue_rate,", Real Revenue Rate : ", leverage_revenu_rate)
        
        #레버리지를 곱한 실제 손절 할 마이너스 수익율
        #레버리지를 곱하고 난 여기가 실제 내 원금 대비 실제 손실율입니다!
        leverage_danger_rate = DANGER_RATE * LEVERAGE
        
        #@TODO: 목표 future_change, 추가매수, 첫구매 비율 공통상수로 빼기
        amount = one_percent_amount * TRADE_RATE
        profit_amount = one_percent_amount * TRADE_RATE * PROFIT_AMOUNT_MULTIPLIER
        if amount < minimun_amount:
            amount = minimun_amount
        if profit_amount < minimun_amount:
            profit_amount = minimun_amount * PROFIT_AMOUNT_MULTIPLIER
        print("Danger Rate : ", DANGER_RATE,", Real Danger Rate : ", leverage_danger_rate)    
        # if leverage_revenu_rate > STOP_PROFIT_RATE:
        if revenue_rate > STOP_REVENUE_PROFIT_RATE:
            if abs(position["amount"]) < profit_amount:
                profit_amount = abs(position["amount"])
            if position["amount"] > 0:
                # 5% 매도
                print("------------------------------------------------------")
                print(f"이익 0.2% 이상이므로 {5 * PROFIT_AMOUNT_MULTIPLIER}% 매도")
                # current_price = binance.get_now_price(TARGET_COIN_TICKER)
                binance.create_market_order(TARGET_COIN_TICKER, "sell", profit_amount)
                # binance.create_order(TARGET_COIN_TICKER, "sell", amount, current_price)
                position["amount"] = position["amount"] - profit_amount
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
            elif position["amount"] < 0:
                print("------------------------------------------------------")
                print(f"이익 0.2% 이상이므로 {5 * PROFIT_AMOUNT_MULTIPLIER}% 매수")
                # current_price = binance.get_now_price(TARGET_COIN_TICKER)
                # binance.create_order(TARGET_COIN_TICKER, "buy", amount, current_price)
                binance.create_market_order(TARGET_COIN_TICKER, "buy", profit_amount)
                position["amount"] = position["amount"] + profit_amount
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)

        # 숏 포지션일 경우
        if position["amount"] < 0 and res_data:
            if futre_change["max_chage"] < MINUS_FUTURE_PRICE_RATE:
                # 5% 추가 매도
                print("------------------------------------------------------")
                print("Sell", amount, TARGET_COIN_TICKER)
                # binance.create_order(TARGET_COIN_TICKER, "sell", amount, current_price)
                binance.create_market_order(TARGET_COIN_TICKER, "sell", amount)
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
                print("------------------------------------------------------")
            elif futre_change["max_chage"] > PLUS_FUTURE_PRICE_RATE and revenue_rate > STOP_REVENUE_PROFIT_RATE: # 손실 방지
                # 포지션 종료, 5% 추가 매수
                print("------------------------------------------------------")
                print("Buy", amount, TARGET_COIN_TICKER)
                # binance.create_order(TARGET_COIN_TICKER, "buy", amount + abs_amt, current_price)
                binance.create_market_order(TARGET_COIN_TICKER, "buy", amount + abs_amt)
                print("------------------------------------------------------")
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
                
            #내 보유 수량의 절반을 손절한다 단!! 매수 비중이 90% 이상이면서 내 수익율이 손절 마이너스 수익율보다 작을 때
            if revenue_rate <= DANGER_RATE and buy_percent >= 90.0:
                
                #주문 취소후
                binance.cancel_all_orders(TARGET_COIN_TICKER)
                time.sleep(0.1)
                """
                #클래스에선 수수료 절감 차원에서 지정가로 잡았지만 단점은 100% 포지션이 종료되거나 잡힌다는 보장이 없다는 점입니다.
                                    
                #해당 코인 가격을 가져온다.
                coin_price = myBinance.GetCoinNowPrice(binanceX, Target_Coin_Ticker)

                #숏 포지션을 잡는다
                print(binanceX.create_limit_sell_order(Target_Coin_Ticker, abs_amt / 2.0, coin_price))
                """
                #따라서 여기서는 시장가로 잡습니다 <- 이렇게 하는걸 권장드려요!
                #숏 포지션을 잡는다
                #print(binanceX.create_market_sell_order(Target_Coin_Ticker, abs_amt / 2.0))
                print(binance.create_market_order(TARGET_COIN_TICKER, "buy", abs_amt / 2.0))

                #스탑 로스 설정을 건다.
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
        
        # 롱 포지션일 경우
        elif position["amount"] > 0 and res_data:
            if futre_change["max_chage"] > PLUS_FUTURE_PRICE_RATE:
                # 5% 추가 매수
                print("------------------------------------------------------")
                print("Buy", amount, TARGET_COIN_TICKER)
                # binance.create_order(TARGET_COIN_TICKER, "buy", amount, current_price)
                binance.create_market_order(TARGET_COIN_TICKER, "buy", amount)
                print("------------------------------------------------------")
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
            elif futre_change["max_chage"] < MINUS_FUTURE_PRICE_RATE and revenue_rate > STOP_REVENUE_PROFIT_RATE: # 손실 방지
                # 포지션 종료, 5% 추가 매도
                print("------------------------------------------------------")
                print("Sell", amount, TARGET_COIN_TICKER)
                # binance.create_order(TARGET_COIN_TICKER, "sell", amount + abs_amt, current_price)
                binance.create_market_order(TARGET_COIN_TICKER, "sell", amount + abs_amt)
                print("------------------------------------------------------")
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)

            #내 보유 수량의 절반을 손절한다 단!! 매수 비중이 90% 이상이면서 내 수익율이 손절 마이너스 수익율보다 작을 때 (스탑로스는 청산 방지용, 이건 손절용)
            if revenue_rate <= DANGER_RATE and buy_percent >= 90.0:
                
                #주문 취소후
                binance.cancel_all_orders(TARGET_COIN_TICKER)
                time.sleep(0.1)
                """
                #클래스에선 수수료 절감 차원에서 지정가로 잡았지만 단점은 100% 포지션이 종료되거나 잡힌다는 보장이 없다는 점입니다.
                                    
                #해당 코인 가격을 가져온다.
                coin_price = myBinance.GetCoinNowPrice(binanceX, Target_Coin_Ticker)

                #숏 포지션을 잡는다
                print(binanceX.create_limit_sell_order(Target_Coin_Ticker, abs_amt / 2.0, coin_price))
                """
                #따라서 여기서는 시장가로 잡습니다 <- 이렇게 하는걸 권장드려요!
                #숏 포지션을 잡는다
                #print(binanceX.create_market_sell_order(Target_Coin_Ticker, abs_amt / 2.0))
                print(binance.create_market_order(TARGET_COIN_TICKER, "sell", abs_amt / 2.0))

                #스탑 로스 설정을 건다.
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
            
    
if __name__ == "__main__":
    main()