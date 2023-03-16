import os
import json
from typing import List, Dict, Tuple

import boto3
import numpy as np
import pandas as pd
import time
from datetime import datetime

from src.component.binance.constraint import (
    TIME_WINDOW, BINANCE_API_KEY, BINANCE_SECRET_KEY, 
    TARGET_COIN_SYMBOL, TARGET_COIN_TICKER,
    LEVERAGE, TARGET_RATE, TARGET_REVENUE_RATE, STOP_LOSS_RATE, DANGER_RATE, PLUS_FUTURE_PRICE_RATE, MINUS_FUTURE_PRICE_RATE,
    STOP_PROFIT_RATE, PROFIT_AMOUNT_MULTIPLIER, STOP_REVENUE_PROFIT_RATE, CURRENT_VARIANCE, FUTURE_CHANGES_DIR, FUTURE_CHANGE_MULTIPLIER, 
    FUTURE_MAX_LEN, FUTURE_MIN_LEN, TRADE_RATE, MOVING_AVERAGE_WINDOW, SWITCHING_CHANGE_MULTIPLIER,
    COLUMN_LIMIT, MINIMUM_FUTURE_PRICE_RATE, EXIT_PRICE_CHANGE, LOSS_CRITERIA_RATE, LOSS_TYPE, TAKE_PROFIT_MULTIPLIER, MA_VARIANT_PREVIOUS_STEP,
    HIGH_RSI, LOW_RSI, RSI_DIR, RSI_MAX_LEN
)
from src.component.binance.binance import Binance
from src.module.db.redis.redis import Redis
from src.module.utills.data import get_ma


def rsi_calc(ohlc: pd.DataFrame, period: int = 14):
    ohlc = ohlc[4].astype(float)
    delta = ohlc.diff()
    gains, declines = delta.copy(), delta.copy()
    gains[gains < 0] = 0
    declines[declines > 0] = 0

    _gain = gains.ewm(com=(period-1), min_periods=period).mean()
    _loss = declines.abs().ewm(com=(period-1), min_periods=period).mean()

    RS = _gain / _loss
    return pd.Series(100-(100/(1+RS)), name="RSI")

# def rsi_binance(itv='1h', simbol='BTC/USDT', ohlcv=None):
def rsi_binance(ohlcv=None):
    
    # ohlcv = binance.fetch_ohlcv(symbol="BTC/USDT", timeframe=itv, limit=200)
    df = pd.DataFrame(ohlcv)
    rsi = rsi_calc(df, 12).iloc[-1]
    return rsi

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
    print([i * 100 for i in res_data])
    future_change = {"max_chage": res_data[max_index] * 100, "max_index": max_index}
    return future_change, future_changes

def get_price_ma_variant(data_list: List, window: int) -> Tuple[float, float]:
    columns = ['open', 'high', 'low', 'close', 'volume', f"ma_{window}"] + [f'bid_{i}' for i in range(COLUMN_LIMIT)] \
                + [f'ask_{i}' for i in range(COLUMN_LIMIT)] + [f'bid_volume_{i}' for i in range(COLUMN_LIMIT)] \
                + [f'ask_volume_{i}' for i in range(COLUMN_LIMIT)]
    df = pd.DataFrame(data_list)
    df = get_ma(df, window)[columns]
    # price_variant = df["close"].iloc[-1] - df["close"].iloc[-window] 
    price_variant = df["close"].iloc[-1] - df["close"].iloc[-15] 
    # ma_vaiant = df[f"ma_{window}"].iloc[-1] - df[f"ma_{window}"].iloc[-window]
    ma_vaiant = df[f"ma_{window}"].iloc[-1] - df[f"ma_{window}"].iloc[-15]
    return price_variant, ma_vaiant

def get_price_ma_variant_with_index(data_list: List, window: int, index: int) -> Tuple[float, float]:
    columns = ['open', 'high', 'low', 'close', 'volume', f"ma_{window}"] + [f'bid_{i}' for i in range(COLUMN_LIMIT)] \
                + [f'ask_{i}' for i in range(COLUMN_LIMIT)] + [f'bid_volume_{i}' for i in range(COLUMN_LIMIT)] \
                + [f'ask_volume_{i}' for i in range(COLUMN_LIMIT)]
    df = pd.DataFrame(data_list)
    df = get_ma(df, window)[columns]
    # price_variant = df["close"].iloc[-1] - df["close"].iloc[-window] 
    price_variant = df["close"].iloc[-1] - df["close"].iloc[-index] 
    # ma_vaiant = df[f"ma_{window}"].iloc[-1] - df[f"ma_{window}"].iloc[-window]
    ma_vaiant = df[f"ma_{window}"].iloc[-1] - df[f"ma_{window}"].iloc[-index]
    ma_vaiant_previous = df[f"ma_{window}"].iloc[-1 - MA_VARIANT_PREVIOUS_STEP] - df[f"ma_{window}"].iloc[-index - MA_VARIANT_PREVIOUS_STEP]
    return price_variant, ma_vaiant, ma_vaiant_previous

def main():
    client = boto3.client("sagemaker-runtime")
    redis = Redis(host="localhost", port=6379, db=0)
    binance = Binance(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    binance.set_leverage(TARGET_COIN_SYMBOL, LEVERAGE)
    position = binance.position_check(TARGET_COIN_SYMBOL)
    future_changes = json.load(open(os.path.join(FUTURE_CHANGES_DIR, "future_changes.json"), "r"))
    now_minute = datetime.now().minute
    now_hour = datetime.now().hour
    around_per_30 = abs(now_minute - 30) <= 10
    around_per_60 = abs(now_minute - 60) <= 10
    around_per_30_5 = abs(now_minute - 30) <= 5
    around_per_60_5 = abs(now_minute - 60) <= 5
    korean_time = 4 <= now_hour <= 9
    
    ohlcv = binance.get_ohlcv(ticker=TARGET_COIN_TICKER, timeframe='1m', limit=200)
    ohlcv = pd.DataFrame(ohlcv)
    volume = ohlcv[5]
    close = ohlcv[4]
    delta = close.diff()
    # delta_list = delta[-RSI_MAX_LEN:].tolist()
    delta_list = delta[-5:].tolist()
    neg_cnt = len([i for i in delta_list[-3:] if i < 0])
    pos_cnt = len([i for i in delta_list[-3:] if i > 0])
    neg_cnt_5 = len([i for i in delta_list[-5:] if i < 0])
    pos_cnt_5 = len([i for i in delta_list[-5:] if i > 0])
    # volume_list = volume[-RSI_MAX_LEN:].tolist()
    volume_list = volume[-5:].tolist()
    max_volume_index = np.argmax(volume_list)
    max_delta_index = np.argmax([abs(i) for i in delta_list])
    top_delta = delta_list[max_volume_index]
    top_variant_delta = delta_list[max_delta_index]
    rsi = rsi_binance(ohlcv)
    rsi_dic = json.load(open(os.path.join(RSI_DIR, "rsi.json"), "r"))
    if "volume" not in rsi_dic.keys():
        rsi_dic["volume"] = []
    if "close" not in rsi_dic.keys():
        rsi_dic["close"] = []
    # rsi_dic["volume"].append(volume)
    # rsi_dic["close"].append(close)
    rsi_dic["rsi"].append(rsi)
    rsi_dic["rsi"] = rsi_dic["rsi"][-RSI_MAX_LEN:]
    # rsi_dic["volume"] = rsi_dic["volume"][-RSI_MAX_LEN:]
    # rsi_dic["close"] = rsi_dic["close"][-RSI_MAX_LEN:]
    with open(os.path.join(RSI_DIR, "rsi.json"), "w") as f:
        json.dump(rsi_dic, f)
    print("------------------------------------------------------")
    print("RSI LIST :", rsi_dic["rsi"][-RSI_MAX_LEN:])
    print("VOLUME LIST :", volume_list)
    print("DELTA LIST :", delta_list)
    print("MAX VOLUME : ", volume_list[max_volume_index])
    print("TOP DELTA : ", top_delta)
    print("TOP VARIANT DELTA : ", top_variant_delta)
    print("------------------------------------------------------")
    #시장가 taker 0.04, 지정가 maker 0.02 -> 시장가가 수수료가 더 비싸다.

    #시장가 숏 포지션 잡기 
    #print(binance.create_market_sell_order(Target_Coin_Ticker, 0.002))
    #print(binance.create_order(Target_Coin_Ticker, "market", "sell", 0.002, None))
    # 지정가 주문 실패된 것 취소하기
    binance.cancel_failed_order(TARGET_COIN_TICKER)
    
    # 수집 시간
    time.sleep(0.1)
    # if redis.size() == TIME_WINDOW:
    if redis.size() == MOVING_AVERAGE_WINDOW + TIME_WINDOW:
        
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
        # res = client.invoke_endpoint(EndpointName="test",
        #                             ContentType="application/json",
        #                             Accept="application/json",
        #                             Body=request_body)
        # next 30분 각각의 예측값을 받아온다. 길이 30
        res_data = json.loads(res["Body"].read().decode("utf-8"))["prediction"]
        # res_data = [24000 for _ in range(30)]
    else:
        print("Not enough data")
        return
    # price_variant, ma_variant = get_price_ma_variant(data_list, 25)
        
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
    # _, ma_100_variant = get_price_ma_variant(data_list, 100)

    # scaling
    res_data = [res_data[i] * (1 + variance / future_variance) for i in range(0, len(res_data))]
    
    max_amount = round(binance.get_amout(position["total"], current_price, 0.5), 3) * LEVERAGE
    
    one_percent_amount = round(max_amount * 0.01, 3)
    
    #첫 매수 비중을 구한다.. 여기서는 5%! 
    # * 20.0 을 하면 20%가 되겠죠? 마음껏 조절하세요!
    first_amount = one_percent_amount * TRADE_RATE * 2

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
    plus_switching_rate = 0.14
    minus_switching_rate = -0.14
    if res_data:
        global PLUS_FUTURE_PRICE_RATE
        global MINUS_FUTURE_PRICE_RATE
        global MINIMUM_FUTURE_PRICE_RATE
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
                if plus_chages[int(len(plus_chages) * FUTURE_CHANGE_MULTIPLIER)] < MINIMUM_FUTURE_PRICE_RATE:
                    PLUS_FUTURE_PRICE_RATE = MINIMUM_FUTURE_PRICE_RATE
                else:
                    PLUS_FUTURE_PRICE_RATE = plus_chages[int(len(plus_chages) * FUTURE_CHANGE_MULTIPLIER)]
                if plus_chages[int(len(plus_chages) * SWITCHING_CHANGE_MULTIPLIER)] < MINIMUM_FUTURE_PRICE_RATE:
                    plus_switching_rate = MINIMUM_FUTURE_PRICE_RATE
                else:
                    plus_switching_rate = plus_chages[int(len(plus_chages) * SWITCHING_CHANGE_MULTIPLIER)]
        if 'minus_future_changes' in future_changes.keys():
            if len(future_changes["minus_future_changes"]) >= FUTURE_MIN_LEN:
                minus_chages = future_changes["minus_future_changes"].copy()
                minus_chages.sort(reverse=True)
                # future_changes["minus_future_changes"].sort(reverse=True)
                # MINUS_FUTURE_PRICE_RATE = future_changes["minus_future_changes"][int(len(future_changes["minus_future_changes"]) * FUTURE_CHANGE_MULTIPLIER)]
                if minus_chages[int(len(minus_chages) * FUTURE_CHANGE_MULTIPLIER)] > -MINIMUM_FUTURE_PRICE_RATE:
                    MINUS_FUTURE_PRICE_RATE = -MINIMUM_FUTURE_PRICE_RATE
                else:
                    MINUS_FUTURE_PRICE_RATE = minus_chages[int(len(minus_chages) * FUTURE_CHANGE_MULTIPLIER)]
                if minus_chages[int(len(minus_chages) * SWITCHING_CHANGE_MULTIPLIER)] > -MINIMUM_FUTURE_PRICE_RATE:
                    minus_switching_rate = -MINIMUM_FUTURE_PRICE_RATE
                else:
                    minus_switching_rate = minus_chages[int(len(minus_chages) * SWITCHING_CHANGE_MULTIPLIER)]
        with open(os.path.join(FUTURE_CHANGES_DIR, "future_changes.json"), "w") as f:
            json.dump(future_changes, f)
        print("------------------------------------------------------")
        print("future price change", futre_change["max_chage"], "%")
        print("plus future price change", PLUS_FUTURE_PRICE_RATE, "%")
        print("minus future price change", MINUS_FUTURE_PRICE_RATE, "%")
        print("plus switching price change", plus_switching_rate, "%")
        print("minus switching price change", minus_switching_rate, "%")
    else:
        futre_change = []
        pass
    
    print("------------------------------------------------------")
    _, ma_100_variant, ma_100_variant_previous = get_price_ma_variant_with_index(data_list, 60, 2)
    _, ma_25_variant, ma_25_variant_previous = get_price_ma_variant_with_index(data_list, 25, 2)
    print("ma_100_variant", ma_100_variant)
    print("ma_100_variant_previous", ma_100_variant_previous)
    print("ma_25_variant", ma_25_variant)
    print("ma_25_variant_previous", ma_25_variant_previous)
    increase_percent = (ma_100_variant - ma_100_variant_previous) / abs(ma_100_variant_previous) * 100
    increase_percent_25 = (ma_25_variant - ma_25_variant_previous) / abs(ma_25_variant_previous) * 100
    print("increase_percent", increase_percent)
    print("increase_percent_25", increase_percent_25)
    sum_rsi = 0
    rsi_variant = [abs(i - rsi) for i in rsi_dic["rsi"]]
    rsi_varint_increase = (sum(rsi_variant) / len(rsi_variant)) > 7
    if (position["amount"] > 0) or (position["amount"] == 0 and ma_100_variant > 0):
        # variant_increase = ma_100_variant > ma_100_variant_previous
        variant_increase = increase_percent > 7
        variant_increase_25 = increase_percent_25 > 6
        sum_rsi = sum([i > rsi for i in rsi_dic["rsi"]])
        rsi_vary = sum_rsi > 5
        sum_5_rsi = sum([i > rsi for i in rsi_dic["rsi"][-6:]])
        rsi_5_vary = sum_5_rsi >= 2
        top_delta_same = top_delta > 0
        top_variant_delta_same = top_variant_delta > 0
        delta_cnt = pos_cnt > neg_cnt
        delta_cnt_5 = pos_cnt_5 > neg_cnt_5
        delta_5_ratio_condition = pos_cnt_5 == 3 and neg_cnt_5 == 2
    elif position["amount"] < 0 or (position["amount"] == 0 and ma_100_variant < 0):
        # variant_increase = ma_100_variant < ma_100_variant_previous
        variant_increase = increase_percent < -7
        variant_increase_25 = increase_percent_25 < -6
        sum_rsi = sum([i < rsi for i in rsi_dic["rsi"]])
        rsi_vary = sum_rsi > 5
        sum_5_rsi = sum([i < rsi for i in rsi_dic["rsi"][-6:]])
        rsi_5_vary = sum_5_rsi >= 2
        top_delta_same = top_delta < 0
        top_variant_delta_same = top_variant_delta < 0
        delta_cnt = neg_cnt > pos_cnt
        delta_cnt_5 = neg_cnt_5 > pos_cnt_5
        delta_5_ratio_condition = neg_cnt_5 == 3 and pos_cnt_5 == 2
    else:
        variant_increase = False
        variant_increase_25 = False
        rsi_vary = False
        rsi_5_vary = False
        top_delta_same = False
        top_variant_delta_same = False
        delta_cnt = False
        delta_cnt_5 = False
    # RSI vary, RSI variant increse는 저점, 고점 판독 위함 False일 경우 고점, 저점임 cf) rsi_variant_increase는 가격 변동성 측정 위함. 균형 이루면 힘이 없을 가능 성 큼.
    print("variant_increase", variant_increase)
    print("variant_increase_25", variant_increase_25)
    print('RSI :', rsi)
    print('RSI vary :', rsi_vary)
    print('RSI 5 vary :', rsi_5_vary)
    print('RSI variant :', rsi_variant)
    print('RSI variant increase :', rsi_varint_increase)
    print('TOP DELTA same :', top_delta_same)
    print("TOP VARIANT DELTA same", top_variant_delta_same)
    print("DELTA CNT", delta_cnt)
    print("DELTA CNT 5", delta_cnt_5)
    print("DELTA 5 RATIO CONDITION", delta_5_ratio_condition)
    print("------------------------------------------------------")
    # 시간차 막기 위해 다시 체크
    current_price = binance.get_now_price(TARGET_COIN_TICKER)
    position = binance.position_check(TARGET_COIN_SYMBOL)
    abs_amt = abs(position["amount"])
    
    # long_criteria = ma_100_variant > 0 and ma_25_variant > 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase and variant_increase_25 and rsi_5_vary and top_delta_same and top_variant_delta_same and delta_cnt
    # short_criteria = ma_100_variant < 0 and ma_25_variant < 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase and variant_increase_25 and rsi_5_vary and top_delta_same and top_variant_delta_same and delta_cnt
    
    # long_criteria_new = ma_100_variant > 0 and ma_25_variant > 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase and variant_increase_25 and rsi_5_vary and not top_delta_same and not top_variant_delta_same and delta_cnt_5
    # short_criteria_new = ma_100_variant < 0 and ma_25_variant < 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase and variant_increase_25 and rsi_5_vary and not top_delta_same and not top_variant_delta_same and delta_cnt_5
    # long_criteria_new = ma_100_variant > 0 and ma_25_variant > 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase_25 and rsi_5_vary and not top_delta_same and not top_variant_delta_same and delta_cnt_5
    # short_criteria_new = ma_100_variant < 0 and ma_25_variant < 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase_25 and rsi_5_vary and not top_delta_same and not top_variant_delta_same and delta_cnt_5
    
    # long_criteria_new = ma_100_variant > 0 and ma_25_variant > 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase_25 and rsi_5_vary and not top_delta_same
    # short_criteria_new = ma_100_variant < 0 and ma_25_variant < 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase_25 and rsi_5_vary and not top_delta_same
    
    # 조정이 아니고 추세를 따라가는 로직
    long_criteria_new_new = ma_100_variant > 0 and ma_25_variant > 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase_25 and variant_increase and not rsi_5_vary and delta_5_ratio_condition and top_delta_same and current_price > 0
    short_criteria_new_new = ma_100_variant < 0 and ma_25_variant < 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase_25 and variant_increase and not rsi_5_vary and delta_5_ratio_condition and top_delta_same and current_price < 0
    
    #0이면 포지션 잡기전
    if abs_amt == 0 and res_data:
        # and (not around_per_30_5 or not around_per_60_5):  
        
        # if futre_change["max_chage"] > PLUS_FUTURE_PRICE_RATE and ma_100_variant > 0 and abs(ma_100_variant) >= 1 and variant_increase and rsi < 66: # 추세 추종, 추세 꺾일때 진입 방지
        # if 0 < futre_change["max_chage"] < PLUS_FUTURE_PRICE_RATE and ma_100_variant > 0 and abs(ma_100_variant) >= 1 and variant_increase and rsi < 66: # 추세 추종, 추세 꺾일때 진입 방지
        # if ma_100_variant > 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi < HIGH_RSI or rsi_5_vary) and not rsi_vary and not (futre_change["max_chage"] < minus_switching_rate) and ma_25_variant > 0 and rsi_varint_increase:  
        # if ma_100_variant > 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi < HIGH_RSI or rsi_5_vary) and rsi_vary and not (futre_change["max_chage"] < minus_switching_rate) and ma_25_variant > 0 and rsi_varint_increase:  
        # if ma_100_variant > 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi < HIGH_RSI or rsi_5_vary) and rsi_vary and not (futre_change["max_chage"] < minus_switching_rate) and rsi_varint_increase and top_delta_same:  
        # long_criteria = ma_100_variant > 0 and ma_25_variant > 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase and variant_increase_25 and rsi_5_vary and top_delta_same and top_variant_delta_same
        # short_criteria = ma_100_variant < 0 and ma_25_variant < 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase and variant_increase_25 and rsi_5_vary and top_delta_same and top_variant_delta_same
        # # if ma_100_variant > 0 and ma_25_variant > 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase and variant_increase_25 and rsi_5_vary and top_delta_same and top_variant_delta_same:
        # if (long_criteria and not korean_time) or (short_criteria and korean_time):
        # if long_criteria or long_criteria_new:
        # if long_criteria_new or long_criteria_new_new:
        if long_criteria_new_new:
        # if ma_100_variant > 0 and abs(ma_100_variant) >= 1 and variant_increase and rsi < 66 and futre_change["max_chage"] > 0 and ma_25_variant >0:  
            print("------------------------------------------------------")
            print("Buy", first_amount, TARGET_COIN_TICKER)
            print("------------------------------------------------------")
            #매수 주문을 넣는다.
            current_price = binance.get_now_price(TARGET_COIN_TICKER)
            binance.create_order(TARGET_COIN_TICKER, "buy", first_amount, current_price)
            # binance.create_market_order(TARGET_COIN_TICKER, "buy", first_amount)
            binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
        # elif futre_change["max_chage"] < MINUS_FUTURE_PRICE_RATE and ma_100_variant < 0 and abs(ma_100_variant) >= 1 and variant_increase and rsi > 30:
        # elif 0 > futre_change["max_chage"] > MINUS_FUTURE_PRICE_RATE and ma_100_variant < 0 and abs(ma_100_variant) >= 1 and variant_increase and rsi > 30:
        # elif ma_100_variant < 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi > LOW_RSI or rsi_5_vary) and not rsi_vary and not (futre_change["max_chage"] > plus_switching_rate) and ma_25_variant < 0 and rsi_varint_increase:
        # elif ma_100_variant < 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi > LOW_RSI or rsi_5_vary) and rsi_vary and not (futre_change["max_chage"] > plus_switching_rate) and ma_25_variant < 0 and rsi_varint_increase:
        # elif ma_100_variant < 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi > LOW_RSI or rsi_5_vary) and rsi_vary and not (futre_change["max_chage"] > plus_switching_rate) and rsi_varint_increase and top_delta_same:
        # elif ma_100_variant < 0 and ma_25_variant < 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase and variant_increase_25 and rsi_5_vary and top_delta_same and top_variant_delta_same:  
        # elif (short_criteria and not korean_time) or (long_criteria and korean_time):
        # elif short_criteria or short_criteria_new:
        # elif short_criteria_new or short_criteria_new_new:
        elif short_criteria_new_new:
        # elif ma_100_variant < 0 and abs(ma_100_variant) >= 1 and variant_increase and rsi > 40 and not futre_change["max_chage"] < 0 and ma_25_variant < 0:
            print("------------------------------------------------------")
            print("Sell", first_amount, TARGET_COIN_TICKER)
            print("------------------------------------------------------")
            #매도 주문을 넣는다.
            current_price = binance.get_now_price(TARGET_COIN_TICKER)
            binance.create_order(TARGET_COIN_TICKER, "sell", first_amount, current_price)
            # binance.create_market_order(TARGET_COIN_TICKER, "sell", first_amount)
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
        # global amount
        amount = one_percent_amount * TRADE_RATE  * 2
        profit_amount = one_percent_amount * TRADE_RATE * PROFIT_AMOUNT_MULTIPLIER
        if amount < minimun_amount:
            amount = minimun_amount
        if profit_amount < minimun_amount:
            profit_amount = minimun_amount 
        print("Danger Rate : ", DANGER_RATE,", Real Danger Rate : ", leverage_danger_rate)    
        # if leverage_revenu_rate > STOP_PROFIT_RATE:
        if revenue_rate > STOP_PROFIT_RATE:
            if abs(position["amount"]) < profit_amount:
                profit_amount = abs(position["amount"])
            if position["amount"] > 0:
                if res_data:
                    if futre_change['max_chage'] < 0: 
                        # 오를 것 같으면 포지션 종료 (손실 방지)
                        if around_per_30 or around_per_60: # 30분, 60분 추세가 반대로 바뀔 것 같으면
                            profit_amount = abs_amt
                        else:
                            profit_amount = profit_amount * 4.0
                            if abs(position["amount"]) < profit_amount:
                                profit_amount = abs(position["amount"])
                        print("------------------------------------------------------")
                        print("떨어질 것 같으므로", profit_amount, "매도")
                # 5% 매도
                print("------------------------------------------------------")
                print(f"이익 0.2% 이상이므로 {5 * PROFIT_AMOUNT_MULTIPLIER}% 매도")
                current_price = binance.get_now_price(TARGET_COIN_TICKER)
                # binance.create_market_order(TARGET_COIN_TICKER, "sell", profit_amount)
                binance.create_order(TARGET_COIN_TICKER, "sell", profit_amount, current_price)
                position["amount"] = position["amount"] - profit_amount
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
            elif position["amount"] < 0:
                if res_data:
                    if futre_change['max_chage'] > 0 and (around_per_30 or around_per_60):
                        if around_per_30 or around_per_60: # 30분, 60분 추세가 반대로 바뀔 것 같으면
                            # 오를 것 같으면 포지션 종료 (손실 방지), 이미 많이 떨어지고 이
                            profit_amount = abs_amt
                        else:
                            profit_amount = profit_amount * 4.0
                            if abs(position["amount"]) < profit_amount:
                                profit_amount = abs(position["amount"])
                            print("------------------------------------------------------")
                            print("오를 것 같으므로", profit_amount, "매수")
                print("------------------------------------------------------")
                print(f"이익 0.2% 이상이므로 {5 * PROFIT_AMOUNT_MULTIPLIER}% 매수")
                current_price = binance.get_now_price(TARGET_COIN_TICKER)
                binance.create_order(TARGET_COIN_TICKER, "buy", profit_amount, current_price)
                # binance.create_market_order(TARGET_COIN_TICKER, "buy", profit_amount)
                position["amount"] = position["amount"] + profit_amount
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
        # 수익은 별로 없지만 남은 잔고가 없을 경우
        elif (position['free'] / (position['total'] + 1)) < 0.01 and leverage_revenu_rate > 0:
            if abs(position["amount"]) < profit_amount:
                profit_amount = abs(position["amount"])
            
            if position["amount"] > 0:
            
                print("------------------------------------------------------")
                print(f"이익이 별로 없지만 남은 잔고가 없음, {5 * PROFIT_AMOUNT_MULTIPLIER}% 매도")
                current_price = binance.get_now_price(TARGET_COIN_TICKER)
                # binance.create_market_order(TARGET_COIN_TICKER, "sell", profit_amount)
                binance.create_order(TARGET_COIN_TICKER, "sell", profit_amount, current_price)
                position["amount"] = position["amount"] - profit_amount
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
            
            elif position["amount"] < 0:
                
                print("------------------------------------------------------")
                print(f"이익이 별로 없지만 남은 잔고가 없음, {5 * PROFIT_AMOUNT_MULTIPLIER}% 매수")
                current_price = binance.get_now_price(TARGET_COIN_TICKER)
                binance.create_order(TARGET_COIN_TICKER, "buy", profit_amount, current_price)
                position["amount"] = position["amount"] + profit_amount
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
                

        # 숏 포지션일 경우
        if position["amount"] < 0 and res_data:
            # if futre_change["max_chage"] < MINUS_FUTURE_PRICE_RATE and ma_100_variant < 0 and abs(ma_100_variant) >= 1 and variant_increase and rsi > 30:
            # if 0 > futre_change["max_chage"] > MINUS_FUTURE_PRICE_RATE and ma_100_variant < 0 and abs(ma_100_variant) >= 1 and variant_increase and rsi > 30:
            # 
            # if ma_100_variant < 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi > LOW_RSI or rsi_5_vary) and not rsi_vary and not (futre_change["max_chage"] > plus_switching_rate) and ma_25_variant < 0 and rsi_varint_increase:
            # if ma_100_variant < 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi > LOW_RSI or rsi_5_vary) and rsi_vary and not (futre_change["max_chage"] > plus_switching_rate) and ma_25_variant < 0 and rsi_varint_increase:
            # if ma_100_variant < 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi > LOW_RSI or rsi_5_vary) and rsi_vary and not (futre_change["max_chage"] > plus_switching_rate) and rsi_varint_increase and top_delta_same:
            # if ma_100_variant < 0 and ma_25_variant < 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase and variant_increase_25 and rsi_5_vary and top_delta_same and top_variant_delta_same:
            # if (short_criteria and not korean_time) or (long_criteria and korean_time):
            # if short_criteria or short_criteria_new:
            # if short_criteria_new or short_criteria_new_new:
            if short_criteria_new_new:
                # and (not around_per_30_5 or not around_per_60_5):
                # price_variant, ma_variant = get_price_ma_variant(data_list, 25)
                # if ma_variant < 0:
                # 5% 추가 매도
                print("------------------------------------------------------")
                print("Sell", amount, TARGET_COIN_TICKER)
                current_price = binance.get_now_price(TARGET_COIN_TICKER)
                binance.create_order(TARGET_COIN_TICKER, "sell", amount, current_price)
                # binance.create_market_order(TARGET_COIN_TICKER, "sell", amount)
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
                print("------------------------------------------------------")
                # position = binance.position_check(TARGET_COIN_SYMBOL)
                # profit_price = position['entry_price'] * (1 + STOP_REVENUE_PROFIT_RATE)
                # take profit
                # profit_price = current_price * (1 - STOP_REVENUE_PROFIT_RATE)
                # binance.create_order(TARGET_COIN_TICKER, "buy", profit_amount, profit_price)
                if revenue_rate < STOP_REVENUE_PROFIT_RATE:
                    take_amount = one_percent_amount * TRADE_RATE * TAKE_PROFIT_MULTIPLIER
                    if abs(position["amount"]) < take_amount:
                        take_amount = abs(position["amount"])
                    print("------------------------------------------------------")
                    print("Take Profit Setting")
                    profit_price = position['entry_price'] * (1 - STOP_REVENUE_PROFIT_RATE / 100)
                    binance.create_order(TARGET_COIN_TICKER, "buy", take_amount, profit_price)
                
                
            # elif futre_change["max_chage"] > PLUS_FUTURE_PRICE_RATE and revenue_rate > STOP_REVENUE_PROFIT_RATE: # 손실 방지
            # 수익은 별로 없지만 반대방향 신호가 강한 경우 or 충분히 수익 있고 반대방향 신호가 적당히 있는 경우
            elif (futre_change["max_chage"] > plus_switching_rate and revenue_rate > 0) \
                 or (futre_change["max_chage"] > PLUS_FUTURE_PRICE_RATE and revenue_rate > STOP_REVENUE_PROFIT_RATE) \
                 or (not variant_increase and not rsi_5_vary and not rsi_varint_increase and revenue_rate > 0):
                  # and ma_100_variant > 0 and abs(ma_100_variant) > 8:
                price_variant, ma_variant = get_price_ma_variant(data_list, 25)
                if ma_variant > 0:
                # 포지션 종료, 5% 추가 매수
                    print("------------------------------------------------------")
                    print("반대 신호가 강해 포지션 스위칭")
                    print("Buy", amount, TARGET_COIN_TICKER)
                    current_price = binance.get_now_price(TARGET_COIN_TICKER)
                    binance.create_order(TARGET_COIN_TICKER, "buy", amount + abs_amt, current_price)
                    # binance.create_market_order(TARGET_COIN_TICKER, "buy", amount + abs_amt)
                    print("------------------------------------------------------")
                    binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
            
            # 손절 로직 물림 방지, 반대방향 신호가 강하고 ma, 가격도 반대방향으로 이동중일 경우 + 매수 비중이 20% 이상일 경우, 손실 보고 있을 경우
                        
            elif (futre_change["max_chage"] > plus_switching_rate):
            # and ma_100_variant > 0:
                price_variant, ma_variant = get_price_ma_variant(data_list, 25)
                # 매수 비중 10% 초과, 조금만 투입헀을 경우 손절
                if (price_variant > 0 and ma_variant > 0) and (1 - (position['free'] / (position['total'] + 1)) < LOSS_CRITERIA_RATE): # and ma_100_variant > 0 and abs(ma_100_variant) > 8:
                    # and (1 - (position['free'] / (position['total'] + 1)) > LOSS_CRITERIA_RATE) \
                    # and (revenue_rate < DANGER_RATE) \
                    
                    # 포지션 종료, 5% 추가 매수
                    print("------------------------------------------------------")
                    print("반대 신호가 강해 손절 후 포지션 스위칭")
                    print("Buy", amount, TARGET_COIN_TICKER)
                    current_price = binance.get_now_price(TARGET_COIN_TICKER)
                    binance.create_order(TARGET_COIN_TICKER, "buy", amount + abs_amt, current_price)
                    # binance.create_market_order(TARGET_COIN_TICKER, "buy", amount + abs_amt)
                    print("------------------------------------------------------")
                    binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
            
            # 포지션 종료, 지정가 매매
            # elif (not variant_increase and not rsi_5_vary and not rsi_varint_increase and revenue_rate > 0) or (ma_100_variant > 0 and abs(ma_100_variant) >= 1):
            elif (ma_100_variant > 0 and abs(ma_100_variant) >= 1):
                print("------------------------------------------------------")
                print("반대 신호가 강해 지정가 포지션 종료")
                # print("Buy", amount, TARGET_COIN_TICKER)
                current_price = binance.get_now_price(TARGET_COIN_TICKER)
                binance.create_order(TARGET_COIN_TICKER, "buy", abs_amt, current_price)
                # binance.create_market_order(TARGET_COIN_TICKER, "buy", amount + abs_amt)
                print("------------------------------------------------------")
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
                    
            # 포지션 종료, exit 시그널
            elif data_list[-1]["close"] - data_list[-3]["close"] > EXIT_PRICE_CHANGE and revenue_rate < DANGER_RATE:
                print("------------------------------------------------------")
                print("Exit Sinal이 강해 손절 감수하고 포지션 종료")
                binance.create_market_order(TARGET_COIN_TICKER, "buy", abs_amt)
                print("------------------------------------------------------")
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
            
            elif revenue_rate < STOP_REVENUE_PROFIT_RATE: # 손실 방지
                
                # take profit - 하지말자 지정가보다 낮게 내려가면 손해임
                take_amount = one_percent_amount * TRADE_RATE * TAKE_PROFIT_MULTIPLIER
                if abs(position["amount"]) < take_amount:
                    take_amount = abs(position["amount"])
                print("------------------------------------------------------")
                print("Take Profit Setting")
                profit_price = position['entry_price'] * (1 - STOP_REVENUE_PROFIT_RATE / 100)
                binance.create_order(TARGET_COIN_TICKER, "buy", take_amount, profit_price)
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
            # if futre_change["max_chage"] > PLUS_FUTURE_PRICE_RATE and ma_100_variant > 0 and abs(ma_100_variant) >= 1 and variant_increase and rsi < 66:
            # if 0 < futre_change["max_chage"] < PLUS_FUTURE_PRICE_RATE and ma_100_variant > 0 and abs(ma_100_variant) >= 1 and variant_increase and rsi < 66:
            # if ma_100_variant > 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi < HIGH_RSI or rsi_5_vary) and not rsi_vary and not (futre_change["max_chage"] < minus_switching_rate) and ma_25_variant > 0 and rsi_varint_increase:  
            # if ma_100_variant > 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi < HIGH_RSI or rsi_5_vary) and rsi_vary and not (futre_change["max_chage"] < minus_switching_rate) and ma_25_variant > 0 and rsi_varint_increase:  
            # if ma_100_variant > 0 and abs(ma_100_variant) >= 1 and variant_increase and variant_increase_25 and (rsi < HIGH_RSI or rsi_5_vary) and rsi_vary and not (futre_change["max_chage"] < minus_switching_rate) and rsi_varint_increase and top_delta_same:
            # if ma_100_variant > 0 and ma_25_variant > 0 and abs(ma_100_variant) >= 1 and abs(ma_25_variant) >= 1 and variant_increase and variant_increase_25 and rsi_5_vary and top_delta_same and top_variant_delta_same:
            # if (long_criteria and not korean_time) or (short_criteria and korean_time):
            # if long_criteria or long_criteria_new:
            # if long_criteria_new or long_criteria_new_new:
            if long_criteria_new_new:
                # and (not around_per_30_5 or not around_per_60_5) 
            #     price_variant, ma_variant = get_price_ma_variant(data_list, 25)
            # if ma_variant > 0:
            # 5% 추가 매수
                print("------------------------------------------------------")
                print("Buy", amount, TARGET_COIN_TICKER)
                current_price = binance.get_now_price(TARGET_COIN_TICKER)
                binance.create_order(TARGET_COIN_TICKER, "buy", amount, current_price)
                # binance.create_market_order(TARGET_COIN_TICKER, "buy", amount)
                print("------------------------------------------------------")
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
                if revenue_rate < STOP_REVENUE_PROFIT_RATE:
                # take profit
                    take_amount = one_percent_amount * TRADE_RATE * TAKE_PROFIT_MULTIPLIER
                    if abs(position["amount"]) < take_amount:
                        take_amount = abs(position["amount"])
                    # take profit
                    print("------------------------------------------------------")
                    print("Take Profit Setting")
                    # position = binance.position_check(TARGET_COIN_TICKER)
                    profit_price = position['entry_price'] * (1 + STOP_REVENUE_PROFIT_RATE / 100)
                    binance.create_order(TARGET_COIN_TICKER, "sell", take_amount, profit_price)
                    binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
            
                # profit_price = current_price * (1 + STOP_REVENUE_PROFIT_RATE)
            # elif futre_change["max_chage"] < MINUS_FUTURE_PRICE_RATE and revenue_rate > STOP_REVENUE_PROFIT_RATE: # 손실 방지
            # 수익은 별로 없지만 반대방향 신호가 강한 경우 or 충분히 수익 있고 반대방향 신호가 적당히 있는 경우
            elif (futre_change["max_chage"] < minus_switching_rate and revenue_rate > 0) \
                 or (futre_change["max_chage"] < MINUS_FUTURE_PRICE_RATE and revenue_rate > STOP_REVENUE_PROFIT_RATE) \
                 or (not variant_increase and not rsi_5_vary and not rsi_varint_increase and revenue_rate > 0):    
                  # and ma_100_variant < 0 and abs(ma_100_variant) > 8:
                price_variant, ma_variant = get_price_ma_variant(data_list, 25)
                if ma_variant < 0:
                # 포지션 종료, 5% 추가 매도
                    print("------------------------------------------------------")
                    print("반대 신호가 강해 포지션 스위칭")
                    print("Sell", amount, TARGET_COIN_TICKER)
                    current_price = binance.get_now_price(TARGET_COIN_TICKER)
                    binance.create_order(TARGET_COIN_TICKER, "sell", amount + abs_amt, current_price)
                    # binance.create_market_order(TARGET_COIN_TICKER, "sell", amount + abs_amt)
                    print("------------------------------------------------------")
                    binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
                
            elif (futre_change["max_chage"] < minus_switching_rate):
            # and ma_100_variant < 0:
                price_variant, ma_variant = get_price_ma_variant(data_list, 25)
                # 매수 비중 10% 초과
                if (price_variant < 0 and ma_variant < 0) and (1 - (position['free'] / (position['total'] + 1)) < LOSS_CRITERIA_RATE): # and ma_100_variant < 0 and abs(ma_100_variant) > 8:
                    # and (1 - (position['free'] / (position['total'] + 1)) > LOSS_CRITERIA_RATE) \
                    # and (revenue_rate < DANGER_RATE):
                    print("------------------------------------------------------")
                    print("반대 신호가 강해 손절 후 포지션 스위칭")
                    print("Sell", amount, TARGET_COIN_TICKER)
                    current_price = binance.get_now_price(TARGET_COIN_TICKER)
                    binance.create_order(TARGET_COIN_TICKER, "sell", amount + abs_amt, current_price)
                    # binance.create_market_order(TARGET_COIN_TICKER, "sell", amount + abs_amt)
                    print("------------------------------------------------------")
                    binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)      
            
            # 포지션 종료, 지정가 매매
            
            # elif (not variant_increase and not rsi_5_vary and not rsi_varint_increase and revenue_rate > 0)  or (ma_100_variant < 0 and abs(ma_100_variant) >= 1):
            elif (ma_100_variant < 0 and abs(ma_100_variant) >= 1):
                print("------------------------------------------------------")
                print("반대 신호가 강해 지정가 포지션 종료")
                # print("Sell", amount, TARGET_COIN_TICKER)
                current_price = binance.get_now_price(TARGET_COIN_TICKER)
                binance.create_order(TARGET_COIN_TICKER, "sell", abs_amt, current_price)
                # binance.create_market_order(TARGET_COIN_TICKER, "sell", amount + abs_amt)
                print("------------------------------------------------------")
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
                                  
            
            # 포지션 종료, exit 시그널
            elif data_list[-1]["close"] - data_list[-3]["close"] < -EXIT_PRICE_CHANGE and revenue_rate < DANGER_RATE:
                print("------------------------------------------------------")
                print("Exit Signal이 강해 손절 감수하고 포지션 종료")
                binance.create_market_order(TARGET_COIN_TICKER, "sell", abs_amt)
                binance.set_stop_loss(TARGET_COIN_TICKER, STOP_LOSS_RATE)
            
            elif revenue_rate < STOP_REVENUE_PROFIT_RATE: # 손실 방지
                take_amount = one_percent_amount * TRADE_RATE * TAKE_PROFIT_MULTIPLIER
                if abs(position["amount"]) < take_amount:
                    take_amount = abs(position["amount"])
                # take profit
                print("------------------------------------------------------")
                print("Take Profit Setting")
                # position = binance.position_check(TARGET_COIN_TICKER)
                profit_price = position['entry_price'] * (1 + STOP_REVENUE_PROFIT_RATE / 100)
                binance.create_order(TARGET_COIN_TICKER, "sell", take_amount, profit_price)
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