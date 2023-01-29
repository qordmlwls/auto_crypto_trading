import os
import json

import boto3

from src.component.binance.constraint import (
    TIME_WINDOW, BINANCE_API_KEY, BINANCE_SECRET_KEY, 
    TARGET_COIN_SYMBOL, TARGET_COIN_TICKER,
    LEVERAGE
)
from src.component.binance.binance import Binance
from src.module.db.redis import Redis


def main():
    client = boto3.client('sagemaker-runtime')
    redis = Redis(host='localhost', port=6379, db=0)
    binance = Binance(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    binance.set_leverage(TARGET_COIN_SYMBOL, LEVERAGE)
    position = binance.position_check(TARGET_COIN_SYMBOL)
        
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
    if res_data:
        pass
    else:
        pass
    
    
if __name__ == '__main__':
    main()