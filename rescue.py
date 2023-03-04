import os
import subprocess
import boto3

from src.component.binance.constraint import (
    TIME_WINDOW, BINANCE_API_KEY, BINANCE_SECRET_KEY, 
    TARGET_COIN_SYMBOL, TARGET_COIN_TICKER,
    LEVERAGE, TARGET_RATE, TARGET_REVENUE_RATE, STOP_LOSS_RATE, DANGER_RATE, PLUS_FUTURE_PRICE_RATE, MINUS_FUTURE_PRICE_RATE,
    STOP_PROFIT_RATE, PROFIT_AMOUNT_MULTIPLIER, STOP_REVENUE_PROFIT_RATE, CURRENT_VARIANCE, FUTURE_CHANGES_DIR, FUTURE_CHANGE_MULTIPLIER, 
    FUTURE_MAX_LEN, FUTURE_MIN_LEN, TRADE_RATE, MOVING_AVERAGE_WINDOW, SWITCHING_CHANGE_MULTIPLIER,
    COLUMN_LIMIT, MINIMUM_FUTURE_PRICE_RATE, EXIT_PRICE_CHANGE, RETRAIN_SIGNAL_RATE
)
from src.component.binance.binance import Binance

SAGEMAKER_EXECUTION_ROLE = os.environ.get('SAGEMAKER_EXECUTION_ROLE', '')

def main():
    sagemaker = boto3.client('sagemaker')
    pipeline_info = sagemaker.list_pipeline_executions(PipelineName='AutotradingTrainPipeline')
    for pipeline in pipeline_info['PipelineExecutionSummaries']:
        if pipeline['PipelineExecutionStatus'] == 'Executing':
            print("Pipeline is already running.")
            return
    binance = Binance(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    position = binance.position_check(TARGET_COIN_SYMBOL)
    current_price = binance.get_now_price(TARGET_COIN_TICKER)
    max_amount = round(binance.get_amout(position["total"], current_price, 0.5), 3) * LEVERAGE
    minimun_amount = binance.get_minimum_amout(TARGET_COIN_TICKER)
    abs_amt = abs(position["amount"])
    if abs_amt == 0:
        return
    else:
        buy_percent = abs_amt / (max_amount * 0.01)
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
        # 물려있고 손실만 보는 상태이면 모델이 깨졌을 가능성이 높다. 재학습 요청
        if (position['free'] / (position['total'] + 1)) < RETRAIN_SIGNAL_RATE and revenue_rate < 0:
            print("Model is broken. Request retraining.")
            subprocess.call("""python3 /home/ubuntu/auto_crypto_trading/pipelines/run_pipeline.py --module-name pipelines.auto_trading_model.pipeline --role-arn %s  --tags '[{"Key": "Test", "Value": "Test"}]' --kwargs '{"region": "ap-northeast-2", "role": "%s"}'
                            """ % (SAGEMAKER_EXECUTION_ROLE, SAGEMAKER_EXECUTION_ROLE), shell=True)
            return
        # print("""python3 /home/ubuntu/auto_crypto_trading/pipelines/run_pipeline.py --module-name pipelines.auto_trading_model.pipeline --role-arn %s  --tags '[{"Key": "Test", "Value": "Test"}]' --kwargs '{"region": "ap-northeast-2", "role": "%s"}'
        #                     """ % (SAGEMAKER_EXECUTION_ROLE, SAGEMAKER_EXECUTION_ROLE))

if __name__ == "__main__":
    main()