import os
from os.path import join
from typing import Dict
import argparse

import json
import tarfile
import shutil
from distutils.dir_util import copy_tree
import logging

from src.component.preprocess.preprocess import preprocess
from src.module.db.s3 import S3


ROOT_DIR = '/opt/ml/processing'
DATA_DIR = join(ROOT_DIR, 'data')
OUTPUT_DIR = join(ROOT_DIR, 'output')
TRAIN_CODE_DIR = join(ROOT_DIR, 'code')
DEPLOY_CODE_DIR = join(ROOT_DIR, 'deploy_code')

BUKET_NAME = 'autocryptotrading'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def main(config: Dict):
    # s3 = S3(BUKET_NAME)
    # for key in s3.s3.Bucket(BUKET_NAME).objects.all():
    #     if key.key.startswith('data/data_'):
    #         s3.download_file(key.key, join(DATA_DIR, key.key.split('/')[-1]))
    file_list = os.listdir(DATA_DIR)
    data_list = []
    for file in file_list:
        if 'data_' in file:
            with open(join(DATA_DIR, file), 'r') as f:
                data = json.load(f)
            data_list.append(data)
    logger.info(f'Number of data: {len(data_list)}')
    logger.info(f'First data: {data_list[0]}')
    logger.info(f'Last data: {data_list[-1]}')
    data_list.sort(key=lambda x: x['ticker']['timestamp'])
    df = preprocess(data_list, config)
    logger.info(f'Number of rows: {len(df)}')
    logger.info(f'First row: {df.iloc[0]}')
    logger.info(f'Last row: {df.iloc[-1]}')
    df.to_csv(join(OUTPUT_DIR, 'preprocessed.csv'), index=False)
    shutil.move('/opt/ml/code/pipelines/auto_trading_model/train.py', '/opt/ml/train.py') 
    shutil.move('/opt/ml/code/pipelines/auto_trading_model/inference.py', '/opt/ml/inference.py')   
    copy_tree('/opt/ml/code/src', '/opt/ml/src')
    with tarfile.open('code.tar.gz', 'w:gz') as f:
        f.add('src')
        f.add('train.py')
        f.add('inference.py')
    shutil.move('code.tar.gz', os.path.join(TRAIN_CODE_DIR, 'code.tar.gz'))
    shutil.move('/opt/ml/code/pipelines/auto_trading_model/deploy_model.py', os.path.join(DEPLOY_CODE_DIR, 'deploy_model.py'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moving_average_window', type=int, default=100)
    parser.add_argument('--data_minute_limit', type=int, default=43200, help='minute data limit, default is 30 days')
    parameters = parser.parse_args()
    config = parameters.__dict__
    main(config)
    