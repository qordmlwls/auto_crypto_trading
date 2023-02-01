import os
from os.path import join
import json
import tarfile
import shutil
from distutils.dir_util import copy_tree

from src.component.preprocess.preprocess import preprocess


ROOT_DIR = '/opt/ml/processing'
DATA_DIR = join(ROOT_DIR, 'data')
OUTPUT_DIR = join(ROOT_DIR, 'output')
TRAIN_CODE_DIR = join(ROOT_DIR, 'code')
DEPLOY_CODE_DIR = join(ROOT_DIR, 'deploy_code')


def main():
    file_list = os.listdir(DATA_DIR)
    data_list = []
    for file in file_list:
        if 'data_' in file:
            with open(join(DATA_DIR, file), 'r') as f:
                data = json.load(f)
            data_list.append(data)
    data_list.sort(key=lambda x: x['ticker']['timestamp'])
    df = preprocess(data_list)
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
    main()
    