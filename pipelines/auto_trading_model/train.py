import logging
import os
import argparse

import pandas as pd
from distutils.dir_util import copy_tree

from src.component.train.train import GruTrainer
from src.component.binance.constraint import TIME_WINDOW, COLUMN_LIMIT


INPUT_DIR = '/opt/ml/input/data'
TRAIN_DIR = '/opt/ml/input/data/crypto_data'
MODEL_DIR = '/opt/ml/model'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_size', type=int, default=TIME_WINDOW, help='input size of sequence length')
    parser.add_argument('--batch_size', type=int, default=150, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of gru')
    parser.add_argument('--num_layers', type=int, default=3, help='number of gru layers')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    # parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--drop_out', type=float, default=0.2, help='drop out rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--cpu_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR, help='model directory')
    parser.add_argument('--test_mode', type=bool, default=False, help='test mode')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')
    parser.add_argument('--fp16', type=bool, default=True, help='fp16 mode')
    parser.add_argument('--patience', type=int, default=1000, help='patience for early stopping')
    parser.add_argument('--column_limit', type=int, default=COLUMN_LIMIT, help='column limit for bid and ask')
    parser.add_argument('--activation_function', type=str, default='none', help='activation function')
    parser.add_argument('--moving_average_window', type=int, default=100, help='moving average window')
    parser.add_argument('--loss_type', type=str, default='huber', help='loss type')
    parser.add_argument('--scaler_x', type=str, default='minmax', help='scaler for x')
    parser.add_argument('--scaler_y', type=str, default='robust', help='scaler for y')
    parser.add_argument('--addtional_layer', type=bool, default=False, help='addtional layer')
    parser.add_argument('--output_size', type=int, default=TIME_WINDOW, help='output size of prediction')
    parser.add_argument('--make_robust', type=bool, default=True, help='make robust')
    parameters = parser.parse_args()
    model_config = parameters.__dict__
    
    logger.info('Loading data...')
    ch_name_train = 'preprocessed'
    train_input_files = [os.path.join(TRAIN_DIR, file) for file in os.listdir(TRAIN_DIR) if
                         ch_name_train in file and 'manifest' not in file]
    
    raw_train = [pd.read_csv(file, engine='python') for file in train_input_files]
    
    train_data = pd.concat(raw_train)
    
    logger.info('Saving data...')
    copy_tree('/opt/ml/code', '/opt/ml/model/code')
    
    logger.info('Start training...')
    gru_trainer = GruTrainer(**model_config)
    gru_trainer.run(train_data)
    