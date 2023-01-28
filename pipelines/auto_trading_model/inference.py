import logging
import sys
import os

import torch
import numpy as np
import json
import pandas as pd
import joblib

from src.component.train.train import GrudModel
from src.component.preprocess.preprocess import preprocess
from src.module.utills.gpu import get_gpu_device
from src.module.utills.data import rolling_window


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info('Start')

MODEL_DIR = '/opt/ml/model'

device = get_gpu_device()
with open(os.path.join(MODEL_DIR, 'hyperparameters.json'), 'r') as f:
    model_config = json.load(f)
scaler_x = joblib.load(os.path.join(MODEL_DIR, 'scaler_x.pkl'))
scaler_y = joblib.load(os.path.join(MODEL_DIR, 'scaler_y.pkl'))


def input_fn(request_body, request_content_type):
    logger.info('input_fn')
    assert request_content_type == 'application/json'
    data_list = json.loads(request_body)['data_list']
    
    df = preprocess(data_list)
    scaled_x = pd.DataFrame(scaler_x.transform(df), columns=df.columns)
    
    idx_array = np.array([i for i in range(len(df))])
    rolling_tensor = torch.tensor(rolling_window(idx_array, model_config['frame_size']))
    
    embedding = torch.tensor(scaled_x.values).to(device)
    time_embedding = embedding[rolling_tensor.long()]
    return {'input_data': time_embedding}


def model_fn(model_dir):
    logger.info('model_fn')
    file_exist = False
    model_file_list = []
    try:
        for file_name in os.listdir(model_dir):
            if file_name.endswith('.ckpt'):
                model_file_list.append(file_name)
                file_exist = True
        if not file_exist:
            raise Exception('No model file')
        highest_index = np.argmax([int(file_name.split('=')[2].split('.')[0]) for file_name in model_file_list])
        highest_file_name = model_file_list[highest_index]
        model = GrudModel.load_from_checkpoint(os.path.join(model_dir, highest_file_name), **model_config)
        logger.info(f'Load model from {highest_file_name}')
    
    except Exception as e:
        raise FileNotFoundError(f'No model file in {model_dir}') from e
    model.to(device).eval()
    return model


def predict_fn(input_data, model):
    logger.info('predict_fn')
    if isinstance(input_data['input_data'], torch.Tensor):
        output = model(input_data)
        out = scaler_y.inverse_transform(output.cpu().detach().numpy())
        logger.info('Predition done')
    else:
        raise Exception('Input data is not tensor')
    return out


def output_fn(prediction, response_content_type):
    logger.info('output_fn')
    return json.dumps({'prediction': prediction.tolist()})

# for testing

if __name__ == '__main__':
    request_body = json.dumps({'data_list': [1,2,3]})
    
    data = input_fn(request_body, 'application/json')
    model = model_fn(MODEL_DIR)
    prediction = predict_fn(data, model)
    out = output_fn(prediction, 'application/json')