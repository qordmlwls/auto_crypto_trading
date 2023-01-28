import os
from os.path import join
import json

from src.component.preprocess.preprocess import preprocess


ROOT_DIR = '/opt/ml/preprocessing'
DATA_DIR = join(ROOT_DIR, 'data')
OUTPUT_DIR = join(ROOT_DIR, 'output')


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


if __name__ == '__main__':
    main()
    