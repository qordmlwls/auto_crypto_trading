import os
import json
from torch import Tensor
from pandas import DataFrame
from typing import NoReturn, Dict, List, Tuple

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import joblib
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from src.module.utills.data import prepare_batch
from src.module.utills.gpu import get_gpu_device


class GrudModel(LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.input_size = args['input_size']
        # self.input_size = input_size
        self.hidden_size = args['hidden_size']
        self.output_size = args['frame_size']  # next step의 모든 close price를 예측
        self.num_layers = args['num_layers']
        self.batch_size = args['batch_size']
        self.learning_rate = args['learning_rate']
        self.sequence_length = args['frame_size']
        self.drop_out = args['drop_out']
        self.activation_function = args['activation_function']
        self.loss_type = args['loss_type']
        self.addtional_layer = args['addtional_layer']
        # self.device = args['device']

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, 
                          dropout=self.drop_out, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size * self.sequence_length,)
        self.intermediate = nn.Linear(self.hidden_size * self.sequence_length, self.hidden_size * self.sequence_length)
        # nn.init.kaiming_normal_(self.intermediate.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.intermediate.weight, nonlinearity='leaky_relu')
        self.weight_initialization(self.intermediate, self.activation_function )
        self.layer_norm2 = nn.LayerNorm(self.hidden_size * self.sequence_length,)
        self.intermediate2 = nn.Linear(self.hidden_size * self.sequence_length, self.hidden_size * self.sequence_length)
        # nn.init.kaiming_normal_(self.intermediate2.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.intermediate2.weight, nonlinearity='leaky_relu')
        self.weight_initialization(self.intermediate2, self.activation_function)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size * self.sequence_length,)
        self.fc = nn.Linear(self.hidden_size * self.sequence_length, self.output_size)
        # nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
        self.weight_initialization(self.fc, self.activation_function)
        self.drop_out_layer = nn.Dropout(self.drop_out)
        if args['loss_type'] == 'mse':
            self.criterion = nn.MSELoss()
        elif args['loss_type'] == 'mae':
            self.criterion = nn.L1Loss()
        elif args['loss_type'] == 'huber':
            self.criterion = nn.SmoothL1Loss()
        elif args['loss_type'] == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        # self.activation_fn = nn.ReLU()
        self.activation_fn = self.activation(self.activation_function)
        
    def activation(self, activation_function):
        if activation_function == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_function == 'tanh':
            return nn.Tanh()
        elif activation_function == 'relu':
            return nn.ReLU()
        elif activation_function == 'gelu': 
            return nn.GELU()
        
    def weight_initialization(self, layer, activation_function):
        if activation_function == 'leaky_relu':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
        elif activation_function == 'tanh':
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
        elif activation_function == 'relu':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        else:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            return
            
        
    def forward(self, x):
        # input x: (batch, seq_len, input_size)
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x.float(), h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # out, _ = self.gru(x.float())  # out: tensor of shape (batch_size, seq_length, hidden_size)
        maximum = out.max()
        if maximum == np.Inf or maximum == np.NINF:
            raise ValueError('ValueError: inf or -inf')
        # many to many
        out = out.reshape(out.shape[0], -1)  # out: (batch_size, seq_length * hidden_size)
        if self.activation_function == 'tanh':
            # loss inf 발산 문제 해결 위해 추가
            out = self.layer_norm1(out)
            out = self.drop_out_layer(self.activation_fn(out))
            # out = self.activation_fn(out)
        elif self.activation_function == 'leaky_relu' or self.activation_function == 'relu':
            out = self.layer_norm1(out)
            out = self.drop_out_layer(self.activation_fn(out))
            
        else:    
            pass
        # to detect saturation effect
        print('out max: ', out.max())
        print('out min: ', out.min())
        # out = F.relu(out)
        # out = self.drop_out_layer(self.activation_fn(out))
        # out = self.drop_out_layer(self.activation_fn(self.intermediate(out)))
        # out = self.drop_out_layer(self.activation_fn(self.intermediate2(out)))
        if self.addtional_layer:
            out = self.drop_out_layer(self.activation_fn(self.layer_norm1(out)))
            out = self.drop_out_layer(self.activation_fn(self.layer_norm2(self.intermediate(out))))
            out = self.drop_out_layer(self.activation_fn(self.layer_norm3(self.intermediate2(out))))
        # out = F.relu(self.layer_norm(out))
        out = self.fc(out)  # out: (batch_size, output_size)
        return out

    def training_step(self, batch, batch_idx):
        # x: (batch, seq_len, input_size)
        # y: (batch, output_size(=frame_size))
        x, y = batch['data'], batch['target'].float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # self.log('train_loss', loss)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss)
    
    def validation_step(self, batch, batch_idx):
        # x: (batch, seq_len, input_size)
        # y: (batch, output_size(=frame_size))
        x, y = batch['data'], batch['target']
        y_hat = self(x)
        if self.loss_type == 'bce':
            loss = self.criterion(y_hat, y.float())
        else:
            loss = self.criterion(y_hat, y)
        # self.log('val_loss', loss)  
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class GrudDataset(Dataset):
    def __init__(self, x: List[DataFrame], y: List[DataFrame]):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx) -> Dict[str, DataFrame]:
        return {
            'x': self.x[idx],
            'y': self.y[idx]
        }
    

class GruTrainer:
    
    def __init__(self, **args):
        self.args = args
        self.train_dataloader = None
        self.val_dataloader = None
        self.device = get_gpu_device()
        # self.args.update({'device': self.device})
        
    def _prepare_batch_wrapper(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return prepare_batch(batch)
    
    # rolling window
    def _build_sequence(self, train: DataFrame, test: DataFrame) -> Tuple[List, List]:
        data_x = []
        data_y = []
        # x는 맨 마지막 step을 제외하고, y는 맨 첫번째 step을 제외하고
        for i in range(len(train) - 2 * self.args['frame_size']):
            _x = train[i: i + self.args['frame_size']]
            data_x.append(_x)
            
        for j in range(self.args['frame_size'], len(train) - self.args['frame_size']):
            _y = test[j: j + self.args['frame_size']]
            # volatilty
            volatility = (_y - test[j-1: j]['close'].values[0]) / test[j-1: j]['close'].values[0]
            # data_y.append(_y)
            data_y.append(volatility)
            
        return data_x, data_y
        
    def _prepare_dataset(self, df: DataFrame) -> NoReturn:
        
        columns = ['open', 'high', 'low', 'close', 'volume', f"ma_{self.args['moving_average_window']}"] + [f'bid_{i}' for i in range(self.args['column_limit'])] \
                    + [f'ask_{i}' for i in range(self.args['column_limit'])] + [f'bid_volume_{i}' for i in range(self.args['column_limit'])] \
                    + [f'ask_volume_{i}' for i in range(self.args['column_limit'])]
        # x = df
        x = df[columns].copy()
        y = df[['close']].copy()
        # 이상치가 많으므로 RobustScaler 사용 -> 다시 Minmax사용, X에는 크기가 많은 값이 많아서 딥러닝 loss가 안줄고 saturation effect 있으므로 MinMaxScaler 사용
        if self.args['scaler_x'] == 'minmax':
            scaler_x = MinMaxScaler()
        elif self.args['scaler_x'] == 'robust':
            scaler_x = RobustScaler()
        if self.args['scaler_y'] == 'minmax':
            scaler_y = MinMaxScaler()
        elif self.args['scaler_y'] == 'robust':
            scaler_y = RobustScaler()
        else:
            scaler_y = None
        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=self.args['test_ratio'], shuffle=False)
        
        # trainset만 스케일링, volatilty지만 아웃라이어가 많아서 scaling 진행
        # scaled_train_x = pd.DataFrame(scaler_x.fit_transform(train_x), columns=train_x.columns)
        # scaled_train_y = pd.DataFrame(scaler_y.fit_transform(train_y), columns=train_y.columns)
        
        # scaled_val_x = pd.DataFrame(scaler_x.transform(val_x), columns=val_x.columns)
        # scaled_val_y = pd.DataFrame(scaler_y.transform(val_y), columns=val_y.columns)
        
        # volatilty
        scaled_train_x = pd.DataFrame(scaler_x.fit_transform(train_x), columns=train_x.columns)
        scaled_train_y = train_y
        
        scaled_val_x = pd.DataFrame(scaler_x.transform(val_x), columns=val_x.columns)
        scaled_val_y = val_y
        
        train_x, train_y = self._build_sequence(scaled_train_x, scaled_train_y)
        val_x, val_y = self._build_sequence(scaled_val_x, scaled_val_y)
        whole_y = pd.concat(train_y)
        if self.args['loss_type'] == 'bce': # binary classification
            train_y = [pd.DataFrame((y > 0).astype(int), columns=y.columns) for y in train_y]
            val_y = [pd.DataFrame((y > 0).astype(int), columns=y.columns) for y in val_y]
        elif scaler_y is not None:
            scaler_y.fit(whole_y)
            train_y = [pd.DataFrame(scaler_y.transform(y), columns=y.columns) for y in train_y]
            val_y = [pd.DataFrame(scaler_y.transform(y), columns=y.columns) for y in val_y]
        else:
            pass
        train_dataset = GrudDataset(train_x, train_y)
        val_dataset = GrudDataset(val_x, val_y)
        self.train_dataloader = DataLoader(train_dataset, 
                                           batch_size=self.args['batch_size'], 
                                           shuffle=False,  # 시계열이라 막 셔플하면 안됨
                                           num_workers=self.args['cpu_workers'],
                                           collate_fn=self._prepare_batch_wrapper,
                                        #    drop_last=True
                                           )
        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=self.args['batch_size'],
                                         shuffle=False,
                                         num_workers=self.args['cpu_workers'],
                                         collate_fn=self._prepare_batch_wrapper,
                                        #  drop_last=True
                                         )
        test_batch = next(iter(self.val_dataloader))
        data, target = test_batch['data'], test_batch['target']
        self.args.update({'input_size': len(data[0][0])})
        
        # save hyperparameters
        with open(os.path.join(self.args['model_dir'], 'hyperparameters.json'), 'w') as f:
            json.dump(self.args, f)
        # save scaler
        joblib.dump(scaler_x, os.path.join(self.args['model_dir'], 'scaler_x.pkl'))
        if scaler_y is not None:
            joblib.dump(scaler_y, os.path.join(self.args['model_dir'], 'scaler_y.pkl'))
        # self.args.update({'device': self.device})
        # with open('scaler.pkl', 'wb') as f:
        #     pickle.dump(scaler, f)
        
    def run(self, df: DataFrame) -> NoReturn:
        
        self._prepare_dataset(df)
        seed_everything(self.args['seed'])
        model = GrudModel(**self.args)
        model.to(self.device)
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.args['model_dir'],
            filename='grud-{epoch:02d}-{val_loss:.6f}',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min',
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=self.args['patience'],
            verbose=False,
            mode='min'
        )
        learning_rate_monitor = LearningRateMonitor(logging_interval='step')
        if pl.__version__ == '1.7.6':
            trainer = Trainer(
                callbacks=[checkpoint_callback, early_stop_callback, learning_rate_monitor],
                max_epochs=self.args['epochs'],
                fast_dev_run=self.args['test_mode'],
                num_sanity_val_steps=None if self.args['test_mode'] else 0,

                # For gpu Setup
                deterministic=True if self.device != torch.device('cpu') else False,
                gpus=-1 if self.device != 'cpu' else None,
                precision=16 if self.args['fp16'] else 32,
                accelerator='cuda' if torch.cuda.is_available() else None,  # pytorch-lightning 1.7.6
                strategy='dp' if torch.cuda.is_available() else None  # pytorch-lightning 1.7.6
            )
        elif pl.__version__ == '1.4.9':
            trainer = Trainer(
                callbacks=[checkpoint_callback, early_stop_callback, learning_rate_monitor],
                max_epochs=self.args['epochs'],
                fast_dev_run=self.args['test_mode'],
                num_sanity_val_steps=None if self.args['test_mode'] else 0,

                # For gpu Setup
                deterministic=True if self.device != torch.device('cpu') else False,
                gpus=-1 if self.device != 'cpu' else None,
                precision=16 if self.args['fp16'] else 32,
                accelerator='dp' if torch.cuda.is_available() else None  # pytorch-lightning 1.4.9
            )
        else:
            raise Exception("pytorch lightning version should be 1.7.6 or 1.4.9")
        # trainer = Trainer(
        #     gpus=1,
        #     max_epochs=self.args['epochs'],
        #     callbacks=[checkpoint_callback, early_stop_callback],
        #     progress_bar_refresh_rate=20,
        #     logger=pl.loggers.TensorBoardLogger(self.args['model_dir'])
        # )
        trainer.fit(model=model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)
        # trainer.save_checkpoint(os.path.join(self.args['model_dir'], 'grud.ckpt'))
        print('Done training')
        
                                         