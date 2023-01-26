import os
import json
from torch import Tensor
from pandas import DataFrame
from typing import NoReturn, Dict

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

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

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * self.sequence_length, self.output_size)

    def forward(self, x):
        # input x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # many to many
        out = out.reshape(out.shape[0], -1)  # out: (batch_size, seq_length * hidden_size)
        out = self.fc(out)  # out: (batch_size, output_size)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # x: (batch, seq_len, input_size)
        # y: (batch, output_size(=frame_size))
        x, y = batch['data'], batch['target']
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    # def train_dataloader(self):
    #     return DataLoader(GrudDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(GrudDataset(val_x, val_y), batch_size=self.batch_size, shuffle=False)

    # def test_dataloader(self):
    #     return DataLoader(GrudDataset(test_x, test_y), batch_size=self.batch_size, shuffle=False)
    

# class GruDataset(Dataset):
#     def __init__(self, x: DataFrame, y: DataFrame):
#         self.x = x  
#         self.y = y

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return self.x.lioc[idx]

class GrudDataset(Dataset):
    def __init__(self, x: DataFrame, y: DataFrame):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx) -> Dict[str, DataFrame]:
        return {
            'x': self.x.iloc[idx],
            'y': self.y.iloc[idx]
        }
    

class GruTrainer:
    
    def __init__(self, **args):
        self.args = args
        self.train_dataloader = None
        self.val_dataloader = None
        self.device = get_gpu_device()
        
    def _prepare_batch_wrapper(self, batch: Dict[str, DataFrame]):
        return prepare_batch(batch, self.args['frame_size'])
        
    def _prepare_dataset(self, df: DataFrame) -> NoReturn:
        
        x = df
        y = df[['close']].copy()
        scaler = MinMaxScaler()
        # scaler_y = MinMaxScaler()
        # scaled_x = scaler_x.fit_transform(x)
        # scaled_y = scaler_y.fit_transform(y)
        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, shuffle=False)
        # train, val = train_test_split(df, test_size=self.args['test_ratio'], shuffle=False)
        # trainset만 스케일링
        scaled_train_x = scaler.fit_transform(train_x)

        scaled_val_x = scaler.transform(val_x)
        train_dataset = GrudDataset(scaled_train_x, train_y)
        val_dataset = GrudDataset(scaled_val_x, val_y)
        self.train_dataloader = DataLoader(train_dataset, 
                                           batch_size=self.args['batch_size'], 
                                           shuffle=False,  # 시계열이라 막 셔플하면 안됨
                                           num_workers=self.args['cpu_workers'],
                                           collate_fn=self._prepare_batch_wrapper)
        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=self.args['batch_size'],
                                         shuffle=False,
                                         num_workers=self.args['cpu_workers'],
                                         collate_fn=self._prepare_batch_wrapper)
        test_batch = next(iter(self.val_dataloader))
        data, target = test_batch['data'], test_batch['target']
        self.args.update({'input_size': len(data[0][0])})
        
        # save hyperparameters
        with open(os.path.join(self.args['save_dir'], 'hyperparameters.json'), 'w') as f:
            json.dump(self.args, f)
        # save scaler
        joblib.dump(scaler, os.path.join(self.args['save_dir'], 'scaler.pkl'))
        # with open('scaler.pkl', 'wb') as f:
        #     pickle.dump(scaler, f)
    def run(self, df: DataFrame) -> NoReturn:
        
        self._prepare_dataset(df)
        seed_everything(self.args['seed'])
        model = GrudModel(**self.args)
        model.to(self.device)
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.args['save_dir'],
            filename='grud-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode='min'
        )
        trainer = Trainer(
            gpus=1,
            max_epochs=self.args['epochs'],
            callbacks=[checkpoint_callback, early_stop_callback],
            progress_bar_refresh_rate=20,
            logger=pl.loggers.TensorBoardLogger(self.args['save_dir'])
        )
        trainer.fit(model, self.train_dataloader, self.val_dataloader)
        trainer.test(model, self.val_dataloader)
        trainer.save_checkpoint(os.path.join(self.args['save_dir'], 'grud.ckpt'))
        print('Done training')
        return model
                                         