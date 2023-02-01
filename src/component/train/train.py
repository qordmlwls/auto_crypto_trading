import os
import json
from torch import Tensor
from pandas import DataFrame
from typing import NoReturn, Dict, List

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

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, 
                          dropout=self.drop_out ,batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hidden_size * self.sequence_length,)
        self.fc = nn.Linear(self.hidden_size * self.sequence_length, self.output_size)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # input x: (batch, seq_len, input_size)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.gru(x.float())  # out: tensor of shape (batch_size, seq_length, hidden_size)
        maximum = out.max()
        if maximum == np.Inf or maximum == np.NINF:
            raise ValueError('ValueError: inf or -inf')
        # many to many
        out = out.reshape(out.shape[0], -1)  # out: (batch_size, seq_length * hidden_size)
        out = F.relu(out)
        out = F.relu(self.layer_norm(out))
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
        
    def _prepare_batch_wrapper(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return prepare_batch(batch, self.args['frame_size'])
        
    def _prepare_dataset(self, df: DataFrame) -> NoReturn:
        
        x = df
        y = df[['close']].copy()
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        # scaled_x = scaler_x.fit_transform(x)
        # scaled_y = scaler_y.fit_transform(y)
        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=self.args['test_ratio'], shuffle=False)
        # train, val = train_test_split(df, test_size=self.args['test_ratio'], shuffle=False)
        # trainset만 스케일링
        scaled_train_x = pd.DataFrame(scaler_x.fit_transform(train_x), columns=train_x.columns)
        scaled_train_y = pd.DataFrame(scaler_y.fit_transform(train_y), columns=train_y.columns)
        
        scaled_val_x = pd.DataFrame(scaler_x.transform(val_x), columns=val_x.columns)
        scaled_val_y = pd.DataFrame(scaler_y.transform(val_y), columns=val_y.columns)
        
        train_dataset = GrudDataset(scaled_train_x, scaled_train_y)
        val_dataset = GrudDataset(scaled_val_x, scaled_val_y)
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
        with open(os.path.join(self.args['model_dir'], 'hyperparameters.json'), 'w') as f:
            json.dump(self.args, f)
        # save scaler
        joblib.dump(scaler_x, os.path.join(self.args['model_dir'], 'scaler_x.pkl'))
        joblib.dump(scaler_y, os.path.join(self.args['model_dir'], 'scaler_y.pkl'))
        # with open('scaler.pkl', 'wb') as f:
        #     pickle.dump(scaler, f)
        
    def run(self, df: DataFrame) -> NoReturn:
        
        self._prepare_dataset(df)
        seed_everything(self.args['seed'])
        model = GrudModel(**self.args)
        model.to(self.device)
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.args['model_dir'],
            filename='grud-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min',
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode='min'
        )
        if pl.__version__ == '1.7.6':
            trainer = Trainer(
                callbacks=[checkpoint_callback, early_stop_callback],
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
                callbacks=[checkpoint_callback, early_stop_callback],
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
        
                                         