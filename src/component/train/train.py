import os
import json
from torch import Tensor
from pandas import DataFrame
from typing import NoReturn

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
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

    def train_dataloader(self):
        return DataLoader(GrudDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(GrudDataset(val_x, val_y), batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(GrudDataset(test_x, test_y), batch_size=self.batch_size, shuffle=False)
    

class GruDataset(Dataset):
    def __init__(self, x: DataFrame):
        self.x = x  

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x.lioc[idx]
    

class GruTrainer:
    
    def __init__(self, **args):
        self.args = args
        self.train_dataloader = None
        self.val_dataloader = None
        self.device = get_gpu_device()
        
    def _prepare_batch_wrapper(self, df: DataFrame):
        return prepare_batch(DataFrame, self.args['frame_size'])
        
    def _prepare_dataset(self, df: DataFrame) -> NoReturn:
        
        frame_size = self.args['frame_size']
        train, val = train_test_split(df, test_size=self.args['test_ratio'], shuffle=False)
        train_dataset = GruDataset(train)
        val_dataset = GruDataset(val)
        self.train_dataloader = DataLoader(train_dataset, 
                                           batch_size=self.args['batch_size'], 
                                           shuffle=True,
                                           num_workers=self.args['cpu_workers'],
                                           collate_fn=self._prepare_batch_wrapper)
        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=self.args['batch_size'],
                                         shuffle=True,
                                         num_workers=self.args['cpu_workers'],
                                         collate_fn=self._prepare_batch_wrapper)
        test_batch = next(iter(self.val_dataloader))
        data, target = test_batch['data'], test_batch['target']
        self.args.update({'input_size': len(data[0][0])})
        
        # save hyperparameters
        with open(os.path.join(self.args['save_dir'], 'hyperparameters.json'), 'w') as f:
            json.dump(self.args, f)
    
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
                                         