import torch
import torch.nn as nn
from torch import optim

from models import DLinear, Linear, NLinear, QuLTSF 
from utils.tools import (EarlyStopping, adjust_learning_rate, set_seed, count_parameters, setup_logger, setup_device)
from utils.metrics import compute_metrics
from utils.algos import validate
from data_provider.data_factory import data_provider_cv

import logging
import numpy as np
from tqdm import tqdm
import time
import os
import warnings

warnings.filterwarnings('ignore')


model_dict = {'DLinear': DLinear, 'NLinear': NLinear, 'Linear': Linear}

config = {
    "model_name": "DLinear",  # options: [Linear, DLinear, NLinear]
    "forecasting_task": "M",  # options: [M, S, MS]
    "target": "OT",
    "freq": "h",  # options: [h: hourly, d: daily, b: business days, w: weekly, m: monthly]
    "time_embed": "timeF",  # options: [timeF, fixed]
    "data_path": "data_provider/weather.csv",
    "split_ratio": 0.9,  #for train/test split
    "k_folds": 5,
    
    "kernel_size":25,
    "individual": 0, # individual head; True 1 False 0
    "num_channels": 7, # DLinear with --individual, use this hyperparameter as the number of channels

    "seq_len": 30,  # input sequence length
    "pred_len": 5,  # prediction sequence length
    "train_epochs": 30,
    "batch_size": 8,
    "patience": 20,  # early stopping patience
    "learning_rate": 0.0001,
    "loss_function": "mse",
    "lradj": "type3",  # adjust learning rate
}

setting = (
    f"{config['model_name']}_tr{config['target']}_epo{config['train_epochs']}_"
    f"lr{config['learning_rate']}_sl{config['seq_len']}_pl{config['pred_len']}_"
    f"kfolds{config['k_folds']}"
)
print(setting)


## SET UP
set_seed(seed=2021)
logger = setup_logger(setting)
device = setup_device(setting)
count_parameters(model_dict[config["model_name"]].Model(config).float())

model_path = './checkpoints/' + setting + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
    

## LOAD DATA
train_datas, train_loaders, val_datas, val_loaders = data_provider_cv(config, flag='train')
test_data, test_loader = data_provider_cv(config, flag='test')


## TRAIN
for fold, train_data, train_loader, val_data, val_loader in zip(range(1,config["k_folds"]+1), train_datas, train_loaders, val_datas, val_loaders): 
    
    logger.info(f"------ Fold"+str(fold)+" ------")
    
    model = model_dict[config["model_name"]].Model(config).float().to(device)
    model_optim = optim.Adam(model.parameters(), lr=config["learning_rate"])
    early_stopping = EarlyStopping(patience=config["patience"], verbose=True)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(config["train_epochs"]), desc="Training Progress"):
        
        logger.info(f"[EPOCH {epoch + 1}/{config['train_epochs']}]")
        epoch_time = time.time()
        train_loss = []        
        model.train()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            
            f_dim = -1 if config["forecasting_task"] == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:].to(device)
                
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            loss.backward()
            model_optim.step()

        train_loss = np.average(train_loss)
        vali_loss = validate(val_data, val_loader, model, criterion, device,  setting, config, fold)

        logger.info("Cost time: {0:.1f} | Train Loss: {1:.7f} | Vali Loss: {2:.7f}".format(time.time() - epoch_time, train_loss, vali_loss))
        
        early_stopping(vali_loss , model, model_path, logger)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

        adjust_learning_rate(model_optim, epoch + 1, config, logger)
    
    final_vali_mse, final_vali_mae = compute_metrics(model, val_data, val_loader, device, config, setting)
    logger.info(f"Validation MSE: {final_vali_mse:.7f}, Validation MAE: {final_vali_mae:.7f}")
    
    best_model_path = model_path + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    logger.info("Best model saved")
    logger.info("\n")
    
    
## TEST
test_mse, test_mae = compute_metrics(model, test_data, test_loader, device, config, setting, plot=True)
logger.info(f"-"*30)
logger.info(f"Test MSE: {test_mse:.7f}, Test MAE: {test_mae:.7f}")
logging.shutdown()