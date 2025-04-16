import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.tools import visualize
import os



def compute_metrics(model, data_set, data_loader, device, config, setting, plot=False):
    model.eval() 
    preds = []
    trues = []

    with torch.no_grad():
        for i, (batch_x, batch_y, _, _) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)

            f_dim = -1 if config["forecasting_task"] == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]

            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())
            
            if plot==True:
                folder_path = './results_test/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    
                if i % 5 == 0:
                    visualize(data_set, batch_x, batch_y.detach().cpu(), outputs.detach().cpu(), folder_path, i)
                

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mse = mean_squared_error(trues.flatten(), preds.flatten())
    mae = mean_absolute_error(trues.flatten(), preds.flatten())

    return mse, mae