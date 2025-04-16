import torch
from utils.tools import visualize
import numpy as np
import os



def validate(vali_data, vali_loader, model, criterion, device,  setting, config, fold):
        
        folder_path = './results_validation/' + setting + '/' + 'fold' + str(fold) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        total_loss = []
        
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(vali_loader):
                
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                outputs = model(batch_x)

                f_dim = -1 if config["forecasting_task"] == 'MS' else 0
                outputs = outputs[:,:, f_dim:]
                batch_y = batch_y[:,:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                if i % 5 == 0:
                    visualize(vali_data, batch_x, true, pred, folder_path, i)
                    
                loss = criterion(pred, true)
                total_loss.append(loss)
                
        model.train()
        return np.average(total_loss)