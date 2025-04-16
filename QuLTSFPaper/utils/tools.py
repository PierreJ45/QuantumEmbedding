import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import random
import os
import logging
import re

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, config, logger, printout=True):
    # lr = config["learning_rate * (0.2 ** (epoch // 2))
    if config["lradj"] == 'type1':
        lr_adjust = {epoch: config["learning_rate"] * (0.5 ** ((epoch - 1) // 1))}
    elif config["lradj"] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif config["lradj"] == 'type3':
        lr_adjust = {epoch: config["learning_rate"] if epoch < 3 else config["learning_rate"] * (0.9 ** ((epoch - 3) // 1))}
    elif config["lradj"] == 'constant':
        lr_adjust = {epoch: config["learning_rate"]}
    elif config["lradj"] == '3':
        lr_adjust = {epoch: config["learning_rate"] if epoch < 10 else config["learning_rate"]*0.1}
    elif config["lradj"] == '4':
        lr_adjust = {epoch: config["learning_rate"] if epoch < 15 else config["learning_rate"]*0.1}
    elif config["lradj"] == '5':
        lr_adjust = {epoch: config["learning_rate"] if epoch < 25 else config["learning_rate"]*0.1}
    elif config["lradj"] == '6':
        lr_adjust = {epoch: config["learning_rate"] if epoch < 5 else config["learning_rate"]*0.1}  
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: logger.info('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, logger):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, logger)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, logger)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, logger):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model..')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visualize(data_set, batch_x, true, pred, folder_path, i):
    """
    Processes input, true values, and predictions by applying inverse transformation,
    concatenates the sequences, and visualizes the results.
    """
    # Apply inverse transformation
    input_inversed = data_set.inverse_transform(batch_x[0, :, :].detach().cpu().numpy())
    true_inversed = data_set.inverse_transform(true[0, :, :].numpy())
    pred_inversed = data_set.inverse_transform(pred[0, :, :].numpy())

    # Concatenate last columns
    gt = np.concatenate((input_inversed[:, -1], true_inversed[:, -1]), axis=0)
    pd = np.concatenate((input_inversed[:, -1], pred_inversed[:, -1]), axis=0)

    input_len = len(input_inversed[:, -1])
    name = os.path.join(folder_path, f"{i}.png")

    plt.figure()
    plt.plot(gt, label='GroundTruth', color='blue', linewidth=2, marker='o', markersize=3)  # Blue for Ground Truth
    plt.plot(range(input_len-1, len(pd)), pd[input_len-1:], label='Prediction', color='red', linewidth=2, marker='o', markersize=2)
    plt.axvline(x=input_len, color='grey', linestyle='--', linewidth=2, label='Prediction Start')
    plt.grid()
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()
    
        
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
def setup_logger(setting: str, log_dir: str = "./logs"):

    folder_path = os.path.join(log_dir, setting)
    os.makedirs(folder_path, exist_ok=True)

    # Define log file path
    log_filename = os.path.join(folder_path, "training.log")

    # Configure logging
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    return logging.getLogger()


def setup_device(setting: str):
    # Select device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device



def plot_loss_and_results(log_file, var_train, max_epoch=None):
    with open(log_file, "r") as file:
        log_text = file.read()

    epoch_pattern = re.compile(r"\[EPOCH (\d+)/\d+\]")
    loss_pattern = re.compile(r"Train Loss: ([\d\.]+) \| Vali Loss: ([\d\.]+)")
    mse_mae_pattern = re.compile(r"Validation MSE: ([\d\.]+), Validation MAE: ([\d\.]+)")

    epochs = {}
    mse_values = []
    mae_values = []

    current_epoch = None

    for line in log_text.split("\n"):
        epoch_match = epoch_pattern.search(line)
        loss_match = loss_pattern.search(line)
        mse_mae_match = mse_mae_pattern.search(line)

        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            if current_epoch not in epochs:
                epochs[current_epoch] = {"train": [], "vali": []}

        if loss_match and current_epoch is not None:
            train_loss, vali_loss = map(float, loss_match.groups())
            epochs[current_epoch]["train"].append(train_loss)
            epochs[current_epoch]["vali"].append(vali_loss)

        if mse_mae_match:
            mse, mae = map(float, mse_mae_match.groups())
            mse_values.append(mse)
            mae_values.append(mae)

    # Compute average losses per epoch
    epoch_numbers = sorted(epochs.keys())[:max_epoch]
    avg_train_loss = [np.mean(epochs[ep]["train"]) for ep in epoch_numbers][:max_epoch]
    avg_vali_loss = [np.mean(epochs[ep]["vali"]) for ep in epoch_numbers][:max_epoch]
    print(avg_train_loss[-1], avg_vali_loss[-1])

    # Compute average MSE and MAE
    avg_mse = np.mean(mse_values)
    avg_mae = np.mean(mae_values)

    # Print the average MSE and MAE
    print(f"Average Validation MSE: {avg_mse:.6f}")
    print(f"Normalized Average Validation MSE: {avg_mse/var_train:.6f}")
    print(f"Average Validation MAE: {avg_mae:.6f}")
    print(f"Normalized Average Validation MAE: {avg_mae/var_train:.6f}")

    # Plot the results
    plt.figure(figsize=(5, 3))
    plt.plot(epoch_numbers, avg_train_loss, label="Avg Train Loss", linestyle='-')
    plt.plot(epoch_numbers, avg_vali_loss, label="Avg Validation Loss", linestyle='-')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Average Train and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()