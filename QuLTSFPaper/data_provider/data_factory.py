from data_provider.data_loader import DatasetStandard, DatasetWithOverlap
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset


class SubsetCV(Subset):
    def __getattr__(self, attr):
        return getattr(self.dataset, attr)
    

def data_provider(config, flag):
    timeenc = 0 if config["time_embed"] != 'timeF' else 1

    drop_last = True
    batch_size = config["batch_size"]
    freq = config["freq"]
    Data = DatasetStandard
    
    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = True

    data_set = Data(
        data_path=config["data_path"],
        flag=flag,
        size=[config["seq_len"], config["pred_len"]],
        forecasting_task=config["forecasting_task"],
        target=config["target"],
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=10,
        drop_last=drop_last)
    
    return data_set, data_loader



def data_provider_cv(config, flag):
    timeenc = 0 if config["time_embed"] != 'timeF' else 1

    drop_last = True
    batch_size = config["batch_size"]
    freq = config["freq"]
    k_folds = config["k_folds"]
    Data = DatasetStandard
    
    data_set = Data(
            data_path=config["data_path"],
            flag=flag,
            size=[config["seq_len"], config["pred_len"]],
            forecasting_task=config["forecasting_task"],
            target=config["target"],
            timeenc=timeenc,
            freq=freq,
            split_ratio=config["split_ratio"]
        )

    if flag == 'test':
        print(flag, len(data_set))
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=10,
            drop_last=drop_last)
        
        return data_set, data_loader
    
    else:
        print(flag, len(data_set), "-", k_folds, "folds")
        
        kf = KFold(n_splits=k_folds, shuffle=False)
        train_loaders, train_sets = [], []
        val_loaders, val_sets = [], []
        for fold, (train_idx, val_idx) in enumerate(kf.split(data_set)):

            train_fold = SubsetCV(data_set, train_idx)
            val_fold = SubsetCV(data_set, val_idx)
            
            train_sets.append(train_fold)
            val_sets.append(val_fold)

            train_loaders.append(DataLoader(train_fold, batch_size=config["batch_size"], shuffle=True))
            val_loaders.append(DataLoader(val_fold, batch_size=config["batch_size"], shuffle=False))
        
        return train_sets, train_loaders, val_sets, val_loaders
    
