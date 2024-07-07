import os
import sys
import torch
import wandb
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import core.utils as utils
import core.models as models
import core.datasets as datasets
import core.training as training
import core.project_paths as project_paths
import core.losses as losses

from os import environ
from accelerate.utils import set_seed
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

IS_LOCAL = bool('LOCAL_TEST' in environ)

config = {
    'lr': 1e-3,
    'wd': 1e-3,
    'epoch': 15,
    'seed': 22,
    'folds': 2,
    'batch_size': 8 if not 'LOCAL_TEST' in environ else 1,
    'model_name': 'timm/efficientnet_b0.ra_in1k',
    'checkpoint_freq': 100 # no checkpoint in the middle
}
file_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator()

def train_one_fold(train_loader, valid_loader, fold_n):
    model = models.ThreeViewModel(model_name=config['model_name'])
    accelerator.print(f'Training for {file_name} on FOLD #{fold_n}...')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader), len(train_loader) * config['epoch'])
    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, valid_loader, lr_scheduler)

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.init(
            project = 'RSNA2024',
            name = f'{file_name}_fold{fold_n}',
            config = {
                **config,
                'fold': fold_n
            },
            group = file_name
        )

    for epoch in range(config['epoch']):
        training.train_one_epoch(model=model, 
                                 loader=train_loader, 
                                 criterion=criterion, 
                                 optimizer=optimizer, 
                                 lr_scheduler=lr_scheduler, 
                                 epoch=epoch, 
                                 accelerator=accelerator)
        
        training.valid_one_epoch(model=model, 
                                 loader=valid_loader, 
                                 criterion=criterion, 
                                 optimizer=optimizer, 
                                 lr_scheduler=lr_scheduler, 
                                 epoch=epoch, 
                                 accelerator=accelerator)
        if accelerator.is_local_main_process and not IS_LOCAL:
            wandb.log({f'epoch': epoch})
        if accelerator.is_local_main_process and (epoch + 1) % config['checkpoint_freq'] == 0:
            torch.save((model.state_dict()), f'./{file_name}_{fold_n}_{epoch}.pt')

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.finish()

    return model

def get_loaders(df, data, fold_n):
    if fold_n is None:
        train_df = df.copy()
        valid_df = df[df['fold'] == 0].copy()
    else:
        train_df = df[df['fold'] != fold_n].copy()
        valid_df = df[df['fold'] == fold_n].copy()
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f'Data is split into train: {len(train_df)}, and valid: {len(valid_df)}')
    
    train_set = datasets.ThreeViewDataset(train_df, data)
    valid_set = datasets.ThreeViewDataset(valid_df, data)
    train_loader =  DataLoader(train_set, 
                                batch_size=config['batch_size'], 
                                shuffle=True, 
                                num_workers=12 if not 'LOCAL_TEST' in environ else 4, 
                                pin_memory=False)
    valid_loader = DataLoader(valid_set, 
                                batch_size=config['batch_size'], 
                                shuffle=False, 
                                num_workers=12 if not 'LOCAL_TEST' in environ else 4, 
                                pin_memory=False)

    return train_loader, valid_loader

def main(): 
    set_seed(config['seed'])
    df = datasets.get_df()
    data = datasets.get_data(df, drop_rate=0.2)
    save_weights = []
    for fold_n in range(config['folds']):
        train_loader, valid_loader = get_loaders(df, data, fold_n)
        model = train_one_fold(train_loader, valid_loader, fold_n)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            torch.save((model.cpu().state_dict()), f'./{fold_n}.pt')

        accelerator.free_memory()
        model = None

    if accelerator.is_local_main_process:
        data = None
        for fold_n in range(config['folds']):
            save_weights.append(torch.load(f'./{fold_n}.pt'))
            os.remove(f'./{fold_n}.pt')
        torch.save(save_weights, f'{project_paths.save_path}/{file_name}.pt')

if __name__ == "__main__":
    main()