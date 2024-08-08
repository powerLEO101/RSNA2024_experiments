import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import sys
import timm
import torch
import wandb
import torch.nn as nn
import numpy as np
import albumentations as A
import torch.optim as optim
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import core.utils as utils
import core.models as models
import core.datasets as datasets
import core.training as training
import core.project_paths as project_paths
import core.keypoints as keypoints
import core.losses as losses

from tqdm import tqdm
from os import environ
from accelerate.utils import set_seed
from accelerate import Accelerator
from types import SimpleNamespace
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule
from sklearn.model_selection import KFold

IS_LOCAL = bool('LOCAL_TEST' in environ)
wandb.require('core')

config = {
    'lr': 5e-4,
    'wd': 1e-3,
    'epoch': 10,
    'seed': 22,
    'folds': 1,
    'batch_size': 16 if not 'LOCAL_TEST' in environ else 1,
    'model_name': 'resnet18',
    'out_feature_divide': 2,
    'checkpoint_freq': 5 # no checkpoint in the middle
}
file_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator(mixed_precision='no')
device = accelerator.device

#%% LOSS


#%% MODEL
class RegModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name=model_name,
                                       pretrained=True,
                                       num_classes=10)
    def forward(self, x):
        x = x['img']
        return self.model(x).sigmoid()

#%% DATASET

class PreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.cfg = SimpleNamespace(
            img_dir = f'{project_paths.base_path}/data',
            n_frames = 3,
        )
        self.records = self.load_coords(df)

    def load_coords(self, df):
        # Convert to dict
        d = df.groupby("series_id")[["relative_x", "relative_y"]].apply(lambda x: list(x.itertuples(index=False, name=None)))
        records = {}
        for i, (k,v) in enumerate(d.items()):
            records[i]= {"series_id": k, "label": np.array(v).flatten()}
            assert len(v) == 5
            
        return records
    
    def pad_image(self, img):
        n= img.shape[-1]
        if n >= self.cfg.n_frames:
            start_idx = (n - self.cfg.n_frames) // 2
            return img[:, :, start_idx:start_idx + self.cfg.n_frames]
        else:
            pad_left = (self.cfg.n_frames - n) // 2
            pad_right = self.cfg.n_frames - n - pad_left
            return np.pad(img, ((0,0), (0,0), (pad_left, pad_right)), 'constant', constant_values=0)
    
    def load_img(self, source, series_id):
        fname= os.path.join(self.cfg.img_dir, "processed_{}/{}.npy".format(source, series_id))
        img= np.load(fname).astype(np.float32)
        img= self.pad_image(img)
        img= np.transpose(img, (2, 0, 1))
        img= (img / 255.0)
        return img
        
        
    def __getitem__(self, idx):
        d= self.records[idx]
        label= d["label"]
        source= d["series_id"].split("_")[0]
        series_id= "_".join(d["series_id"].split("_")[1:])     
                
        img= self.load_img(source, series_id)
        return {
            'img': torch.from_numpy(img).float(),
            'label': torch.from_numpy(label).float(),
            }
    
    def __len__(self,):
        return len(self.records)

#%% TRAINING
def train_one_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    model.train()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    for step, batch in bar:
        # B C X Y
        optimizer.zero_grad()
        pred_labels = model(batch)
        loss = criterion(pred_labels, batch['label'])
        running_loss += (loss.item() - running_loss) / (step + 1)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        if accelerator.is_local_main_process and not IS_LOCAL:
            wandb.log({
                'lr': lr, 
                'train_step_loss': loss.item(),
                })
        bar.set_postfix_str(f'epoch: {epoch}, lr: {lr:.2e}, train_loss: {running_loss: .4e}')
        accelerator.free_memory()

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.log({
            'train_epoch_loss': running_loss,
            })

def valid_one_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    global global_step
    model.eval()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    for step, batch in bar:
        # B C X Y
        with torch.no_grad():
            pred_label = model(batch)
        loss = criterion(pred_label, batch['label'])
        running_loss += (loss.item() - running_loss) / (step + 1)
        bar.set_postfix_str(f'Epoch: {epoch}, valid_loss: {running_loss}')
        accelerator.free_memory()

    accelerator.print(f'Valid loss: {running_loss}')

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.log({
            'valid_epoch_loss': running_loss,
            })

def train_one_fold(train_loader, valid_loader, fold_n):
    model = RegModel(model_name=config['model_name'])
    accelerator.print(f'Training for {file_name} on FOLD #{fold_n}...')
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    #lr_scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader), len(train_loader) * config['epoch'])
    lr_scheduler = get_constant_schedule(optimizer)
    model, optimizer, train_loader, valid_loader, lr_scheduler, criterion = \
        accelerator.prepare(model, optimizer, train_loader, valid_loader, lr_scheduler, criterion)

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
        train_one_epoch(model=model, 
                        loader=train_loader, 
                        criterion=criterion, 
                        optimizer=optimizer, 
                        lr_scheduler=lr_scheduler, 
                        epoch=epoch, 
                        accelerator=accelerator)
        
        valid_one_epoch(model=model, 
                        loader=valid_loader, 
                        criterion=criterion, 
                        optimizer=optimizer, 
                        lr_scheduler=lr_scheduler, 
                        epoch=epoch, 
                        accelerator=accelerator)
        if accelerator.is_local_main_process and not IS_LOCAL:
            wandb.log({f'epoch': epoch})
        if accelerator.is_local_main_process and \
            (epoch + 1) % config['checkpoint_freq'] == 0 and \
            (epoch + 1) != config['epoch']:
            torch.save((model.state_dict()), f'{project_paths.save_path}/checkpoints/{file_name}_{fold_n}_{epoch}.pt')

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.finish()

    return model

def get_loaders(df, fold_n):
    train_df = df[df['source'] != 'spider'].reset_index(drop=True)
    valid_df = df[df['source'] == 'spider'].reset_index(drop=True)

    print(f'Data is split into train: {len(train_df)}, and valid: {len(valid_df)}')
    
    train_set = PreTrainDataset(train_df)
    valid_set = PreTrainDataset(valid_df)
    
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
#%% EXPERIMENT
def main(): 
    set_seed(config['seed'])
    df = pd.read_csv(f'{project_paths.base_path}/coords_pretrain.csv')
    df= df.sort_values(["source", "filename", "level"]).reset_index(drop=True)
    df["filename"] = df["filename"].str.replace(".jpg", ".npy")
    df["series_id"] = df["source"] + "_" + df["filename"].str.split(".").str[0]
    # folds = KFold(n_splits=5, shuffle=True, random_state=23)
    # for fold, (train_index, valid_index) in enumerate(folds.split(df)):
    #     df.loc[valid_index, 'fold'] = int(fold)

    save_weights = []
    for fold_n in range(config['folds']):
        train_loader, valid_loader = get_loaders(df, fold_n)
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