import os
import sys
import timm
import torch
import wandb
import torch.nn as nn
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
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

IS_LOCAL = bool('LOCAL_TEST' in environ)
wandb.require('core')

config = {
    'lr': 1e-3,
    'wd': 1e-3,
    'epoch': 10,
    'seed': 22,
    'folds': 1,
    'batch_size': 128 if not 'LOCAL_TEST' in environ else 1,
    'model_name': 'timm/efficientnet_b0.ra_in1k',
    'out_feature_divide': 2,
    'checkpoint_freq': 5 # no checkpoint in the middle
}
file_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator()
device = accelerator.device

#%% LOSS
class PerLevelMSELoss(nn.Module):
    def __init__(self, normalize_label=True, max_label=256):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.normalize_label = normalize_label
        self.max_label = max_label
    
    def forward(self, pred, label, have_label):
        real_pred, real_label = [], []
        for p, l, h in zip(pred, label, have_label):
            real_pred.append(p[h])
            real_label.append(l[h])
        real_pred = torch.cat(real_pred, dim=0)
        real_label = torch.cat(real_label, dim=0)
        if self.normalize_label:
            real_label = real_label / self.max_label
        return self.loss(real_pred, real_label)

class PerLevelCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.tensor([1., 2., 4.], device=device)
        self.loss = nn.BCEWithLogitsLoss(weight=weight)
    
    def forward(self, pred, label, have_label):
        real_pred, real_label = [], []
        for p, l, h in zip(pred, label, have_label):
            real_pred.append(p[h])
            real_label.append(l[h])
        real_pred = torch.cat(real_pred, dim=0)
        real_label = torch.cat(real_label, dim=0) # batch_size * have_label_size, 3
        return self.loss(real_pred, real_label)

class PerLevelTwoWayLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = PerLevelMSELoss()
        self.cls_loss = PerLevelCrossEntropyLoss()
    
    def forward(self, pred, label, have_label):
        loss1 = self.mse_loss(pred[0], label[0], have_label)
        loss2 = self.cls_loss(pred[1], label[1], have_label)
        return loss1 + loss2, loss1, loss2

#%% MODEL
class RSNAModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        base_model = timm.create_model(model_name=model_name,
                                       pretrained=True)
        try:
            in_features = base_model.fc.in_features
        except:
            in_features = base_model.classifier.in_features
        
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(in_features, 30)
        self.reg_head = nn.Linear(in_features, 20)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        x = self.global_pool(x).view(batch_size, -1)
        reg = self.reg_head(x).view(batch_size, 10, 2)
        cls = self.cls_head(x).view(batch_size, 10, 3)
        return reg, cls

#%% DATASET
class SegmentDataset(Dataset):
    def __init__(self, 
                 df,
                 df_co, 
                 data,
                 augment_level=0,
                 rough_pos_factor=1,
                 image_size=[256, 256],
                 length=25):

        super().__init__()
        self.augment_level = augment_level
        self.rough_pos_factor = rough_pos_factor
        self.df = df
        self.df_co = df_co
        self.data = data
        self.image_size = image_size
        self.length = length

        if augment_level == 0:
            self.augment = A.ReplayCompose([
                A.Resize(*self.image_size),
                ToTensorV2()],
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        elif self.augment_level == 1:
            self.augment = A.ReplayCompose([ # remove vertical flip because model cannot practically tell up/down
                A.Resize(*self.image_size),
                A.Perspective(p=0.5),
                A.HorizontalFlip(p=0.5),
                #A.VerticalFlip(p=0.5),
                A.Rotate(p=0.5, limit=(-25, 25)),
                ToTensorV2()], 
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.df_co)

    def __getitem__(self, idx):
        meta = dict(self.df_co.iloc[idx])

        image = self._get_slice(meta)
        result = self.augment(image=image.numpy().astype('f'), keypoints=meta['keypoints'])
        image = result['image']
        image = torch.cat([image] * 3, dim=0)
        keypoints = torch.tensor([[int(x[0] // self.rough_pos_factor), int(x[1] // self.rough_pos_factor)] \
                                  for x in result['keypoints']])
        # masks = []
        # for y, x in keypoints: # positions are inverted in numpy axis
        #     masks.append(mask_from_keypoint(x, y, self.length, *self.image_size))

        return {
            'image': image,
            'loc': keypoints,
            'label': torch.tensor(meta['label'], dtype=torch.float32),
            #'masks': masks,
            'have_keypoints': torch.tensor(meta['have_keypoints'])
        }
    
    def _get_slice(self, meta):
        volumes, desc, s_ids = self._get_data_ram_or_disk(meta)
        volume = volumes[s_ids.index(str(meta['series_id']))]
        return volume[int(meta['instance_number'])].unsqueeze(-1)
    
    def _get_data_ram_or_disk(self, meta):
        if isinstance(self.data[meta['study_id']], str):
            return datasets.dicom_to_3d_tensors(self.data[meta['study_id']])
        else:
            return self.data[meta['study_id']]

#%% TRAINING
def train_one_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    running_mse_loss = 0.0
    running_cls_loss = 0.0
    model.train()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    for step, batch in bar:
        # B C X Y
        image = batch['image']
        label = [batch['loc'], batch['label']]
        have_label = batch['have_keypoints']
        optimizer.zero_grad()
        pred_labels = model(image)
        loss, mse_loss, cls_loss = criterion(pred_labels, label, have_label)
        running_loss += (loss.item() - running_loss) / (step + 1)
        running_mse_loss += (mse_loss.item() - running_mse_loss) / (step + 1)
        running_cls_loss += (cls_loss.item() - running_cls_loss) / (step + 1)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        if accelerator.is_local_main_process and not IS_LOCAL:
            wandb.log({
                'lr': lr, 
                'train_step_loss': loss.item(),
                'train_step_mse_loss': mse_loss.item(),
                'train_step_cls_loss': cls_loss.item()
                })
        bar.set_postfix_str(f'epoch: {epoch}, lr: {lr:.2e}, train_loss: {running_loss: .4e}')
        accelerator.free_memory()

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.log({
            'train_epoch_loss': running_loss,
            'train_epoch_mse_loss': running_mse_loss,
            'train_epoch_cls_loss': running_cls_loss
            })

def valid_one_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    running_mse_loss = 0.0
    running_cls_loss = 0.0
    global global_step
    model.eval()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    for step, batch in bar:
        # B C X Y
        image = batch['image']
        label = [batch['loc'], batch['label']]
        have_label = batch['have_keypoints']
        with torch.no_grad():
            pred_label = model(image)
        loss, mse_loss, cls_loss = criterion(pred_label, label, have_label)
        running_loss += (loss.item() - running_loss) / (step + 1)
        running_mse_loss += (mse_loss.item() - running_mse_loss) / (step + 1)
        running_cls_loss += (cls_loss.item() - running_cls_loss) / (step + 1)
        bar.set_postfix_str(f'Epoch: {epoch}, valid_loss: {running_loss}')
        accelerator.free_memory()

    accelerator.print(f'Valid loss: {running_loss}')
    accelerator.print(f'Valid mse loss: {running_mse_loss}')
    accelerator.print(f'Valid cls loss: {running_cls_loss}')

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.log({
            'valid_epoch_loss': running_loss,
            'valid_epoch_mse_loss': running_mse_loss,
            'valid_epoch_cls_loss': running_cls_loss
            })

def train_one_fold(train_loader, valid_loader, fold_n):
    model = RSNAModel(model_name=config['model_name'])
    accelerator.print(f'Training for {file_name} on FOLD #{fold_n}...')
    criterion = PerLevelTwoWayLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader), len(train_loader) * config['epoch'])
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

def get_loaders(df, df_co, data, fold_n):
    if fold_n is None:
        train_df = df_co.copy()
        valid_df = df_co[df_co['fold'] == 0].copy()
    else:
        train_df = df_co[df_co['fold'] != fold_n].copy()
        valid_df = df_co[df_co['fold'] == fold_n].copy()
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f'Data is split into train: {len(train_df)}, and valid: {len(valid_df)}')
    
    train_set = SegmentDataset(df, train_df, data, augment_level=1, rough_pos_factor=1)
    valid_set = SegmentDataset(df, valid_df, data, augment_level=0, rough_pos_factor=1)
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
    df = datasets.get_df()
    df_co = keypoints.df_label_co
    df_co = df_co[df_co['study_id'].isin(df['study_id'])].reset_index(drop=True)
    df_co = keypoints.get_df_co(df_co, df)
    df_co.to_csv(f'./{file_name}_df_co.csv', index=False)
    data = datasets.get_data(df, drop_rate=0.2)
    save_weights = []
    for fold_n in range(config['folds']):
        train_loader, valid_loader = get_loaders(df, df_co, data, fold_n)
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