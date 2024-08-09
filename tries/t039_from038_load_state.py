import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import sys
import cv2
import timm
import torch
import wandb
import pydicom
import torch.nn as nn
import numpy as np
import pandas as pd
import albumentations as A
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import core.utils as utils
import core.models as models
import core.datasets as datasets
import core.training as training
import core.project_paths as project_paths
import core.keypoints as keypoints
import core.losses as losses
from core.fold_for_all import fold_for_all

from glob import glob
from tqdm import tqdm
from os import environ
from einops import rearrange
from collections import defaultdict
from accelerate.utils import set_seed
from accelerate import Accelerator
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

IS_LOCAL = bool('LOCAL_TEST' in environ)
wandb.require('core')

config = {
    'lr': 1e-4,
    'wd': 1e-3,
    'epoch': 15,
    'seed': 22,
    'folds': 5,
    'batch_size': 32 if not 'LOCAL_TEST' in environ else 1,
    'model_name': 'resnet18',
    'grad_acc': 4,
    'checkpoint_freq': 5,
}
file_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator()
device = accelerator.device

#%% LOSS

class PerLevelCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight) if weight is not None else nn.CrossEntropyLoss()
    
    def forward(self, pred, label):
        label = label['label']

        pred = pred.view(-1, 3)
        label = label.view(-1, 3)
        return self.loss(pred, label), {}

#%% MODEL

class RegModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name=model_name,
                                       pretrained=False,
                                       num_classes=10)
    def forward(self, x):
        return self.model(x).sigmoid()

class RSNAModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name=model_name,
                                       pretrained=True,
                                       in_chans=8,
                                       num_classes=15)
    def forward(self, x):
        x = x['image']
        return self.model(x)

#%% DATASET
class SegmentDataset(Dataset):
    def __init__(self, 
                 df,
                 df_series,
                 df_co, 
                 data,
                 sagittal_keypoints,
                 augment_level=0,
                 image_size=[256, 256]):

        super().__init__()
        self.augment_level = augment_level
        self.df = df
        self.df_series = df_series
        self.df_co = df_co
        self.data = data
        self.image_size = image_size
        self.sagittal_t2_keypoints = sagittal_keypoints

        if self.augment_level == 0:
            self.augment = A.Compose([
                A.Resize(*self.image_size)])
                #keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        elif self.augment_level == 1:
            self.augment = A.Compose([
                A.Resize(*self.image_size),
                A.Perspective(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(p=0.5, limit=(-25, 25))])
                #keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.df_series)

    def __getitem__(self, idx):
        meta = dict(self.df_series.iloc[idx])

        midpoint = int(meta['total_instance_number'] // 2)
        chosen_index = midpoint + np.random.randint(-2, 3)
        image = self._get_slice(meta, chosen_index).float()

        masks = self._get_keypoints_pred(meta['series_id'], chosen_index)
        image = torch.from_numpy(self.augment(image=image.numpy())['image'])
        image = torch.stack([image] * 3, dim=0)
        final_image = torch.zeros(8, *self.image_size)
        final_image[:3, ...] = image
        final_image[3:, ...] = masks
        label = self._query_spinal_label(meta['study_id'])

        return {
            'image': final_image,
            'label': label
        }
    
    def _get_slice(self, meta, instance_number):
        volume = self._get_data_ram_or_disk(meta)
        v_min, v_max = volume.min(), volume.max()
        return (volume[int(instance_number)] - v_min) / (v_max - v_min + 0.00001)
    
    def _get_data_ram_or_disk(self, meta):
        if isinstance(self.data[meta['series_id']], str):
            return datasets.get_dicom_volume(self.data[meta['series_id']])
        else:
            return self.data[meta['series_id']]
    
    def _query_spinal_label(self, study_id):
        shortlist = self.df[self.df['study_id'] == study_id].iloc[0]
        conditions = ['spinal_canal_stenosis']
        levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1'] 
        result = []
        for condition in conditions:
            for level in levels:
                tmp = [0] * 3
                tmp[shortlist[f'{condition}_{level}']] = 1
                result.append(tmp)
        result = torch.tensor(result, dtype=torch.float32)
        return result
    
    def _get_keypoints_pred(self, series_id, instance_number):
        pred = self.sagittal_t2_keypoints[(series_id, instance_number)]
        result = torch.zeros(5, *self.image_size)
        for i in range(5):
            image_len = self.image_size[0]
            posx, posy = int(pred[i][1] * image_len), int(pred[i][0] * image_len)
            result[i, ...] = losses.mask_from_keypoint(posx, posy, 10, *self.image_size)
        return result

#%% TRAINING
def train_one_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    running_losses = defaultdict(float)
    model.train()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    for step, batch in bar:
        # B C X Y
        optimizer.zero_grad()
        pred_labels = model(batch)
        loss, losses = criterion(pred_labels, batch)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        if accelerator.is_local_main_process and not IS_LOCAL:
            wandb.log({
                'lr': lr, 
                'train_step_loss': loss.item(),
                **losses
                })
        running_loss += (loss.item() - running_loss) / (step + 1)
        for key in losses.keys():
            running_losses[key] += (losses[key] - running_losses[key]) / (step + 1)

        bar.set_postfix_str(f'epoch: {epoch}, lr: {lr:.2e}, train_loss: {running_loss: .4e}')
        accelerator.free_memory()

    if accelerator.is_local_main_process and not IS_LOCAL:
        train_running_losses = {f'train_epoch_{k}': running_losses[k] for k in running_losses.keys()}
        wandb.log({
            'train_epoch_loss': running_loss,
            **train_running_losses
            })

def valid_one_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    running_losses = defaultdict(float)
    global global_step
    model.eval()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    for step, batch in bar:
        # B C X Y
        with torch.no_grad():
            pred_label = model(batch)
        loss, losses = criterion(pred_label, batch)
        running_loss += (loss.item() - running_loss) / (step + 1)
        for key in losses.keys():
            running_losses[key] += (losses[key] - running_losses[key]) / (step + 1)
        bar.set_postfix_str(f'Epoch: {epoch}, valid_loss: {running_loss}')
        accelerator.free_memory()

    accelerator.print(f'Valid loss: {running_loss}')
    accelerator.print(running_losses)

    if accelerator.is_local_main_process and not IS_LOCAL:
        valid_running_losses = {f'valid_epoch_{k}': running_losses[k] for k in running_losses.keys()}
        wandb.log({
            'valid_epoch_loss': running_loss,
            **valid_running_losses
        })

def train_one_fold(train_loader, valid_loader, fold_n):
    model = RSNAModel(model_name=config['model_name'])
    model_dict = '/Users/leo101/Downloads/t035_from034_pretrain_coor.pt' if IS_LOCAL \
        else '/root/autodl-fs/t035_from034_pretrain_coor.pt'
    model_dict = torch.load(model_dict)[0]
    del model_dict['model.fc.weight']
    del model_dict['model.fc.bias']
    del model_dict['model.conv1.weight']
    model.load_state_dict(model_dict, strict=False)
    accelerator.print(f'Training for {file_name} on FOLD #{fold_n}...')
    criterion = PerLevelCrossEntropyLoss(torch.tensor([1., 2., 4.], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader), len(train_loader) * config['epoch'], 0.3)
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

def get_loaders(df, df_series, df_co, data, sagittal_keypoints, fold_n):
    if fold_n is None:
        train_df = df_series.copy()
        valid_df = df_series[df_series['fold'] == 0].copy()
    else:
        train_df = df_series[df_series['fold'] != fold_n].copy()
        valid_df = df_series[df_series['fold'] == fold_n].copy()
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f'Data is split into train: {len(train_df)}, and valid: {len(valid_df)}')
    
    train_set = SegmentDataset(df, train_df, df_co, data, sagittal_keypoints, augment_level=0)
    valid_set = SegmentDataset(df, valid_df, df_co, data, sagittal_keypoints, augment_level=0)

    # weights = []
    # weight_multiplier = [1., 2., 4.]
    # for element in train_set:
    #     weights.append(weight_multiplier[element['label'].tolist().index(1)])
    # weighted_sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_set))

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

def get_df_co(df_co, df):

    def query_condition(condition, l, study_id, df):
        condition_full = '_'.join([condition.lower().replace(' ', '_'), l.lower().replace('/', '_')])
        shortlist = df[df['study_id'] == study_id].iloc[0]
        result = [0] * 3
        result[shortlist[condition_full]] = 1
        return result
    # only sagittal view
    only_sagittal = []
    for s, id in df_co[['study_id', 'series_id']].values:
        d = datasets.find_description(s, id)
        if 'Axial' in d:
            only_sagittal.append(False)
        else:
            only_sagittal.append(True)
    df_co_only_sag = df_co[only_sagittal].reset_index(drop=True)

    new_df_co = []
    levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    for series_id, sub_df1 in df_co_only_sag.groupby('series_id'):
        for instance_number, sub_df2 in sub_df1.groupby('instance_number'):
            if len(sub_df2['condition'].unique()) != 1:
                continue
            sub_df2 = sub_df2.reset_index(drop=True)
            condition = sub_df2.iloc[0]['condition']
            coors = [[0, 0]] * 10
            have_keypoints = [False] * 10
            label = [[0., 0., 0.]] * 10

            if 'Spinal' in condition:
                for idx, l in enumerate(levels):
                    tmp = sub_df2[sub_df2['level'] == l].reset_index(drop=True)
                    if len(tmp) == 0:
                        continue
                    tmp = tmp.iloc[0]
                    label[idx] = query_condition(condition, l, tmp['study_id'], df)
                    coors[idx] = [int(tmp['x']), int(tmp['y'])]
                    have_keypoints[idx] = True
            if 'Left Neural' in condition:
                for idx, l in enumerate(levels):
                    tmp = sub_df2[sub_df2['level'] == l].reset_index(drop=True)
                    if len(tmp) == 0:
                        continue
                    tmp = tmp.iloc[0]
                    label[idx + 5] = query_condition(condition, l, tmp['study_id'], df)
                    coors[idx + 5] = [int(tmp['x']), int(tmp['y'])]
                    have_keypoints[idx + 5] = True
            elif 'Right Neural' in condition: # some slices have both left and right neural, it is impossible and will cause error, hence elif
                for idx, l in enumerate(levels):
                    tmp = sub_df2[sub_df2['level'] == l].reset_index(drop=True)
                    if len(tmp) == 0:
                        continue
                    tmp = tmp.iloc[0]
                    label[idx + 5] = query_condition(condition, l, tmp['study_id'], df)
                    coors[idx + 5] = [int(tmp['x']), int(tmp['y'])] # coordinates keeps the same
                    have_keypoints[idx + 5] = True
            
            study_id = sub_df2.iloc[0]['study_id']

            new_df_co.append([study_id, series_id, instance_number, label, coors, have_keypoints])

    new_df_co = pd.DataFrame(new_df_co, columns=['study_id', 'series_id', 'instance_number', 'label', 'keypoints', 'have_keypoints'])
    new_df_co['fold'] = new_df_co['study_id'].apply(fold_for_all.get)
    return new_df_co

def get_df_series():
    def get_dicom_metadata(filename):
        dcm = pydicom.dcmread(filename)
        result = {}
        for element in dcm:
            if element.name == 'Pixel Data': continue
            result[element.name] = element.value
        return result
    def get_neural_direction(folder_path):
        files = glob(f'{folder_path}/*.dcm')
        if len(files) == 0:
            return False
        files.sort(key=lambda x: int(x[:-4].split('/')[-1]))
        x_position_first = get_dicom_metadata(files[0])['Image Position (Patient)'][0]
        x_position_final = get_dicom_metadata(files[-1])['Image Position (Patient)'][0]
        return x_position_first < x_position_final
    df_series = pd.read_csv(f'{project_paths.base_path}/train_series_descriptions.csv')
    df_series = df_series[df_series['series_description'].str.contains('Sagittal')].reset_index(drop=True)
    def count_files(folder):
        files = glob(f'{folder}/*.dcm')
        return len(files)
    df_series['filepath'] = df_series.apply(lambda x: f"{project_paths.base_path}/train_images/{x['study_id']}/{x['series_id']}", axis=1)
    df_series['total_instance_number'] = df_series['filepath'].apply(count_files)
    df_series['fold'] = df_series['study_id'].apply(fold_for_all.get)
    df_series['reverse'] = df_series['filepath'].apply(get_neural_direction)
    return df_series

def main(): 
    set_seed(config['seed'])
    df = datasets.get_df()
    df_series = get_df_series()
    df_series = df_series[df_series['study_id'].isin(df['study_id'])].reset_index(drop=True)
    df_series = df_series[df_series['series_description'] == 'Sagittal T2/STIR'].reset_index(drop=True)
    df_co = keypoints.df_label_co.copy()
    df_co = df_co[df_co['study_id'].isin(df['study_id'])].reset_index(drop=True)
    df_co = get_df_co(df_co, df)
    sagittal_keypoints = torch.load('/Users/leo101/Downloads/sagittal_t2_keypoints.pt') if IS_LOCAL \
        else torch.load(f'{project_paths.base_path}/sagittal_t2_keypoints.pt')
    data = datasets.get_data_w_series(df_series, drop_rate=0.2)

    save_weights = []
    for fold_n in range(config['folds']):
        train_loader, valid_loader = get_loaders(df, df_series, df_co, data, sagittal_keypoints, fold_n)
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