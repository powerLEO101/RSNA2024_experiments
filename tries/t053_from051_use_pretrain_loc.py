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
    'epoch': 10,
    'seed': 22,
    'folds': 5,
    'batch_size': 8 if not 'LOCAL_TEST' in environ else 1,
    'model_name': 'efficientnet_b0.ra_in1k',
    'grad_acc': 4,
    'checkpoint_freq': 5,
}
file_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator(split_batches=True, dispatch_batches=True)
device = accelerator.device

#%% LOSS

class PerLevelCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pred_, label_):
        losses = []
        for i in range(3):
            pred = pred_[:, 15 * i : 15 * (i + 1)].contiguous()
            label = label_[:, 15 * i : 15 * (i + 1)].contiguous()
            pred = pred.view(-1, 3)
            label = label.view(-1, 3)
            label_weights = torch.tensor([2 ** x.tolist().index(1) for x in label], device=device) ## 2 ** index == (1, 2, 4)
            loss = self.loss(pred, label)
            loss *= label_weights
            loss = loss.mean()
            losses.append(loss)
        
        return sum(losses) / 3, {'spinal_loss': losses[0], 'neural_loss': (losses[1] + losses[2]) / 2}

class ImportanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, label):
        pred = pred.view(-1, 2)
        label = label.view(-1, 2)
        return self.loss(pred, label)

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = PerLevelCrossEntropyLoss()
        self.importance_loss = ImportanceLoss()
    
    def forward(self, pred, label):
        cls_label = label['label']
        importance_label = label['importance']
        cls_loss, loss_dict = self.cls_loss(pred[0], cls_label)
        importance_loss = self.importance_loss(pred[1], importance_label)
        return cls_loss + importance_loss, {'cls_loss': cls_loss, 'importance_loss': importance_loss, **loss_dict}


#%% MODEL

class RSNAModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        base_model = timm.create_model(model_name=model_name,
                                       pretrained=True,
                                       in_chans=3)
        try:
            in_features = base_model.fc.in_features
        except:
            in_features = base_model.classifier.in_features
        
        layers = list(base_model.children())[:-1]
        self.encoder = nn.Sequential(*layers)
        self.importance_head = nn.Linear(in_features, 2)
        self.spinal_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
        )
        self.neural_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
        )
    def forward(self, input):
        x = input['image']

        B, N, C, X, Y = x.shape
        x = rearrange(x, 'b n c x y -> (b n) c x y')
        x = self.encoder(x)
        importance_logit = self.importance_head(x)
        importance_logit = rearrange(importance_logit, '(b n) f -> b n f', b=B)
        importance = importance_logit.sigmoid()

        x = rearrange(x, '(b n) f -> b n f', b=B)
        x_spinal = x * importance[:, :, 0 : 1]
        x_spinal = x_spinal.mean(1)
        x_neural = x * importance[:, :, 1 : 2]
        x_neural_left = x_neural[:, : 25, :]
        x_neural_left = x_neural_left.mean(1)
        x_neural_right = x_neural[:,25 :, :]
        x_neural_right = x_neural_right.mean(1)

        x_spinal = self.spinal_head(x_spinal)
        x_neural_left = self.neural_head(x_neural_left)
        x_neural_right = self.neural_head(x_neural_right)
        x_all = torch.cat([x_spinal, x_neural_left, x_neural_right], dim=1)
        return x_all, importance_logit

#%% DATASET

class SegmentDataset(Dataset):
    def __init__(self, 
                 df,
                 df_series,
                 df_co, 
                 data,
                 importance_decay=0.4,
                 augment_level=0,
                 image_size=[256, 256]):

        super().__init__()
        self.augment_level = augment_level
        self.df = df
        self.df_series = list(df_series.groupby('study_id'))
        self.df_co = df_co
        self.data = data
        self.image_size = image_size
        self.importance_decay = importance_decay

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
        study_id, metas = self.df_series[idx]

        views = ['Sagittal T2/STIR', 'Sagittal T1']

        final_images = []
        is_important = []
        slice_pos = []
        for view in views:
            meta = metas[metas['series_description'] == view].iloc[0]
            if len(meta) == 0:
                continue
            for i in range(meta['total_instance_number']):
                image, slice_position = self._get_slice(meta, i)
                image, slice_position = image.float(), float(slice_position)
                image = torch.from_numpy(self.augment(image=image.numpy())['image'])
                image = torch.stack([image] * 3, dim=0)
                final_images.append(image)

                conditions = ['Spinal Canal Stenosis', 'Left Neural Foraminal Narrowing', 'Right Neural Foraminal Narrowing']
                is_important.append([self._is_important(meta['series_id'], conditions[0], i), \
                                     max(self._is_important(meta['series_id'], conditions[1], i), \
                                         self._is_important(meta['series_id'], conditions[2], i))])

                slice_pos.append(slice_position)

        packed = zip(final_images, is_important, slice_pos) # sorting three objects together based on slice_pos
        packed = sorted(packed, key=lambda x : x[-1], reverse=True) # the higher the slice_pos, the more left
        final_images, is_important, slice_pos = zip(*packed) # unpacking

        final_images = torch.stack(final_images, dim=0)
        is_important = torch.tensor(is_important, dtype=torch.float32)
        for i in range(2):
            is_important[:, i] = self._process_importance(is_important[:, i])
        final_images_pad = []
        is_important_pad = []
        for i in np.arange(0, len(final_images), len(final_images) / 50)[:50]:
            index = int(i)
            final_images_pad.append(final_images[index])
            is_important_pad.append(is_important[index])
        final_images_pad = torch.stack(final_images_pad, dim=0)
        is_important_pad = torch.stack(is_important_pad, dim=0)
        
        label = self._query_label(study_id)

        return {
            'image': final_images_pad,
            'label': label,
            'importance': is_important_pad,
            'study_id': study_id
        }
    
    def _get_slice(self, meta, instance_number):
        volume, pos = self._get_data_ram_or_disk(meta)
        v_min, v_max = volume.min(), volume.max()
        return (volume[int(instance_number)] - v_min) / (v_max - v_min + 0.00001), pos[instance_number]
    
    def _get_data_ram_or_disk(self, meta):
        if isinstance(self.data[meta['series_id']], str):
            return datasets.get_dicom_volume(self.data[meta['series_id']])
        else:
            return self.data[meta['series_id']]
    
    def _query_label(self, study_id):
        shortlist = self.df[self.df['study_id'] == study_id].iloc[0]
        conditions = ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing']
        levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1'] 
        result = []
        for condition in conditions:
            for level in levels:
                tmp = [0.] * 3
                tmp[shortlist[f'{condition}_{level}']] = 1.
                result.extend(tmp)
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

    def _find_distance(self, series_id, condition, id):
        shortlist = self.df_co[(self.df_co['series_id'] == series_id) & (self.df_co['condition'] == condition)].reset_index(drop=True)
        result = 100
        for i in shortlist['instance_number'].values:
            result = min(result, abs(i - id))
        return result
    
    def _is_important(self, series_id, condition, id):
        shortlist = self.df_co[(self.df_co['series_id'] == series_id) & (self.df_co['condition'] == condition)]
        if len(shortlist) == 0:
            return 0
        return 1 if id in shortlist['instance_number'].values else 0
    
    def _process_importance(self, is_important):
        expand_index = [-1, 1, -2, 2]
        for i in range(len(is_important)):
            if is_important[i] == 1:
                for x in expand_index:
                    if i + x >= 0 and i + x < len(is_important):
                        is_important[i + x] = max(is_important[i + x], 0.5)
        return is_important


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
    global ckpt
    model = RSNAModel(model_name=config['model_name'])
    model.encoder.load_state_dict(ckpt[fold_n], strict=True)
    accelerator.print(f'Training for {file_name} on FOLD #{fold_n}...')
    criterion = CustomLoss()
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

def get_loaders(df, df_series, df_co, data, fold_n):
    if fold_n is None:
        train_df = df_series.copy()
        valid_df = df_series[df_series['fold'] == 0].copy()
    else:
        train_df = df_series[df_series['fold'] != fold_n].copy()
        valid_df = df_series[df_series['fold'] == fold_n].copy()
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f'Data is split into train: {len(train_df)}, and valid: {len(valid_df)}')
    
    train_set = SegmentDataset(df, train_df, df_co, data, augment_level=0)
    valid_set = SegmentDataset(df, valid_df, df_co, data, augment_level=0)

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
def get_dicom_metadata(filename):
    dcm = pydicom.dcmread(filename)
    result = {}
    for element in dcm:
        if element.name == 'Pixel Data': continue
        result[element.name] = element.value
    return result

def get_dicom_volume(subfolder_path):
    dicom_files = [f for f in os.listdir(subfolder_path) if f.endswith('.dcm')]
    slice_position = []
    if os.path.isfile(os.path.join(subfolder_path, 'data.pt')):
        return torch.load(os.path.join(subfolder_path, 'data.pt'))
    else:
        dicom_files.sort(key=lambda x: int(x[:-4]))
        first_slice = pydicom.dcmread(os.path.join(subfolder_path, dicom_files[0]))
        img_shape = first_slice.pixel_array.shape
        num_slices = len(dicom_files)
        volume = torch.zeros((num_slices, *img_shape), dtype=torch.float16)
        for i, file in enumerate(dicom_files):
            ds = pydicom.dcmread(os.path.join(subfolder_path, file))
            metadata = get_dicom_metadata(os.path.join(subfolder_path, file))
            x = ds.pixel_array.astype(float)
            if x.shape == volume.shape[1:]:
                volume[i, :, :] = torch.tensor(x)
                slice_position.append(float(metadata['Image Position (Patient)'][0]))
            else:
                volume = volume[:i]
                break
    return volume, slice_position

def get_data_w_series(df, drop_rate=0.0):
    print('Loading data into RAM')
    data = {}
    for filepath, series_id in tqdm(df[['filepath', 'series_id']].values):
        if np.random.rand() < drop_rate:
            data[series_id] = filepath
        else:
            data[series_id] = get_dicom_volume(filepath)
    return data

def get_df_co():
    df_co = pd.read_csv(f'{project_paths.base_path}/train_label_coordinates.csv')
    df_co['instance_number'] = df_co['instance_number'].apply(lambda x: x - 1)
    only_sagittal = []
    for s, id in df_co[['study_id', 'series_id']].values:
        d = datasets.find_description(s, id)
        if 'Axial' in d:
            only_sagittal.append(False)
        else:
            only_sagittal.append(True)
    df_co_only_sag = df_co[only_sagittal].reset_index(drop=True)
    df_co_only_sag = df_co_only_sag.drop('x', axis=1).drop('y', axis=1).drop('level', axis=1).reset_index(drop=True)
    df_co_only_sag = df_co_only_sag.drop_duplicates().reset_index(drop=True)
    return df_co_only_sag

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
    bad_study_ids = [490052995, 1261271580, 2492114990, 2507107985, 2626030939, 2773343225, 2780132468, 3008676218, 3109648055, 3387993595, 3637444890]
    df_series = df_series.drop(np.where(df_series['study_id'].isin(bad_study_ids))[0])
    df_series = df_series[df_series['study_id'].isin(df['study_id'])].reset_index(drop=True)
    df_co = get_df_co()
    data = get_data_w_series(df_series, drop_rate=0.0) if accelerator.is_local_main_process else {}
    global ckpt
    ckpt = torch.load('/media/workspace/RSNA2024_checkpoints/t052_from051_pretrain_on_loc_dataset.pt')
    ckpt = [{k[7:] : v for k, v in one_ckpt.items()} for one_ckpt in ckpt]
    ckpt = [{k[8:] : v for k, v in one_ckpt.items() if 'encoder' in k}for one_ckpt in ckpt]
    print(ckpt[0].keys())

    save_weights = []
    for fold_n in range(config['folds']):
        train_loader, valid_loader = get_loaders(df, df_series, df_co, data, fold_n)
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