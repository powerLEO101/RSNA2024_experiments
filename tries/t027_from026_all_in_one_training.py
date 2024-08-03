import os
import sys
import timm
import torch
import wandb
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
from accelerate.utils import set_seed
from accelerate import Accelerator
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

IS_LOCAL = bool('LOCAL_TEST' in environ)
wandb.require('core')

config = {
    'lr': 1e-3,
    'wd': 1e-3,
    'epoch': 10,
    'seed': 22,
    'folds': 5,
    'batch_size': 32 if not 'LOCAL_TEST' in environ else 1,
    'model_name': 'timm/efficientnet_b0.ra_in1k',
    'out_feature_divide': 2,
    'checkpoint_freq': 5 
}
file_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator()
device = accelerator.device

#%% LOSS
class PerLevelCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight) if weight is not None else nn.CrossEntropyLoss()
    
    def forward(self, pred, label, have_label):
        have_label = have_label.flatten()
        pred = pred.view(-1, 3)[have_label]
        label = label.view(-1, 3)[have_label]
        return self.loss(pred, label)

class LevelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, label):
        pred = rearrange(pred, 'b n l f -> (b n l) f')
        label = rearrange(label, 'b n l f -> (b n l) f')
        is_keep = pred.sum(dim=1) != 0
        pred = pred[is_keep]
        label = label[is_keep]
        return self.loss(pred, label)

class CustomLoss(nn.Module):
    def __init__(self, weight=None):
        ''' Loss in order of level_cls, slice_cls, slice_levels, volume_cls'''
        super().__init__()
        self.ce_loss = PerLevelCrossEntropyLoss(weight=weight)
        self.level_loss = LevelLoss()
    
    def forward(self, pred, label):
        

#%% MODEL
class LevelModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        base_model = timm.create_model(model_name=model_name,
                                       pretrained=True,
                                       drop_rate=0.5,
                                       drop_path_rate=0.5)
        try:
            in_features = base_model.fc.in_features
        except:
            in_features = base_model.classifier.in_features
        
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(nn.Dropout(p=0.5), 
                                      nn.Linear(in_features, 3))
        self.in_features = in_features
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        x = self.global_pool(x).view(batch_size, -1)
        return x

class RSNA25dModel(nn.Module):
    def __init__(self, model_name, total_level=16):
        super().__init__()
        self.total_level = total_level

        self.encoder = LevelModel(model_name)
        self.lstm = nn.LSTM(input_size=self.encoder.in_features, 
                            hidden_size=self.encoder.in_features,
                            num_layers=2,
                            batch_first=True)
        self.cls_head = nn.Linear(self.encoder.in_features, 30)
        self.level_head = nn.Linear(self.encoder.in_features, 5)
        self.level_cls_head = nn.Linear(self.encoder.in_features, 3)

        self.volume_lstm = nn.LSTM(input_size=self.encoder.in_features, 
                            hidden_size=self.encoder.in_features,
                            num_layers=2,
                            batch_first=True)
        self.volume_cls_head = nn.Linear(self.encoder.in_features, 45)
    
    def forward(self, x):
        ''' volume: [b, n, 16, 3, 128, 256]'''

        x = x['volume']
        assert self.total_level == x.shape[2]

        batch_size = x.shape[0]
        x = rearrange(x, 'b n s c x y -> (b n s) c x y')
        x = self.encoder(x) # B * N * S IN_FEATURES

        x = rearrange(x, '(b n s) f -> (b n) s f', b=batch_size, s=self.total_level)
        x_lstm, _ = self.lstm(x) # (b n) s f
        slice_levels = self.level_head(x_lstm)
        slice_cls = self.cls_head(x_lstm[:, -1, :])
        slice_levels = rearrange(slice_levels, '(b n) s f -> b n s f', b=batch_size)
        slice_cls = rearrange(slice_cls, '(b n) (f1 f2) -> b n f1 f2', b=batch_size, f1=10, f2=3)

        level_cls = self.level_cls_head(x)
        level_cls = rearrange(level_cls, '(b n) s f -> b n s f', b=batch_size)

        x_lstm = rearrange(x_lstm, '(b n) s f -> b n s f', b=batch_size)
        x_lstm = x_lstm[:, :, -1, :]
        volume_features, _ = self.volume_lstm(x_lstm)
        volume_cls = self.volume_cls_head(volume_features[:, -1, :])
        volume_cls = rearrange(volume_cls, 'b (x f) -> b x f', f=3)

        return {
            'slice_levels': slice_levels,
            'slice_cls': slice_cls,
            'level_cls': level_cls,
            'volume_cls': volume_cls
        }

#%% DATASET
class SegmentDataset(Dataset):
    def __init__(self, 
                 df,
                 df_series,
                 df_co, 
                 data,
                 augment_level=0,
                 window_len=32,
                 step_size=16,
                 total_level=16,
                 image_size=[256, 256],
                 image_size1=[128, 256]):

        super().__init__()
        self.augment_level = augment_level
        self.df = df
        self.df_series = df_series
        self.df_co = df_co
        self.data = data
        self.image_size = image_size
        self.image_size1 = image_size1
        self.window_len = window_len # be careful here, sometimes I divide by 2
        self.step_size = step_size
        self.total_level = total_level

        self.resize = A.Compose([
                A.Resize(*self.image_size)],
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        if self.augment_level == 0:
            self.augment = A.Compose([
                A.Resize(*self.image_size1)])
                #keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        elif self.augment_level == 1:
            self.augment = A.Compose([
                A.Resize(*self.image_size1),
                A.Perspective(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(p=0.5, limit=(-25, 25))])
                #keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.df_series)

    def __getitem__(self, idx):
        meta = dict(self.df_series.iloc[idx])

        volume = []
        volume_original = []
        slice_labels = []
        slice_level_labels = []
        slice_have_labels = []
        for instance_number in range(meta['total_instance_number']):
            labels, kps, have_label = self._query_label(meta['series_id'], instance_number)

            image = self._get_slice(meta, instance_number)
            result = self.resize(image=image.numpy().astype('f'), keypoints=kps)
            image = result['image']
            kps = result['keypoints']
            images, strong_labels = self._split_image(image, have_label, kps)

            images_full = []
            level_label = []
            for idx, i in enumerate(np.arange(0, len(images), len(images) / self.total_level)):
                images_full.append(images[int(i)].copy())
                level_label.append(strong_labels[int(i)])
            images_full = self.augment(images=images_full)['images']
            images_full = torch.stack([torch.tensor(x, dtype=torch.float32) for x in images_full], dim=0)
            images_full = torch.stack([images_full] * 3, dim=1)

            volume.append(images_full)
            volume_original.append(torch.from_numpy(image))
            slice_labels.append(labels)
            slice_level_labels.append(level_label)
            slice_have_labels.append(have_label)
        
        volume = torch.stack(volume, dim=0)
        volume_original = torch.stack(volume_original, dim=0)
        slice_labels = torch.tensor(slice_labels)
        slice_level_labels = torch.tensor(slice_level_labels)
        slice_have_labels = torch.tensor(slice_have_labels)
        volume_label = torch.tensor(self._query_volume_label(meta['study_id']))

        return {
            'volume': volume,
            'volume_original': volume_original,
            'volume_label': volume_label,
            'slice_label': slice_labels,
            'slice_level_label': slice_level_labels,
            'slice_have_label': slice_have_labels
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
    
    def _query_label(self, series_id, instance_number):
        shortlist = self.df_co[self.df_co['series_id'] == series_id]
        shortlist = shortlist[shortlist['instance_number'] == instance_number]
        if len(shortlist) == 0:
            return [[0] * 3] * 10, [[0] * 2] * 10, [False] * 10
        shortlist = shortlist.iloc[0]
        return shortlist['label'], shortlist['keypoints'], shortlist['have_keypoints']
    
    def _query_volume_label(self, study_id):
        shortlist = self.df[self.df['study_id'] == study_id].iloc[0]
        conditions = ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing']
        levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1'] 
        result = []
        for condition in conditions:
            for level in levels:
                tmp = [0] * 3
                tmp[shortlist[f'{condition}_{level}']] = 1
                result.append(tmp)
        return result
    
    def _split_image(self, image, have_label, kps):
        strong_labels = []
        images = []
        for i in range(0, image.shape[0], self.step_size):
            if i + self.window_len > image.shape[0]: break
            images.append(image[i : i + self.window_len, :])
            strong_label = [0] * 5
            for j in range(10):
                if have_label[j] == False: continue
                if (kps[j][1] <= i + self.window_len) and (kps[j][1] >= i):
                    strong_label[j % 5] = 1
            strong_labels.append(strong_label)
        return images, strong_labels



#%% TRAINING
def train_one_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    model.train()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    for step, batch in bar:
        # B C X Y
        image = batch['image']
        label = [batch['label'], batch['level_label']]
        have_label = batch['have_label']
        optimizer.zero_grad()
        pred_labels = model(image)
        loss, loss_ce, loss_level = criterion(pred_labels, label, have_label)
        running_loss += (loss.item() - running_loss) / (step + 1)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        if accelerator.is_local_main_process and not IS_LOCAL:
            wandb.log({
                'lr': lr, 
                'train_step_loss': loss.item(),
                'train_step_ce_loss': loss_ce.item(),
                })
        bar.set_postfix_str(f'epoch: {epoch}, lr: {lr:.2e}, train_loss: {running_loss: .4e}')
        accelerator.free_memory()

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.log({
            'train_epoch_loss': running_loss,
            })

def valid_one_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    running_ce_loss = 0.0
    global global_step
    model.eval()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    for step, batch in bar:
        # B C X Y
        image = batch['image']
        label = [batch['label'], batch['level_label']]
        have_label = batch['have_label']
        with torch.no_grad():
            pred_label = model(image)
        loss, loss_ce, loss_level = criterion(pred_label, label, have_label)
        running_loss += (loss.item() - running_loss) / (step + 1)
        running_ce_loss += (loss_ce.item() - running_ce_loss) / (step + 1)
        bar.set_postfix_str(f'Epoch: {epoch}, valid_loss: {running_loss}')
        accelerator.free_memory()

    accelerator.print(f'Valid loss: {running_loss}')

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.log({
            'valid_epoch_loss': running_loss,
            'valid_epoch_ce_loss': running_ce_loss,
            })

def train_one_fold(train_loader, valid_loader, fold_n):
    model = RSNA25dModel(model_name=config['model_name'])
    accelerator.print(f'Training for {file_name} on FOLD #{fold_n}...')
    criterion = TwoWayLoss(torch.tensor([1., 2., 4.], device=device))
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
    
    train_set = SegmentDataset(df, train_df, data, augment_level=1)
    valid_set = SegmentDataset(df, valid_df, data, augment_level=0)

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
            label = [[0, 0, 0]] * 10

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
    df_series = pd.read_csv(f'{project_paths.base_path}/train_series_descriptions.csv')
    df_series = df_series[df_series['series_description'].str.contains('Sagittal')].reset_index(drop=True)
    def count_files(folder):
        files = glob(f'{folder}/*.dcm')
        return len(files)
    df_series['filepath'] = df_series.apply(lambda x: f"../input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{x['study_id']}/{x['series_id']}", axis=1)
    df_series['total_instance_number'] = df_series['filepath'].apply(count_files)
    return df_series

def main(): 
    set_seed(config['seed'])
    df = datasets.get_df()
    df_series = get_df_series()
    df_series = df_series[df_series['study_id'].isin(df['study_id'])].reset_index(drop=True)
    df_co = keypoints.df_label_co.copy()
    df_co = df_co[df_co['study_id'].isin(df['study_id'])].reset_index(drop=True)
    df_co = get_df_co(df_co, df)
    data = datasets.get_data_w_series(df_series, drop_rate=0.2)

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