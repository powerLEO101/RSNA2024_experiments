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
import segmentation_models_pytorch as smp

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
from accelerate import DistributedDataParallelKwargs


IS_LOCAL = bool('LOCAL_TEST' in environ)
wandb.require('core')

config = {
    'lr': 1e-4,
    'wd': 1e-3,
    'epoch': 10,
    'seed': 22,
    'folds': 5,
    'batch_size': 4 if not 'LOCAL_TEST' in environ else 1,
    'model_name': 'efficientnet_b0.ra_in1k',
    'grad_acc': 4,
    'checkpoint_freq': 2,
}
file_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator(split_batches=True, dispatch_batches=True)
device = accelerator.device

#%% LOSS

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pred_, label_):
        label_ = label_['label']
        losses = []
        for i in range(2):
            pred = pred_[:, 15 * i : 15 * (i + 1)].contiguous()
            label = label_[:, 15 * i : 15 * (i + 1)].contiguous()
            pred = pred.view(-1, 3)
            label = label.view(-1, 3)
            label_weights = torch.tensor([2 ** x.tolist().index(1) for x in label], device=device) ## 2 ** index == (1, 2, 4)
            loss = self.loss(pred, label)
            loss *= label_weights
            loss = loss.mean()
            losses.append(loss)
        
        return sum(losses) / 2, {'left_loss': losses[0], 'right_loss': losses[1]}


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
        self.head = nn.Linear(in_features, 3)
    def forward(self, input):
        x = input['image']
        B, N, C, X, Y = x.shape
        x = rearrange(x, 'b n c x y -> (b n) c x y')
        x = self.encoder(x)
        x = self.head(x)
        x = rearrange(x, '(b n) f -> b n f', b=B)
        x = x.flatten(-2)
        return x
#%% DATASET

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

class LocDataset(Dataset):
    def __init__(self, df, df_for_label, is_infer=False):
        super().__init__()
        self.df = df
        self.df_for_label = df_for_label
        self.data = {}
        if is_infer:
            self.augment = A.Compose([A.Resize(384, 384)])
        else:
            self.augment = A.Compose([
                A.Resize(384, 384),
                A.Perspective(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(p=0.5, limit=(-25, 25))
                ])
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        meta = self.df.iloc[index]
        study_id = meta['study_id']
        image = self._get_data_from_cache_and_disk(meta['filepath'])
        locs_info = eval(meta['locs_info'])
        select_image = []
        conditions = ['Left Subarticular Stenosis', 'Right Subarticular Stenosis']
        for condition in conditions:
            one_locs_info = locs_info[condition]
            for level in range(5):
                instance_number = one_locs_info[level][0]
                select_image.append(image[:, instance_number, ...])
        select_image = torch.stack(select_image, dim=1)
        select_image = torch.cat([select_image] * 3, dim=0)
        select_image = select_image.permute(1, 0, 2, 3)
        label = self._query_label(study_id)

        return {
            'image': select_image,
            'label': label,
            #'condition': meta['condition']
            'study_id': study_id
        }
    
    def _get_data_from_cache_and_disk(self, filepath):
        if filepath not in self.data:
            volume, _ = get_dicom_volume(filepath)
            volume_resized = [torch.from_numpy(x.copy()) for x in self.augment(images=volume.float().numpy())['images']]
            volume = torch.stack(volume_resized, dim=0)
            volume = (volume - volume.min()) / (volume.max() - volume.min())
            volume = volume.unsqueeze(0)
            self.data[filepath] = volume.float()
        return self.data[filepath]
    
    def _query_label(self, study_id):
        shortlist = self.df_for_label[self.df_for_label['study_id'] == study_id].iloc[0]
        conditions = ['left_subarticular_stenosis', 'right_subarticular_stenosis'] # !!! the order of label here is different
        levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1'] 
        result = []
        for condition in conditions:
            for level in levels:
                tmp = [0.] * 3
                tmp[shortlist[f'{condition}_{level}']] = 1.
                result.extend(tmp)
        result = torch.tensor(result, dtype=torch.float32)
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
    model = RSNAModel(config['model_name'])
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

def get_loaders(df, df_for_label, fold_n):
    if fold_n is None:
        train_df = df.copy()
        valid_df = df[df['fold'] == 0].copy()
    else:
        train_df = df[df['fold'] != fold_n].copy()
        valid_df = df[df['fold'] == fold_n].copy()
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f'Data is split into train: {len(train_df)}, and valid: {len(valid_df)}')
    
    train_set = LocDataset(train_df, df_for_label)
    valid_set = LocDataset(valid_df, df_for_label, is_infer=True)

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

def main(): 
    set_seed(config['seed'])
    df = pd.read_csv('/media/workspace/RSNA2024_input/pretrain_locs_dataset_v3/df_series_axial.csv')
    df['fold'] = df['study_id'].apply(fold_for_all.get)
    df_for_label = datasets.get_df()

    save_weights = []
    for fold_n in range(config['folds']):
        train_loader, valid_loader = get_loaders(df, df_for_label, fold_n)
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