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
    'lr': 1e-3,
    'wd': 1e-3,
    'epoch': 20,
    'seed': 22,
    'folds': 5,
    'batch_size': 2 if not 'LOCAL_TEST' in environ else 1,
    'model_name': 'efficientnet_b0.ra_in1k',
    'grad_acc': 4,
    'checkpoint_freq': 2,
}
file_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator(split_batches=True, dispatch_batches=True, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
device = accelerator.device

#%% LOSS

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, label):
        label = label['label']
        B, C, N, X, Y = pred.shape
        pred = pred.view(B, -1)
        label = label.view(B, -1)
        return self.loss(pred, label), {}


#%% MODEL

class Model(nn.Module):
    def __init__(self, backbone=None, segtype='unet', pretrained=True):
        super(Model, self).__init__()
        
        n_blocks = 4
        self.n_blocks = n_blocks
        
        self.encoder = timm.create_model(
            'efficientnet_b0.ra_in1k',
            in_chans=3,
            features_only=True,
            drop_rate=0.1,
            drop_path_rate=0.1,
            pretrained=pretrained
        )
        g = self.encoder(torch.rand(1, 3, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )

        self.segmentation_head = nn.Conv2d(decoder_channels[n_blocks-1], 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self,x):
        x = x['image']
        
        global_features = [0] + self.encoder(x)[:self.n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features
    
#from timm.models.layers.conv2d_same import Conv2dSame
class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return timm.models.layers.conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1, 1), value: float = 0):
    ih, iw, iz = x.size()[-3:]
    pad_h = get_same_padding(ih, k[0], s[0], d[0])
    pad_w = get_same_padding(iw, k[1], s[1], d[1])
    pad_z = get_same_padding(iz, k[2], s[2], d[2])
    if pad_h > 0 or pad_w > 0 or pad_z > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_z // 2, pad_z - pad_z // 2], value=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def conv3d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0), dilation: Tuple[int, int, int] = (1, 1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-3:], stride, dilation)
    return F.conv3d(x, weight, bias, stride, (0, 0, 0), dilation, groups)


class Conv3dSame(nn.Conv3d):
    """ Tensorflow like 'SAME' convolution wrapper for 3d convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv3d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def create_conv3d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv3dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv3d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


def convert_3d(module):

    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
            
    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )
    del module

    return module_output


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
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.data = {}
        self.augment = A.Compose([A.Resize(384, 384)])
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        meta = self.df.iloc[index]
        study_id = meta['study_id']
        image = self._get_data_from_cache_and_disk(meta['filepath'])
        locs_info = eval(meta['locs_info'])
        label = self._get_locs_label_3d(locs_info, image.shape[1])
        image, label = self._pad(image, label)

        image = torch.cat([image] * 3, dim=0)
        return {
            'image': image,
            'label': label,
            #'condition': meta['condition']
        }
    
    def _get_data_from_cache_and_disk(self, filepath):
        if filepath not in self.data:
            volume, _ = get_dicom_volume(filepath)
            volume_resized = [torch.from_numpy(x) for x in self.augment(images=volume.float().numpy())['images']]
            volume = torch.stack(volume_resized, dim=0)
            volume = (volume - volume.min()) / (volume.max() - volume.min())
            volume = volume.unsqueeze(0)
            self.data[filepath] = volume.float()
        return self.data[filepath]
    
    def _get_locs_label_3d(self, locs_info, total_slice):
        length = 6
        def in_bound(x, lim=384):
            if x < 0 or x >= lim:
                return False
            return True

        result = torch.zeros(10, total_slice, 384, 384)
        conditions = ['Left Subarticular Stenosis', 'Right Subarticular Stenosis']
        for condition_id, condition in enumerate(conditions):
            one_locs_info = locs_info[condition]
            for level in range(5):
                label_id = condition_id * 5 + level
                z, x, y = one_locs_info[level]
                x = int(x * 384)
                y = int(y * 384)
                for i in range(-length + 1, length):
                    for j in range(-length + 1, length):
                        if in_bound(x + i) and in_bound(y + j):
                            result[label_id, z, x + i, y + j] = 1
                            if in_bound(z + 1, total_slice):
                                result[label_id, z + 1, x + i, y + j] = 0.5
                            if in_bound(z - 1, total_slice):
                                result[label_id, z - 1, x + i, y + j] = 0.5
        return result
    
    def _pad(self, image, label):
        result_image = torch.zeros(1, 80, 384, 384)
        result_label = torch.zeros(10, 80, 384, 384)
        if image.shape[1] < 80:
            result_image[:, :image.shape[1], ...] = image
            result_label[:, :label.shape[1], ...] = label
        else:
            result_image[:, :, ...] = image[:, :80, ...]
            result_label[:, :, ...] = label[:, :80, ...]
        return result_image, result_label

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
    model = convert_3d(Model())
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

def get_loaders(df, fold_n):
    if fold_n is None:
        train_df = df.copy()
        valid_df = df[df['fold'] == 0].copy()
    else:
        train_df = df[df['fold'] != fold_n].copy()
        valid_df = df[df['fold'] == fold_n].copy()
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f'Data is split into train: {len(train_df)}, and valid: {len(valid_df)}')
    
    train_set = LocDataset(train_df)
    valid_set = LocDataset(valid_df)

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
    bad_study_ids = [208289456, 2581283971, 3324678907, 3369277408, 3867046855, 114899184, 335455502, 582364168, 594735110, 766494595, 809072026, 1258848546, 1271819130, 1537608176, 1538136131, 1603568458, 1973833645, 2083466060, 2334206006, 2470505035, 2568819355, 2755347468, 2780132468, 3339741647, 3542237003, 3651144029, 3707028884, 3884015124, 3885334932, 3930841971, 3956571539, 4193490688]
    df = df[~df['study_id'].isin(bad_study_ids)].reset_index(drop=True)

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