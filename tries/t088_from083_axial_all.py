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
import torch.nn.functional as F

sys.path.append('/media/workspace/RSNA2024_experiments')
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


IS_LOCAL = False
wandb.require('core')

config = {
    'lr': 1e-4,
    'wd': 1e-3,
    'epoch': 10,
    'seed': 22,
    'folds': 5,
    'batch_size': 1 if not 'LOCAL_TEST' in environ else 1,
    'model_name': 'efficientnet_b0.ra_in1k',
    'grad_acc': 4,
    'checkpoint_freq': 2,
}
file_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator(gradient_accumulation_steps=4, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
device = accelerator.device

#%% LOSS

class ClassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pred, label, have_label):
        pred = pred.view(-1, 3)
        label = label.view(-1, 3)
        have_label = have_label.view(-1)
        pred = pred[have_label]
        label = label[have_label]

        label_weights = torch.tensor([2 ** x.tolist().index(1) for x in label], device=pred.device) ## 2 ** index == (1, 2, 4)
        loss = self.loss(pred, label)
        loss *= label_weights
        loss = loss.mean()
        return loss, {}

class LocsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, pred, label, have_label):
        pred = pred.view(-1, 3)
        label = label.view(-1, 3)
        have_label = have_label.view(-1)
        pred = pred[have_label]
        label = label[have_label]
        
        loss_z = self.loss(pred[:, 0], label[:, 0])
        loss_xy = self.loss(pred[: , 1 : ], label[: , 1 : ])


        return loss_z + loss_xy, {'loss_z' : loss_z, 'loss_xy' : loss_xy}

class JensonShannonDiv(nn.Module):
    def forward(self, pred, label, dims, have_label):
        pred = pred.permute(1, 0, 2, 3)
        label = label.permute(1, 0, 2, 3)

        heatmap =  torch.split_with_sizes(pred, dims, 0)
        truth =  torch.split_with_sizes(label, dims, 0)
        num_image = len(heatmap)

        loss =0
        for i in range(num_image):
            p,q = truth[i], heatmap[i]
            p, q = p[:, have_label[i]], q[:, have_label[i]]
            D,num_point,H,W = p.shape

            eps = 1e-8
            p = torch.clamp(p.transpose(1,0).flatten(1),eps,1-eps)
            q = torch.clamp(q.transpose(1,0).flatten(1),eps,1-eps)
            m = (0.5 * (p + q)).log()

            kl = lambda x,t: F.kl_div(x,t, reduction='batchmean', log_target=True)
            loss += 0.5 * (kl(m, p.log()) + kl(m, q.log()))
        loss = loss/num_image
        return loss, {}


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_loss = ClassLoss()
        self.locs_loss = LocsLoss()
        self.js_div = JensonShannonDiv()

    def forward(self, pred, label):
        class_loss, dict0 = self.class_loss(pred['grade'], label['label'], label['have_label'])
        locs_loss, dict1 = self.locs_loss(pred['coordinates'], label['coordinates'], label['have_label'])
        js_div, dict2 = self.js_div(pred['mask'], label['seg_label'], label['dims'].tolist(), label['have_label'])

        result_dict = {'class_loss' : class_loss, 'locs_loss' : locs_loss, 'js_div' : js_div, **dict0, **dict1, **dict2}
        result_dict = {k : v.detach() for k, v, in result_dict.items()}

        #return locs_loss / (96 ** 2) + js_div, result_dict
        return class_loss + locs_loss * 0.01 + js_div * 0.2, result_dict

#%% MODEL

class MyDecoderBlock3d(nn.Module):
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
    
class MyUnetDecoder3d(nn.Module):
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock3d(i, s, o)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):  
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode

def heatmap_to_coord(heatmap):
    # hengck
    num_image = len(heatmap)
    device = heatmap[0].device
    _,_, H, W = heatmap[0].shape
    D = max([h.shape[1] for h in heatmap])

    # create coordinates grid.
    x = torch.linspace(0, W - 1, W, device=device)
    y = torch.linspace(0, H - 1, H, device=device)
    z = torch.linspace(0, D - 1, D, device=device)

    point_xy=[]
    point_z =[]
    for i in range(num_image):
        num_point, D, H, W = heatmap[i].shape
        pos_x = x.reshape(1,1,1,W)
        pos_y = y.reshape(1,1,H,1)
        pos_z = z[:D].reshape(1,D,1,1)

        py = torch.sum(pos_y * heatmap[i], dim=(1,2,3))
        px = torch.sum(pos_x * heatmap[i], dim=(1,2,3))
        pz = torch.sum(pos_z * heatmap[i], dim=(1,2,3))

        point_xy.append(torch.stack([py, px]).T) # !!!x, y is reverse for my experiments here!!!
        point_z.append(pz)

    xy = torch.stack(point_xy)
    z = torch.stack(point_z).unsqueeze(-1)
    return torch.cat([z, xy], dim=-1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = timm.create_model(model_name='pvt_v2_b4',
                                    pretrained=True, 
                                    in_chans=3,
                                    features_only=True)
        self.encoder = encoder

        sample_input = torch.rand(1, 3, 64, 64)
        encoder_channels = [x.shape[1] for x in self.encoder(sample_input)]
        decoder_channels = [384, 192, 96]

        self.unet_decoder = MyUnetDecoder3d(
            in_channel=encoder_channels[-1],
            skip_channel=encoder_channels[: -1][::-1],
            out_channel=decoder_channels
        )
        self.seg_mask_head = nn.Conv3d(decoder_channels[-1], 5, kernel_size=1)
        self.grade_mask_head = nn.Conv3d(decoder_channels[-1], 128, kernel_size=1)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        ) 
    
    def forward(self, x):

        dims = x['dims']
        B = len(dims)
        x = x['image']
        BD, C, X, Y = x.shape

        x = self.encoder(x)
        features_batch = [torch.split_with_sizes(feature, split_sizes=dims.tolist(), dim=0) for feature in x] # without padding

        seg = []
        grade = []
        for i in range(B):
            features = [one_feature[i].permute(1, 0, 2, 3).unsqueeze(0) for one_feature in features_batch]
            x, _ = self.unet_decoder(features[-1], features[: -1][::-1])
            one_seg = self.seg_mask_head(x).squeeze(0)
            tmp_c, tmp_d, tmp_x, tmp_y = one_seg.shape
            one_seg = rearrange(one_seg.flatten(1).softmax(-1), 'c (d x y) -> c d x y', d=tmp_d, x=tmp_x)
            one_grade = self.grade_mask_head(x).squeeze(0)

            seg.append(one_seg)
            grade.append(one_grade)


        grade_after_mask = []
        for i in range(B):
            one_seg = rearrange(seg[i], '(c c1) d x y -> c c1 d x y', c1=1) # unsqueeze one dim for broadcasting, use einops for better readability
            one_grade = rearrange(grade[i], '(c1 c) d x y -> c1 c d x y', c1=1)
            grade_after_mask.append((one_grade * one_seg).sum(dim=(2, 3, 4))) # after multiply: c_grade(128), c_seg(10~), d, x, y
            # after taking sum, c_seg(10), c_grade(128), taking sum here because one_seg acts like attention and sum to 1
        grade_after_mask = rearrange(torch.stack(grade_after_mask, dim=0), 'b c0 c1 -> (b c0) c1')
        final_grade = rearrange(self.head(grade_after_mask), '(b c0) f -> b c0 f', b=B) # B, C, 3

        seg_cat = torch.cat(seg, dim=1)
        
        return {
            'grade': final_grade,
            'mask': seg_cat,
            'coordinates': heatmap_to_coord(seg),
        }
        

#%% DATASET

def get_dicom_metadata(filename):
    dcm = pydicom.dcmread(filename)
    result = {}
    for element in dcm:
        if element.name == 'Pixel Data': continue
        result[element.name] = element.value
    return result

def get_dicom_volume(subfolder_path, slice_pos_index=0, use_cache=True):
    """ !!! careful, when doing axial, do not use data.pt directly"""
    dicom_files = [f for f in os.listdir(subfolder_path) if f.endswith('.dcm')]
    slice_position = []
    if use_cache and os.path.isfile(os.path.join(subfolder_path, 'data.pt')):
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
                slice_position.append(float(metadata['Image Position (Patient)'][slice_pos_index]))
            else:
                volume = volume[:i]
                break
    return volume, slice_position


class RSNADataset(Dataset):
    def __init__(self, df, df_for_label, is_infer=False):
        super().__init__()
        self.df = df
        self.df_for_label = df_for_label
        self.resize = A.Resize(384, 384)
        self.data = {}
        self.conditions = [
            'left_neural_foraminal_narrowing_l1_l2',
            'left_neural_foraminal_narrowing_l2_l3',
            'left_neural_foraminal_narrowing_l3_l4',
            'left_neural_foraminal_narrowing_l4_l5',
            'left_neural_foraminal_narrowing_l5_s1',
            'right_neural_foraminal_narrowing_l1_l2',
            'right_neural_foraminal_narrowing_l2_l3',
            'right_neural_foraminal_narrowing_l3_l4',
            'right_neural_foraminal_narrowing_l4_l5',
            'right_neural_foraminal_narrowing_l5_s1',
            'spinal_canal_stenosis_l1_l2',
            'spinal_canal_stenosis_l2_l3',
            'spinal_canal_stenosis_l3_l4',
            'spinal_canal_stenosis_l4_l5',
            'spinal_canal_stenosis_l5_s1',
            'left_subarticular_stenosis_l1_l2',
            'left_subarticular_stenosis_l2_l3',
            'left_subarticular_stenosis_l3_l4',
            'left_subarticular_stenosis_l4_l5',
            'left_subarticular_stenosis_l5_s1',
            'right_subarticular_stenosis_l1_l2',
            'right_subarticular_stenosis_l2_l3',
            'right_subarticular_stenosis_l3_l4',
            'right_subarticular_stenosis_l4_l5',
            'right_subarticular_stenosis_l5_s1'
        ]
        if is_infer:
            self.augment = A.ReplayCompose([
            ], keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))
        else:
            self.augment = A.ReplayCompose([
                A.Perspective(p=0.5),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                A.Rotate(p=0.5, limit=(-25, 25))
            ], keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        meta = self.df.iloc[index]
        volume = self._get_data_from_cache_and_disk(meta['filepath'])
        #volume = volume[eval(meta['index_order'])]
        from copy import deepcopy
        locs = deepcopy(eval(meta['locs']))  # !!! be careful here that locs is copied by reference from the pandas! therefore any change to locs will affect df, this message is out dated

        random_level = np.random.randint(5)  # apply cluster
        volume_mask = [one_cluster == random_level for one_cluster in eval(meta['cluster'])]
        volume_mask_index = np.where(volume_mask)[0]
        volume_mask_new_index = {volume_mask_index[i]: i for i in range(len(volume_mask_index))}
        levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
        level_conditions = [condition for condition in self.conditions if levels[random_level] in condition]
        new_locs = {}
        for condition in level_conditions:
            z, x, y = locs[condition]  # xy already swapped
            if z not in volume_mask_index:
                continue
            new_locs[condition] = (volume_mask_new_index[z], x, y)
        volume = volume[volume_mask]
        locs = new_locs

        volume, locs = self._apply_augment(volume, locs, level_conditions)
        seg_label, have_label = self._make_seg_label(volume, locs, level_conditions)
        coordinates = self._make_coordinates(locs, level_conditions)
        label = self._query_label(meta['study_id'], level_conditions)

        volume = torch.stack([volume] * 3, dim=1)
        if len(volume) > 32:
            volume = volume[: 32]
            seg_label = seg_label[:, : 32]

        return {
            'study_id': meta['study_id'],
            'image': volume,
            'label': label,
            'seg_label': seg_label,
            'coordinates': coordinates,
            'have_label': have_label
        }

    def _get_data_from_cache_and_disk(self, filepath):
        if filepath not in self.data:
            volume, _ = get_dicom_volume(filepath)
            volume_resized = [torch.from_numpy(x) for x in self.resize(images=volume.float().numpy())['images']]
            volume = torch.stack(volume_resized, dim=0)
            upper = torch.quantile(volume, 0.995)
            lower = torch.quantile(volume, 0.005)
            volume = volume.clamp(lower, upper)
            volume = (volume - volume.min()) / (volume.max() - volume.min())
            self.data[filepath] = volume.float()
        return self.data[filepath]

    def _make_seg_label(self, volume, locs, conditions, fwhm=5):
        """
            :param fwhm: The effective radius
        """

        def in_bound(x, lim=96):
            if x < 0 or x >= lim:
                return False
            return True

        have_label = []
        # conditions = self.conditions
        seg_label = torch.zeros(len(conditions), len(volume), 96, 96)
        length = 3
        for condition_id, condition in enumerate(conditions):
            if condition not in locs:
                have_label.append(False)
                continue
            have_label.append(True)
            z, x, y = locs[condition]
            x = int(x * 96)
            y = int(y * 96)
            for i in range(96):
                for j in range(96):
                    # Use Gaussian Distribution here
                    seg_label[condition_id, z, i, j] = np.exp(
                        -4 * np.log(2) * ((i - x) ** 2 + (j - y) ** 2) / fwhm ** 2)
        for i in range(len(conditions)):
            seg_label[i] = seg_label[i] / seg_label[i].sum()
        # seg_label[seg_label == 0] = -1000
        # seg_label = seg_label.flatten(1).softmax(-1).view(len(conditions), len(volume), 96, 96)
        have_label = torch.tensor(have_label)
        return seg_label, have_label

    def _query_label(self, study_id, conditions):
        shortlist = self.df_for_label[self.df_for_label['study_id'] == study_id].iloc[0]
        # conditions = self.conditions
        result = []
        for condition in conditions:
            tmp = [0.] * 3
            tmp[shortlist[condition]] = 1.
            result.append(tmp)
        result = torch.tensor(result, dtype=torch.float32)
        return result

    def _make_coordinates(self, locs, conditions):
        coordinates = []
        # conditions = self.conditions
        for condition in conditions:
            if condition not in locs:
                coordinates.append([0, 0, 0])
                continue
            z, x, y = locs[condition]
            x *= 96
            y *= 96
            coordinates.append([z, x, y])
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
        return coordinates

    def _apply_augment(self, image, locs, conditions):
        # image: D, X, Y; locs: {name : [z, x, y]}
        keypoints_array = []
        for condition in conditions:
            if condition not in locs:
                keypoints_array.append([0, 0])
            else:
                keypoints_array.append([locs[condition][1] * 384, locs[condition][2] * 384])
        transformed = self.augment(image=image[0].numpy(), keypoints=keypoints_array)
        replay = transformed['replay']

        transformed_volume = []
        for one_image in image:
            transformed_volume.append(
                torch.tensor(A.ReplayCompose.replay(replay, image=one_image.float().numpy())['image']))
        transformed_volume = torch.stack(transformed_volume, dim=0)

        new_locs = {}
        for idx, condition in enumerate(conditions):
            if condition not in locs:
                continue
            new_locs[condition] = [locs[condition][0], transformed['keypoints'][idx][0] / 384,
                                   transformed['keypoints'][idx][1] / 384]

        return transformed_volume, new_locs

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
    model = Model()
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

def custom_collate(data):
    result = {}
    result['study_id'] = torch.tensor([x['study_id'] for x in data])
    result['label'] = torch.stack([x['label'] for x in data], dim=0)
    result['coordinates'] = torch.stack([x['coordinates'] for x in data], dim=0)
    result['have_label'] = torch.stack([x['have_label'] for x in data], dim=0)

    result['seg_label'] = torch.cat([x['seg_label'] for x in data], dim=1)
    result['image'] = torch.cat([x['image'] for x in data], dim=0)

    result['dims'] = torch.tensor([len(x['image']) for x in data], dtype=torch.int)

    return result

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
    
    train_set = RSNADataset(train_df, df_for_label)
    valid_set = RSNADataset(valid_df, df_for_label, is_infer=True)

    # weights = []
    # weight_multiplier = [1., 2., 4.]
    # for element in train_set:
    #     weights.append(weight_multiplier[element['label'].tolist().index(1)])
    # weighted_sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_set))

    train_loader =  DataLoader(train_set, 
                                batch_size=config['batch_size'], 
                                shuffle=True,
                                num_workers=12 if not 'LOCAL_TEST' in environ else 4, 
                                pin_memory=False,
                                collate_fn=custom_collate)
    valid_loader = DataLoader(valid_set, 
                                batch_size=config['batch_size'], 
                                shuffle=False, 
                                num_workers=12 if not 'LOCAL_TEST' in environ else 4, 
                                pin_memory=False,
                                collate_fn=custom_collate)

    return train_loader, valid_loader

#%% EXPERIMENT

def get_volume_size(filepath):
    volume, _ = get_dicom_volume(filepath)
    return volume.shape[1 : ]

def normalize_name(condition, level):
    return condition.lower().replace(' ', '_') + '_' + level.lower().replace('/', '_')


df_co = pd.read_csv(f'{project_paths.base_path}/train_label_coordinates.csv')
df_co['instance_number'] = df_co['instance_number'] - 1
def get_coordinates_from_series_id(series_id, filepath, index_order):
    shortlist = df_co[df_co['series_id'] == series_id]
    X, Y = get_volume_size(filepath)

    locs = {}
    for i in range(len(shortlist)):
        z, x, y, condition, level = shortlist.iloc[i][['instance_number', 'y', 'x', 'condition', 'level']]
        locs[normalize_name(condition, level)] = [index_order[z], x / X, y / Y]
    return locs

def get_index_order(filepath):
    _, patient_indices = get_dicom_volume(filepath)
    index_order = list(range(len(patient_indices)))
    index_order.sort(key=lambda x: patient_indices[x])
    return index_order

def get_df_series(filter='Sagittal T1'):
    df_series = pd.read_csv(f'{project_paths.base_path}/train_series_descriptions.csv')
    df_series = df_series[df_series['series_description'].str.contains(filter)].reset_index(drop=True)
    def count_files(folder):
        files = glob(f'{folder}/*.dcm')
        return len(files)
    df_series['filepath'] = df_series.apply(lambda x: f"{project_paths.base_path}/train_images/{x['study_id']}/{x['series_id']}", axis=1)
    df_series['total_instance_number'] = df_series['filepath'].apply(count_files)
    df_series['fold'] = df_series['study_id'].apply(fold_for_all.get)
    df_series['index_order'] = df_series['filepath'].apply(get_index_order)
    df_series['locs'] = df_series.apply(lambda x : get_coordinates_from_series_id(x['series_id'], x['filepath'], x['index_order']), axis=1)
    return df_series

def main(): 
    set_seed(config['seed'])
    #df = get_df_series()
    df = pd.read_csv('/media/workspace/RSNA2024_input/rsna-2024-lumbar-spine-degenerative-classification/0924axial_df_all.csv')
    df['fold'] = df['study_id'].apply(fold_for_all.get)
    df_for_label = datasets.get_df()
    df = df[df['study_id'].isin(df_for_label['study_id'])].reset_index(drop=True)

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