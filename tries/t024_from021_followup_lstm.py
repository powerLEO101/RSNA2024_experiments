import os
import sys
import timm
import torch
import wandb
import torch.nn as nn
import numpy as np
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
    'epoch': 20,
    'seed': 22,
    'folds': 1,
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
        if weight is not None:
            self.loss = nn.CrossEntropyLoss(weight=weight)
        else:
            self.loss = nn.CrossEntropyLoss()
    
    def forward(self, pred, label, have_label):
        real_pred, real_label = [], []
        for p, l, h in zip(pred, label, have_label):
            real_pred.append(p[h])
            real_label.append(l[h])
        real_pred = torch.cat(real_pred, dim=0)
        real_label = torch.cat(real_label, dim=0) # batch_size * have_label_size, 3
        return self.loss(real_pred, real_label)

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

class RSNA2dModel(nn.Module):
    def __init__(self, model_name, fold_n, total_level=16, supervise_level=False, freeze_encoder=True):
        super().__init__()
        self.supervise_level = supervise_level
        self.total_level = total_level
        self.freeze_encoder = freeze_encoder

        self.encoder = LevelModel(model_name)
        if IS_LOCAL:
            self.encoder.load_state_dict(torch.load('/Users/leo101/Downloads/t022_from021_regulation_0_4.pt', map_location='cpu'), strict=False)
        else:
            self.encoder.load_state_dict(torch.load('/root/autodl-fs/t023_from022_more_aug_norm.pt')[fold_n], strict=False)
        self.encoder.requires_grad_(not self.freeze_encoder)
        self.lstm = nn.LSTM(input_size=self.encoder.in_features, 
                            hidden_size=self.encoder.in_features,
                            num_layers=2,
                            batch_first=True)
        self.cls_head = nn.Linear(self.encoder.in_features, 30)
        self.level_head = nn.Linear(self.encoder.in_features, 5)
    
    def forward(self, x):
        '''
            Accepts batched serieses of split images. Needs padding (shoud pad at the front?)
        '''

        assert self.total_level == x.shape[1]

        batch_size = x.shape[0]
        x = rearrange(x, 'b s c x y -> (b s) c x y')
        x = self.encoder(x) # B * S FN
        x = rearrange(x, '(b s) f -> b s f', b=batch_size)
        x, _ = self.lstm(x)
        cls = self.cls_head(x[:, -1, :]).view(batch_size, -1, 3)
        if self.supervise_level:
            level = self.level_head(x)
            return cls, level
        else:
            return cls

#%% DATASET
class SegmentDataset(Dataset):
    def __init__(self, 
                 df,
                 df_co, 
                 data,
                 augment_level=0,
                 window_len=32,
                 step_size=15,
                 total_level=16,
                 image_size=[256, 256],
                 image_size1=[128, 256]):

        super().__init__()
        self.augment_level = augment_level
        self.df = df
        self.df_co = df_co
        self.data = data
        self.image_size = image_size
        self.image_size1 = image_size1
        self.window_len = int(window_len // 2)
        self.step_size = step_size
        self.total_level = total_level

        self.resize = A.Resize(*self.image_size)
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
        return len(self.df_co)

    def __getitem__(self, idx):
        meta = dict(self.df_co.iloc[idx])

        image = self._get_slice(meta)
        image = self.resize(image=image.numpy().astype('f'))['image']

        images = []
        for i in range(0, image.shape[0], self.step_size):
            if i + self.step_size > image.shape[0]:
                break
            images.append(image[i : i + self.step_size, :])
        images_full = []
        for idx, i in enumerate(np.arange(0, len(images), len(images) / self.total_level)):
            images_full.append(images[int(i)].copy())
        images_full = self.augment(images=images_full)['images']
        images_full = torch.stack([torch.tensor(x, dtype=torch.float32) for x in images_full], dim=0)
        images_full = torch.stack([images_full] * 3, dim=1)

        return {
            'image': images_full,
            'label': torch.tensor(meta['label'], dtype=torch.float32),
            #'masks': masks,
            'have_label': torch.tensor(meta['have_keypoints'])
        }
    
    def _get_slice(self, meta):
        volumes, desc, s_ids = self._get_data_ram_or_disk(meta)
        volume = volumes[s_ids.index(str(meta['series_id']))]
        v_min, v_max = volume.min(), volume.max()
        return (volume[int(meta['instance_number'])] - v_min) / (v_max - v_min + 0.00001)
    
    def _get_data_ram_or_disk(self, meta):
        if isinstance(self.data[meta['study_id']], str):
            return datasets.dicom_to_3d_tensors(self.data[meta['study_id']])
        else:
            return self.data[meta['study_id']]

#%% TRAINING
def train_one_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, accelerator):
    running_loss = 0.0
    model.train()
    bar = tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_local_main_process)

    for step, batch in bar:
        # B C X Y
        image = batch['image']
        label = batch['label']
        have_label = batch['have_label']
        optimizer.zero_grad()
        pred_labels = model(image)
        loss = criterion(pred_labels, label, have_label)
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
        image = batch['image']
        label = batch['label']
        have_label = batch['have_label']
        with torch.no_grad():
            pred_label = model(image)
        loss = criterion(pred_label, label, have_label)
        running_loss += (loss.item() - running_loss) / (step + 1)
        bar.set_postfix_str(f'Epoch: {epoch}, valid_loss: {running_loss}')
        accelerator.free_memory()

    accelerator.print(f'Valid loss: {running_loss}')

    if accelerator.is_local_main_process and not IS_LOCAL:
        wandb.log({
            'valid_epoch_loss': running_loss,
            })

def train_one_fold(train_loader, valid_loader, fold_n):
    model = RSNA2dModel(model_name=config['model_name'], fold_n=fold_n)
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
def main(): 
    set_seed(config['seed'])
    df = datasets.get_df()
    df_co = keypoints.df_label_co.copy()
    df_co = df_co[df_co['study_id'].isin(df['study_id'])].reset_index(drop=True)
    df_co = keypoints.get_df_co(df_co, df)


    df_co_ = keypoints.df_label_co.copy()
    df_co_ = df_co_[df_co_['study_id'].isin(df['study_id'])].reset_index(drop=True)
    df_co_ = keypoints.get_only_sagittal_df_co_w_fold(df_co_)
    old_df_co_folds = {s: f for s, f in df_co_[['study_id', 'fold']].values}
    df_co['fold'] = df_co['study_id'].apply(old_df_co_folds.get)

    if not IS_LOCAL:
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