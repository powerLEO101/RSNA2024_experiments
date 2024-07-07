# %%
import os
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A

from tqdm import tqdm
from os import environ
from torch.utils.data import Dataset
from torchvision.transforms import v2
from sklearn.model_selection import KFold
from .project_paths import base_path
from albumentations.pytorch import ToTensorV2
from .utils import display_images, scale_normalize

IS_INFER = False
df_meta_f = pd.read_csv(f'{base_path}/train_series_descriptions.csv')
df_meta_f_ = pd.read_csv(f'{base_path}/test_series_descriptions.csv')
df_label_co = pd.read_csv(f'{base_path}/train_label_coordinates.csv')
df_label_co['instance_number'] = df_label_co['instance_number'].apply(lambda x: x - 1) ### DEBUG!!!
# df_label_co['new_id'] = [f'{x}_{y}' for x, y in zip(df_label_co['study_id'], df_label_co['series_id'])]
# df_label_co = df_label_co.set_index('series_id')
kfold_random_seed = 23

def find_description(study_id, series_id):
    if IS_INFER:
        return df_meta_f_[(df_meta_f_['study_id'] == int(study_id)) & 
                    (df_meta_f_['series_id'] == int(series_id))]['series_description'].iloc[0]
    else:
        return df_meta_f[(df_meta_f['study_id'] == int(study_id)) & 
                    (df_meta_f['series_id'] == int(series_id))]['series_description'].iloc[0]

def get_df():
    df = pd.read_csv(f'{base_path}/train_clean.csv')
    df = df.fillna(-1)
    df = df.replace({'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2})
    df['filepath'] = df['study_id'].map(lambda x: f'{base_path}/train_images/{x}')
    folds = KFold(n_splits=5, shuffle=True, random_state=kfold_random_seed)
    for fold, (train_index, valid_index) in enumerate(folds.split(df)):
        df.loc[valid_index, 'fold'] = int(fold)
    return df

def get_df_infer():
    part_1 = os.listdir(f'../input/rsna-2024-lumbar-spine-degenerative-classification/test_images')
    part_1 = list(filter(lambda x: x.find('.DS') == -1, part_1))
    df = [{'study_id': x, 'filepath': f"../input/rsna-2024-lumbar-spine-degenerative-classification/test_images/{x}"} for x in part_1]
    df = pd.DataFrame(df)
    return df

def dicom_to_3d_tensors(main_folder_path):
    result = []
    desc = []
    series_id = []
    study_id = main_folder_path.split('/')[-1]
    for subfolder in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        dicom_files = [f for f in os.listdir(subfolder_path) if f.endswith('.dcm')]
        if os.path.isfile(os.path.join(subfolder_path, 'data.pt')):
            volume = torch.load(os.path.join(subfolder_path, 'data.pt'))
        else:
            if not dicom_files:
                continue
            dicom_files.sort(key=lambda x: int(x[:-4]))
            first_slice = pydicom.dcmread(os.path.join(subfolder_path, dicom_files[0]))
            img_shape = first_slice.pixel_array.shape
            series_description = subfolder
            num_slices = len(dicom_files)
            volume = torch.zeros((num_slices, *img_shape), dtype=torch.float16)
            for i, file in enumerate(dicom_files):
                ds = pydicom.dcmread(os.path.join(subfolder_path, file))
                x = ds.pixel_array.astype(float)
                if x.shape == volume.shape[1:]:
                    volume[i, :, :] = torch.tensor(x)
                else:
                    volume = volume[:i]
                    break
        result.append(volume)
        desc.append(find_description(study_id, subfolder))
        series_id.append(subfolder)
    return result, desc, series_id

def get_data(df, drop_rate=0.1):
    print('Loading data into RAM')
    data = {}
    for filepath, study_id in tqdm(df[['filepath', 'study_id']].values):
        if np.random.rand() < drop_rate:
            data[study_id] = filepath
        else:
            data[study_id] = dicom_to_3d_tensors(filepath)
    return data

def get_label(meta, label_name: str=None, drop_partial_label=False):
    keys = ['spinal_canal_stenosis_l1_l2',
       'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4',
       'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1',
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
       'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3',
       'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5',
       'left_subarticular_stenosis_l5_s1', 'right_subarticular_stenosis_l1_l2',
       'right_subarticular_stenosis_l2_l3',
       'right_subarticular_stenosis_l3_l4',
       'right_subarticular_stenosis_l4_l5',
       'right_subarticular_stenosis_l5_s1']
    label = []
    for i, name in enumerate(keys):
        if meta[name] == -1:
            if (not drop_partial_label) or label_name in name:
                label.extend([0] * 3)
            continue
        if label_name is not None and not label_name in name:
            if not drop_partial_label:
                label.extend([0] * 3)
            continue
        tmp = [0] * 3
        tmp[int(meta[name])] = 1
        label.extend(tmp)
    if drop_partial_label and len(label) < 30:
        label.extend([0] * 15)
    label = torch.tensor(label, dtype=torch.float32)
    return label

def generate_weights(meta, desc, weights=[1, 2, 4]):
    weights = {idx: x for idx, x in enumerate(weights)}
    result = []
    for one_desc in desc:
        one_desc = {'Sagittal T2/STIR': 'spinal',
                    'Sagittal T1': 'neural',
                    'Axial T2': 'subart'}.get(one_desc)
        verdict = meta[meta.keys().str.contains(one_desc)].max()
        verdict = weights[verdict]
        result.append(verdict)
    result = np.array(result)
    result = result / result.sum()
    return result
        
def get_verdict(meta, meta_file, weights=[1, 2, 4]):
    weights = {idx: x for idx, x in enumerate(weights)}
    study_id = meta['study_id']
    label_name = f"{meta['condition'].replace(' ', '_')}_{meta['level'].replace('/','_')}".lower()
    try:
        verdict = weights[meta_file[label_name]]
    except KeyError:
        verdict = 0
    return verdict

class RSNADataset(Dataset):
    def __init__(self, 
                 df,
                 data,
                 augment_level=0,
                 rough_pos_factor=0,
                 drop_partial_label=True,
                 image_size=[256, 256],
                 severity_weights=[1, 2, 4]):

        super().__init__()
        self.augment_level = augment_level
        self.drop_partial_label = drop_partial_label
        self.rough_pos_factor = rough_pos_factor
        self.df = df
        self.data = data
        self.image_size = image_size
        self.severity_weights = severity_weights
        self.augment_level = augment_level

        if augment_level == 0:
            self.augment = v2.Compose([
                v2.Resize(image_size, interpolation=cv2.INTER_CUBIC)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        meta_file = self.df.iloc[idx]
        image, desc, series_id = self.get_data_ram_or_disk(meta_file)

        #p = generate_weights(meta_file, desc)
        random_index = np.random.choice(len(desc))

        image, desc = image[random_index], desc[random_index]
        min_, max_ = image.min(), image.max()
        meta = df_label_co[df_label_co['series_id'] == int(series_id[random_index])] # require clean data/ all series must have coordinates
        p = np.array([get_verdict(meta.iloc[i], meta_file, weights=self.severity_weights) for i in range(len(meta))])
        if p.sum() == 0:
            p = [1]
        else:
            p = p / p.sum()
        meta = meta.iloc[np.random.choice(len(p), p=p)]

        result = torch.zeros(3, *self.image_size)
        pos = [float(meta['x'] / image.shape[1] * self.image_size[0]), 
               float(meta['y'] / image.shape[2] * self.image_size[1])]

        for i in [-1, 0, 1]:
            current_index = int(meta['instance_number']) + i
            if current_index < 0 or current_index >= image.shape[0]:
                continue

            data = image[current_index].unsqueeze(0)
            data = scale_normalize(data, min=min_, max=max_)
            data = self.augment(data)

            result[i + 1, :, :] = data

        # some post-processing for input data
        label_name = f"{meta['condition'].replace(' ', '_').lower()}".replace('left_', '').replace('right_', '')
        if 'spinal' in label_name:
            label_name = 'spinal'
        elif 'neural' in label_name:
            label_name = 'neural'
        elif 'subart' in label_name:
            label_name = 'subart'
        else:
            print('Error in converting label name. Check whether data is clean?')
        if self.rough_pos_factor != 0:
            pos = [x // self.rough_pos_factor for x in pos]
        label = get_label(meta_file, label_name, self.drop_partial_label)

        return {
            'image': result,
            'label': label,
            'pos': torch.tensor(pos, dtype=torch.float32),
            'head': label_name
        }
    
    def get_data_ram_or_disk(self, meta):
        if isinstance(self.data[meta['study_id']], str):
            return dicom_to_3d_tensors(self.data[meta['study_id']])
        else:
            return self.data[meta['study_id']]
            
class RSNADatasetInfer(RSNADataset):
    def __getitem__(self, idx):

        meta = dict(self.df.iloc[idx])
        image, desc, series_id = self.get_data_ram_or_disk(meta)

        result_all = []
        label_names = []
        for index in range(len(desc)):
            min_, max_ = image[index].min(), image[index].max()
            for instance_number in range(image[index].shape[0]):
                result = torch.zeros(3, 256, 256)
                for i in [-1, 0, 1]:
                    current_index = instance_number + i
                    if current_index < 0 or current_index >= image[index].shape[0]:
                        continue
                    data = image[index][current_index].unsqueeze(0)
                    data = scale_normalize(data, min=min_, max=max_)
                    data = self.augment(data)
                    result[i + 1, :, :] = data
                result_all.append(result)
                label_names.append({'Sagittal T2/STIR': 'spinal',
                                    'Sagittal T1': 'neural',
                                    'Axial T2': 'subart'}.get(desc[index]))
        result_all = torch.stack(result_all, dim=0)

        return {
            'image': result_all,
            'head': label_names
        }

VIEWS = ['Sagittal T2/STIR', 'Sagittal T1', 'Axial T2']
class ThreeViewDataset(RSNADataset):
    def __init__(self, 
                 df, 
                 data,
                 view_slice_count=[10, 10, 20],
                 **kargs):
        super().__init__(df, data, **kargs)
        self.view_slice_count = view_slice_count
        self.total_slice = sum(view_slice_count)
        
        if self.augment_level == 0:
            self.augment = A.ReplayCompose([
                A.Resize(*self.image_size),
                ToTensorV2(),
            ])
        elif self.augment_level == 1:
            self.augment = A.ReplayCompose([
                A.Resize(*self.image_size),
                A.Perspective(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(p=0.5, limit=(-25, 25)),
                ToTensorV2(),
            ])

    def __getitem__(self, idx):
        meta = dict(self.df.iloc[idx])
        image, desc, series_id = self.get_data_ram_or_disk(meta)

        result = torch.zeros(self.total_slice, 3, *self.image_size)
        current_index = 0
        for slice_count, one_view in zip(self.view_slice_count, VIEWS):
            if one_view in desc:
                volume = self._get_n_slice(slice_count, image[desc.index(one_view)])
                volume = self._apply_augment_on_volume(volume)
                result_view = torch.zeros(len(volume), 3, *self.image_size)
                for i in range(len(volume)):
                    for offset in [-1, 0, 1]:
                        if i + offset < 0 or i + offset >= len(volume):
                            continue
                        result_view[i, offset + 1] = volume[i + offset]
                result[current_index : current_index + slice_count] = result_view
            current_index += slice_count
        
        label = get_label(meta)

        return {
            'image': result,
            'label': label
        }

    def _get_n_slice(self, n, x):
        step = len(x) / n
        result = torch.zeros(n, *x.shape[1:])
        norm_param = [x.min(), x.max()]
        for i in range(n):
            one_slice = x[int(i * step)]
            one_slice = scale_normalize(one_slice, *norm_param)
            result[i] = one_slice
        return result
    
    def _apply_augment_on_volume(self, x):
        result = torch.zeros(len(x), *self.image_size)
        transformed = self.augment(image=x[0].numpy())
        replay = transformed['replay']
        result[0] = transformed['image']
        for i in range(1, len(x)):
            transformed = A.ReplayCompose.replay(replay, image=x[i].numpy())
            result[i] = transformed['image']
        return result
