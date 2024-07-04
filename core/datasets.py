# %%
import os
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd

from tqdm import tqdm
from os import environ
from torch.utils.data import Dataset
from torchvision.transforms import v2
from sklearn.model_selection import KFold
from .project_paths import base_path
from .utils import display_images

IS_INFER = False
df_meta_f = pd.read_csv(f'{base_path}/train_series_descriptions.csv')
df_meta_f_ = pd.read_csv(f'{base_path}/test_series_descriptions.csv')
df_label_co = pd.read_csv(f'{base_path}/train_label_coordinates.csv')
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
    df = pd.read_csv(f'{base_path}/train.csv')
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

def get_label(meta, label_name=None, drop_partial_label=False):
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
    if label_name == -1:
        if drop_partial_label:
            return torch.zeros(30)
        else:
            return torch.zeros(75)
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

class Normalizer(object):
    def __init__(self, n_type):
        self.n_type = n_type
        if n_type == 'scale':
            self.n_fun = self.scale_normalize
        else:
            self.n_fun = self.no_normalize
    
    def __call__(self, x, **kargs):
        return self.n_fun(x, **kargs)
    
    def scale_normalize(self, x, min, max):
        x = (x - min) / (max - min)
        return x
    
    def no_normalize(self, x):
        return x


class ImageAugmentor(object):
    def __init__(self, method):
        self.method = method
        if method == 'baseline':
            self.augment = v2.Compose([
                v2.Resize([256, 256], interpolation=cv2.INTER_CUBIC)
            ])
        else:
            self.augment = v2.Compose([v2.Identity()])
    
    def __call__(self, x):
        return self.augment(x)

class ImagePreprocessor(object):
    def __init__(self, 
                 method='baseline',
                 normalize='scale',
                 augment='baseline'):
        self.method = method
        self.normalize = Normalizer(normalize)
        self.augment = ImageAugmentor(augment)
        self.is_infer = IS_INFER
        if method == 'baseline':
            self.preprocess = self.baseline_preprocess
        elif method == '10slice':
            self.preprocess = self.baseline_v2_preprocess
        elif method == 't004':
            self.preprocess = self.t004_preprocess
        elif method == 't006':
            self.preprocess = self.t006_preprocess
        else:
            self.preprocess = self.no_preprocess
    
    def __call__(self, *args):
        return self.preprocess(*args)
    
    def no_preprocess(self, x, desc):
        return x
    
    def baseline_preprocess(self, x, desc):
        result = torch.zeros(6, 256, 256)
        if 'Sagittal T1' in desc:
            data = x[desc.index('Sagittal T1')]
            min, max = data.min(), data.max()
            data_ = data[int(len(data) / 2) - 1 : int(len(data) / 2) + 2]
            data_ = self.normalize(data_, min=min, max=max)
            data_ = self.augment(data_)
            result[:3, :, :] = data_
        if 'Sagittal T2/STIR' in desc:
            data = x[desc.index('Sagittal T2/STIR')]
            min, max = data.min(), data.max()
            data_ = data[int(len(data) / 2) - 1 : int(len(data) / 2) + 2]
            data_ = self.normalize(data_, min=min, max=max)
            data_ = self.augment(data_)
            result[3:6, :, :] = data_
        return result
    
    def baseline_v2_preprocess(self, x, desc):
        result = torch.zeros(20, 256, 256)
        if 'Sagittal T1' in desc:
            data = x[desc.index('Sagittal T1')]
            min_, max_ = data.min(), data.max()
            data_ = data[max(int(len(data) / 2) - 5, 0) : int(len(data) / 2) + 5]
            data_ = self.normalize(data_, min=min_, max=max_)
            data_ = self.augment(data_)
            result[0:data_.shape[0], :, :] = data_
        if 'Sagittal T2/STIR' in desc:
            data = x[desc.index('Sagittal T2/STIR')]
            min_, max_ = data.min(), data.max()
            data_ = data[max(int(len(data) / 2) - 5, 0) : int(len(data) / 2) + 5]
            data_ = self.normalize(data_, min=min_, max=max_)
            data_ = self.augment(data_)
            result[10:10 + data_.shape[0], :, :] = data_
        return result
    
    def t004_preprocess(self, x, desc, series_id):
        if self.is_infer:
            result_all = []
            for index in range(len(desc)):
                min_, max_ = x[index].min(), x[index].max()
                for instance_number in range(x[index].shape[0]):
                    result = torch.zeros(3, 256, 256)
                    for i in [-1, 0, 1]:
                        current_index = instance_number + i
                        if current_index < 0 or current_index >= x[index].shape[0]:
                            continue
                        data = x[index][current_index].unsqueeze(0)
                        data = self.normalize(data, min=min_, max=max_)
                        data = self.augment(data)
                        result[i + 1, :, :] = data
                    result_all.append(result)
            result_all = torch.stack(result_all, dim=0)
            return result_all, -1

        random_index = np.random.randint(len(desc))
        min_, max_ = x[random_index].min(), x[random_index].max()
        meta = df_label_co[df_label_co['series_id'] == int(series_id[random_index])]
        result = torch.zeros(3, 256, 256)
        if len(meta) == 0: # 2 studies have diagnoses without label coor
            return result, -1
        meta = meta.sample(1)
        for i in [-1, 0, 1]:
            current_index = int(meta['instance_number'].values[0]) + i
            if current_index < 0 or current_index >= x[random_index].shape[0]:
                continue
            data = x[random_index][current_index].unsqueeze(0)
            data = self.normalize(data, min=min_, max=max_)
            data = self.augment(data)
            result[i + 1, :, :] = data
        label_name = f"{meta['condition'].values[0].replace(' ', '_').lower()}_{meta['level'].values[0].replace('/', '_').lower()}"
        return result, label_name

    def t006_preprocess(self, x, desc, series_id):
        if self.is_infer:
            result_all = []
            label_names = []
            for index in range(len(desc)):
                min_, max_ = x[index].min(), x[index].max()
                for instance_number in range(x[index].shape[0]):
                    result = torch.zeros(3, 256, 256)
                    for i in [-1, 0, 1]:
                        current_index = instance_number + i
                        if current_index < 0 or current_index >= x[index].shape[0]:
                            continue
                        data = x[index][current_index].unsqueeze(0)
                        data = self.normalize(data, min=min_, max=max_)
                        data = self.augment(data)
                        result[i + 1, :, :] = data
                    result_all.append(result)
                    label_names.append({'Sagittal T2/STIR': 'spinal',
                                        'Sagittal T1': 'neural',
                                        'Axial T2': 'subart'}.get(desc[index]))
            result_all = torch.stack(result_all, dim=0)
            return result_all, label_names, 0, 0

        random_index = np.random.randint(len(desc))
        min_, max_ = x[random_index].min(), x[random_index].max()
        meta = df_label_co[df_label_co['series_id'] == int(series_id[random_index])]
        result = torch.zeros(3, 256, 256)
        if len(meta) == 0: # 2 studies have diagnoses without label coor
            return result, 'spinal_canal_stenosis', 0., 0. # spinal canal steo is a placeholder, could by anything
        meta = meta.sample(1)
        original_size = x[random_index].shape
        pos_x, pos_y = float(meta['y'].iloc[0] / original_size[1] * 256), float(meta['x'].iloc[0] / original_size[2] * 256)# xy is inverted in numpy
        for i in [-1, 0, 1]:
            current_index = int(meta['instance_number'].values[0]) + i
            if current_index < 0 or current_index >= x[random_index].shape[0]:
                continue
            data = x[random_index][current_index].unsqueeze(0)
            data = self.normalize(data, min=min_, max=max_)
            data = self.augment(data)
            result[i + 1, :, :] = data
        label_name = f"{meta['condition'].values[0].replace(' ', '_').lower()}".replace('left_', '').replace('right_', '')
        return result, label_name, pos_x, pos_y


class RSNADataset(Dataset):
    def __init__(self, 
                 df,
                 data,
                 method='baseline', 
                 normalize='scale', 
                 augment='baseline',
                 exact_pos=True):
        super().__init__()
        self.method = method
        self.normalize = normalize
        self.augment = augment

        self.df = df
        self.data = data
        self.preprocess = ImagePreprocessor(method, normalize, augment)
        self.drop_partial_label = False
        self.is_infer = IS_INFER
        self.exact_pos = exact_pos

        if method == 't006':
            self.drop_partial_label = True
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx): # every subject has at least one saggital view
        meta = dict(self.df.iloc[idx])
        image, desc, series_id = self.get_data_ram_or_disk(meta)
        if self.method == 't004':
            image, label_name = self.preprocess(image, desc, series_id)
        elif self.method == 't006':
            image, label_name, pos_x, pos_y = self.preprocess(image, desc, series_id)
        else:
            image = self.preprocess(image, desc)
        
        if self.is_infer:
            return {
                'image': image.unsqueeze(0),
                'name': meta['study_id']
            }
        elif self.method == 't006':
            label = get_label(meta, label_name, self.drop_partial_label)
            if 'spinal' in label_name:
                label_name = 'spinal'
            elif 'neural' in label_name:
                label_name = 'neural'
            elif 'subart' in label_name:
                label_name = 'subart'
            if self.exact_pos:
                return {
                    'image': image,
                    'label': label,
                    'pos': torch.tensor([pos_x, pos_y], dtype=torch.float32),
                    'head': label_name
                }
            else:
                return {
                    'image': image,
                    'label': label,
                    'pos': torch.tensor([pos_x // 32, pos_y // 32], dtype=torch.float32),
                    'head': label_name
                }
        else:
            label = get_label(meta, label_name, self.drop_partial_label)
            return {
                'image': image,
                'label': label,
            }
    
    def get_data_ram_or_disk(self, meta):
        if isinstance(self.data[meta['study_id']], str):
            return dicom_to_3d_tensors(self.data[meta['study_id']])
        else:
            return self.data[meta['study_id']]
            