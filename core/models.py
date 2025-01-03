import timm
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from einops import rearrange, pack, unpack

class BaselineModel(nn.Module):
    def __init__(self, model_name, in_chans=6):
        super().__init__()
        self.model = timm.create_model(model_name=model_name,
                                       pretrained=True,
                                       num_classes=25 * 3,
                                       in_chans=in_chans)
    def forward(self, x):
        x = self.model(x)
        return x


class ThreeViewModel(nn.Module):
    def __init__(self, 
                 model_name, 
                 in_chans=3, 
                 global_pool='avg',
                 view_slice_count=[10, 10, 20],
                 head_dropout_rate=0.1,
                 out_feature_divide=2):
        super().__init__()
        base_model = timm.create_model(model_name=model_name,
                                       pretrained=True,
                                       in_chans=in_chans,
                                       global_pool=global_pool)
        try:
            in_features = base_model.fc.in_features
        except:
            in_features = base_model.classifier.in_features
        
        layers = list(base_model.children())[:-1]
        self.encoder = nn.Sequential(*layers)
        self.lstm = []
        self.heads = []
        for i in range(3):
            self.lstm.append(nn.LSTM(in_features, in_features // out_feature_divide, bidirectional=True, batch_first=True))
            self.heads.append(nn.Sequential(
                #nn.Dropout(head_dropout_rate),
                nn.Linear(in_features // out_feature_divide * 2 * view_slice_count[i], 30 if i != 0 else 15)
            ))
        self.lstm = nn.ModuleList(self.lstm)
        self.heads = nn.ModuleList(self.heads)
        
        self.view_slice_count = view_slice_count
        self.out_feature_divide = out_feature_divide

    def forward(self, x):
        B, S, C, X, Y = x.shape
        x = rearrange(x, 'b s c x y -> (b s) c x y')
        features = self.encoder(x)
        features = rearrange(features, '(b s) f -> b s f', b=B, s=S)
        current_slice = 0
        result = []
        for i in range(3):
            view_features = features[:, current_slice : current_slice + self.view_slice_count[i], :]
            view_features, _ = self.lstm[i](view_features)
            view_features = rearrange(view_features, 'b s f -> b (s f)')
            result.append(self.heads[i](view_features))
            current_slice += self.view_slice_count[i]
        result, _ = pack(result, 'b *')
        return result
    
    def _to_device(self, device):
        pass
        # for i in range(3):
        #     self.lstm[i].to(device)
        #     self.heads[i].to(device)


class ThreeViewModel_old(nn.Module):
    def __init__(self, model_name, in_chans=3, global_pool='avg', predict_bb=True):
        super().__init__()
        base_model = timm.create_model(model_name=model_name,
                                       pretrained=True,
                                       in_chans=in_chans,
                                       global_pool=global_pool)
        try:
            in_features = base_model.fc.in_features
        except:
            in_features = base_model.classifier.in_features
        
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.spinal_head = nn.Linear(in_features, 30)
        self.neural_head = nn.Linear(in_features, 30)
        self.subart_head = nn.Linear(in_features, 30)
        bb_head_in_features = in_features * (self.encoder(torch.rand(1, 3, 256, 256)).shape[-1] ** 2)
        self.bb_head = nn.Linear(bb_head_in_features, 2) # TODO bb head could be improved for each diagnoses
        self.predict_bb = predict_bb

    def forward(self, x, heads):
        # X: [B, C, X, Y]
        assert len(x.shape) == 4
        features = self.encoder(x)
        features_pooled = self.global_pool(features)
        features_pooled = features_pooled.flatten(start_dim=1)
        features = features.flatten(start_dim=1)
        result = []
        for head, feature in zip(heads, features_pooled):
            if head == 'spinal':
                result.append(self.spinal_head(feature.unsqueeze(0)))
            elif head == 'neural':
                result.append(self.neural_head(feature.unsqueeze(0)))
            elif head == 'subart':
                result.append(self.subart_head(feature.unsqueeze(0)))
            else:
                print('Error in model, cannot find head')
        result = torch.cat(result, dim=0)
        if self.predict_bb:
            result_co = self.bb_head(features)
            return result, result_co
        else:
            return result

class SegmentationModel(nn.Module):
    def __init__(self, pool_kernel_size=None):
        super().__init__()
        backbone = smp.Unet('resnet18', classes=10)
        self.backbone = backbone
        if pool_kernel_size is None:
            self.pool = None
        else:
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size)
    
    def forward(self, x):
        x = self.backbone(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

class RSNAModel(nn.Module):
    def __init__(self, model_name):
        base_model = timm.create_model(model_name=model_name,
                                       pretrained=True)
        try:
            in_features = base_model.fc.in_features
        except:
            in_features = base_model.classifier.in_features
        
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.cls_head = nn.Linear(in_features, 75)
        self.reg_head = nn.Linear(in_features, 10)
    
    def forward(self, x):
        x = self.encoder(x)
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        return cls, reg