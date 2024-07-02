import timm
import torch
import torch.nn as nn

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
    def __init__(self, model_name, in_chans=3, global_pool='avg'):
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
        self.spinal_head = nn.Linear(in_features, 30)
        self.neural_head = nn.Linear(in_features, 30)
        self.subart_head = nn.Linear(in_features, 30)
        self.bb_head = nn.Linear(in_features, 2)

    def forward(self, x, heads):
        # X: [B, C, X, Y]
        assert len(x.shape) == 4
        features = self.encoder(x)
        result = []
        for head, feature in zip(heads, features):
            if head == 'spinal':
                result.append(self.spinal_head(feature.unsqueeze(0)))
            elif head == 'neural':
                result.append(self.neural_head(feature.unsqueeze(0)))
            elif head == 'subart':
                result.append(self.subart_head(feature.unsqueeze(0)))
            else:
                print('Error in model, cannot find head')
        result = torch.cat(result, dim=0)
        result_co = self.bb_head(features)
        return result, result_co
