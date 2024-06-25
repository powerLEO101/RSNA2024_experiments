import timm
import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name=model_name,
                                       pretrained=True,
                                       num_classes=25 * 3,
                                       in_chans=6)
    def forward(self, x):
        x = self.model(x)
        return x