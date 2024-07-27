import torch
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp

def mask_from_keypoint(x, y, length, image_x, image_y):
    # reference: Object as Points paper, Gaussian kernel. Could be inaccruate
    def in_bound(a, b):
        if a < 0 or a >= image_x:
            return False
        if b < 0 or b >= image_y:
            return False
        return True

    result = torch.zeros(image_x, image_y)
    for i in range(-length + 1, length):
        for j in range(-length + 1, length):
            if in_bound(x + i, y + j):
                result[x + i, y + j] = np.exp(-1 * (i ** 2 + j ** 2) / length)
    return result

class TwoWayLoss(nn.Module):
    def __init__(self, w1=1., w2=1.):
        super(TwoWayLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l2_loss = nn.MSELoss()
        self.w1 = w1
        self.w2 = w2

    def forward(self, predictions, targets):
        pred1, pred2 = predictions
        target1, target2 = targets
        bce_loss = self.bce_loss(pred1, target1)
        l2_loss = self.l2_loss(pred2, target2)
        total_loss = bce_loss * self.w1 + l2_loss * self.w2
        return total_loss, bce_loss, l2_loss
    
class PerLabelCrossEntropyLoss(nn.Module):
    def __init__(self, weight=[1., 1., 1.]):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(weight))
        self.n_label = 25
    
    def forward(self, p, y):
        loss = 0
        for i in range(self.n_label):
            p_, y_ = p[:, i * 3 : (i + 1) * 3], y[:, i * 3 : (i + 1) * 3]
            loss = loss + self.cross_entropy(p_, y_) / self.n_label
        return loss

class PerLevelDiceLoss(nn.Module):
    def __init__(self, device, from_keypoint=True, length=15, image_size=[256, 256]):
        super().__init__()
        self.loss = smp.losses.DiceLoss(mode='multilabel')
        self.from_keypoint = from_keypoint
        self.length = length
        self.image_size = image_size
        self.device = device

    def forward(self, preds, labels, have_labels): # possible slow because need to process one by one in batch?
        loss = []
        for pred, label, have_label in zip(preds, labels, have_labels):
            pred = pred[have_label]
            label = label[have_label]
            true_label = []
            if self.from_keypoint:
                for i in range(len(label)):
                    y, x = label[i] # inverted in numpy
                    true_label.append(mask_from_keypoint(x, y, self.length, *self.image_size).to(self.device))
                label = torch.stack(true_label, dim=0)
            loss.append(self.loss(pred, label))
        
        loss = torch.stack(loss, dim=0).mean()
                
        return loss