import torch
import torch.nn as nn

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
        total_loss = bce_loss + l2_loss
        return total_loss