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
