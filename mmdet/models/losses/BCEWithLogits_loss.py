import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weighted_loss


@LOSSES.register_module
class WeightedMultilabelLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(WeightedMultilabelLoss, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = self.loss_weight * self.cerition(pred, target)
        return loss.mean()
