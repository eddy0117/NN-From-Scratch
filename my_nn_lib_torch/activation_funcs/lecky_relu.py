import numpy as np
import torch
from ..core import BaseModule

class LeckyReLU(BaseModule):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        self.in_feat = x
        return torch.where(x > 0, x, self.alpha * x)
    
    def backward(self, delta):
        return delta * torch.where(self.in_feat > 0, 1, self.alpha)
