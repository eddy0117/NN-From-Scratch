import numpy as np
from ..core import BaseModule
import torch
class ReLU(BaseModule):
    def forward(self, x):
        self.in_feat = x
        
        return torch.maximum(x, torch.tensor(0))

    def backward(self, delta):
        return delta * (self.in_feat > 0)
    
   