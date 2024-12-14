import numpy as np
import torch
from ..core import BaseModule

class Softmax(BaseModule):
    # 限搭配 CrossEntropyLoss 使用
    def forward(self, x):
        exp_x = torch.exp(x)
        self.activations = exp_x / torch.sum(exp_x, axis=1, keepdims=True)
        return self.activations
    
    def backward(self, delta):
        # 通常為 output layer 的 activation func
        # 將 Softmax 與 CrossEntropyLoss 的導數合併在 delta 項計算
        # 所以 activateion 項係數為 1

    
 
        return (self.activations - delta) * 1

