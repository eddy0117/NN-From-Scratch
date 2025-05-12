import torch

from ..core import BaseOptimizer

class Momentum(BaseOptimizer):
    def __init__(self, layers, lr=0.01, alpha=0.9):
        super().__init__(lr)
        self.alpha = alpha
        self.v_w = []
        self.v_b = []
        
        for idx, layer in enumerate(layers):
            self.v_w[idx] = self.alpha * self.v_w[idx] - self.lr * layer.params_delta['dW']
            self.v_b[idx] = self.alpha * self.v_b[idx] - self.lr * layer.params_delta['db']
           

    def update_params(self, layers):
        if len(self.v_w) == 0:
            self.v_w = [torch.zeros_like(layer.w).cuda() for layer in layers]
            self.v_b = [torch.zeros_like(layer.b).cuda() for layer in layers]

        for idx, layer in enumerate(layers):
            self.v_w[idx] = self.alpha * self.v_w[idx] - self.lr * layer.params_delta['dW']
            self.v_b[idx] = self.alpha * self.v_b[idx] - self.lr * layer.params_delta['db']

        return 