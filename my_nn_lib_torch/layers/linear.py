import numpy as np
import torch
from ..core import BaseModule

class Linear(BaseModule):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
    

        self.w, self.b, self.velocity = self.weight_init()
        self.params_delta = {'dW': None, 'db': None}

    def weight_init(self) -> dict:
        velocity = {}
        # 使用 xavier initalization 初始化 weight
        # source: https://www.numpyninja.com/post/weight-initialization-techniques

        # w 是高斯分佈的隨機數，所以使用 xavier init 時分子為 2
        w = (torch.randn(self.in_features, self.out_features) * np.sqrt(2 / self.in_features)).cuda()
        b = torch.zeros((1, self.out_features)).cuda()
   
        velocity['w'] = torch.zeros_like(w).cuda()
        velocity['b'] = torch.zeros_like(b).cuda()
        return w, b, velocity
    
    def update_params(self, opt_params: dict):
        self.velocity['w'] = opt_params['alpha'] * self.velocity['w'] - opt_params['lr'] * self.params_delta['dW']
        self.velocity['b'] = opt_params['alpha'] * self.velocity['b'] - opt_params['lr'] * self.params_delta['db']
        self.w += self.velocity['w']
        self.b += self.velocity['b']
    
    def forward(self, x) -> dict:
        self.in_feat = x
        I = torch.matmul(x, self.w) + self.b
        return I
    
    def backward(self, delta):
        # delta 是前一層的 dLdZ

        norm_factor = 1 / self.in_feat.shape[0]
        dLdZ = torch.matmul(delta, self.w.T)
        
        # dL/dW
        self.params_delta['dW'] = norm_factor * torch.matmul(self.in_feat.T, delta)
        # dL/db
        self.params_delta['db'] = norm_factor * torch.sum(delta, axis=0, keepdims=True)

        return dLdZ