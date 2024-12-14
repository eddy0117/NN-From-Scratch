import numpy as np
import torch
from ..core import BaseModule

class Flatten(BaseModule):
    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def forward(self, x):
        '''
        x: (N, C, H, W),
        out: (N, C*H*W)
        '''
        self.input_shape = x.shape
        self.output_shape = (self.input_shape[0], np.prod(self.input_shape[1:]))
        
        return x.reshape(self.output_shape)

    def backward(self, delta):
        '''
        delta: (N, C*H*W) -reshape-> (N, C, H, W)
        '''
        return delta.reshape(self.input_shape)
       
    
