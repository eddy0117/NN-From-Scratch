import numpy as np
from abc import ABC, abstractmethod

class BaseModule(ABC):
    def __init__(self):
        # in_feat 是forward 的 input x
        self.in_feat = None
        self.params_delta = {'dW': None, 'db': None}
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, delta):
        pass
    
    def update_params(self, opt_params):
        pass


# class MyLayerModule(BaseModule):
