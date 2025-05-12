import torch
import numpy as np
from abc import ABC, abstractmethod
from .base import BaseModule
class BaseOptimizer(ABC):
    def __init__(self, lr=0.01):
        self.lr = lr

    @abstractmethod
    def update_params(self):
        pass

   