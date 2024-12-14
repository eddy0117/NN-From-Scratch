import torch

class CrossEntropyLoss:
    def cal_loss(y, label):
        return -torch.sum(label * torch.log(y))
    
class SquareLoss:
    def cal_loss(y, label):
        return torch.sum((1 / 2) * ((y - label) ** 2))
