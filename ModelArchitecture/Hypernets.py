import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNet(nn.Module):
    def __init__(self, main_network):
        super().__init__()
        self.main_network = main_network

    def forward(self, x):
        pass