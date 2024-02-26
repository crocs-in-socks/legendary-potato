import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from ModelArchitecture.Encoders import *

class HyperNet(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target

        self.device = 'cuda:1'

        number_of_target_weights = [p.numel() for p in target.parameters()]
        total_target_weights = sum(number_of_target_weights)

        self.hyper_encoder = ResNet3D_Encoder(image_channels=1).to(self.device)

        # self.generator = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(128*4*4*4, total_target_weights // 16),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(total_target_weights // 16, total_target_weights // 8),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(total_target_weights // 8, total_target_weights // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(total_target_weights // 4, total_target_weights // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(total_target_weights // 2, total_target_weights)
        # )

        ### head input shape = (128, 4, 4, 4)

        print('Initializing model.')
        self.hyper_heads = []
        for size in tqdm(number_of_target_weights):
            self.hyper_heads.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(128*4*4*4, 128*4),
                nn.ReLU(inplace=True),
                nn.Linear(128*4, 128*4),
                nn.ReLU(inplace=True),
                nn.Linear(128*4, size)
            ))
        print()

    def forward(self, x):

        _, encoded = self.hyper_encoder(x)

        for head, param in (zip(self.hyper_heads, self.target.parameters())):
            head = head.to(self.device)
            weights = head(encoded)
            matrix = torch.reshape(weights, param.shape)
            param.data = matrix.to(self.device)
            head = head.cpu()
        
        prediction, _ = self.target(x)
        return prediction

        # offset = 0
        # for param in enumerate(tqdm(self.target.parameters())):

        #     number_of_weights = param.numel()

        #     weights = target_weights[offset:offset+number_of_weights]
        #     offset += number_of_weights

        #     matrix = torch.reshape(weights, param.size())
        #     if param.shape == matrix.shape:
        #         param.data = matrix.clone().detach()
        #     else:
        #         print('Skip')

        # prediction, _ = self.target(x)
        # return prediction
        pass