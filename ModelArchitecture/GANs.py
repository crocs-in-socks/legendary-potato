import torch
import torch.nn as nn
import torch.nn.functional as F

import random

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class ReplayBuffer():
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        
        return torch.cat(to_return)

class LambdaLR():
    def __init__(self, num_epochs, offset, decay_start_epoch):
        self.num_epochs = num_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.num_epochs - self.decay_start_epoch)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm3d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm3d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)

class CycleGAN_Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)
    
class CycleGAN_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, residual_blocks=9):
        super().__init__()

        model = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [
                nn.Conv3d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features*2
        
        for _ in range(residual_blocks):
            model += [ResidualBlock(in_features)]
        
        out_features = in_features//2
        for _ in range(2):
            model += [
                nn.ConvTranspose3d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm3d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features//2
        
        model += [
            nn.ReflectionPad3d(3),
            nn.Conv3d(64, out_channels, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)