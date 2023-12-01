import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from ModelArchitecture.Encoders import *
from ModelArchitecture.DUCK_Net import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
# from ModelArchitecture.Losses import *

import os
import json
import glob
import numpy as np
from tqdm import tqdm

DUCK_model = DuckNet(input_channels=1, out_classes=2, starting_filters=17)
ResNet18_model = ResNet3D_Encoder(image_channels=1)
projector_model = Projector(num_layers=4, layer_sizes=[64, 128, 256, 512])

DUCK_params = sum(p.numel() for p in DUCK_model.parameters() if p.requires_grad)
ResNet18_params = sum(p.numel() for p in ResNet18_model.parameters() if p.requires_grad)
projector_params = sum(p.numel() for p in projector_model.parameters() if p.requires_grad)

print(f'DUCK: {DUCK_params/1000000}M\tResNet18: {ResNet18_params/1000000}M\t Projec')

def determine_class_accuracy(pred, target):
    pred_vect = (pred > 0.5).float()
    target_vect = (target > 0.5).float()

    for i in range(pred_vect.shape[0]):
        print(pred_vect[i].data, pred[i].data, target_vect[i].data, target[i].data)

    correct_cases = (pred_vect == target_vect)
    true_pos = torch.sum(correct_cases)

    accuracy = true_pos/(pred_vect.shape[0] * pred_vect.shape[1])
    return accuracy

out = torch.tensor([[0.6, 0.8, 0, 0, 0.4]]).float()
label = torch.tensor([[1, 0, 0, 0, 1]]).float()

accuracy = determine_class_accuracy(out, label)
print(f'Accuracy: {accuracy}')