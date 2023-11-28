import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from ModelArchitecture.Encoders import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import os
import json
import glob
import numpy as np
from tqdm import tqdm

batch_size = 1
device = 'cuda:1'

DUCKmodel_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/Duck1wmh_focal + dice_state_dict_best_loss97.pth'

from_Sim1000_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*FLAIR.nii.gz'))
from_sim2211_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/brats/**/*FLAIR.nii.gz'))

from_Sim1000_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*mask.nii.gz'))
from_sim2211_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/brats/**/*mask.nii.gz'))

from_Sim1000_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*.json'))
from_sim2211_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/brats/**/*.json'))

composed_transform = transforms.Compose([
        ToTensor3D(labeled=True)
    ])

from_Sim1000 = ImageLoader3D(paths=from_Sim1000_data_paths, gt_paths=from_Sim1000_gt_paths, json_paths=from_Sim1000_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)

from_sim2211 = ImageLoader3D(paths=from_sim2211_data_paths,gt_paths=from_sim2211_gt_paths, json_paths=from_sim2211_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)

fullset = ConcatDataset([from_Sim1000, from_sim2211])

train_size = int(0.7 * len(fullset))
validation_size = int(0.1 * len(fullset))
test_size = len(fullset) - (train_size + validation_size)

trainset, validationset, testset = random_split(fullset, (train_size, validation_size, test_size))

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=0)

for data in trainloader:
    print(type(data))
    exit(0)

# unique_sar = []

# for idx, path in enumerate(from_sim2211_json_paths):

#     with open(path, 'r') as file:
#         sample = json.load(file)
#         # for key in sample.keys():
#         #     print(key, sample[key])
#         # exit(0)
#         num_lesions = sample['num_lesions']
#         # print(f'idx: {idx}')
#         for lesion_idx in range(num_lesions):
#             if sample[f'{lesion_idx}_semi_axes_range'] not in unique_sar:
#                 unique_sar.append(sample[f'{lesion_idx}_semi_axes_range'])
#             # print(sample[f'{lesion_idx}_semi_axes_range'])

# print(unique_sar)