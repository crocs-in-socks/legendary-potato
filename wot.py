import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import *

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *
from ModelArchitecture.Hypernets import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# trainset, validationset, testset = load_dataset('lits:window:0_400', '70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a', ToTensor3D(labeled=True))

# trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)

# for idx, sample_dict in enumerate(tqdm(trainloader)):
#     sample_dict = trainset[idx]
#     image = sample_dict['input']
#     gt = sample_dict['gt']

#     plt.figure(figsize=(20, 15))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image[0, :, :, 64].detach().cpu())
#     plt.subplot(1, 2, 2)
#     plt.imshow(gt[1, :, : , 64].detach().cpu())
#     plt.savefig(f'./temporary/{idx}')
#     plt.close()

data = np.load('../server_stare_indexes.npy', allow_pickle=True).item()

print(data.keys())

# for key in data.keys():
#     sub = data[key]
#     for idx, item in enumerate(sub):
#         item = item.replace('04d05e02-a59c-4a91-8c16-28a8c9f1c14f', '70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a')
#         data[key][idx] = item

# np.save('../brats_2020_indexes.npy', data)

print('Script executed.')