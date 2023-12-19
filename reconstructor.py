import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from Utilities.Generic import *

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
import numpy as np
from tqdm import tqdm

c = Constants(
    batch_size = 1,
    patience = 5,
    num_workers = 8,
    number_of_epochs = 100,
    date = '13_12_2023',
    to_save_folder = 'Dec13',
    to_load_folder = 'Dec13',
    device = 'cuda:1',
    proxy_type = 'UNETproxy_',
    train_task = 'reconstruction_simulated_noise_bg_>_sim_wmh_&_sim_brats_healthy_occluded',
    to_load_encoder_path = None,
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'sim_2211_brats',
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))

testloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)

model = UNet(out_channels=1).to(c.device)
model.load_state_dict(torch.load(f'{c.to_save_folder}UNETproxy_reconstruction_simulated_noise_bg_>_sim_wmh_&_sim_brats_healthy_occluded_13_12_2023_state_dict30.pth'))

print()
print('Testing reconstruction.')

model.eval()
count = 1

for data in tqdm(testloader):
    image = data['input'].to(c.device)
    gt = data['gt'].to(c.device)
    masked_image = (image[:, 0] * gt[:, 1]).unsqueeze(1)

    with torch.no_grad():
        reconstruction = model(masked_image)

    plt.figure(figsize=(20, 15))
    plt.subplot(1, 4, 1)
    plt.imshow(image[0, 0, :, :, 64].detach().cpu())
    plt.colorbar()
    plt.title('Image')
    plt.subplot(1, 4, 2)
    plt.imshow(gt[0, 1, :, :, 64].detach().cpu())
    plt.colorbar()
    plt.title('gt')
    plt.subplot(1, 4, 3)
    plt.imshow(masked_image[0, 0, :, :, 64].detach().cpu())
    plt.colorbar()
    plt.title('masked_image')
    plt.subplot(1, 4, 4)
    plt.imshow(reconstruction[0, 0, :, :, 64].detach().cpu())
    plt.colorbar()
    plt.title('reconstruction')
    plt.savefig(f'./temporary/count#{count}')
    plt.close()

    count += 1

    del image
    del gt
    del reconstruction


print()
print('Script executed.')