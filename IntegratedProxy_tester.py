import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

import numpy as np
from tqdm import tqdm

batch_size = 1
patience = 15
num_workers = 16
device = 'cuda:0'
number_of_epochs = 100
date = '05_12_2023'

proxy_encoder_path = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/Dec04/VGGproxy_encoder_weightedBCEPbatch12_then_VoxCFT_brainmask_04_12_2023_state_dict100.pth'
proxy_projector_path = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/Dec04/VGGproxy_projector_weightedBCEPbatch12_then_VoxCFT_brainmask_04_12_2023_state_dict100.pth'

segmentation_path = './ModelArchitecture/unet_wts_proxy.pth'

composed_transform = transforms.Compose([
        ToTensor3D(labeled=True)
    ])

test_data = np.load('../wmh_indexes.npy', allow_pickle=True).item()
testset = ImageLoader3D(paths=test_data['test_names'], gt_paths=None, json_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)

testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

proxy_encoder = VGG3D_Encoder(input_channels=1).to(device)
proxy_encoder.load_state_dict(torch.load(proxy_encoder_path))
proxy_projector = Projector(num_layers=5, layer_sizes=[32, 64, 128, 256, 512], test=True).to(device)
proxy_projector.load_state_dict(torch.load(proxy_projector_path))

segmentation_model = SA_UNet().to(device)
segmentation_model.load_state_dict(torch.load(segmentation_path), strict=False)

test_loss = 0
test_dice = 0
test_f1_accuracy = 0

proxy_encoder.eval()
proxy_projector.eval()
segmentation_model.eval()

print()
print('Testing Integrated model.')

for idx, data in enumerate(tqdm(testloader), 0):
    with torch.no_grad():
        image = data['input'].to(device)
        gt = data['gt'].to(device)

        to_projector, _ = proxy_encoder(image)
        combined_projection, projection_maps = proxy_projector(to_projector)

        # combined_projection = combined_projection * -1
        # for idx, map in enumerate(projection_maps):
        #     projection_maps[idx] = map * -1
        segmentation = segmentation_model(image, projection_maps)

        dice = Dice_Score(segmentation[0].cpu().numpy(), gt[:,1].cpu().numpy())
        f1_acc = F1_score(segmentation[0].cpu().numpy(), gt[:,1].cpu().numpy())
        test_dice += dice.item()
        test_f1_accuracy += f1_acc.item()

        combined_projection = F.interpolate(combined_projection, size=(128, 128, 128))

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 4, 1)
        plt.imshow(image[0, 0, :, :, 64].detach().cpu())
        plt.title(f'Input Sample #{idx}')
        plt.colorbar()
        plt.subplot(1, 4, 2)
        plt.imshow(gt[0, 1, :, :, 64].detach().cpu())
        plt.title(f'GT #{idx}')
        plt.colorbar()
        plt.subplot(1, 4, 3)
        plt.imshow(segmentation[0, 0, :, :, 64].detach().cpu())
        plt.title(f'Segmentation output #{idx}')
        plt.colorbar()
        plt.subplot(1, 4, 4)
        plt.imshow(combined_projection[0, 0, :, :, 64].detach().cpu())
        plt.title(f'Projection map #{idx}')
        plt.colorbar()
        plt.savefig(f'./temporary/sample#{idx}')
        plt.close()
    
test_dice /= len(testloader)
test_f1_accuracy /= len(testloader)
print(f'Test dice score: {test_dice}')
print(f'Test f1 accuracy: {test_f1_accuracy}')

print()
print('Script executed.')