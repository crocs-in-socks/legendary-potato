import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
device = 'cuda:1'
number_of_epochs = 100
date = '07_12_2023'

proxy_type = 'randomBGUNET_CascadeStyle'
proxy_encoder_type = 'randomBGUNET_CascadeStyle_encoder'
proxy_projector_type = 'randomBGUNET_CascadeStyle_projector'

save_model_path = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/Dec07/'
# proxy_encoder_path = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/Dec06/UNETcopy_encoder_VoxCFT18000_randomBG_06_12_2023_state_dict40.pth'
# proxy_projector_path = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/Dec06/UNETcopy_projector_VoxCFT18000_randomBG_06_12_2023_state_dict40.pth'

segmentation_path = './ModelArchitecture/unet_focal + dice_state_dict_best_loss85.pth'
proxy_encoder_path = './ModelArchitecture/unet_focal + dice_state_dict_best_loss85.pth'

composed_transform = transforms.Compose([
        ToTensor3D(labeled=True)
    ])

data = np.load('../wmh_indexes.npy', allow_pickle=True).item()
trainset = ImageLoader3D(paths=data['train_names'], gt_paths=None, json_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)

validationset = ImageLoader3D(paths=data['val_names'], gt_paths=None, json_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)

testset = ImageLoader3D(paths=data['test_names'], gt_paths=None, json_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

proxy_encoder = SA_UNet_Encoder(out_channels=2).to(device)
proxy_encoder.load_state_dict(torch.load(proxy_encoder_path)['model_state_dict'], strict=False)

proxy_projector = Projector(num_layers=4, layer_sizes=[64, 128, 256, 512], test=True).to(device)

segmentation_model = SA_UNet(out_channels=2).to(device)
segmentation_model.load_state_dict(torch.load(segmentation_path)['model_state_dict'], strict=False)

num_voxels = 10500
segmentation_criterion = DiceLoss().to(device)
proxy_critertion = VoxelwiseSupConLoss_inImage(device=device, num_voxels=num_voxels).to(device)

proxy_optimizer = optim.Adam([*proxy_encoder.parameters(), *proxy_projector.parameters()], lr = 0.0001, eps = 0.0001)

proxy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(proxy_optimizer, mode='max', patience=5, factor=0.5, verbose=True)

for parameter in segmentation_model.parameters():
    parameter.requires_grad = False

train_dice_loss_list = []
train_dice_score_list = []
train_f1_accuracy_list = []

best_validation_score = None
validation_dice_loss_list = []
validation_dice_score_list = []
validation_f1_accuracy_list = []

print()
print('Training Cascade Style.')

for epoch in range(1, number_of_epochs+1):
    
    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    proxy_encoder.train()
    proxy_projector.train()
    segmentation_model.eval()

    train_dice_loss = 0
    train_dice_score = 0
    train_f1_accuracy = 0

    for data in tqdm(trainloader):
        image = data['input'].to(device)
        gt = data['gt'].to(device)

        to_projector, _ = proxy_encoder(image)
        # _, projection_maps = proxy_projector(to_projector)
        final_out, projection_maps = proxy_projector(to_projector)
        segmentation = segmentation_model(final_out, projection_maps)

        projection_maps = [F.interpolate(projection_map, size=(128, 128, 128), mode='trilinear') for projection_map in projection_maps]
        projection_maps = torch.cat(projection_maps, dim=1)

        proxy_optimizer.zero_grad()

        segmentation_loss = segmentation_criterion(segmentation[:, 1], gt[:, 1])
        proxy_loss = proxy_critertion(projection_maps, gt)

        train_dice_loss += segmentation_loss.item()
        total_loss = segmentation_loss + 0.5*(proxy_loss/num_voxels)
        total_loss.backward()

        proxy_optimizer.step()

        dice = Dice_Score(segmentation[:, 1].detach().cpu().numpy(), gt[:,1].detach().cpu().numpy())
        f1_acc = F1_score(segmentation[:, 1].detach().cpu().numpy(), gt[:,1].cpu().numpy())
        train_dice_score += dice.item()
        train_f1_accuracy += f1_acc.item()

        del image
        del final_out
        del gt
        del to_projector
        del projection_maps
        del segmentation
        del segmentation_loss
        del proxy_loss
        del total_loss
    
    train_dice_loss_list.append(train_dice_loss / len(trainloader))
    train_dice_score_list.append(train_dice_score / len(trainloader))
    train_f1_accuracy_list.append(train_f1_accuracy / len(trainloader))
    print(f'Train dice loss at epoch#{epoch}: {train_dice_loss_list[-1]}')
    print(f'Train dice score at epoch#{epoch}: {train_dice_score_list[-1]}')
    print(f'Train f1 accuracy at epoch#{epoch}: {train_f1_accuracy_list[-1]}')

    print()
    # Validation loop
    proxy_encoder.eval()
    proxy_projector.eval()
    segmentation_model.eval()

    validation_dice_loss = 0
    validation_dice_score = 0
    validation_f1_accuracy = 0

    for idx, data in enumerate(tqdm(validationloader), 0):
        with torch.no_grad():
            image = data['input'].to(device)
            gt = data['gt'].to(device)

            to_projector, _ = proxy_encoder(image)
            # _, projection_maps = proxy_projector(to_projector)
            final_out, projection_maps = proxy_projector(to_projector)
            segmentation = segmentation_model(final_out, projection_maps)

            # projection_maps = [F.interpolate(projection_map, size=(128, 128, 128), mode='trilinear') for projection_map in projection_maps]
            # projection_maps = torch.cat(projection_maps, dim=1)

            segmentation_loss = segmentation_criterion(segmentation[:, 1], gt[:, 1])
            # proxy_loss = proxy_critertion(projection_maps, gt)

            validation_dice_loss += segmentation_loss.item()
            dice = Dice_Score(segmentation[:, 1].cpu().numpy(), gt[:, 1].detach().cpu().numpy())
            f1_acc = F1_score(segmentation[:, 1].cpu().numpy(), gt[:, 1].detach().cpu().numpy())
            validation_dice_score += dice.item()
            validation_f1_accuracy += f1_acc.item()
    
    validation_dice_loss_list.append(validation_dice_loss / len(validationloader))
    validation_dice_score_list.append(validation_dice_score / len(validationloader))
    validation_f1_accuracy_list.append(validation_f1_accuracy / len(validationloader))
    print(f'Validation dice loss at epoch#{epoch}: {validation_dice_loss_list[-1]}')
    print(f'Validation dice score at epoch#{epoch}: {validation_dice_score_list[-1]}')
    print(f'Validation f1 accuracy at epoch#{epoch}: {validation_f1_accuracy_list[-1]}')

    proxy_scheduler.step(validation_dice_score_list[-1])

    np.save(f'./results/{proxy_type}_{date}_losses.npy', [train_dice_score_list, train_f1_accuracy_list, validation_dice_score_list, validation_f1_accuracy_list])
    
    if best_validation_score is None:
        best_validation_score = validation_dice_score_list[-1]
    elif validation_dice_score_list[-1] > best_validation_score:
        patience = 15
        best_validation_score = validation_dice_score_list[-1]
        torch.save(proxy_encoder.state_dict(), f'{save_model_path}{proxy_encoder_type}_{date}_state_dict_best_score{epoch}.pth')
        torch.save(proxy_projector.state_dict(), f'{save_model_path}{proxy_projector_type}_{date}_state_dict_best_score{epoch}.pth')

    if epoch % 10 == 0:
        torch.save(proxy_encoder.state_dict(), f'{save_model_path}{proxy_encoder_type}_{date}_state_dict{epoch}.pth')
        torch.save(proxy_projector.state_dict(), f'{save_model_path}{proxy_projector_type}_{date}_state_dict{epoch}.pth')

print()
print('Script executed.')