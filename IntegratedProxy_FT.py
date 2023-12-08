import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import Constants, load_dataset

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

import numpy as np
from tqdm import tqdm

c = Constants(
    batch_size = 1,
    patience = 15,
    num_workers = 16,
    num_epochs = 100,
    date = '07_12_2023',
    to_save_folder = 'Dec07',
    to_load_folder = 'pretrained',
    device = 'cuda:0',
    proxy_type = 'Integrated_RandomBGUNET',
    train_task = 'Dice_&_Vox_FT',
    to_load_encoder_path = 'unet_focal + dice_state_dict_best_loss85.pth',
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'wmh'
)

trainset, validationset, testst = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
validationloader = DataLoader(validationset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

proxy_encoder = SA_UNet_Encoder(out_channels=2).to(c.device)
proxy_encoder.load_state_dict(torch.load(c.to_load_encoder_path)['model_state_dict'], strict=False)

proxy_projector = IntegratedProjector(num_layers=4, layer_sizes=[64, 128, 256, 512]).to(c.device)

segmentation_model = SA_UNet(out_channels=2).to(c.device)
segmentation_model.load_state_dict(torch.load(c.to_load_encoder_path)['model_state_dict'], strict=False)

num_voxels = 16000
segmentation_criterion = DiceLoss().to(c.device)
proxy_critertion = VoxelwiseSupConLoss_inImage(device=c.device, num_voxels=num_voxels).to(c.device)

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
print('Finetuning Integrated model.')

for epoch in range(1, c.num_epochs+1):
    
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
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)

        to_projector, _ = proxy_encoder(image)
        # _, projection_maps = proxy_projector(to_projector)
        projection_maps = proxy_projector(to_projector)
        segmentation = segmentation_model(image, projection_maps)

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
            image = data['input'].to(c.device)
            gt = data['gt'].to(c.device)

            to_projector, _ = proxy_encoder(image)
            # _, projection_maps = proxy_projector(to_projector)
            projection_maps = proxy_projector(to_projector)
            segmentation = segmentation_model(image, projection_maps)

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

    np.save(f'./results/{c.proxy_type}_{c.date}_losses.npy', [train_dice_score_list, train_f1_accuracy_list, validation_dice_score_list, validation_f1_accuracy_list])
    
    if best_validation_score is None:
        best_validation_score = validation_dice_score_list[-1]
    elif validation_dice_score_list[-1] > best_validation_score:
        patience = 15
        best_validation_score = validation_dice_score_list[-1]
        torch.save(proxy_encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict_best_score{epoch}.pth')
        torch.save(proxy_projector.state_dict(), f'{c.to_save_folder}{c.projector_type}_{c.date}_state_dict_best_score{epoch}.pth')

    if epoch % 10 == 0:
        torch.save(proxy_encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict{epoch}.pth')
        torch.save(proxy_projector.state_dict(), f'{c.to_save_folder}{c.projector_type}_{c.date}_state_dict{epoch}.pth')

print()
print('Script executed.')