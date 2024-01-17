import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import *

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

c = Constants(
    batch_size = 1,
    patience = 5,
    num_workers = 12,
    num_epochs = 100,
    date = '11_01_2024',
    to_save_folder = 'Jan11',
    to_load_folder = None,
    device = 'cuda:1',
    proxy_type = 'LiTS_Unet_preprocessing_0_>_200_window_init_features_64_with_ahe',
    train_task = 'segmentation',
    to_load_encoder_path = None,
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'lits:window:0_200'
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))

trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
validationloader = DataLoader(validationset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

model = UNet(out_channels=2, init_features=64).to(c.device)
criterion = DiceLoss().to(c.device)
optimizer = optim.Adam(model.parameters(), lr = 0.001, eps = 0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=0.00001, verbose=True)

train_dice_loss_list = []
train_dice_score_list = []

best_validation_score = None
validation_dice_loss_list = []
validation_dice_score_list = []

print()
print('Training segmentation model.')

for epoch in range(1, c.num_epochs+1):
    
    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    model.train()

    train_dice_loss = 0
    train_dice_score = 0

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:

    for data in tqdm(trainloader):
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)

        segmentation = model(image)
        optimizer.zero_grad()
        loss = criterion(segmentation[:, 1], gt[:, 1])
        train_dice_loss += loss.item()
        loss.backward()
        optimizer.step()
        dice = Dice_Score(segmentation[:, 1].detach().cpu().numpy(), gt[:,1].detach().cpu().numpy())
        train_dice_score += dice.item()

        plt.figure(figsize=(20, 15))
        plt.subplot(1, 3, 1)
        plt.imshow(image[0, 0, :, :, 64].detach().cpu())
        plt.subplot(1, 3, 2)
        plt.imshow(segmentation[0, 1, :, : , 64].detach().cpu())
        plt.subplot(1, 3, 3)
        plt.imshow(gt[0, 1, :, : , 64].detach().cpu())
        plt.savefig('./temp')
        plt.close()

        del image
        del gt
        del segmentation
        del loss
    train_dice_loss_list.append(train_dice_loss / len(trainloader))
    train_dice_score_list.append(train_dice_score / len(trainloader))
    print(f'Train dice loss at epoch#{epoch}: {train_dice_loss_list[-1]}')
    print(f'Train dice score at epoch#{epoch}: {train_dice_score_list[-1]}')

    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    # break

    print()

    # Validation loop
    model.eval()

    validation_dice_loss = 0
    validation_dice_score = 0

    for idx, data in enumerate(tqdm(validationloader), 0):
        with torch.no_grad():
            image = data['input'].to(c.device)
            gt = data['gt'].to(c.device)
            segmentation = model(image)

            loss = criterion(segmentation[:, 1], gt[:, 1])
            validation_dice_loss += loss.item()
            dice = Dice_Score(segmentation[:, 1].cpu().numpy(), gt[:, 1].detach().cpu().numpy())
            validation_dice_score += dice.item()

            plt.subplot(1, 3, 1)
            plt.imshow(image[0, 0, :, :, 64].detach().cpu())
            plt.subplot(1, 3, 2)
            plt.imshow(segmentation[0, 1, :, : , 64].detach().cpu())
            plt.subplot(1, 3, 3)
            plt.imshow(gt[0, 1, :, : , 64].detach().cpu())
            plt.savefig('./temp')
            plt.close()
    
    validation_dice_loss_list.append(validation_dice_loss / len(validationloader))
    validation_dice_score_list.append(validation_dice_score / len(validationloader))
    print(f'Validation dice loss at epoch#{epoch}: {validation_dice_loss_list[-1]}')
    print(f'Validation dice score at epoch#{epoch}: {validation_dice_score_list[-1]}')

    scheduler.step(validation_dice_loss_list[-1])

    np.save(f'./results/{c.proxy_type}_{c.date}_losses.npy', [train_dice_loss_list, train_dice_score_list, validation_dice_loss_list, validation_dice_score_list])
    
    if best_validation_score is None:
        best_validation_score = validation_dice_score_list[-1]

    elif validation_dice_score_list[-1] > best_validation_score:
        best_validation_score = validation_dice_score_list[-1]
        torch.save(model.state_dict(), f'{c.to_save_folder}{c.segmentor_type}_{c.date}_state_dict_best_score{epoch}.pth')

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'{c.to_save_folder}{c.segmentor_type}_{c.date}_state_dict{epoch}.pth')

print()
print('Script executed.')