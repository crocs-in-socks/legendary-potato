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

c = Constants(
    batch_size = 2,
    patience = 5,
    num_workers = 8,
    num_epochs = 100,
    date = '18_12_2023',
    to_save_folder = 'Dec18',
    to_load_folder = None,
    device = 'cuda:0',
    proxy_type = 'Integrated_Unet_&_Unetproxy_pat5_dice_feature_addition_1e-4_>_1e-5_lr',
    train_task = 'seg_&_proxy(scratch)_simulated_brain_bg_>_real_wmh_ratiod_wrt_wmh_simulated_brain_bg',
    to_load_encoder_path = 'unet_focal + dice_state_dict_best_loss85.pth',
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'sim_2211_ratios:72_80_18_17+wmh'
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
wmh_trainset, wmh_validationset, wmh_testset = load_dataset('wmh', c.drive, ToTensor3D(labeled=True))

trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
validationloader = DataLoader(wmh_testset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

proxy_encoder = SA_UNet_Encoder(out_channels=2).to(c.device)

segmentation_model = SA_UNet(out_channels=2).to(c.device)
segmentation_model.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/segmentation_models/unet_focal + dice_state_dict_best_loss85.pth')['model_state_dict'], strict=False)

segmentation_criterion = DiceLoss().to(c.device)
proxy_optimizer = optim.Adam(proxy_encoder.parameters(), lr = 0.0001, eps = 0.0001)
proxy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(proxy_optimizer, mode='max', patience=5, factor=0.5, min_lr=0.00001, verbose=True)

for parameter in segmentation_model.parameters():
    parameter.requires_grad = False

train_dice_loss_list = []
train_dice_score_list = []

best_validation_score = None
validation_dice_loss_list = []
validation_dice_score_list = []

print()
print('Finetuning proxy for integrated model.')

for epoch in range(1, c.num_epochs+1):
    
    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    proxy_encoder.train()
    segmentation_model.eval()

    train_dice_loss = 0
    train_dice_score = 0

    for data in tqdm(trainloader):
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)

        to_projector, _ = proxy_encoder(image)
        segmentation = segmentation_model(image, to_projector)

        proxy_optimizer.zero_grad()
        segmentation_loss = segmentation_criterion(segmentation[:, 1], gt[:, 1])
        train_dice_loss += segmentation_loss.item()
        segmentation_loss.backward()
        proxy_optimizer.step()

        dice = Dice_Score(segmentation[:, 1].detach().cpu().numpy(), gt[:,1].detach().cpu().numpy())
        train_dice_score += dice.item()

        del image
        del gt
        del to_projector
        del segmentation
        del segmentation_loss
    
    train_dice_loss_list.append(train_dice_loss / len(trainloader))
    train_dice_score_list.append(train_dice_score / len(trainloader))
    print(f'Train dice loss at epoch#{epoch}: {train_dice_loss_list[-1]}')
    print(f'Train dice score at epoch#{epoch}: {train_dice_score_list[-1]}')

    print()

    # Validation loop
    proxy_encoder.eval()
    segmentation_model.eval()

    validation_dice_loss = 0
    validation_dice_score = 0

    for idx, data in enumerate(tqdm(validationloader), 0):
        with torch.no_grad():
            image = data['input'].to(c.device)
            gt = data['gt'].to(c.device)

            to_projector, _ = proxy_encoder(image)
            segmentation = segmentation_model(image, to_projector)
            segmentation_loss = segmentation_criterion(segmentation[:, 1], gt[:, 1])

            validation_dice_loss += segmentation_loss.item()
            dice = Dice_Score(segmentation[:, 1].cpu().numpy(), gt[:, 1].detach().cpu().numpy())
            validation_dice_score += dice.item()
    
    validation_dice_loss_list.append(validation_dice_loss / len(validationloader))
    validation_dice_score_list.append(validation_dice_score / len(validationloader))
    print(f'Validation dice loss at epoch#{epoch}: {validation_dice_loss_list[-1]}')
    print(f'Validation dice score at epoch#{epoch}: {validation_dice_score_list[-1]}')

    proxy_scheduler.step(validation_dice_score_list[-1])

    np.save(f'./results/{c.proxy_type}_{c.date}_losses.npy', [train_dice_loss_list, train_dice_score_list, validation_dice_loss_list, validation_dice_score_list])
    
    if best_validation_score is None:
        best_validation_score = validation_dice_score_list[-1]

    elif validation_dice_score_list[-1] > best_validation_score:
        best_validation_score = validation_dice_score_list[-1]
        torch.save(proxy_encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict_best_score{epoch}.pth')

    if epoch % 10 == 0:
        torch.save(proxy_encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict{epoch}.pth')

print()
print('Script executed.')