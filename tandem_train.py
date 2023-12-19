import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import Constants, load_dataset

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
    patience = 10,
    num_workers = 8,
    num_epochs = 100,
    date = '14_12_2023',
    to_save_folder = 'Dec14',
    to_load_folder = None,
    device = 'cuda:1',
    proxy_type = 'Integrated_Unet_&_VGGproxy_tandem_(segmentation_>_proxy)_pat10_feature_add_1e-3_>_1e-5_lr',
    train_task = 'seg_&_proxy(reconstruction)_simulated_brain_bg_>_real_wmh_ratiod_wrt_wmh_simulated_brain_bg',
    encoder_load_path = None,
    projector_load_path = None,
    classifier_load_path = None,
    proxy_load_path = None,
    dataset = 'sim_2211_ratios:72_80_18_17+wmh'
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
validationloader = DataLoader(validationset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

# proxy_encoder = VGG3D_Encoder(input_channels=1).to(c.device)
proxy_encoder = SA_UNet_Encoder().to(c.device)
proxy_encoder.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/experiments/Dec12/UNETproxy_reconstruction_simulated_noise_bg_>_sim_wmh_&_sim_brats_occluded_12_12_2023_state_dict_best_loss7.pth'), strict=False)

# proxy_projector = IntegratedChannelProjector(num_layers=4, layer_sizes=[64, 128, 256, 512], layer_dimensions=[64, 32, 32, 16]).to(c.device)

segmentation_model = SA_UNet(out_channels=2).to(c.device)
segmentation_model.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/segmentation_models/unet_focal + dice_state_dict_best_loss85.pth')['model_state_dict'], strict=False)

criterion = DiceLoss().to(c.device)

segmentation_optimizer = optim.Adam(segmentation_model.parameters(), lr = 0.001, eps = 0.0001)
# proxy_optimizer = optim.Adam([*proxy_encoder.parameters(), *proxy_projector.parameters()], lr = 0.001, eps = 0.0001)
proxy_optimizer = optim.Adam(proxy_encoder.parameters(), lr = 0.001, eps = 0.0001)

segmentation_optimizer = optim.Adam(segmentation_model.parameters(), lr = 0.001, eps = 0.0001)
# proxy_optimizer = optim.Adam([*proxy_encoder.parameters(), *proxy_projector.parameters()], lr = 0.001, eps = 0.0001)
proxy_optimizer = optim.Adam(proxy_encoder.parameters(), lr = 0.001, eps = 0.0001)

segmentation_scheduler = optim.lr_scheduler.ReduceLROnPlateau(segmentation_optimizer, mode='max', factor=0.5, patience=c.patience, min_lr=0.00001,verbose=True)
proxy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(proxy_optimizer, mode='max', factor=0.5, patience=c.patience, min_lr=0.00001, verbose=True)

segmentation_train_dice_loss_list = []
segmentation_train_dice_score_list = []
proxy_train_dice_loss_list = []
proxy_train_dice_score_list = []

best_validation_score = None

validation_dice_loss_list = []
validation_dice_score_list = []

print()
print('Tandem training segmentation and proxy models.')

for epoch in range(1, c.num_epochs+1):
    
    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop

    segmentation_train_dice_loss = 0
    segmentation_train_dice_score = 0
    proxy_train_dice_loss = 0
    proxy_train_dice_score = 0

    for data in tqdm(trainloader):

        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)

        # segmentation model step
        segmentation_model.train()
        proxy_encoder.eval()
        # proxy_projector.eval()

        for p in segmentation_model.parameters():
            p.requires_grad = True
        # for p in [*proxy_encoder.parameters(), *proxy_projector.parameters()]:
        for p in proxy_encoder.parameters():
            p.requires_grad = False

        to_projector, _ = proxy_encoder(image)

        # projection_maps = proxy_projector(to_projector)
        # segmentation = segmentation_model(image, projection_maps)
        segmentation = segmentation_model(image, to_projector)

        segmentation_optimizer.zero_grad()

        segmentation_loss = criterion(segmentation[:, 1], gt[:, 1])
        segmentation_train_dice_loss += segmentation_loss.item()
        segmentation_loss.backward()

        segmentation_optimizer.step()

        dice = Dice_Score(segmentation[:, 1].detach().cpu().numpy(), gt[:,1].detach().cpu().numpy())
        segmentation_train_dice_score += dice.item()

        del to_projector
        del _
        # del projection_maps
        del segmentation
        del segmentation_loss
        del dice

        # proxy model step
        segmentation_model.eval()
        proxy_encoder.train()
        # proxy_projector.train()

        for p in segmentation_model.parameters():
            p.requires_grad = False
        # for p in [*proxy_encoder.parameters(), *proxy_projector.parameters()]:
        #     p.requires_grad = True
        for p in proxy_encoder.parameters():
            p.requires_grad = True

        to_projector, _ = proxy_encoder(image)
        # projection_maps = proxy_projector(to_projector)
        # segmentation = segmentation_model(image, projection_maps)
        segmentation = segmentation_model(image, to_projector)


        proxy_optimizer.zero_grad()

        proxy_loss = criterion(segmentation[:, 1], gt[:, 1])
        proxy_train_dice_loss += proxy_loss.item()
        proxy_loss.backward()

        proxy_optimizer.step()

        dice = Dice_Score(segmentation[:, 1].detach().cpu().numpy(), gt[:,1].detach().cpu().numpy())
        proxy_train_dice_score += dice.item()


        del image
        del gt
        del to_projector
        del _
        # del projection_maps
        del segmentation
        del proxy_loss
        del dice
    
    segmentation_train_dice_loss_list.append(segmentation_train_dice_loss / len(trainloader))
    segmentation_train_dice_score_list.append(segmentation_train_dice_score / len(trainloader))
    proxy_train_dice_loss_list.append(proxy_train_dice_loss / len(trainloader))
    proxy_train_dice_score_list.append(proxy_train_dice_score / len(trainloader))
    print(f'Segmentation train dice loss at epoch#{epoch}: {segmentation_train_dice_loss_list[-1]}')
    print(f'Segmentation train dice score at epoch#{epoch}: {segmentation_train_dice_score_list[-1]}')
    print(f'Proxy train dice loss at epoch#{epoch}: {proxy_train_dice_loss_list[-1]}')
    print(f'Proxy train dice score at epoch#{epoch}: {proxy_train_dice_score_list[-1]}')

    print()
    # Validation loop
    proxy_encoder.eval()
    # proxy_projector.eval()
    segmentation_model.eval()

    validation_dice_loss = 0
    validation_dice_score = 0

    for idx, data in enumerate(tqdm(validationloader), 0):
        with torch.no_grad():
            image = data['input'].to(c.device)
            gt = data['gt'].to(c.device)

            to_projector, _ = proxy_encoder(image)
            # projection_maps = proxy_projector(to_projector)
            # segmentation = segmentation_model(image, projection_maps)
            segmentation = segmentation_model(image, to_projector)

            segmentation_loss = criterion(segmentation[:, 1], gt[:, 1])
            validation_dice_loss += segmentation_loss.item()

            dice = Dice_Score(segmentation[:, 1].cpu().numpy(), gt[:, 1].detach().cpu().numpy())
            validation_dice_score += dice.item()

            del image
            del gt
            del to_projector
            del _
            # del projection_maps
            del segmentation
            del segmentation_loss
            del dice
    
    validation_dice_loss_list.append(validation_dice_loss / len(validationloader))
    validation_dice_score_list.append(validation_dice_score / len(validationloader))
    print(f'Validation dice loss at epoch#{epoch}: {validation_dice_loss_list[-1]}')
    print(f'Validation dice score at epoch#{epoch}: {validation_dice_score_list[-1]}')

    segmentation_scheduler.step(validation_dice_score_list[-1])
    proxy_scheduler.step(validation_dice_score_list[-1])

    np.save(f'./results/{c.proxy_type}_{c.date}_losses.npy', [segmentation_train_dice_loss_list, segmentation_train_dice_score_list, proxy_train_dice_loss_list, proxy_train_dice_score_list, validation_dice_loss_list, validation_dice_score_list])
    
    if best_validation_score is None:
        best_validation_score = validation_dice_score_list[-1]
    elif validation_dice_score_list[-1] > best_validation_score:
        best_validation_score = validation_dice_score_list[-1]
        torch.save(segmentation_model.state_dict(), f'{c.to_save_folder}{c.segmentor_type}_{c.date}_state_dict_best_score{epoch}.pth')
        torch.save(proxy_encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict_best_score{epoch}.pth')
        # torch.save(proxy_projector.state_dict(), f'{c.to_save_folder}{c.projector_type}_{c.date}_state_dict_best_score{epoch}.pth')

    if epoch % 10 == 0:
        torch.save(segmentation_model.state_dict(), f'{c.to_save_folder}{c.segmentor_type}_{c.date}_state_dict{epoch}.pth')
        torch.save(proxy_encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict{epoch}.pth')
        # torch.save(proxy_projector.state_dict(), f'{c.to_save_folder}{c.projector_type}_{c.date}_state_dict{epoch}.pth')

print()
print('Script executed.')