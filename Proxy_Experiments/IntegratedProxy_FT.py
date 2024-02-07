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
    date = '14_12_2023',
    to_save_folder = 'Dec14',
    to_load_folder = None,
    device = 'cuda:1',
    proxy_type = 'Integrated_Unet_&_VGGproxy_pat5_dice_stepped4_multichannel_projection_1e-4_>_1e-5_lr',
    train_task = 'seg(decoder_step_per_epoch)_&_proxy(classifier)_simulated_brain_bg_>_real_wmh_ratiod_wrt_wmh_simulated_brain_bg',
    to_load_encoder_path = 'unet_focal + dice_state_dict_best_loss85.pth',
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'sim_2211_ratios:72_80_18_17+wmh'
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
validationloader = DataLoader(validationset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

# proxy_encoder = SA_UNet_Encoder(out_channels=2).to(c.device)
# proxy_encoder.load_state_dict(torch.load(c.to_load_encoder_path)['model_state_dict'], strict=False)
proxy_encoder = VGG3D_Encoder(input_channels=1).to(c.device)
proxy_encoder.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/experiments/Dec08/VGGproxy_weightedBCE_wLRScheduler_simulated_lesions_on_brain_encoder_08_12_2023_state_dict_best_loss80.pth'))

proxy_projector = IntegratedSpatialProjector(num_layers=4, layer_sizes=[64, 128, 256, 512]).to(c.device)

segmentation_model = SA_UNet(out_channels=2).to(c.device)
segmentation_model.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/segmentation_models/unet_focal + dice_state_dict_best_loss85.pth')['model_state_dict'], strict=False)

segmentation_criterion = DiceLoss().to(c.device)
# proxy_critertion = VoxelwiseSupConLoss_inImage(device=c.device, num_voxels=num_voxels).to(c.device)

segmentation_optimizer = optim.Adam(segmentation_model.parameters(), lr = 0.0001, eps = 0.0001)
proxy_optimizer = optim.Adam([*proxy_encoder.parameters(), *proxy_projector.parameters()], lr = 0.0001, eps = 0.0001)

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
    proxy_projector.train()
    segmentation_model.eval()

    train_dice_loss = 0
    train_dice_score = 0

    segmentation_step = True

    for data in tqdm(trainloader):
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)

        to_projector, _ = proxy_encoder(image)
        # _, projection_maps = proxy_projector(to_projector)
        projection_maps = proxy_projector(to_projector)
        segmentation = segmentation_model(image, projection_maps)

        # projection_maps = [F.interpolate(projection_map, size=(128, 128, 128), mode='trilinear') for projection_map in projection_maps]
        # projection_maps = torch.cat(projection_maps, dim=1)

        if segmentation_step:
            for name, module in segmentation_model.named_modules():
                if name.startswith("upconv") or name.startswith("decoder") or name.startswith("seg"):
                    for param in module.parameters():
                        param.requires_grad = True

            segmentation_model.train()
            segmentation_optimizer.zero_grad()

        proxy_optimizer.zero_grad()

        segmentation_loss = segmentation_criterion(segmentation[:, 1], gt[:, 1])
        # proxy_loss = proxy_critertion(projection_maps, gt)

        train_dice_loss += segmentation_loss.item()
        segmentation_loss.backward()

        if segmentation_step:
            segmentation_optimizer.step()
            print('Stepped the segmentor.')
            for parameter in segmentation_model.parameters():
                parameter.requires_grad = False
            segmentation_model.eval()
            segmentation_step = False

        proxy_optimizer.step()

        dice = Dice_Score(segmentation[:, 1].detach().cpu().numpy(), gt[:,1].detach().cpu().numpy())
        train_dice_score += dice.item()

        del image
        del gt
        del to_projector
        del projection_maps
        del segmentation
        del segmentation_loss
        # del proxy_loss
    
    train_dice_loss_list.append(train_dice_loss / len(trainloader))
    train_dice_score_list.append(train_dice_score / len(trainloader))
    print(f'Train dice loss at epoch#{epoch}: {train_dice_loss_list[-1]}')
    print(f'Train dice score at epoch#{epoch}: {train_dice_score_list[-1]}')

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
    print(f'Validation dice loss at epoch#{epoch}: {validation_dice_loss_list[-1]}')
    print(f'Validation dice score at epoch#{epoch}: {validation_dice_score_list[-1]}')

    proxy_scheduler.step(validation_dice_score_list[-1])

    np.save(f'./results/{c.proxy_type}_{c.date}_losses.npy', [train_dice_loss_list, train_dice_score_list, validation_dice_loss_list, validation_dice_score_list])
    
    if best_validation_score is None:
        best_validation_score = validation_dice_score_list[-1]

    elif validation_dice_score_list[-1] > best_validation_score:
        best_validation_score = validation_dice_score_list[-1]
        torch.save(proxy_encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict_best_score{epoch}.pth')
        torch.save(proxy_projector.state_dict(), f'{c.to_save_folder}{c.projector_type}_{c.date}_state_dict_best_score{epoch}.pth')
        torch.save(segmentation_model.state_dict(), f'{c.to_save_folder}{c.segmentor_type}_{c.date}_state_dict_best_score{epoch}.pth')

    if epoch % 10 == 0:
        torch.save(proxy_encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict{epoch}.pth')
        torch.save(proxy_projector.state_dict(), f'{c.to_save_folder}{c.projector_type}_{c.date}_state_dict{epoch}.pth')
        torch.save(segmentation_model.state_dict(), f'{c.to_save_folder}{c.segmentor_type}_{c.date}_state_dict{epoch}.pth')

print()
print('Script executed.')