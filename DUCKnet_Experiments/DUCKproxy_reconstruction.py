import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
import numpy as np
from tqdm import tqdm

batch_size = 1
patience = 15
num_workers = 16
device = 'cuda:0'
number_of_epochs = 100
date = '30_11_2023'
encoder_type = 'DUCK_WMH_reconproxy_encoder_simFinetuned'
reconstructor_type = 'DUCK_WMH_reconproxy_reconstructor_simFinetuned'
projector_type = 'DUCK_WMH_reconproxy_projector_simFinetuned'

save_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov30/'
DUCKmodel_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/Duck1wmh_focal + dice_state_dict_best_loss97.pth'

from_Sim1000_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*FLAIR.nii.gz'))
from_sim2211_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/**/*FLAIR.nii.gz'))

from_Sim1000_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*mask.nii.gz'))
from_sim2211_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/**/*mask.nii.gz'))

from_Sim1000_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*.json'))
from_sim2211_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/**/*.json'))

clean_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
clean_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))

composed_transform = transforms.Compose([
        ToTensor3D(labeled=True)
    ])

from_Sim1000 = ImageLoader3D(paths=from_Sim1000_data_paths, gt_paths=from_Sim1000_gt_paths, json_paths=from_Sim1000_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)

from_sim2211 = ImageLoader3D(paths=from_sim2211_data_paths,gt_paths=from_sim2211_gt_paths, json_paths=from_sim2211_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)

clean = ImageLoader3D(paths=clean_data_paths, gt_paths=clean_gt_paths, json_paths=None, image_size=128, type_of_imgs='nifty', transform=composed_transform)

fullset = ConcatDataset([from_Sim1000, from_sim2211, clean])

train_size = int(0.8 * len(fullset))
validation_size = len(fullset) - train_size

trainset, validationset = random_split(fullset, (train_size, validation_size))

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

DUCKnet_encoder = DuckNet(input_channels=1, out_classes=2, starting_filters=17).to(device)
DUCKnet_encoder.load_state_dict(torch.load(DUCKmodel_path))

projection_head = Projector(num_layers=5, layer_sizes=[17, 34, 68, 136, 272]).to(device)

reconstruction_head = Decoder().to(device)

# Freezing DUCKnet
# for param in DUCKnet_encoder.parameters():
#     param.requires_grad = False

projector_optimizer = optim.Adam([*DUCKnet_encoder.parameters(), *projection_head.parameters()], lr = 0.0001, eps = 0.0001)
# projector_optimizer = optim.Adam(projection_head.parameters(), lr = 0.0001, eps = 0.0001)
reconstructor_optimizer = optim.Adam(reconstruction_head.parameters(), lr = 0.0001, eps = 0.0001)

reconstruction_criterion = nn.MSELoss().to(device)
projection_criterion = VoxelwiseSupConLoss_inImage(device=device).to(device)

projection_train_loss_list = []
projection_validation_loss_list = []
reconstruction_train_loss_list = []
reconstruction_validation_loss_list = []

best_validation_loss = None

print()
print('Training ReconProxy.')
for epoch in range(1, number_of_epochs+1):

    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    DUCKnet_encoder.train()
    projection_head.train()
    reconstruction_head.train()

    projection_train_loss = 0
    reconstruction_train_loss = 0

    # patience -= 1
    # if patience == 0:
    #     print()
    #     print(f'Breaking at epoch #{epoch} due to lack of patience. Best validation loss was: {best_validation_loss}')

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:

    for data in tqdm(trainloader):
        image = data['input'].to(device)
        gt = data['gt'].to(device)
        oneHot_label = data['lesion_labels'].float().to(device)

        if torch.unique(gt[:, 1]).shape[0] == 2:
            to_projector, to_reconstructor = DUCKnet_encoder(image)
            reconstruction = reconstruction_head(to_reconstructor)

            reconstruction_loss = reconstruction_criterion(reconstruction, image)
            reconstruction_train_loss += reconstruction_loss.item()

            projection = projection_head(to_projector)
            projection = F.interpolate(projection, size=(128, 128, 128))

            projection_loss = projection_criterion(projection, gt)
            projection_train_loss += projection_loss.item()

            projector_optimizer.zero_grad()
            reconstructor_optimizer.zero_grad()
            loss = projection_loss + reconstruction_loss
            loss.backward()
            projector_optimizer.step()
            reconstructor_optimizer.step()

            del projection
            del projection_loss
            
            del image
            del gt
            del oneHot_label
            del to_projector
            del to_reconstructor
            del reconstruction
            del reconstruction_loss
            del loss
    
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    # break
    
    projection_train_loss_list.append(projection_train_loss / len(trainloader))
    reconstruction_train_loss_list.append(reconstruction_train_loss / len(trainloader))
    print(f'Projection train loss at epoch #{epoch}: {projection_train_loss_list[-1]}')
    print(f'Reconstruction train loss at epoch #{epoch}: {reconstruction_train_loss_list[-1]}')

    print()
    torch.cuda.empty_cache()

    # Validation loop
    DUCKnet_encoder.eval()
    projection_head.eval()
    reconstruction_head.eval()

    projection_validation_loss = 0
    reconstruction_validation_loss = 0

    for data in tqdm(validationloader):
        image = data['input'].to(device)
        gt = data['gt'].to(device)
        oneHot_label = data['lesion_labels'].float().to(device)

        if torch.unique(gt[:, 1]).shape[0] == 2:
            to_projector, to_reconstructor = DUCKnet_encoder(image)
            reconstruction = reconstruction_head(to_reconstructor)

            reconstruction_loss = reconstruction_criterion(reconstruction, image)
            reconstruction_validation_loss += reconstruction_loss.item()

            projection = projection_head(to_projector)
            projection = F.interpolate(projection, size=(128, 128, 128))

            projection_loss = projection_criterion(projection, gt)
            projection_validation_loss += projection_loss.item()

            del projection
            del projection_loss

            del image
            del gt
            del oneHot_label
            del to_projector
            del to_reconstructor
            del reconstruction
            del reconstruction_loss
    
    projection_validation_loss_list.append(projection_validation_loss / len(validationloader))
    reconstruction_validation_loss_list.append(reconstruction_validation_loss / len(validationloader))
    print(f'Projection validation loss at epoch #{epoch}: {projection_validation_loss_list[-1]}')
    print(f'Reconstruction validation loss at epoch #{epoch}: {reconstruction_validation_loss_list[-1]}')

    np.save(f'./results/{projector_type}_{date}_losses.npy', [projection_train_loss_list, projection_validation_loss_list])
    np.save(f'./results/{reconstructor_type}_{date}_accuracies.npy', [reconstruction_train_loss_list, reconstruction_validation_loss_list])

    if best_validation_loss is None:
        best_validation_loss = reconstruction_validation_loss_list[-1]
    elif reconstruction_validation_loss_list[-1] < best_validation_loss:
        patience = 15
        best_validation_loss = reconstruction_validation_loss_list[-1]
        # torch.save(DUCKnet_encoder.state_dict(), f'{save_model_path}{encoder_type}_{date}_state_dict_best_loss{epoch}.pth')
        torch.save(projection_head.state_dict(), f'{save_model_path}{projector_type}_{date}_state_dict_best_loss{epoch}.pth')
        torch.save(reconstruction_head.state_dict(), f'{save_model_path}{reconstructor_type}_{date}_state_dict_best_loss{epoch}.pth')
        # torch.save(encoder_projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict_best_loss{epoch}.pth')
        torch.save(projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict_best_loss{epoch}.pth')
        torch.save(reconstructor_optimizer.state_dict(), f'{save_model_path}{reconstructor_type}_ optimizer_{date}_state_dict_best_loss{epoch}.pth')
        print(f'New best validation loss at epoch #{epoch}')

    if epoch % 10 == 0:
        torch.save(DUCKnet_encoder.state_dict(), f'{save_model_path}{encoder_type}_{date}_state_dict{epoch}.pth')
        torch.save(projection_head.state_dict(), f'{save_model_path}{projector_type}_{date}_state_dict{epoch}.pth')
        torch.save(reconstruction_head.state_dict(), f'{save_model_path}{reconstructor_type}_{date}_state_dict{epoch}.pth')
        # torch.save(encoder_projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict{epoch}.pth')
        torch.save(projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict{epoch}.pth')
        torch.save(reconstructor_optimizer.state_dict(), f'{save_model_path}{reconstructor_type}_ optimizer_{date}_state_dict{epoch}.pth')

    print()

torch.save(DUCKnet_encoder.state_dict(), f'{save_model_path}{encoder_type}_{date}_state_dict{number_of_epochs+1}.pth')
torch.save(projection_head.state_dict(), f'{save_model_path}{projector_type}_{date}_state_dict{number_of_epochs+1}.pth')
torch.save(reconstruction_head.state_dict(), f'{save_model_path}{reconstructor_type}_{date}_state_dict{number_of_epochs+1}.pth')
# torch.save(encoder_projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict{number_of_epochs+1}.pth')
torch.save(projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict{number_of_epochs+1}.pth')
torch.save(reconstructor_optimizer.state_dict(), f'{save_model_path}{reconstructor_type}_ optimizer_{date}_state_dict{number_of_epochs+1}.pth')

print()
print('Script executed.')