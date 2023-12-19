import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import Constants, load_dataset

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import numpy as np
from tqdm import tqdm

c = Constants(
    batch_size = 1,
    patience = 15,
    num_workers = 8,
    number_of_epochs = 100,
    date = '06_12_2023',
    to_save_folder = 'Dec06',
    to_load_folder = 'pretrained',
    device = 'cuda:1',
    proxy_type = 'UNETcopy',
    train_task = 'VoxCFT_wRandomBG',
    to_load_encoder_path = 'unet_focal + dice_state_dict_best_loss28.pth',
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'wmh',
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))

trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
validationloader = DataLoader(validationset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

encoder = SA_UNet_Encoder(out_channels=2).to(c.device)
encoder.load_state_dict(torch.load(c.to_load_encoder_path)['model_state_dict'], strict=False)
projection_head = Projector(num_layers=4, layer_sizes=[64, 128, 256, 512], test=True).to(c.device)

projector_optimizer = optim.Adam([*encoder.parameters(), *projection_head.parameters()], lr = 0.001, eps = 0.0001)
projection_criterion = VoxelwiseSupConLoss_inImage(device=c.device).to(c.device)

projection_train_loss_list = []
projection_validation_loss_list = []

print()
print('Training Proxy.')
for epoch in range(1, c.num_epochs+1):

    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    encoder.train()
    projection_head.train()

    projection_train_loss = 0

    # patience -= 1
    # if patience == 0:
    #     print()
    #     print(f'Breaking at epoch #{epoch} due to lack of patience. Best validation loss was: {best_validation_loss}')

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:

    for data_idx, data in tqdm(enumerate(trainloader)):
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)
        # subtracted = data['subtracted'].to(device)

        to_projector, _ = encoder(image)

        if torch.unique(gt[:, 1]).shape[0] == 2:
            # brain_mask = torch.zeros_like(image)
            # brain_mask[image != 0] = 1
            # brain_mask = brain_mask.float().to(device)

            _, stacked_projections = projection_head(to_projector)
            projection = F.interpolate(stacked_projections, size=(128, 128, 128))

            projection_loss = projection_criterion(projection, gt)
            projection_train_loss += projection_loss.item()

            projector_optimizer.zero_grad()
            loss = projection_loss
            loss.backward()

            del projection
            del projection_loss
            # del brain_mask

        if (data_idx+1) % 6 == 0:
            projector_optimizer.step()

        del image
        del gt
        del to_projector
    
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    # break
    
    projection_train_loss_list.append(projection_train_loss / len(trainloader))
    print(f'Projection train loss at epoch #{epoch}: {projection_train_loss_list[-1]}')

    print()
    torch.cuda.empty_cache()

    # Validation loop
    encoder.eval()
    projection_head.eval()

    projection_validation_loss = 0

    for data in tqdm(validationloader):
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)
        # subtracted = data['subtracted'].to(device)

        to_projector, _ = encoder(image)

        if torch.unique(gt[:, 1]).shape[0] == 2:
            # brain_mask = torch.zeros_like(image)
            # brain_mask[image != 0] = 1
            # brain_mask = brain_mask.float().to(device)

            _, stacked_projections = projection_head(to_projector)
            projection = F.interpolate(stacked_projections, size=(128, 128, 128))

            projection_loss = projection_criterion(projection, gt)
            projection_validation_loss += projection_loss.item()


            del projection
            del projection_loss
            # del brain_mask

        del image
        del gt
        del to_projector

    
    projection_validation_loss_list.append(projection_validation_loss / len(validationloader))
    print(f'Projection validation loss at epoch #{epoch}: {projection_validation_loss_list[-1]}')

    np.save(f'./results/{c.projector_type}_{c.date}_losses.npy', [projection_train_loss_list, projection_validation_loss_list])

    if epoch % 10 == 0:
        torch.save(encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict{epoch}.pth')
        torch.save(projection_head.state_dict(), f'{c.to_save_folder}{c.projector_type}_{c.date}_state_dict{epoch}.pth')

    print()

torch.save(encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict{c.num_epochs+1}.pth')
torch.save(projection_head.state_dict(), f'{c.to_save_folder}{c.projector_type}_{c.date}_state_dict{c.num_epochs+1}.pth')

print()
print('Script executed.')