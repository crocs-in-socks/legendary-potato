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
    num_workers = 16,
    num_epochs = 100,
    date = '05_12_2023',
    to_save_folder = 'Dec05',
    to_load_folder = 'Dec05',
    device = 'cuda:1',
    proxy_type = 'VGGproxy',
    train_task = 'weightedBCE_>_VoxCFT_wBrainMask',
    to_load_encoder_path = 'VGGproxy_encoder_weightedBCEpretrain_withLRScheduler_05_12_2023_state_dict_best_loss26.pth',
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'simulated_lesions_on_brain_with_clean',
)

num_voxels = 10500

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
validationloader = DataLoader(validationset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

encoder = VGG3D_Encoder(input_channels=1).to(c.device)
encoder.load_state_dict(torch.load(c.to_load_encoder_path))
projection_head = Projector(num_layers=5, layer_sizes=[32, 64, 128, 256, 512]).to(c.device)

projector_optimizer = optim.Adam([*encoder.parameters(), *projection_head.parameters()], lr = 0.001, eps = 0.0001)
projection_criterion = VoxelwiseSupConLoss_inImage(device=c.device).to(c.device)

projection_train_loss_list = []
projection_validation_loss_list = []

best_validation_loss = None

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

    for data in tqdm(trainloader):
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)
        oneHot_label = data['lesion_labels'].float().to(c.device)
        # subtracted = data['subtracted'].to(device)

        to_projector, to_classifier = encoder(image)

        if torch.unique(gt[:, 1]).shape[0] == 2:
            brain_mask = torch.zeros_like(image)
            brain_mask[image != 0] = 1
            brain_mask = brain_mask.float().to(c.device)

            projection, stacked = projection_head(to_projector)
            projection = F.interpolate(stacked, size=(128, 128, 128))

            projection_loss = (projection_criterion(projection, gt) / num_voxels)
            projection_train_loss += projection_loss.item()

            projector_optimizer.zero_grad()
            loss = projection_loss
            loss.backward()
            projector_optimizer.step()

            del projection
            del projection_loss

        del image
        del gt
        del oneHot_label
        del to_projector
        del to_classifier
    
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
        oneHot_label = data['lesion_labels'].float().to(c.device)
        # subtracted = data['subtracted'].to(device)

        to_projector, to_classifier = encoder(image)

        if torch.unique(gt[:, 1]).shape[0] == 2:
            brain_mask = torch.zeros_like(image)
            brain_mask[image != 0] = 1
            brain_mask = brain_mask.float().to(c.device)

            projection, stacked = projection_head(to_projector)
            projection = F.interpolate(stacked, size=(128, 128, 128))

            projection_loss = (projection_criterion(projection, gt) / num_voxels)
            projection_validation_loss += projection_loss.item()

            del projection
            del projection_loss

        del image
        del gt
        del oneHot_label
        del to_projector
        del to_classifier

    
    projection_validation_loss_list.append(projection_validation_loss / len(validationloader))
    print(f'Projection validation loss at epoch #{epoch}: {projection_validation_loss_list[-1]}')

    np.save(f'./results/{c.proxy_type}_{c.train_task}_{c.date}_losses.npy', [projection_train_loss_list, projection_validation_loss_list])

    if epoch % 10 == 0:
        torch.save(encoder.state_dict(), f'{c.model_save_path}{c.encoder_type}_{c.date}_state_dict{epoch}.pth')
        torch.save(projection_head.state_dict(), f'{c.model_save_path}{c.projector_type}_{c.date}_state_dict{epoch}.pth')

    print()

torch.save(encoder.state_dict(), f'{c.model_save_path}{c.encoder_type}_{c.date}_state_dict{c.num_epochs+1}.pth')
torch.save(projection_head.state_dict(), f'{c.model_save_path}{c.projector_type}_{c.date}_state_dict{c.num_epochs+1}.pth')

print()
print('Script executed.')