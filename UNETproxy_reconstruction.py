import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from Utilities.Generic import *

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import numpy as np
from tqdm import tqdm

c = Constants(
    batch_size = 2,
    patience = 5,
    num_workers = 8,
    number_of_epochs = 100,
    date = '13_12_2023',
    to_save_folder = 'Dec13',
    to_load_folder = None,
    device = 'cuda:1',
    proxy_type = 'UNETproxy',
    train_task = 'reconstruction_simulated_noise_bg_>_sim_wmh_&_sim_brats_healthy_occluded',
    to_load_encoder_path = None,
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'sim_2211_wmh+sim_2211_brats',
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))

trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)
validationloader = DataLoader(validationset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

model = UNet(out_channels=1).to(c.device)

optimizer = optim.Adam(model.parameters(), lr = 0.001, eps = 0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=c.patience, min_lr=0.00001, verbose=True)

criterion = MS_SSIMLoss().to(c.device)

train_loss_list = []
validation_loss_list = []
best_validation_loss = None

print()
print('Training ReconProxy.')
for epoch in range(1, c.num_epochs+1):

    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    model.train()

    train_loss = 0

    # patience -= 1
    # if patience == 0:
    #     print()
    #     print(f'Breaking at epoch #{epoch} due to lack of patience. Best validation loss was: {best_validation_loss}')

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:

    for data in tqdm(trainloader):
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)
        masked_image = (image[:, 0] * gt[:, 1]).unsqueeze(1)

        # print(image.shape, gt.shape, masked_image.shape)

        reconstruction = model(masked_image)
        loss = criterion(reconstruction, image)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        del image
        del gt
        del reconstruction
        del loss
    
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    # break
    
    train_loss_list.append(train_loss / len(trainloader))
    print(f'Reconstruction train loss at epoch #{epoch}: {train_loss_list[-1]}')

    print()
    torch.cuda.empty_cache()

    # Validation loop
    model.eval()

    validation_loss = 0

    for data in tqdm(validationloader):
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)
        masked_image = (image[:, 0] * gt[:, 1]).unsqueeze(1)

        with torch.no_grad():
            reconstruction = model(masked_image)
            loss = criterion(reconstruction, image)
            validation_loss += loss.item()

        del image
        del gt
        del reconstruction
        del loss
    
    validation_loss_list.append(validation_loss / len(validationloader))
    print(f'Reconstruction validation loss at epoch #{epoch}: {validation_loss_list[-1]}')

    np.save(f'./results/{c.proxy_type}_{c.train_task}_{c.date}_losses.npy', [train_loss_list, validation_loss_list])

    if best_validation_loss is None:
        best_validation_loss = validation_loss_list[-1]
        
    elif validation_loss_list[-1] < best_validation_loss:
        best_validation_loss = validation_loss_list[-1]
        torch.save(model.state_dict(), f'{c.to_save_folder}{c.proxy_type}_{c.train_task}_{c.date}_state_dict_best_loss{epoch}.pth')

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'{c.to_save_folder}{c.proxy_type}_{c.train_task}_{c.date}_state_dict{epoch}.pth')

    print()

torch.save(model.state_dict(), f'{c.to_save_folder}{c.proxy_type}_{c.train_task}_{c.date}_state_dict{c.num_epochs+1}.pth')

print()
print('Script executed.')