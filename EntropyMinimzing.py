import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Utilities.Generic import *

from ModelArchitecture.SlimUNETR.SlimUNETR import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

import numpy as np
from tqdm import tqdm
from itertools import cycle

c = Constants(
    batch_size = 1,
    patience = 10,
    num_workers = 16,
    num_epochs = 200,
    date = '22_12_2023',
    to_save_folder = 'Dec22',
    to_load_folder = None,
    device = 'cuda:0',
    proxy_type = 'SlimUNETR_dynamic_entropy_maps',
    train_task = 'normalized_KLD_&_FOCALdice',
    to_load_encoder_path = None,
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'sim_2211_wmh'
)

eps = 1e-7
KLD_weight = 10

simulated_trainset, simulated_validationset, simulated_testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
real_trainset, real_validationset, real_testset = load_dataset('wmh', c.drive, ToTensor3D(labeled=True))

simulated_trainloader = DataLoader(simulated_trainset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
simulated_validationloader = DataLoader(simulated_validationset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
simulated_testloader = DataLoader(simulated_testset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

real_trainloader = DataLoader(real_trainset, batch_size=1, shuffle=True, num_workers=c.num_workers)
real_validationloader = DataLoader(real_validationset, batch_size=1, shuffle=True, num_workers=c.num_workers)
real_testloader = DataLoader(real_testset, batch_size=1, shuffle=True, num_workers=c.num_workers)

infinite_real_trainloader = cycle(real_trainloader)
infinite_real_validationloader = cycle(real_validationloader)
infinite_real_testloader = cycle(real_testloader)

model = SlimUNETR(in_channels=1, out_channels=2).to(c.device)
model.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/segmentation_models/slimunetr_focal + dice_state_dict_best_loss147.pth')['model_state_dict'])

KLD_criterion = nn.KLDivLoss(reduction='none').to(c.device)
segmentation_criterion = FocalDICELoss().to(c.device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=0.0001)
optimizer.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/segmentation_models/slimunetr_focal + dice_state_dict_best_loss147.pth')['optimizer_state_dict'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=c.patience, factor=0.5, min_lr=1e-4, verbose=True)
scheduler.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/segmentation_models/slimunetr_focal + dice_state_dict_best_loss147.pth')['lr_scheduler_state_dict'])

### getting average static entropy maps

# static_entropy_maps = [
#     torch.zeros(size=(1, 24, 32, 32, 32)).to(c.device),
#     torch.zeros(size=(1, 48, 16, 16, 16)).to(c.device),
#     torch.zeros(size=(1, 60, 8, 8, 8)).to(c.device)
# ]
# dynamic_model.eval()

# print()
# print('Calculating average static entropy maps.')

# for data in tqdm(static_trainloader):
#     image = data['input'].to(c.device)
#     _, static_hidden_states = dynamic_model(image)
#     for idx, state in enumerate(static_hidden_states):
#         state = ((state - torch.min(state)) / (torch.max(state) - torch.min(state))) + eps
#         static_entropy_maps[idx] += (-(state * torch.log(state)) / len(static_trainloader))

###

train_loss_list = []
train_dice_score_list = []

validation_loss_list = []
validation_dice_score_list = []

best_validation_score = None

print()
print('Finetuning model using entropy maps.')

for epoch in range(1, c.num_epochs+1):

    print()
    print(f'Epoch #{epoch}')

    # Train loop
    model.train()

    train_loss = 0
    train_dice_score = 0
    count = 0

    for real_data, simulated_data in tqdm(zip(infinite_real_trainloader, simulated_trainloader)):
        
        count += 1

        real_image = real_data['input'].to(c.device)
        real_gt = real_data['gt'].to(c.device)
        simulated_image = simulated_data['input'].to(c.device)
        simulated_gt = simulated_data['gt'].to(c.device)

        real_segmentation, real_hidden_states = model(real_image)
        simulated_segmentation, simulated_hidden_states = model(simulated_image)

        real_hidden_states = [(((state - torch.min(state)) / (torch.max(state) - torch.min(state))) + eps) for state in real_hidden_states]
        simulated_hidden_states = [(((state - torch.min(state)) / (torch.max(state) - torch.min(state))) + eps) for state in simulated_hidden_states]

        real_entropy_maps = [-(state * torch.log(state)) for state in real_hidden_states]
        simulated_entropy_maps = [-(state * torch.log(state)) for state in simulated_hidden_states]

        optimizer.zero_grad()

        # loss = sum([KLD_criterion(F.log_softmax(dynamic_map, dim=1), F.softmax(static_map, dim=1)).sum() for dynamic_map, static_map in zip(dynamic_entropy_maps, static_entropy_maps)]) / entropy_weight + segmentation_criterion(segmentation[:, 1], gt[:, 1])

        divergence_loss = sum([KLD_criterion(F.log_softmax(simulated_map, dim=1), F.softmax(real_map, dim=1)).sum() for real_map, simulated_map in zip(real_entropy_maps, simulated_entropy_maps)])
        segmentation_loss = segmentation_criterion(real_segmentation[:, 1], real_gt[:, 1]) + segmentation_criterion(simulated_segmentation[:, 1], simulated_gt[:, 1])

        loss = (divergence_loss / KLD_weight) + segmentation_loss
        train_loss += loss.item()

        optimizer.step()

        train_dice_score += (Dice_Score(real_segmentation[:, 1].detach().cpu().numpy(), real_gt[:,1].detach().cpu().numpy()).item() + Dice_Score(simulated_segmentation[:, 1].detach().cpu().numpy(), simulated_gt[:,1].detach().cpu().numpy()).item()) / 2

        del real_image
        del simulated_image
        del real_gt
        del simulated_gt
        del real_segmentation
        del simulated_segmentation
        del real_hidden_states
        del simulated_hidden_states
        del real_entropy_maps
        del simulated_entropy_maps
        del divergence_loss
        del segmentation_loss
        del loss
    
    train_loss_list.append(train_loss / count)
    train_dice_score_list.append(train_dice_score / count)
    print(f'Train loss at epoch#{epoch}: {train_loss_list[-1]}')
    print(f'Train dice score at epoch#{epoch}: {train_dice_score_list[-1]}')

    print()

    # Validation loop

    model.eval()

    validation_loss = 0
    validation_dice_score = 0
    count = 0

    for real_data, simulated_data in tqdm(zip(infinite_real_validationloader, simulated_validationloader)):

        count += 1

        real_image = real_data['input'].to(c.device)
        real_gt = real_data['gt'].to(c.device)
        simulated_image = simulated_data['input'].to(c.device)
        simulated_gt = simulated_data['gt'].to(c.device)

        real_segmentation, real_hidden_states = model(real_image)
        simulated_segmentation, simulated_hidden_states = model(simulated_image)

        real_hidden_states = [(((state - torch.min(state)) / (torch.max(state) - torch.min(state))) + eps) for state in real_hidden_states]
        simulated_hidden_states = [(((state - torch.min(state)) / (torch.max(state) - torch.min(state))) + eps) for state in simulated_hidden_states]

        real_entropy_maps = [-(state * torch.log(state)) for state in real_hidden_states]
        simulated_entropy_maps = [-(state * torch.log(state)) for state in simulated_hidden_states]

        # loss = sum([KLD_criterion(F.log_softmax(dynamic_map, dim=1), F.softmax(static_map, dim=1)).sum() for dynamic_map, static_map in zip(dynamic_entropy_maps, static_entropy_maps)]) / entropy_weight + segmentation_criterion(segmentation[:, 1], gt[:, 1])

        divergence_loss = sum([KLD_criterion(F.log_softmax(simulated_map, dim=1), F.softmax(real_map, dim=1)).sum() for real_map, simulated_map in zip(real_entropy_maps, simulated_entropy_maps)])
        segmentation_loss = segmentation_criterion(real_segmentation[:, 1], real_gt[:, 1]) + segmentation_criterion(simulated_segmentation[:, 1], simulated_gt[:, 1])

        loss = (divergence_loss / KLD_weight) + segmentation_loss
        validation_loss += loss.item()

        validation_dice_score += (Dice_Score(real_segmentation[:, 1].detach().cpu().numpy(), real_gt[:,1].detach().cpu().numpy()).item() + Dice_Score(simulated_segmentation[:, 1].detach().cpu().numpy(), simulated_gt[:,1].detach().cpu().numpy()).item()) / 2

        del real_image
        del simulated_image
        del real_gt
        del simulated_gt
        del real_segmentation
        del simulated_segmentation
        del real_hidden_states
        del simulated_hidden_states
        del real_entropy_maps
        del simulated_entropy_maps
        del divergence_loss
        del segmentation_loss
        del loss
    
    validation_loss_list.append(validation_loss / count)
    validation_dice_score_list.append(validation_dice_score / count)
    print(f'Validation loss at epoch#{epoch}: {validation_loss_list[-1]}')
    print(f'Validation dice score at epoch#{epoch}: {validation_dice_score_list[-1]}')

    scheduler.step(validation_loss_list[-1])

    np.save(f'./results/{c.proxy_type}_{c.train_task}_{c.date}_losses.npy', [train_loss_list, train_dice_score_list, validation_loss_list, validation_dice_score_list])

    if best_validation_score is None:
        best_validation_score = validation_dice_score_list[-1]
    
    elif validation_dice_score_list[-1] > best_validation_score:
        best_validation_score = validation_dice_score_list[-1]
        torch.save(model.state_dict(), f'{c.to_save_folder}{c.proxy_type}_{c.train_task}_{c.date}_state_dict_best_score{epoch}.pth')

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'{c.to_save_folder}{c.proxy_type}_{c.train_task}_{c.date}_state_dict{epoch}.pth')
print()
print('Script executed.')