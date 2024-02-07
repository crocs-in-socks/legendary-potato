import torch
from torch.utils.data import DataLoader, ConcatDataset

from Utilities.Generic import *

from ModelArchitecture.SlimUNETR.SlimUNETR import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

from skimage.feature import graycoprops

import numpy as np
from tqdm import tqdm
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

c = Constants(
    batch_size = 1,
    patience = None,
    num_workers = 16,
    num_epochs = None,
    date = None,
    to_save_folder = None,
    to_load_folder = None,
    device = 'cuda:1',
    proxy_type = '',
    train_task = '',
    to_load_encoder_path = None,
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'analogous_sim_bright'
)

eps = 1e-7
real_count = sim_count = 25

blur = RandomGaussianBlur3D(p=2.0)

simulated_trainset, simulated_validationset, simulated_testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
simulated_allset = ConcatDataset([simulated_trainset, simulated_validationset, simulated_testset])
real_trainset, real_validationset, real_testset = load_dataset('wmh', c.drive, ToTensor3D(labeled=True))
real_allset = ConcatDataset([real_trainset, real_validationset, real_testset])

simulated_dataloader = DataLoader(simulated_allset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)
real_dataloader = DataLoader(real_allset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)

model = SlimUNETR(in_channels=1, out_channels=2).to(c.device)
model.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/segmentation_models/slimunetr_focal + dice_state_dict_best_loss147.pth')['model_state_dict'])

def graycomatrix_3d(p):
    offset = 1
    levels = torch.tensor(256).to(c.device)
    results = torch.zeros((levels, levels, 1, 1)).to(c.device)
    p = p.flatten()
    maximum = p.shape[0]
    max_valid_idx = maximum - offset
    valid_indices = torch.arange(max_valid_idx)
    results[p[valid_indices], p[valid_indices + offset], 0, 0] += 1
    
    return results

real_contrast = []
real_homogeneity = []
real_dissimilarity = []
real_asm = []
glcm_real_list = []

model.eval()

count = 0
for data_idx, data in tqdm(enumerate(real_dataloader)):

    if count == real_count:
        break
    count += 1

    image = data['input'].to(c.device)
    gt = data['gt'].to(c.device)
    # mask = ((gt[:, 0] == 1) & (image[:, 0] > 0.1)).unsqueeze(0)
    # image[mask] = 0.3
    segmentation, hidden_states = model(image)

    # plt.figure(figsize=(20, 15))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image[0, 0, :, :, 64].detach().cpu())
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(gt[0, 1, :, :, 64].detach().cpu())
    # plt.colorbar()
    # plt.savefig(f'./temporary/real{data_idx}')
    # plt.close()

    # hidden_states = [(((state - torch.min(state)) / (torch.max(state) - torch.min(state))) + eps) for state in hidden_states]
    # hidden_states = [-(state * torch.log(state)) for state in hidden_states]

    for state_idx, state in enumerate(hidden_states):
        state = (((state - torch.min(state)) / (torch.max(state) - torch.min(state))) * 255).int()
        glcm = graycomatrix_3d(state[0]).detach().cpu().numpy()
        glcm_real_list.append(glcm)
        real_contrast.append(graycoprops(glcm, 'contrast')[0, 0])
        real_homogeneity.append(graycoprops(glcm, 'homogeneity')[0, 0])
        real_dissimilarity.append(graycoprops(glcm, 'dissimilarity')[0, 0])
        real_asm.append(graycoprops(glcm, 'ASM')[0, 0])

sim_contrast = []
sim_homogeneity = []
sim_dissimilarity = []
sim_asm = []
glcm_sim_list = []

count = 0
for data_idx, data in tqdm(enumerate(simulated_dataloader)):

    if count == sim_count:
        break
    count += 1

    image = data['input'].to(c.device)
    gt = data['gt'].to(c.device)
    # mask = ((gt[:, 0] == 1) & (image[:, 0] > 0.1)).unsqueeze(0)
    # image[mask] = 0.3
    segmentation, hidden_states = model(image)

    # plt.figure(figsize=(20, 15))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image[0, 0, :, :, 64].detach().cpu())
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(gt[0, 1, :, :, 64].detach().cpu())
    # plt.colorbar()
    # plt.savefig(f'./temporary/sim{data_idx}')
    # plt.close()

    # hidden_states = [(((state - torch.min(state)) / (torch.max(state) - torch.min(state))) + eps) for state in hidden_states]
    # hidden_states = [-(state * torch.log(state)) for state in hidden_states]

    for state_idx, state in enumerate(hidden_states):
        state = (((state - torch.min(state)) / (torch.max(state) - torch.min(state))) * 255).int()
        glcm = graycomatrix_3d(state[0]).detach().cpu().numpy()
        glcm_sim_list.append(glcm)
        sim_contrast.append(graycoprops(glcm, 'contrast')[0, 0])
        sim_homogeneity.append(graycoprops(glcm, 'homogeneity')[0, 0])
        sim_dissimilarity.append(graycoprops(glcm, 'dissimilarity')[0, 0])
        sim_asm.append(graycoprops(glcm, 'ASM')[0, 0])

blurred_sim_contrast = []
blurred_sim_homogeneity = []
blurred_sim_dissimilarity = []
blurred_sim_asm = []
blurred_glcm_sim_list = []

count = 0
for data_idx, data in tqdm(enumerate(simulated_dataloader)):

    if count == sim_count:
        break
    count += 1

    image = data['input']
    image = blur(image)
    image = image.to(c.device)
    gt = data['gt'].to(c.device)
    # mask = ((gt[:, 0] == 1) & (image[:, 0] > 0.1)).unsqueeze(0)
    # image[mask] = 0.3
    segmentation, hidden_states = model(image)

    # plt.figure(figsize=(20, 15))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image[0, 0, :, :, 64].detach().cpu())
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(gt[0, 1, :, :, 64].detach().cpu())
    # plt.colorbar()
    # plt.savefig(f'./temporary/blurredsim{data_idx}')
    # plt.close()

    # hidden_states = [(((state - torch.min(state)) / (torch.max(state) - torch.min(state))) + eps) for state in hidden_states]
    # hidden_states = [-(state * torch.log(state)) for state in hidden_states]

    for state_idx, state in enumerate(hidden_states):
        state = (((state - torch.min(state)) / (torch.max(state) - torch.min(state))) * 255).int()
        glcm = graycomatrix_3d(state[0]).detach().cpu().numpy()
        blurred_glcm_sim_list.append(glcm)
        blurred_sim_contrast.append(graycoprops(glcm, 'contrast')[0, 0])
        blurred_sim_homogeneity.append(graycoprops(glcm, 'homogeneity')[0, 0])
        blurred_sim_dissimilarity.append(graycoprops(glcm, 'dissimilarity')[0, 0])
        blurred_sim_asm.append(graycoprops(glcm, 'ASM')[0, 0])

print('Number of real:', len(glcm_real_list))
print('Number of simulated:', len(glcm_sim_list))
print('Number of blurred simulated:', len(blurred_glcm_sim_list))

count = 0
for real, sim in zip(glcm_real_list, glcm_sim_list):
    diff = sim - real
    plt.imshow(diff[:, :, 0, 0])
    plt.colorbar()
    plt.savefig(f'./temporary/diff/{count}')
    plt.close()
    count += 1

contrast = real_contrast + sim_contrast
dissimilarity = real_dissimilarity + sim_dissimilarity
homogeneity = real_homogeneity + sim_homogeneity
asm = real_asm + sim_asm

contrast_ylim = max(contrast)
dissimilarity_ylim = max(dissimilarity)
homogeneity_ylim = max(homogeneity)
asm_ylim = max(asm)

plt.figure(figsize=(40, 20))
plt.subplot(1, 2, 1)
plt.bar(range(sim_count*3), blurred_sim_contrast)
plt.ylim(0, contrast_ylim)
plt.ylabel('GLCM blurred_sim Contrast')
plt.subplot(1, 2, 2)
plt.bar(range(sim_count*3), sim_contrast)
plt.ylim(0, contrast_ylim)
plt.ylabel('GLCM sim Contrast')
plt.grid()

plt.savefig('./temporary/-Contrast')
plt.close()

plt.figure(figsize=(40, 20))
plt.subplot(1, 2, 1)
plt.bar(range(sim_count*3), blurred_sim_dissimilarity)
plt.ylim(0, dissimilarity_ylim)
plt.ylabel('GLCM blurred_sim Dissimilarity')
plt.subplot(1, 2, 2)
plt.bar(range(sim_count*3), sim_dissimilarity)
plt.ylim(0, dissimilarity_ylim)
plt.ylabel('GLCM sim Dissimilarity')
plt.grid()

plt.savefig('./temporary/-Dissimilarity')
plt.close()

plt.figure(figsize=(40, 20))
plt.subplot(1, 2, 1)
plt.bar(range(sim_count*3), blurred_sim_homogeneity)
plt.ylim(0, homogeneity_ylim)
plt.ylabel('GLCM blurred_sim Homogeneity')
plt.subplot(1, 2, 2)
plt.bar(range(sim_count*3), sim_homogeneity)
plt.ylim(0, homogeneity_ylim)
plt.ylabel('GLCM sim Homogeneity')
plt.grid()

plt.savefig('./temporary/-Homogeneity')
plt.close()

plt.figure(figsize=(40, 20))
plt.subplot(1, 2, 1)
plt.bar(range(sim_count*3), blurred_sim_asm)
plt.ylim(0, asm_ylim)
plt.ylabel('GLCM blurred_sim ASM')
plt.subplot(1, 2, 2)
plt.bar(range(sim_count*3), sim_asm)
plt.ylim(0, asm_ylim)
plt.ylabel('GLCM sim ASM')
plt.grid()

plt.savefig('./temporary/-ASM')
plt.close()

# real_state0_contrast = real_contrast[::3]
# real_state1_contrast = real_contrast[1::3]
# real_state2_contrast = real_contrast[2::3]

# real_state0_dissimilarity = real_dissimilarity[::3]
# real_state1_dissimilarity = real_dissimilarity[1::3]
# real_state2_dissimilarity = real_dissimilarity[2::3]

# real_state0_homogeneity = real_homogeneity[::3]
# real_state1_homogeneity = real_homogeneity[1::3]
# real_state2_homogeneity = real_homogeneity[2::3]

# real_state0_asm = real_asm[::3]
# real_state1_asm = real_asm[1::3]
# real_state2_asm = real_asm[2::3]

# sim_state0_contrast = contrast[::3]
# sim_state1_contrast = contrast[1::3]
# sim_state2_contrast = contrast[2::3]

# sim_state0_dissimilarity = dissimilarity[::3]
# sim_state1_dissimilarity = dissimilarity[1::3]
# sim_state2_dissimilarity = dissimilarity[2::3]

# sim_state0_homogeneity = homogeneity[::3]
# sim_state1_homogeneity = homogeneity[1::3]
# sim_state2_homogeneity = homogeneity[2::3]

# sim_state0_asm = asm[::3]
# sim_state1_asm = asm[1::3]
# sim_state2_asm = asm[2::3]

# print('Contrast kurtosis:')
# print(kurtosis(real_state0_contrast), kurtosis(sim_state0_contrast))
# print(kurtosis(real_state1_contrast), kurtosis(sim_state1_contrast))
# print(kurtosis(real_state2_contrast), kurtosis(sim_state2_contrast))
# print()
# print('Homogeneity kurtosis:')
# print(kurtosis(real_state0_homogeneity), kurtosis(sim_state0_homogeneity))
# print(kurtosis(real_state1_homogeneity), kurtosis(sim_state1_homogeneity))
# print(kurtosis(real_state2_homogeneity), kurtosis(sim_state2_homogeneity))
# print()
# print('Dissimilarity kurtosis:')
# print(kurtosis(real_state0_dissimilarity), kurtosis(sim_state0_dissimilarity))
# print(kurtosis(real_state1_dissimilarity), kurtosis(sim_state1_dissimilarity))
# print(kurtosis(real_state2_dissimilarity), kurtosis(sim_state2_dissimilarity))
# print()
# print('ASM kurtosis:')
# print(kurtosis(real_state0_asm), kurtosis(sim_state0_asm))
# print(kurtosis(real_state1_asm), kurtosis(sim_state1_asm))
# print(kurtosis(real_state2_asm), kurtosis(sim_state2_asm))
# print()

# plt.figure(figsize=(20, 20))
# # plt.scatter(contrast[len(glcm_real_list)::3], homogeneity[len(glcm_real_list)::3], label='state0 simulated data')
# # plt.scatter(contrast[len(glcm_real_list)+1::3], homogeneity[len(glcm_real_list)+1::3], label='state1 simulated data')
# # plt.scatter(contrast[len(glcm_real_list)+2::3], homogeneity[len(glcm_real_list)+2::3], label='state2 simulated data')

# # plt.scatter(contrast[:len(glcm_real_list):3], homogeneity[:len(glcm_real_list):3], label='state0 real data')
# # plt.scatter(contrast[1:len(glcm_real_list):3], homogeneity[1:len(glcm_real_list):3], label='state1 real data')
# # plt.scatter(contrast[2:len(glcm_real_list):3], homogeneity[2:len(glcm_real_list):3], label='state2 real data')

# plt.scatter(contrast[len(glcm_real_list):], homogeneity[len(glcm_real_list):], label='real data')
# plt.scatter(contrast[:len(glcm_real_list)], homogeneity[:len(glcm_real_list)], label='simulated data')

# plt.xlabel('GLCM Contrast')
# plt.ylabel('GLCM Homogeneity')
# plt.grid()
# plt.legend()

# plt.savefig('./temporary/-Contrast-Homogeneity')
# plt.close()

# ###

# plt.figure(figsize=(20, 20))
# # plt.scatter(contrast[len(glcm_real_list)::3], dissimilarity[len(glcm_real_list)::3], label='state0 simulated data')
# # plt.scatter(contrast[len(glcm_real_list)+1::3], dissimilarity[len(glcm_real_list)+1::3], label='state1 simulated data')
# # plt.scatter(contrast[len(glcm_real_list)+2::3], dissimilarity[len(glcm_real_list)+2::3], label='state2 simulated data')

# # plt.scatter(contrast[:len(glcm_real_list):3], dissimilarity[:len(glcm_real_list):3], label='state0 real data')
# # plt.scatter(contrast[1:len(glcm_real_list):3], dissimilarity[1:len(glcm_real_list):3], label='state1 real data')
# # plt.scatter(contrast[2:len(glcm_real_list):3], dissimilarity[2:len(glcm_real_list):3], label='state2 real data')

# plt.scatter(contrast[len(glcm_real_list):], dissimilarity[len(glcm_real_list):], label='real data')
# plt.scatter(contrast[:len(glcm_real_list)], dissimilarity[:len(glcm_real_list)], label='simulated data')

# plt.xlabel('GLCM Contrast')
# plt.ylabel('GLCM Dissimilarity')
# plt.grid()
# plt.legend()

# plt.savefig('./temporary/-Contrast-Dissimilarity')
# plt.close()

# ###

# plt.figure(figsize=(20, 20))
# # plt.scatter(contrast[len(glcm_real_list)::3], asm[len(glcm_real_list)::3], label='state0 simulated data')
# # plt.scatter(contrast[len(glcm_real_list)+1::3], asm[len(glcm_real_list)+1::3], label='state1 simulated data')
# # plt.scatter(contrast[len(glcm_real_list)+2::3], asm[len(glcm_real_list)+2::3], label='state2 simulated data')

# # plt.scatter(contrast[:len(glcm_real_list):3], asm[:len(glcm_real_list):3], label='state0 real data')
# # plt.scatter(contrast[1:len(glcm_real_list):3], asm[1:len(glcm_real_list):3], label='state1 real data')
# # plt.scatter(contrast[2:len(glcm_real_list):3], asm[2:len(glcm_real_list):3], label='state2 real data')

# plt.scatter(contrast[len(glcm_real_list):], asm[len(glcm_real_list):], label='real data')
# plt.scatter(contrast[:len(glcm_real_list)], asm[:len(glcm_real_list)], label='simulated data')

# plt.xlabel('GLCM Contrast')
# plt.ylabel('GLCM ASM')
# plt.grid()
# plt.legend()

# plt.savefig('./temporary/-Contrast-ASM')
# plt.close()

# ###

# plt.figure(figsize=(20, 20))
# # plt.scatter(homogeneity[len(glcm_real_list)::3], dissimilarity[len(glcm_real_list)::3], label='state0 simulated data')
# # plt.scatter(homogeneity[len(glcm_real_list)+1::3], dissimilarity[len(glcm_real_list)+1::3], label='state1 simulated data')
# # plt.scatter(homogeneity[len(glcm_real_list)+2::3], dissimilarity[len(glcm_real_list)+2::3], label='state2 simulated data')

# # plt.scatter(homogeneity[:len(glcm_real_list):3], dissimilarity[:len(glcm_real_list):3], label='state0 real data')
# # plt.scatter(homogeneity[1:len(glcm_real_list):3], dissimilarity[1:len(glcm_real_list):3], label='state1 real data')
# # plt.scatter(homogeneity[2:len(glcm_real_list):3], dissimilarity[2:len(glcm_real_list):3], label='state2 real data')

# plt.scatter(homogeneity[len(glcm_real_list):], dissimilarity[len(glcm_real_list):], label='real data')
# plt.scatter(homogeneity[:len(glcm_real_list)], dissimilarity[:len(glcm_real_list)], label='simulated data')

# plt.xlabel('GLCM Homogeneity')
# plt.ylabel('GLCM Dissimilarity')
# plt.grid()
# plt.legend()

# plt.savefig('./temporary/-Homogeneity-Dissimilarity')
# plt.close()

# ###

# plt.figure(figsize=(20, 20))
# # plt.scatter(homogeneity[len(glcm_real_list)::3], asm[len(glcm_real_list)::3], label='state0 simulated data')
# # plt.scatter(homogeneity[len(glcm_real_list)+1::3], asm[len(glcm_real_list)+1::3], label='state1 simulated data')
# # plt.scatter(homogeneity[len(glcm_real_list)+2::3], asm[len(glcm_real_list)+2::3], label='state2 simulated data')

# # plt.scatter(homogeneity[:len(glcm_real_list):3], asm[:len(glcm_real_list):3], label='state0 real data')
# # plt.scatter(homogeneity[1:len(glcm_real_list):3], asm[1:len(glcm_real_list):3], label='state1 real data')
# # plt.scatter(homogeneity[2:len(glcm_real_list):3], asm[2:len(glcm_real_list):3], label='state2 real data')

# plt.scatter(homogeneity[len(glcm_real_list):], asm[len(glcm_real_list):], label='real data')
# plt.scatter(homogeneity[:len(glcm_real_list)], asm[:len(glcm_real_list)], label='simulated data')

# plt.xlabel('GLCM Homogeneity')
# plt.ylabel('GLCM ASM')
# plt.grid()
# plt.legend()

# plt.savefig('./temporary/-Homogeneity-ASM')
# plt.close()

# ###

# plt.figure(figsize=(20, 20))
# # plt.scatter(dissimilarity[len(glcm_real_list)::3], asm[len(glcm_real_list)::3], label='state0 simulated data')
# # plt.scatter(dissimilarity[len(glcm_real_list)+1::3], asm[len(glcm_real_list)+1::3], label='state1 simulated data')
# # plt.scatter(dissimilarity[len(glcm_real_list)+2::3], asm[len(glcm_real_list)+2::3], label='state2 simulated data')

# # plt.scatter(dissimilarity[:len(glcm_real_list):3], asm[:len(glcm_real_list):3], label='state0 real data')
# # plt.scatter(dissimilarity[1:len(glcm_real_list):3], asm[1:len(glcm_real_list):3], label='state1 real data')
# # plt.scatter(dissimilarity[2:len(glcm_real_list):3], asm[2:len(glcm_real_list):3], label='state2 real data')

# plt.scatter(dissimilarity[len(glcm_real_list):], asm[len(glcm_real_list):], label='real data')
# plt.scatter(dissimilarity[:len(glcm_real_list)], asm[:len(glcm_real_list)], label='simulated data')

# plt.xlabel('GLCM Dissimilarity')
# plt.ylabel('GLCM ASM')
# plt.grid()
# plt.legend()

# plt.savefig('./temporary/-Dissimilarity-ASM')
# plt.close()

print()
print('Script executed.')