import time
import glob
import numpy as np
import seaborn as sns
from einops import rearrange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go

from ModelArchitecture.Transformations import *
from ImageLoader.ImageLoader3D import ImageLoader3D
from ImageLoader.ContrastiveLoader3D import ContrastivePatchLoader3D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from ModelArchitecture.DUCK_Net import DuckNet
from ModelArchitecture.Encoders import *
from tqdm import tqdm

from skimage.morphology import binary_dilation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

batch_size = 8
num_workers = 16
device = 'cuda:0'

encoder_path = '/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/Dec06/UNETproxy_encoder_weightedBCEpretrain_withLRScheduler_05_12_2023_state_dict_best_loss34.pth'

Sim1000_train_data_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*FLAIR.nii.gz'))
Sim1000_train_gt_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*mask.nii.gz'))
Sim1000_train_json_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*.json'))

Sim1000_validation_data_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*FLAIR.nii.gz'))
Sim1000_validation_gt_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*mask.nii.gz'))
Sim1000_validation_json_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*.json'))

sim2211_train_data_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TrainSet/*FLAIR.nii.gz'))
sim2211_train_gt_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TrainSet/*mask.nii.gz'))
sim2211_train_json_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TrainSet/*.json'))

sim2211_validation_data_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/ValSet/*FLAIR.nii.gz'))
sim2211_validation_gt_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/ValSet/*mask.nii.gz'))
sim2211_validation_json_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/ValSet/*.json'))

clean_data_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
clean_gt_paths = sorted(glob.glob('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))

composed_transform = transforms.Compose([
        ToTensor3D(labeled=True)
    ])

Sim1000_trainset = ImageLoader3D(paths=Sim1000_train_data_paths, gt_paths=Sim1000_train_gt_paths, json_paths=Sim1000_train_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)
Sim1000_validationset = ImageLoader3D(paths=Sim1000_validation_data_paths, gt_paths=Sim1000_validation_gt_paths, json_paths=Sim1000_validation_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)

sim2211_trainset = ImageLoader3D(paths=sim2211_train_data_paths, gt_paths=sim2211_train_gt_paths, json_paths=sim2211_train_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)
sim2211_validationset = ImageLoader3D(paths=sim2211_validation_data_paths, gt_paths=sim2211_validation_gt_paths, json_paths=sim2211_validation_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)

clean = ImageLoader3D(paths=clean_data_paths, gt_paths=clean_gt_paths, json_paths=None, image_size=128, type_of_imgs='nifty', transform=composed_transform)
train_size = int(0.8 * len(clean))
validation_size = len(clean) - train_size
clean_trainset, clean_validationset = random_split(clean, (train_size, validation_size))

trainset = ConcatDataset([Sim1000_trainset, sim2211_trainset, clean_trainset])
validationset = ConcatDataset([Sim1000_validationset, sim2211_validationset, clean_validationset])

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

encoder = SA_UNet_Encoder(out_channels=2).to(device)
encoder.load_state_dict(torch.load(encoder_path))
encoder.eval()

activations = {}
input_labels = []

def getActivation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def get_layer_TSNE(layer_name, folder_name, perplexity):

    layer = []
    input_labels = []

    for i in tqdm(range(len(trainset))):
        torch.cuda.empty_cache()

        hook1 = getattr(encoder, layer_name).register_forward_hook(getActivation(layer_name))

        sample_dict = trainset[i]
        sample_input = sample_dict['input'].to(device).unsqueeze(0)
        sample_labels = sample_dict['lesion_labels'].float().to(device)

        _, to_classifier = encoder(sample_input)
        
        layer.append(activations[layer_name].cpu())
        input_labels.append(sample_labels)

    print()

    hook1.remove()

    layer = torch.stack(layer)
    input_labels = torch.stack(input_labels)

    print(f'layer shape: {layer.shape}')
    print(f'input labels shape: {input_labels.shape}')
    # print(f'Input labels shape: {input_labels.shape}')
    # layer = layer.squeeze(1)
    # input_labels = input_labels.squeeze(1)
    layer = torch.reshape(layer, shape=(layer.shape[0], -1))
    # input_labels = torch.reshape(input_labels, shape=(input_labels.shape[0] * input_labels.shape[1], -1))
    # input_labels = input_labels[:, 0]
    print(f'layer shape: {layer.shape}')
    print(f'input labels shape: {input_labels.shape}')

    pca_result = PCA(n_components=128).fit_transform(layer)
    print(pca_result.shape)

    print()
    start_time = time.time()
    print(f'Started t-SNE.')

    tsne_pca_results = TSNE(n_components=2, perplexity=perplexity, n_iter=1000).fit_transform(pca_result)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print(f'TSNE result: {tsne_pca_results.shape}')
    print(f'Time taken for TSNE: {time_elapsed}')

    healthy_points = []
    very_small_lesions = []
    small_lesions = []
    medium_lesions = []
    large_lesions = []

    ### For healthy samples

    for label, point in zip(input_labels, tsne_pca_results):
        if label[0] == 1:
            healthy_points.append(point)
        if label[1] == 1 and label[0] == 0:
            very_small_lesions.append(point)
        if label[2] == 1 and label[0] == 0:
            small_lesions.append(point)
        if label[3] == 1 and label[0] == 0:
            medium_lesions.append(point)
        if label[4] == 1 and label[0] == 0:
            large_lesions.append(point)

    all_points = healthy_points + very_small_lesions + small_lesions + medium_lesions + large_lesions
    all_labels = [1]*len(healthy_points) + [0]*len(very_small_lesions + small_lesions + medium_lesions + large_lesions)

    all_points = np.stack(all_points)
    print(all_points.shape)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.scatterplot(
        x=all_points[:,0],
        y=all_points[:,1],
        hue=all_labels,
        alpha=0.5
    )
    leg = ax.axes.get_legend()
    new_labels = ['Out of Class', 'In Class']

    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    plt.title('Healthy Samples')
    plt.savefig(f'./Healthy_Samples_tsne')
    plt.close()

    ### For very small lesions

    for label, point in zip(input_labels, tsne_pca_results):
        if label[0] == 1 and label[1] == 0:
            healthy_points.append(point)
        if label[1] == 1:
            very_small_lesions.append(point)
        if label[2] == 1 and label[1] == 0:
            small_lesions.append(point)
        if label[3] == 1 and label[1] == 0:
            medium_lesions.append(point)
        if label[4] == 1 and label[1] == 0:
            large_lesions.append(point)

    all_points = very_small_lesions + healthy_points + small_lesions + medium_lesions + large_lesions
    all_labels = [1]*len(very_small_lesions) + [0]*len(healthy_points + small_lesions + medium_lesions + large_lesions)

    all_points = np.stack(all_points)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.scatterplot(
        x=all_points[:,0],
        y=all_points[:,1],
        hue=all_labels,
        alpha=0.5
    )
    leg = ax.axes.get_legend()
    new_labels = ['Out of Class', 'In Class']

    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    plt.title('very_small_lesions')
    plt.savefig(f'./very_small_lesions_tsne')
    plt.close()

    ### For small lesions

    for label, point in zip(input_labels, tsne_pca_results):
        if label[0] == 1 and label[2] == 0:
            healthy_points.append(point)
        if label[1] == 1 and label[2] == 0:
            very_small_lesions.append(point)
        if label[2] == 1:
            small_lesions.append(point)
        if label[3] == 1 and label[2] == 0:
            medium_lesions.append(point)
        if label[4] == 1 and label[2] == 0:
            large_lesions.append(point)

    all_points = small_lesions + very_small_lesions + healthy_points + medium_lesions + large_lesions
    all_labels = [1]*len(small_lesions) + [0]*len(healthy_points + very_small_lesions + medium_lesions + large_lesions)

    all_points = np.stack(all_points)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.scatterplot(
        x=all_points[:,0],
        y=all_points[:,1],
        hue=all_labels,
        alpha=0.5
    )
    leg = ax.axes.get_legend()
    new_labels = ['Out of Class', 'In Class']

    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    plt.title('small_lesions')
    plt.savefig(f'./small_lesions_tsne')
    plt.close()

    ### For medium lesions

    for label, point in zip(input_labels, tsne_pca_results):
        if label[0] == 1 and label[3] == 0:
            healthy_points.append(point)
        if label[1] == 1 and label[3] == 0:
            very_small_lesions.append(point)
        if label[2] == 1 and label[3] == 0:
            small_lesions.append(point)
        if label[3] == 1:
            medium_lesions.append(point)
        if label[4] == 1 and label[3] == 0:
            large_lesions.append(point)

    all_points = medium_lesions + small_lesions + very_small_lesions + healthy_points + large_lesions
    all_labels = [1]*len(medium_lesions) + [0]*len(healthy_points + very_small_lesions + small_lesions + large_lesions)

    all_points = np.stack(all_points)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.scatterplot(
        x=all_points[:,0],
        y=all_points[:,1],
        hue=all_labels,
        alpha=0.5
    )
    leg = ax.axes.get_legend()
    new_labels = ['Out of Class', 'In Class']

    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    plt.title('meium_lesions')
    plt.savefig(f'./medium_lesions_tsne')
    plt.close()

    ### For large lesions

    for label, point in zip(input_labels, tsne_pca_results):
        if label[0] == 1 and label[4] == 0:
            healthy_points.append(point)
        if label[1] == 1 and label[4] == 0:
            very_small_lesions.append(point)
        if label[2] == 1 and label[4] == 0:
            small_lesions.append(point)
        if label[3] == 1 and label[4] == 0:
            medium_lesions.append(point)
        if label[4] == 1:
            large_lesions.append(point)

    all_points = large_lesions + medium_lesions + small_lesions + very_small_lesions + healthy_points
    all_labels = [1]*len(large_lesions) + [0]*len(healthy_points + very_small_lesions + small_lesions + medium_lesions)

    all_points = np.stack(all_points)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.scatterplot(
        x=all_points[:,0],
        y=all_points[:,1],
        hue=all_labels,
        alpha=0.5
    )
    leg = ax.axes.get_legend()
    new_labels = ['Out of Class', 'In Class']

    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    plt.title('large_lesions')
    plt.savefig(f'./large_lesions_tsne')
    plt.close()


get_layer_TSNE('encoder4', 'temporary', 30)

print()
print('Script complete.')