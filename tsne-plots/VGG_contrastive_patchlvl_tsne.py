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
from torch.utils.data import DataLoader, ConcatDataset
from ModelArchitecture.DUCK_Net import DuckNet
from ModelArchitecture.Encoders import *
from tqdm import tqdm

from skimage.morphology import binary_dilation
from sklearn.manifold import TSNE

VGG_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/contrastive_models/'
patches_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/patches/'

batch_size = 1
number_of_epochs = 50
number_of_test_samples = 20
number_of_features = 0
patch_size = 32
device = 'cuda:1'

model_type = f'VGG3D_patchlvl_{patch_size}'
classifier_type = f'classifier_patchlvl_{patch_size}'

test_sample_paths = []
test_patch_labels = []
oneHot_test_patch_labels = []

print('Generating labels for testset.')
for sample_idx in tqdm(range(number_of_test_samples)):
    patch_dict = np.load(f'{patches_path}/patch_size{patch_size}_test_patch_and_label_idx{sample_idx}.npy', allow_pickle=True).item()

    test_sample_paths.append(f'{patches_path}/patch_size{patch_size}_test_patch_and_label_idx{sample_idx}.npy')

    patches = patch_dict['patches']
    label_patches = patch_dict['labels']

    number_of_features = patches.shape[2]

    temp_labels = []
    oneHot_temp_labels = []

    number_of_patches = label_patches.shape[1]
    for patch_idx in range(number_of_patches):
        unique_values = torch.unique(label_patches[0, patch_idx, 0])
        
        if unique_values.shape[0] == 2 or unique_values == 0:
            temp_labels.append(1)
            oneHot_temp_labels.append([1, 0])
        else:
            temp_labels.append(0)
            oneHot_temp_labels.append([0, 1])
    test_patch_labels.append(temp_labels)
    oneHot_test_patch_labels.append(oneHot_temp_labels)
print()

test_patch_dataset = ContrastivePatchLoader3D(test_sample_paths, oneHot_test_patch_labels, device=device, transform=None)

VGG_testloader = DataLoader(test_patch_dataset, batch_size=4, shuffle=True, num_workers=0)

VGG_Model = VGG3D(input_channels=number_of_features, output_classes=2).to(device)
classification_head = Classifier(input_channels=16384, output_channels=2).to(device)

VGG_Model.load_state_dict(torch.load(f'{VGG_model_path}VGG3D_patchlvl_32_state_dict49.pth'))
classification_head.load_state_dict(torch.load(f'{VGG_model_path}classifier_patchlvl_32_state_dict49.pth'))

classifier_criterion = nn.BCELoss().to(device)

activations = {}
input_labels = []

def getActivation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def get_layer_TSNE(layer_name, folder_name, perplexity):

    layer = []
    input_labels = []

    print()
    print('Runnning Model.')
    for (X, oneHot_X_labels) in tqdm(VGG_testloader):
        torch.cuda.empty_cache()
        hook1 = getattr(VGG_Model, layer_name).register_forward_hook(getActivation(layer_name))

        X = X.to(device)
        oneHot_X_labels = oneHot_X_labels.to(device)

        X = torch.reshape(X, shape=(X.shape[0]*X.shape[1], number_of_features, patch_size, patch_size, patch_size))
        oneHot_X_labels = torch.reshape(oneHot_X_labels, shape=(oneHot_X_labels.shape[0]*oneHot_X_labels.shape[1], 2))

        Z = VGG_Model(X)
        Z = torch.reshape(Z, shape=(Z.shape[0], -1))
        Y = classification_head(Z.detach())

        layer.append(activations[layer_name].cpu())
        input_labels.append(oneHot_X_labels.cpu())
        torch.cuda.empty_cache()
    print()

    hook1.remove()

    layer = torch.stack(layer)
    input_labels = torch.stack(input_labels)

    print(f'layer shape: {layer.shape}')
    print(f'input labels shape: {input_labels.shape}')
    # print(f'Input labels shape: {input_labels.shape}')
    # layer = layer.squeeze(1)
    # input_labels = input_labels.squeeze(1)
    layer = torch.reshape(layer, shape=(layer.shape[0]*layer.shape[1], -1))
    input_labels = torch.reshape(input_labels, shape=(input_labels.shape[0] * input_labels.shape[1], -1))
    input_labels = input_labels[:, 0]
    print(f'layer shape: {layer.shape}')
    print(f'input labels shape: {input_labels.shape}')

    pca_result = layer.cpu()

    print()
    start_time = time.time()
    print(f'Started t-SNE.')

    tsne_pca_results = TSNE(n_components=2, perplexity=perplexity, n_iter=1000).fit_transform(pca_result)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print(f'TSNE result: {tsne_pca_results.shape}')
    print(f'Time taken for TSNE: {time_elapsed}')

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.scatterplot(
        x=tsne_pca_results[:,0],
        y=tsne_pca_results[:,1],
        hue=input_labels,
        alpha=0.5
    )
    leg = ax.axes.get_legend()
    new_labels = ['Normal', 'Anomalous']

    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    plt.title('VGG Encoder Patchlvl Supervised Contrastive')
    plt.savefig(f'./{folder_name}/{layer_name}_p{perplexity}')
    plt.close()

get_layer_TSNE('enc_layer7', 'temporary', 30)

print()
print('Script complete.')