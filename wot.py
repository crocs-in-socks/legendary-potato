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

batch_size = 4
number_of_epochs = 20
device = 'cuda:0'
model_type = 'ResNetClassifier'
date = '08_11_2023'

model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov06/'

trainset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TrainSet_5_11_23/*.npz'))
validationset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/ValidSet_5_11_23/*.npz'))
testset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TestSet_5_11_23/*.npz'))
print(f'Trainset size: {len(trainset_paths)}')
print(f'Validationset size: {len(validationset_paths)}')
print(f'Testset size: {len(testset_paths)}')

composed_transform = transforms.Compose([
        ToTensor3D(True, clean=True)
    ])

trainset = ImageLoader3D(paths=trainset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True)
validationset = ImageLoader3D(paths=validationset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True)
testset = ImageLoader3D(paths=testset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True)

allset = ConcatDataset([trainset, validationset, testset])

ResNet_Model = ResNet3D_Encoder(image_channels=1).to(device)
projection_head = nn.Conv3d(960, 1, kernel_size=1).to(device)
# projection_head = nn.Conv3d(512, 1, kernel_size=1).to(device)

ResNet_allloader = DataLoader(allset, batch_size=batch_size, shuffle=True, num_workers=0)

ResNet_Model.load_state_dict(torch.load(f'{model_path}Contrastive_ResNet_06_11_2023_ResNet_state_dict101.pth'))
# projection_head.load_state_dict(torch.load(f'{model_path}Contrastive_ProjectionHead_StackedOnlyAnomalous_wNIMH_09_11_2023_ProjectorHead_state_dict101.pth'))

activations = {}
input_labels = []

def getActivation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def get_layer_TSNE(layer_name, folder_name, perplexity):

    layer = []
    input_labels = []

    for i in tqdm(range(len(allset))):
        torch.cuda.empty_cache()

        hook1 = getattr(ResNet_Model, layer_name).register_forward_hook(getActivation(layer_name))

        sample_dict = allset[i]
        sample_input = sample_dict['input'].to(device).unsqueeze(0)
        sample_clean = sample_dict['clean'].to(device).unsqueeze(0)
        sample_label = (sample_dict['gt'])[1].to(device).unsqueeze(0).unsqueeze(0)

        sample_mixed = torch.cat([sample_input, sample_clean])

        out_dict = ResNet_Model(sample_mixed)
        layer1 = out_dict['out1']
        layer2 = out_dict['out2']
        layer3 = out_dict['out3']
        z_mixed = out_dict['out4']

        layer.append(z_mixed[0].detach().cpu())
        layer.append(z_mixed[1].detach().cpu())
        input_labels.append(1)
        input_labels.append(0)

    print()

    hook1.remove()

    layer = torch.stack(layer)
    input_labels = torch.tensor(input_labels)

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
    plt.title('ResNet Encoder Instance-wise Supervised Contrastive without MSE')
    plt.savefig(f'./{folder_name}/{layer_name}_p{perplexity}')
    plt.close()

get_layer_TSNE('layer4', 'temporary', 30)

print()
print('Script complete.')