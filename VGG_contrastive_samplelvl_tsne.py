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
from ImageLoader.ContrastiveLoader3D import ContrastivePatchLoader3D, ContrastiveLoader3D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from ModelArchitecture.DUCK_Net import DuckNet
from ModelArchitecture.Encoders import VGG3D, Classifier
from tqdm import tqdm

from skimage.morphology import binary_dilation
from sklearn.manifold import TSNE

# print(torch.cuda.mem_get_info())

wmh_indexes = np.load('../wmh_indexes.npy', allow_pickle=True).item()
DUCK_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/models_retrained/'
VGG_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/contrastive_models/'

batch_size = 1
device = 'cuda:1'

model_type = f'VGG3D_samplelvl'

healthy_data = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
healthy_seg = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))
healthy_masks = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*biascorr_brain_mask.nii.gz*'))

train_fraction = int(len(healthy_data)*0.7)
validation_fraction = int(len(healthy_data)*0.1)
test_fraction = int(len(healthy_data)*0.2)

healthy_train_data = healthy_data[:train_fraction]
healthy_train_seg = healthy_seg[:train_fraction]

healthy_validation_data = healthy_data[train_fraction:train_fraction+validation_fraction]
healthy_validation_seg = healthy_seg[train_fraction:train_fraction+validation_fraction]

healthy_test_data = healthy_data[train_fraction+validation_fraction:]
healthy_test_seg = healthy_seg[train_fraction+validation_fraction:]

DUCK_model = DuckNet(input_channels=1, out_classes=2, starting_filters=17).to(device)
DUCK_model.load_state_dict(torch.load(DUCK_model_path + '/DUCK_wmh_24_10_23_state_dict77.pth'))

t2_test = []

test_class_labels = []
oneHot_test_class_labels = []

activations = {}
def getActivation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

wmh_indexes = np.load('../wmh_indexes.npy', allow_pickle=True).item()

sick_trainset = ImageLoader3D(
    wmh_indexes['train_names'],
    None,
    type_of_imgs='numpy',
    transform=ToTensor3D(True)
)

healthy_trainset = ImageLoader3D(
    healthy_train_data,
    healthy_train_seg,
    type_of_imgs='nifty',
    transform=ToTensor3D(True)
)

sick_validationset = ImageLoader3D(
    wmh_indexes['val_names'],
    None,
    type_of_imgs='numpy',
    transform=ToTensor3D(True)
)

healthy_validationset = ImageLoader3D(
    healthy_validation_data,
    healthy_validation_seg,
    type_of_imgs='nifty',
    transform=ToTensor3D(True)
)

sick_testset = ImageLoader3D(
    wmh_indexes['test_names'],
    None,
    type_of_imgs='numpy',
    transform=ToTensor3D(True)
)

healthy_testset = ImageLoader3D(
    healthy_test_data,
    healthy_test_seg,
    type_of_imgs='nifty',
    transform=ToTensor3D(True)
)

print(f'Number of sick test samples: {len(sick_testset)}')
print(f'Number of healthy test samples: {len(healthy_testset)}')
mixed_testset = ConcatDataset([sick_trainset, healthy_trainset, sick_validationset, healthy_validationset, sick_testset, healthy_testset])
print(f'Number of mixed test samples: {len(mixed_testset)}')

DUCK_testloader = DataLoader(mixed_testset, batch_size=batch_size, shuffle=False, num_workers=0)

# print(torch.cuda.mem_get_info())

print()
print('Runnning DUCK_Model for test set.')
for data in tqdm(DUCK_testloader):

    if list(data['input'].size())[0] == batch_size:
        image = data['input'].to(device)
        label = data['gt'].to(device)

        if torch.unique(label[:,0]).shape[0] == 2:
            test_class_labels.append(0)
            oneHot_test_class_labels.append([0, 1])
        else:
            test_class_labels.append(1)
            oneHot_test_class_labels.append([1, 0])

        hook1 = DUCK_model.t2.register_forward_hook(getActivation('t2'))

        # print(torch.cuda.mem_get_info())

        out = DUCK_model(image)
        t2_test.append(activations['t2'].cpu())
        torch.cuda.empty_cache()

        del image
        del label
        del out
print()
hook1.remove()

test_class_labels = torch.tensor(test_class_labels).float().to(device)
oneHot_test_class_labels = torch.tensor(oneHot_test_class_labels).float().to(device)

t2_test = torch.stack(t2_test)
t2_test = t2_test.squeeze(1)

number_of_features = t2_test.shape[1]

print(f't2_train shape: {t2_test.shape}')
print(f'Number of features: {number_of_features}')

feature_map_dataset = ContrastiveLoader3D(t2_test)

VGG_Model = VGG3D(input_channels=number_of_features, output_classes=2).to(device)
classification_head = Classifier(input_channels=16384, output_channels=2).to(device)
# VGG_trainloader = DataLoader(t2_train, batch_size=batch_size, shuffle=True, num_workers=0)
VGG_testloader = DataLoader(feature_map_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

VGG_Model.load_state_dict(torch.load(f'{VGG_model_path}VGG3D_encoder_samplelvl_state_dict99.pth'))
classification_head.load_state_dict(torch.load(f'{VGG_model_path}VGG3D_classifier_samplelvl_state_dict99.pth'))

activations = {}
input_labels = []

def getActivation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def get_layer_TSNE(layer_name, folder_name, perplexity):

    layer = []

    print()
    print('Runnning Model.')
    for X in tqdm(VGG_testloader):
        torch.cuda.empty_cache()
        hook1 = getattr(VGG_Model, layer_name).register_forward_hook(getActivation(layer_name))
        X = X.to(device)
        Z = VGG_Model(X)
        Z = torch.reshape(Z, shape=(-1,))
        Y = classification_head(Z.detach())
        layer.append(activations[layer_name])
        torch.cuda.empty_cache()
    print()

    hook1.remove()

    layer = torch.stack(layer)
    input_labels = test_class_labels

    print(f'layer shape: {layer.shape}')
    print(f'input labels shape: {input_labels.shape}')
    layer = torch.reshape(layer, shape=(layer.shape[0]*layer.shape[1], -1))
    # input_labels = torch.reshape(input_labels, shape=(-1,))
    print(f'layer shape: {layer.shape}')
    # print(f'input labels shape: {input_labels.shape}')

    pca_result = layer.detach().cpu()

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
        hue=input_labels.cpu(),
        alpha=0.5
    )
    leg = ax.axes.get_legend()
    new_labels = ['Normal', 'Anomalous']

    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    plt.title('Instance-level Supervised Contrastive Loss')
    plt.savefig(f'./{folder_name}/{layer_name}_p{perplexity}')
    plt.close()

# for folder_name, perplexity in folders:
#     for layer in (layer_names):
#         print()
#         print(f'Starting next layer. {layer}, {folder_name}, {perplexity}')
#         get_layer_TSNE(layer, folder_name, perplexity)
        # try:
        #     get_layer_TSNE(layer, folder_name, perplexity)
        # except Exception as e:
        #     print(f'Error: {e}, layer: {layer}')

get_layer_TSNE('enc_layer7', 'temporary', 10)

print()
print('Script complete.')