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
from ModelArchitecture.Encoders import VGG3D, Classifier
from tqdm import tqdm

from skimage.morphology import binary_dilation
from sklearn.manifold import TSNE

# layer_names = [
#     'p1', 'p2', 'p3', 'p4', 'p5', 't0', 'l1i', 't1', 'l2i', 't2', 'l3i', 't3', 'l4i', 't4', 'l5i', 't51', 't53', 'l5o', 'q4', 'l4o', 'q3', 'l3o', 'q6', 'l2o', 'q1', 'l1o', 'z1'
# ]
layer_names = [
    'layer7'
]

# layer_names = ['layer4']

folders = [
    ('TSNE-patchlvl-32-p50', 30)
]

wmh_indexes = np.load('../wmh_indexes.npy', allow_pickle=True).item()
VGG_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/contrastive_models/'
# ResNet_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov08/'
# patches_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/patches/'

batch_size = 4
number_of_epochs = 50
number_of_test_samples = 20
number_of_features = 256
patch_size = 32
device = 'cuda:0'

model_type = f'VGG3D_samplelvl'
# model_type = f'VGG3D_samplelvl_{patch_size}'
# classifier_type = f'classifier_patchlvl_{patch_size}'
# encoder_type = f'ResNet3D_encoder_patchlvl_{patch_size}'
# classifier_type = f'ResNet3D_classifier_patchlvl_{patch_size}'

dataset = ImageLoader3D(
    wmh_indexes['test_names'],
    None,
    type_of_imgs='numpy',
    transform=ToTensor3D(True)
)

# test_sample_paths = []
# test_patch_labels = []
# oneHot_test_patch_labels = []

# patches_list = []

# print('Generating labels for testset.')
# for sample_idx in tqdm(range(number_of_test_samples)):
#     patch_dict = np.load(f'{patches_path}/patch_size{patch_size}_test_patch_and_label_idx{sample_idx}.npy', allow_pickle=True).item()

#     test_sample_paths.append(f'{patches_path}/patch_size{patch_size}_test_patch_and_label_idx{sample_idx}.npy')

#     patches = patch_dict['patches']
#     label_patches = patch_dict['labels']
    
#     # downsampled = F.interpolate(patches[0], size=(4, 4, 4), mode='nearest')

#     # patches_list.append(downsampled)

#     number_of_features = patches.shape[2]

#     temp_labels = []
#     oneHot_temp_labels = []

#     number_of_patches = label_patches.shape[1]
#     for patch_idx in range(number_of_patches):
#         unique_values = torch.unique(label_patches[0, patch_idx, 0])
        
#         if unique_values.shape[0] == 2 or unique_values == 0:
#             temp_labels.append(1)
#             oneHot_temp_labels.append([1, 0])
#         else:
#             temp_labels.append(0)
#             oneHot_temp_labels.append([0, 1])
#     test_patch_labels.append(temp_labels)
#     oneHot_test_patch_labels.append(oneHot_temp_labels)
# print()

# test_patch_dataset = ContrastivePatchLoader3D(test_sample_paths, test_patch_labels, oneHot_test_patch_labels, device=device, transform=None)

# # VGG_testloader = DataLoader(test_patch_dataset, batch_size=1, shuffle=True, num_workers=0)
# ResNet_testloader = DataLoader(test_patch_dataset, batch_size=1, shuffle=True, num_workers=0)

testloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

model = DuckNet(input_channels=1, out_classes=2, starting_filters=17).to(device)
model.load_state_dict(torch.load('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/models_retrained/DUCK_wmh_24_10_23_state_dict77.pth'))

VGG_Model = VGG3D(input_channels=number_of_features, output_classes=2).to(device)
classification_head = Classifier(input_channels=16384, output_channels=2).to(device)
# ResNet_Model = ResNet3D(image_channels=number_of_features).to(device)
# classification_head = Classifier(input_channels=2048, output_channels=2).to(device)

VGG_Model.load_state_dict(torch.load(f'{VGG_model_path}VGG3D_samplelvl_state_dict99.pth'))
classification_head.load_state_dict(torch.load(f'{VGG_model_path}VGG3D_samplelvl_classifier_loss_state_dict99.pth'))
# ResNet_Model.load_state_dict(torch.load(f'{ResNet_model_path}ResNet3D_encoder_patchlvl_32_state_dict95.pth'))
# classification_head.load_state_dict(torch.load(f'{VGG_model_path}ResNet3D_classifier_patchlvl_32_state_dict145.pth'))

# trainset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TrainSet_5_11_23/*.npz'))
# validationset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/ValidSet_5_11_23/*.npz'))
# testset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TestSet_5_11_23/*.npz'))

# print(f'Trainset size: {len(trainset_paths)}')
# print(f'Validationset size: {len(validationset_paths)}')
# print(f'Testset size: {len(testset_paths)}')

# composed_transform = transforms.Compose([
#         RandomRotation3D([10, 10], clean=True),
#         RandomIntensityChanges(clean=True),
#         ToTensor3D(True, clean=True)
#     ])
# composed_transform = transforms.Compose([
#         ToTensor3D(True, clean=True)
#     ])

# trainset = ImageLoader3D(paths=trainset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)
# validationset = ImageLoader3D(paths=validationset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)
# testset = ImageLoader3D(paths=testset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)

# allset = ConcatDataset([validationset, testset])
# allset = ConcatDataset([trainset, validationset, testset])

# composed_transform = transforms.Compose([
#         ToTensor3D(True, True)
#     ])

# ResNet_Model = ResNet3D(image_channels=1).to(device)
# classification_head = Classifier(input_channels=32768, output_channels=2).to(device)

# ResNet_Model.load_state_dict(torch.load(f'{ResNet_model_path}Contrastive_ResNet_08_11_2023_ResNet_state_dict101.pth'))
# classification_head.load_state_dict(torch.load(f'{ResNet_model_path}Contrastive_ClassificationHead_08_11_2023_ClassifierHead_state_dict101.pth'))

# ResNet_testloader = DataLoader(allset, batch_size=batch_size, shuffle=False, num_workers=0)

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
    # for (X, X_labels, oneHot_X_labels) in tqdm(VGG_testloader):
    # for (X, X_labels, oneHot_X_labels) in tqdm(ResNet_testloader):
    # for data in tqdm(ResNet_testloader):
    for data in tqdm(testloader):
        torch.cuda.empty_cache()
        image = data['input'].to(device)
        label = data['gt'].to(device)

        # hook1 = getattr(VGG_Model, layer_name).register_forward_hook(getActivation(layer_name))
        # hook1 = getattr(ResNet_Model, layer_name).register_forward_hook(getActivation(layer_name))
        hook1 = getattr(model, layer_name).register_forward_hook(getActivation(layer_name))
        # hook1 = VGG_Model.enc_layer7.register_forward_hook(getActivation(layer_name))

        out = model(image)

        # X = X.to(device)
        # X_labels = X_labels.to(device)
        # oneHot_X_labels = oneHot_X_labels.to(device)

        # image = data['input'].to(device)
        # clean = data['clean'].to(device)
        # mixed = torch.stack([image, clean])

        # mixed = rearrange(mixed, 'ic b c h w d -> (ic b) c h w d')
        # labels = (torch.tensor([1] * batch_size + [0] * batch_size)).float().to(device)

        # z = ResNet_Model.forward(mixed)
        # z = torch.reshape(z, shape=(z.shape[0], -1))

        layer.append(activations[layer_name].cpu())
        input_labels.append(label.detach().cpu())

        # X = torch.reshape(X, shape=(X.shape[0]*X.shape[1], number_of_features, patch_size, patch_size, patch_size))
        # X_labels = torch.reshape(X_labels, shape=(-1,))
        # oneHot_X_labels = torch.reshape(oneHot_X_labels, shape=(oneHot_X_labels.shape[0]*oneHot_X_labels.shape[1], 2))

        # Z = VGG_Model(X)
        # Z = ResNet_Model(X)
        # Z = torch.reshape(Z, shape=(Z.shape[0], -1))
        # Y = classification_head(Z.detach())

        # out = VGG_Model(image)
        # input_labels.append(label.detach().cpu())

        # layer.append(activations[layer_name].cpu())
        # input_labels.append(X_labels.cpu())
        torch.cuda.empty_cache()
    print()

    hook1.remove()

    layer = torch.stack(layer)
    # layer = torch.stack(patches_list)
    # layer = layer.squeeze(1)
    input_labels = torch.stack(input_labels)
    # input_labels = torch.stack(input_labels)

    print(f'layer shape: {layer.shape}')
    print(f'input labels shape: {input_labels.shape}')
    # print(f'Input labels shape: {input_labels.shape}')
    # layer = layer.squeeze(1)
    # input_labels = input_labels.squeeze(1)
    layer = torch.reshape(layer, shape=(layer.shape[0]*layer.shape[1], -1))
    input_labels = torch.reshape(input_labels, shape=(-1,))
    print(f'layer shape: {layer.shape}')
    print(f'input labels shape: {input_labels.shape}')

    exit(0)

    # print(f'Input labels shape: {input_labels.shape}')

    # number_of_features = layer.shape[1]
    # sample_number = 4
    # offset = 4
    # slice_number = layer.shape[-1] // 2

    # feature_maps = layer
    # upsampled_maps = F.interpolate(feature_maps, size=(128, 128, 128), mode='trilinear')

    # upsampled_maps = upsampled_maps.permute(1, 0, 2, 3, 4)
    # masks = input_labels.permute(1, 0, 2, 3, 4)

    # dilated_masks = masks[1].clone()

    # struct_element = np.ones((3, 3, 3), dtype=bool)

    # print()
    # print('Dilating masks.')
    # for sample in tqdm(range(dilated_masks.shape[0])):
    #     dilated = torch.from_numpy(binary_dilation(dilated_masks[sample], struct_element)).int()
    #     dilated_masks[sample] = dilated
    # print()

    # masks = masks[:, sample_number:sample_number+offset]
    # dilated_masks = dilated_masks[sample_number:sample_number+offset]
    # upsampled_maps = upsampled_maps[:, sample_number:sample_number+offset]

    # print(f'Feature maps shape: {feature_maps.shape}')
    # print(f'Upsampled maps shape: {upsampled_maps.shape}')
    # print(f'Masks shape: {masks.shape}')
    # print(f'Dilated masks shape: {dilated_masks.shape}')

    # true_class_maps = []
    # false_class_maps = []

    # print()
    # print('Applying masks to feature maps.')
    # for feature in tqdm(range(number_of_features)):
    #     true_mask = masks[1] == 1
    #     false_mask = (dilated_masks == 1) & (masks[0] == 1)

    #     true_class_maps.append(upsampled_maps[feature][true_mask])
    #     false_class_maps.append(upsampled_maps[feature][false_mask])
    # print()

    # true_class_maps = torch.stack(true_class_maps)
    # false_class_maps = torch.stack(false_class_maps)

    # true_class_maps = true_class_maps.T
    # false_class_maps = false_class_maps.T

    # print(f'True class maps shape: {true_class_maps.shape}')
    # print(f'False class maps shape: {false_class_maps.shape}')

    # class_maps = torch.cat((true_class_maps, false_class_maps), dim=0)
    # class_labels = [1] * true_class_maps.shape[0] + [0] * false_class_maps.shape[0]

    # class_labels = torch.tensor(class_labels)
    # print(f'Class maps shape: {class_maps.shape}')
    # print(f'Class labels shape: {class_labels.shape}')

    # torch.cuda.empty_cache()

    # del layer
    # del input_labels
    # del feature_maps
    # del upsampled_maps
    # del masks
    # del dilated_masks
    # del true_class_maps
    # del false_class_maps

    # class_maps = class_maps.cpu()
    # class_labels = class_labels.cpu()

    # pca_result = class_maps
    pca_result = layer.cpu()

    print()
    start_time = time.time()
    print(f'Started t-SNE.')

    tsne_pca_results = TSNE(n_components=3, perplexity=perplexity, n_iter=1000).fit_transform(pca_result)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print(f'TSNE result: {tsne_pca_results.shape}')
    print(f'Time taken for TSNE: {time_elapsed}')

    # fig = plt.figure(figsize=(9, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # ax = sns.scatterplot(
    #     x=tsne_pca_results[:,0],
    #     y=tsne_pca_results[:,1],
    #     hue=input_labels,
    #     alpha=0.5
    # )
    # leg = ax.axes.get_legend()
    # new_labels = ['Anomalous', 'Not Anomalous']

    # for t, l in zip(leg.texts, new_labels):
    #     t.set_text(l)
    # plt.savefig(f'./{folder_name}/{layer_name}_p{perplexity}.jpg')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(
    #     tsne_pca_results[:, 0],
    #     tsne_pca_results[:, 1],
    #     tsne_pca_results[:, 2],
    #     c=input_labels,
    #     alpha=0.5
    # )

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # leg = plt.legend(input_labels.unique(), loc='upper left')
    # plt.title(f'Layer: {layer_name} Perplexity: {perplexity}')

    # plt.savefig('./3d_tsne.jpg')

    # Purple is normal, Yellow is anomalous
    colours = ['#964bf1', '#ffcb1d']
    input_colours = [colours[label.int()] for label in input_labels]

    fig = go.Figure(data=[go.Scatter3d(
        x=tsne_pca_results[:, 0],
        y=tsne_pca_results[:, 1],
        z=tsne_pca_results[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=input_colours,
            opacity=0.8
        )
    )])

    # Set plot title and axis labels
    fig.update_layout(
        title=f'Layer: {layer_name} Perplexity: {perplexity}', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        autosize=False,
        width=1000,
        height=1000
    )

    # Save the interactive plot as an HTML file
    fig.write_html('3D-ResNet-TSNE-100ep-Contrastive-Generated.html')

    # plt.close()

for folder_name, perplexity in folders:
    for layer in (layer_names):
        print()
        print(f'Starting next layer. {layer}, {folder_name}, {perplexity}')
        get_layer_TSNE(layer, folder_name, perplexity)
        # try:
        #     get_layer_TSNE(layer, folder_name, perplexity)
        # except Exception as e:
        #     print(f'Error: {e}, layer: {layer}')

# get_layer_TSNE('p1')

print()
print('Script complete.')