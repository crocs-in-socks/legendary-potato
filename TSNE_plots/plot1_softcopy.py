import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from ModelArchitecture.Transformations import *
from ImageLoader.ImageLoader3D import ImageLoader3D

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ModelArchitecture.DUCK_Net import DuckNet
from tqdm import tqdm

from skimage.morphology import binary_dilation
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
# from openTSNE import TSNE
# from tsnecuda import TSNE
# from tsne_torch import TorchTSNE as TSNE

activations = {}
input_labels = []
p1 = []

wmh_indexes = np.load('../wmh_indexes.npy', allow_pickle=True).item()

dataset = ImageLoader3D(
    wmh_indexes['test_names'],
    None,
    type_of_imgs='numpy',
    transform=ToTensor3D(True)
)

testloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

device = 'cuda:1'

model = DuckNet(input_channels=1, out_classes=2, starting_filters=17).to(device)
model.load_state_dict(torch.load('./mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/models_retrained/DUCK_wmh_24_10_23_state_dict77.pth'))

def getActivation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

print()
print('Runnning Model.')
for data in tqdm(testloader):
    torch.cuda.empty_cache()
    image = data['input'].to(device)
    label = data['gt'].to(device)

    hook1 = model.p1.register_forward_hook(getActivation('p1'))

    out = model(image)
    input_labels.append(label.detach().cpu())

    p1.append(activations['p1'].cpu())
    torch.cuda.empty_cache()
print()

hook1.remove()

p1 = torch.stack(p1)
input_labels = torch.stack(input_labels)

print(f'p1 shape: {p1.shape}')
print(f'Input labels shape: {input_labels.shape}')
p1 = p1.squeeze(1)
input_labels = input_labels.squeeze(1)
print(f'p1 shape: {p1.shape}')
print(f'Input labels shape: {input_labels.shape}')

number_of_features = p1.shape[1]
sample_number = 7
slice_number = p1.shape[-1] // 2

feature_maps = p1
upsampled_maps = F.interpolate(feature_maps, size=(128, 128, 128), mode='nearest')

upsampled_maps = upsampled_maps.permute(1, 0, 2, 3, 4)
masks = input_labels.permute(1, 0, 2, 3, 4)

dilated_masks = masks[1].clone()

struct_element = np.ones((3, 3, 3), dtype=bool)

print()
print('Dilating masks.')
for sample in tqdm(range(dilated_masks.shape[0])):
    dilated = torch.from_numpy(binary_dilation(dilated_masks[sample], struct_element)).int()
    # dilated = dilated + masks[1, sample]
    dilated_masks[sample] = dilated
print()

plt.figure(figsize=(20, 15))
plt.subplot(1, 3, 1)
plt.imshow(masks[0, sample_number, :, :, slice_number])
plt.title('Mask 0')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(masks[1, sample_number, :, :, slice_number])
plt.title('Mask 1')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(dilated_masks[sample_number, :, :, slice_number])
plt.title('Dilated Mask 1')
plt.colorbar()

plt.savefig('./temp/temp')

# upsampled_maps = torch.reshape(upsampled_maps, shape=(upsampled_maps.shape[0],  -1))
# masks = torch.reshape(masks, shape=(masks.shape[0], -1))
# dilated_masks = torch.reshape(dilated_masks, shape=(-1,))

masks = masks[:, sample_number]
dilated_masks = dilated_masks[sample_number]
upsampled_maps = upsampled_maps[:, sample_number]

print(f'Feature maps shape: {feature_maps.shape}')
print(f'Upsampled maps shape: {upsampled_maps.shape}')
print(f'Masks shape: {masks.shape}')
print(f'Dilated masks shape: {dilated_masks.shape}')

true_class_maps = []
false_class_maps = []

# true_class_maps = upsampled_maps.clone()
# false_class_maps = upsampled_maps.clone()

print()
print('Applying masks to feature maps.')
for feature in tqdm(range(number_of_features)):
    true_mask = masks[1] == 1
    false_mask = (dilated_masks == 1) & (masks[0] == 1)

    true_class_maps.append(upsampled_maps[feature][true_mask])
    false_class_maps.append(upsampled_maps[feature][false_mask])
print()

# for feature in tqdm(range(number_of_features)):
#     true_class_maps[feature][masks[1] == 0] = 0
#     false_class_maps[feature][(dilated_masks == 0) | (masks[0] == 0)] = 0

# plt.figure(figsize=(20, 15))
# plt.subplot(1, 2, 1)
# plt.imshow(true_class_maps[0, sample_number, :, :, slice_number])
# plt.title('true map')
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.imshow(false_class_maps[1, sample_number, :, :, slice_number])
# plt.title('false map')
# plt.colorbar()

# plt.savefig('./temp/cut-outs')

# # largest_false_map = max([tensor.size(0) for tensor in false_class_maps])
# # for feature in tqdm(range(number_of_features)):
# #     false_class_maps[feature] = F.pad(false_class_maps[feature], (0, largest_false_map - false_class_maps[feature].size(0)), mode='constant', value=background_values[feature])

true_class_maps = torch.stack(true_class_maps)
false_class_maps = torch.stack(false_class_maps)

true_class_maps = true_class_maps.T
false_class_maps = false_class_maps.T

print(f'True class maps shape: {true_class_maps.shape}')
print(f'False class maps shape: {false_class_maps.shape}')

class_maps = torch.cat((true_class_maps, false_class_maps), dim=0)
class_labels = [1] * true_class_maps.shape[0] + [0] * false_class_maps.shape[0]

class_labels = torch.tensor(class_labels)
print(f'Class maps shape: {class_maps.shape}')
print(f'Class labels shape: {class_labels.shape}')

torch.cuda.empty_cache()

del model
del testloader
del p1
del input_labels
del feature_maps
del upsampled_maps
del masks
del dilated_masks
del true_class_maps
del false_class_maps

class_maps = class_maps.cpu()
class_labels = class_labels.cpu()

# start_time = time.time()

# pca = PCA(n_components=30)
# pca_result = pca.fit_transform(class_maps)

# end_time = time.time()
# time_elapsed = end_time - start_time

# print(f'PCA result: {pca_result.shape}')
# print(f'Time taken for PCA: {time_elapsed}')

pca_result = class_maps

print()
start_time = time.time()
print(f'Started t-SNE at {start_time}.')

tsne_pca_results = TSNE(n_components=2, perplexity=30.0, verbose=1, n_iter=1000).fit_transform(pca_result)

end_time = time.time()
time_elapsed = end_time - start_time

print(f'TSNE result: {tsne_pca_results.shape}')
print(f'Time taken for TSNE: {time_elapsed}')

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(1, 1, 1)
ax = sns.scatterplot(
    x=tsne_pca_results[:,0],
    y=tsne_pca_results[:,1],
    hue=class_labels,
    alpha=0.5
)
leg = ax.axes.get_legend()
new_labels = ['Hyperintensity', 'Not Hyperintensity']

for t, l in zip(leg.texts, new_labels):
    t.set_text(l)
plt.savefig('.TSNE-Voxelwise/p1-tsne-30.jpg')

print()
print('Script complete.')

# print()
# for feature in tqdm(range(number_of_features)):

#     plt.figure(figsize=(20, 15))
#     plt.subplot(1, 2, 1)
#     plt.imshow(true_class_maps[sample_number, feature, :, :, slice_number], cmap='gray')
#     plt.title('Class 1')
#     plt.colorbar()

#     plt.subplot(1, 2, 2)
#     plt.imshow(false_class_maps[sample_number, feature, :, :, slice_number], cmap='gray')
#     plt.title('Class 0')
#     plt.colorbar()

#     plt.savefig(f'./class_maps/#{feature+1}.jpg')
#     plt.close()
# print()

# flat_true_maps = torch.reshape(true_class_maps, shape=(feature_combinations, -1))
# flat_false_maps = torch.reshape(false_class_maps, shape=(feature_combinations, -1))

# flat_true_maps = true_class_maps.permute(1, 0, 2, 3, 4)
# flat_false_maps = false_class_maps.permute(1, 0, 2, 3, 4)

# flat_true_maps = torch.reshape(flat_true_maps, shape=(flat_true_maps.shape[0], -1))
# flat_false_maps = torch.reshape(flat_false_maps, shape=(flat_true_maps.shape[0], -1))

# print(f'flat_true_maps shape: {flat_true_maps.shape}')
# print(f'flat_false_maps shape: {flat_false_maps.shape}')

# flattened_class_maps = class_maps
# print(f'Flattened class maps shape: {flattened_class_maps.shape}')
# print(f'Flattened class maps shape (transpose): {flattened_class_maps.T.shape}')

# flattened_class_maps = torch.reshape(class_maps, shape=(class_maps.shape[0], -1))
# print(f'Flattened class maps shape: {flattened_class_maps.shape}')