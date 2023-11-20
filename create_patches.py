import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.DUCK_Net import DuckNet
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
from tqdm import tqdm
from einops import rearrange

DUCK_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/models_retrained/'
patches_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/patches/'

batch_size = 1
device = 'cuda:1'
patch_size = 32

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

print(f'Number of sick train samples: {len(sick_trainset)}')
print(f'Number of healthy train samples: {len(healthy_trainset)}')
mixed_trainset = ConcatDataset([sick_trainset, healthy_trainset])
print(f'Number of mixed train samples: {len(mixed_trainset)}')

print(f'Number of sick validation samples: {len(sick_validationset)}')
print(f'Number of healthy validation samples: {len(healthy_validationset)}')
mixed_validationset = ConcatDataset([sick_validationset, healthy_validationset])
print(f'Number of mixed validation samples: {len(mixed_validationset)}')

print(f'Number of sick test samples: {len(sick_testset)}')
print(f'Number of healthy test samples: {len(healthy_testset)}')
mixed_testset = ConcatDataset([sick_testset, healthy_testset])
print(f'Number of mixed test samples: {len(mixed_testset)}')

DUCK_trainloader = DataLoader(mixed_trainset, batch_size=batch_size, shuffle=True, num_workers=0)
DUCK_validationloader = DataLoader(mixed_validationset, batch_size=batch_size, shuffle=True, num_workers=0)
DUCK_testloader = DataLoader(mixed_testset, batch_size=batch_size, shuffle=True, num_workers=0)

data_idx = 0

print()
print('Runnning DUCK_Model for train set.')
for data in tqdm(DUCK_trainloader):
    if list(data['input'].size())[0] == batch_size:
        image = data['input'].to(device)
        label = data['gt'].to(device)

        hook1 = DUCK_model.t2.register_forward_hook(getActivation('t2'))
        out = DUCK_model(image)

        upsampled_map = F.interpolate(activations['t2'], size=(128, 128, 128), mode='nearest')
        # b = batches
        # c = channels
        # nph = number of patches high
        # ph = patch height
        # npw = number of patches wide
        # pw = patch width
        # npd = number of patches deep
        # pd = patch depth
        patches = rearrange(upsampled_map, 'b c (nph ph) (npw pw) (npd pd) -> b (nph npw npd) c ph pw pd', ph=patch_size, pw=patch_size, pd=patch_size)

        patch_labels = rearrange(label, 'b c (nph ph) (npw pw) (npd pd) -> b (nph npw npd) c ph pw pd', ph=patch_size, pw=patch_size, pd=patch_size)

        to_save = {'patches': patches, 'labels': patch_labels}
        np.save(f'{patches_path}patch_size{patch_size}_train_patch_and_label_idx{data_idx}.npy', to_save)

        torch.cuda.empty_cache()
        data_idx += 1

        del image
        del label
        del out
print()
hook1.remove()


data_idx = 0
print()
print('Runnning DUCK_Model for validation set.')
for data in tqdm(DUCK_validationloader):
    if list(data['input'].size())[0] == batch_size:
        image = data['input'].to(device)
        label = data['gt'].to(device)

        hook1 = DUCK_model.t2.register_forward_hook(getActivation('t2'))
        out = DUCK_model(image)

        upsampled_map = F.interpolate(activations['t2'], size=(128, 128, 128), mode='nearest')
        # b = batches
        # c = channels
        # nph = number of patches high
        # ph = patch height
        # npw = number of patches wide
        # pw = patch width
        # npd = number of patches deep
        # pd = patch depth
        patches = rearrange(upsampled_map, 'b c (nph ph) (npw pw) (npd pd) -> b (nph npw npd) c ph pw pd', ph=patch_size, pw=patch_size, pd=patch_size)

        patch_labels = rearrange(label, 'b c (nph ph) (npw pw) (npd pd) -> b (nph npw npd) c ph pw pd', ph=patch_size, pw=patch_size, pd=patch_size)

        to_save = {'patches': patches, 'labels': patch_labels}
        np.save(f'{patches_path}patch_size{patch_size}_validation_patch_and_label_idx{data_idx}.npy', to_save)

        torch.cuda.empty_cache()
        data_idx += 1

        del image
        del label
        del out
print()
hook1.remove()

data_idx = 0
print()
print('Runnning DUCK_Model for test set.')
for data in tqdm(DUCK_testloader):
    if list(data['input'].size())[0] == batch_size:
        image = data['input'].to(device)
        label = data['gt'].to(device)

        hook1 = DUCK_model.t2.register_forward_hook(getActivation('t2'))
        out = DUCK_model(image)

        upsampled_map = F.interpolate(activations['t2'], size=(128, 128, 128), mode='nearest')
        # b = batches
        # c = channels
        # nph = number of patches high
        # ph = patch height
        # npw = number of patches wide
        # pw = patch width
        # npd = number of patches deep
        # pd = patch depth
        patches = rearrange(upsampled_map, 'b c (nph ph) (npw pw) (npd pd) -> b (nph npw npd) c ph pw pd', ph=patch_size, pw=patch_size, pd=patch_size)

        patch_labels = rearrange(label, 'b c (nph ph) (npw pw) (npd pd) -> b (nph npw npd) c ph pw pd', ph=patch_size, pw=patch_size, pd=patch_size)

        to_save = {'patches': patches, 'labels': patch_labels}
        np.save(f'{patches_path}patch_size{patch_size}_test_patch_and_label_idx{data_idx}.npy', to_save)

        torch.cuda.empty_cache()
        data_idx += 1

        del image
        del label
        del out
print()
hook1.remove()

print('Script executed.')