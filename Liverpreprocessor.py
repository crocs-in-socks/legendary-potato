import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import *

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

import numpy as np
import nibabel as nib
from tqdm import tqdm
import pickle

c = Constants(
    batch_size = 1,
    patience = None,
    num_workers = 8,
    num_epochs = None,
    date = None,
    to_save_folder = None,
    to_load_folder = None,
    device = 'cuda:1',
    proxy_type = None,
    train_task = 'Preprocessing',
    encoder_load_path = None,
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'lits:window:-100_400'
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))

trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)
validationloader = DataLoader(validationset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)
testloader = DataLoader(testset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)

print()

for idx, data in enumerate(tqdm(trainloader), 0):
    with torch.no_grad():
        image = data['input'].numpy()
        gt = data['gt'].numpy()

        nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
        nifti_gt = nib.Nifti1Image(gt, affine=np.eye(4))

        nib.save(nifti_image, f'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/cv2_MedFilt_and_Bilateral_and_CLAHE/TrainSet/image_{idx}.nii.gz')
        nib.save(nifti_gt, f'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/cv2_MedFilt_and_Bilateral_and_CLAHE/TrainSet/gt_{idx}.nii.gz')

        np.save(f'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/cv2_MedFilt_and_Bilateral_and_CLAHE/TrainSet/data_{idx}.npy', data)

for idx, data in enumerate(tqdm(validationloader), 0):
    with torch.no_grad():
        image = data['input'].numpy()
        gt = data['gt'].numpy()

        nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
        nifti_gt = nib.Nifti1Image(gt, affine=np.eye(4))

        nib.save(nifti_image, f'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/cv2_MedFilt_and_Bilateral_and_CLAHE/ValSet/image_{idx}.nii.gz')
        nib.save(nifti_gt, f'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/cv2_MedFilt_and_Bilateral_and_CLAHE/ValSet/gt_{idx}.nii.gz')

        np.save(f'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/cv2_MedFilt_and_Bilateral_and_CLAHE/ValSet/data_{idx}.npy', data)

for idx, data in enumerate(tqdm(testloader), 0):
    with torch.no_grad():
        image = data['input'].numpy()
        gt = data['gt'].numpy()

        nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
        nifti_gt = nib.Nifti1Image(gt, affine=np.eye(4))

        nib.save(nifti_image, f'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/cv2_MedFilt_and_Bilateral_and_CLAHE/TestSet/image_{idx}.nii.gz')
        nib.save(nifti_gt, f'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/cv2_MedFilt_and_Bilateral_and_CLAHE/TestSet/gt_{idx}.nii.gz')

        np.save(f'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/experiments/cv2_MedFilt_and_Bilateral_and_CLAHE/TestSet/data_{idx}.npy', data)

print()
print('Script executed.')