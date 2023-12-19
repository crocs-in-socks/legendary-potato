import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 1
patience = 15
num_workers = 8
device = 'cuda:1'

Sim1000_train_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*FLAIR.nii.gz'))
Sim1000_train_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*mask.nii.gz'))
Sim1000_train_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*.json'))

Sim1000_validation_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*FLAIR.nii.gz'))
Sim1000_validation_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*mask.nii.gz'))
Sim1000_validation_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*.json'))

sim2211_train_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TrainSet/*FLAIR.nii.gz'))
sim2211_train_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TrainSet/*mask.nii.gz'))
sim2211_train_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TrainSet/*.json'))

sim2211_validation_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/ValSet/*FLAIR.nii.gz'))
sim2211_validation_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/ValSet/*mask.nii.gz'))
sim2211_validation_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/ValSet/*.json'))

clean_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
clean_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))

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