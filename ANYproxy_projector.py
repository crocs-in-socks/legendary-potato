import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 1
patience = 15
num_workers = 16
device = 'cuda:1'
number_of_epochs = 100
date = '01_12_2023'
encoder_type = 'DUCK_WMHfrozen_proxy_encoder_simFinetuned'
classifier_type = 'DUCK_WMHfrozen_proxy_classifier_simFinetuned'
projector_type = 'DUCK_WMHfrozen_proxy_projector_simFinetuned'

save_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov29/'
DUCKmodel_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/Duck1wmh_focal + dice_state_dict_best_loss97.pth'
encoder_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Dec01/VGGproxy_encoder_simFinetuned_weightedBCE_negativeRing_01_12_2023_state_dict100.pth'
projector_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Dec01/VGGproxy_projector_simFinetuned_weightedBCE_negativeRing_01_12_2023_state_dict100.pth'

from_Sim1000_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*FLAIR.nii.gz'))
from_sim2211_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/**/*FLAIR.nii.gz'))

from_Sim1000_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*mask.nii.gz'))
from_sim2211_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/**/*mask.nii.gz'))

from_Sim1000_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*.json'))
from_sim2211_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/**/*.json'))

clean_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
clean_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))

composed_transform = transforms.Compose([
        ToTensor3D(labeled=True)
    ])

from_Sim1000 = ImageLoader3D(paths=from_Sim1000_data_paths, gt_paths=from_Sim1000_gt_paths, json_paths=from_Sim1000_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)

from_sim2211 = ImageLoader3D(paths=from_sim2211_data_paths,gt_paths=from_sim2211_gt_paths, json_paths=from_sim2211_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)

clean = ImageLoader3D(paths=clean_data_paths, gt_paths=clean_gt_paths, json_paths=None, image_size=128, type_of_imgs='nifty', transform=composed_transform)

fullset = ConcatDataset([from_Sim1000, from_sim2211, clean])

train_size = int(0.8 * len(fullset))
validation_size = len(fullset) - train_size
# validation_size = int(0.1 * len(fullset))
# test_size = len(fullset) - (train_size + validation_size)

trainset, validationset = random_split(fullset, (train_size, validation_size))

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# encoder = DuckNet(input_channels=1, out_classes=2, starting_filters=17).to(device)
# encoder = ResNet3D_Encoder(image_channels=1).to(device)
encoder = VGG3D_Encoder(input_channels=1).to(device)
encoder.load_state_dict(torch.load(encoder_model_path))

# classification_head = Classifier(input_channels=17408, output_channels=5).to(device)
# classification_head = Classifier(input_channels=2176, output_channels=5).to(device)

# projection_head = Projector(num_layers=5, layer_sizes=[17, 34, 68, 136, 272]).to(device)
# projection_head = Projector(num_layers=4, layer_sizes=[64, 128, 256, 512]).to(device)
projection_head = Projector(num_layers=5, layer_sizes=[32, 64, 128, 256, 512]).to(device)
projection_head.load_state_dict(torch.load(projector_model_path))

encoder.eval()
projection_head.eval()

for idx, data in enumerate(tqdm(validationloader), 0):
    image = data['input'].to(device)
    gt = data['gt'].to(device)
    oneHot_label = data['lesion_labels'].float().to(device)

    to_projector, to_classifier = encoder(image)

    if torch.unique(gt[:, 1]).shape[0] == 2:
        projection = projection_head(to_projector)
        projection = F.interpolate(projection, size=(128, 128, 128))

        plt.figure(figsize=(20, 15))
        plt.subplot(1, 3, 1)
        plt.imshow(image[0, 0, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.title(f'Input Sample #{idx+1}')
        plt.subplot(1, 3, 2)    
        plt.imshow(gt[0, 1, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.title(f'Input gt #{idx+1}')
        plt.subplot(1, 3, 3)
        plt.imshow(1-projection[0, 0, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.title(f'Projection #{idx+1}')

        plt.savefig(f'./temporary/valset#{idx+1}')
        plt.close()

        del projection
            
    del image
    del gt
    del oneHot_label
    del to_projector
    del to_classifier