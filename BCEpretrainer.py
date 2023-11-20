import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from ModelArchitecture.Encoders import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 4
number_of_epochs = 100
device = 'cuda:1'
encoder_type = 'BCEpretrain'
classifier_type = 'BCEpretrain_ResNet'
date = '17_11_2023'

model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov17/'

clean_data = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
clean_labels = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))
clean_masks = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*biascorr_brain_mask.nii.gz*'))

train_fraction = int(len(clean_data)*0.7)
validation_fraction = int(len(clean_data)*0.1)

clean_trainset_data = clean_data[:train_fraction]
clean_trainset_labels = clean_labels[:train_fraction]

clean_validationset_data = clean_data[train_fraction:train_fraction+validation_fraction]
clean_validationset_labels = clean_labels[train_fraction:train_fraction+validation_fraction]

anomalous_trainset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TrainSet_5_11_23/*.npz'))
anomalous_validationset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/ValidSet_5_11_23/*.npz'))

print(f'Anomalous Trainset size: {len(anomalous_trainset_paths)}')
print(f'Anomalous Validationset size: {len(anomalous_validationset_paths)}')
print(f'Clean Trainset size: {len(clean_trainset_data)}')
print(f'Clean Validationset size: {len(clean_validationset_data)}')

composed_transform = transforms.Compose([
        ToTensor3D(True, clean=True, subtracted=True)
    ])

anomalous_trainset = ImageLoader3D(paths=anomalous_trainset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True, subtracted=True)
anomalous_validationset = ImageLoader3D(paths=anomalous_validationset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)

clean_trainset = ImageLoader3D(paths=clean_trainset_data, gt_paths=clean_trainset_labels, type_of_imgs='nifty', transform=composed_transform, clean=True, subtracted=True)
clean_validationset = ImageLoader3D(paths=clean_validationset_data, gt_paths=clean_validationset_labels, type_of_imgs='nifty', transform=composed_transform, clean=True, subtracted=True)

trainset = ConcatDataset([anomalous_trainset, clean_trainset])
validationset = ConcatDataset([anomalous_validationset, clean_validationset])

ResNet_encoder = ResNet3D_Encoder(image_channels=1).to(device)
classification_head = Classifier(input_channels=32768, output_channels=2).to(device)

ResNet_trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)
ResNet_validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=0)

optimizer = optim.Adam([*ResNet_encoder.parameters(), *classification_head.parameters()], lr = 0.0001, eps = 0.0001)

criterion = nn.BCELoss().to(device)

train_loss = []
validation_loss = []

train_accuracy = []
validation_accuracy = []

for epoch in range(1, number_of_epochs+1):

    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    ResNet_encoder.train()
    classification_head.train()

    epoch_train_loss = 0
    epoch_train_accuracy = 0

    for data in tqdm(ResNet_trainloader):

        image = data['input'].to(device)
        gt = data['gt'].to(device)
            
        oneHot_labels = []
            
        for sample_idx in range(batch_size):
            if torch.unique(gt[sample_idx, 1]).shape[0] == 2:
                # anomalous
                oneHot_labels.append([1, 0])
            else:
                # normal
                oneHot_labels.append([0, 1])

        oneHot_labels = torch.tensor(oneHot_labels).float().to(device)

        z = ResNet_encoder.forward(image)

        z = torch.reshape(z, shape=(z.shape[0], -1))
        y = classification_head.forward(z)

        loss = criterion(y, oneHot_labels)
        epoch_train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_accuracy += determine_class_accuracy(y, oneHot_labels).cpu()

        del image
        del gt
        del oneHot_labels
        del z
        del y
        del loss
    
    train_loss.append(epoch_train_loss / len(ResNet_trainloader))
    train_accuracy.append(epoch_train_accuracy / len(ResNet_trainloader))

    print(f'Training loss at epoch #{epoch}: {train_loss[-1]}')
    print(f'Training accuracy at epoch #{epoch}: {train_accuracy[-1]}')

    print()
    torch.cuda.empty_cache()

    # Validation loop
    ResNet_encoder.eval()
    classification_head.eval()

    epoch_validation_loss = 0
    epoch_validation_accuracy = 0

    for data in tqdm(ResNet_validationloader):
        with torch.no_grad():

            image = data['input'].to(device)
            gt = data['gt'].to(device)

            oneHot_labels = []
            
            for sample_idx in range(batch_size):
                if torch.unique(gt[sample_idx, 1]).shape[0] == 2:
                    # anomalous
                    oneHot_labels.append([1, 0])
                else:
                    # normal
                    oneHot_labels.append([0, 1])
            oneHot_labels = torch.tensor(oneHot_labels).float().to(device)

            z = ResNet_encoder.forward(image)

            z = torch.reshape(z, shape=(z.shape[0], -1))
            y = classification_head.forward(z)

            loss = criterion(y, oneHot_labels)
            epoch_validation_loss += loss.item()

            epoch_validation_accuracy += determine_class_accuracy(y, oneHot_labels).cpu()

            del image
            del gt
            del z
            del y
            del loss
    
    validation_loss.append(epoch_validation_loss / len(ResNet_validationloader))
    validation_accuracy.append(epoch_validation_accuracy / len(ResNet_validationloader))

    print(f'Validation loss at epoch #{epoch}: {validation_loss[-1]}')
    print(f'Validation accuracy at epoch #{epoch}: {validation_accuracy[-1]}')


    np.save(f'./results/{classifier_type}_{date}_accuracies.npy', [train_loss, validation_loss, train_accuracy, validation_accuracy])

    if epoch % 5 == 0:
        torch.save(ResNet_encoder.state_dict(), f'{model_path}{encoder_type}_{date}_ResNetEncoder_state_dict{epoch}.pth')
        torch.save(classification_head.state_dict(), f'{model_path}{classifier_type}_{date}_ClassifierHead_state_dict{epoch}.pth')

    print()

torch.save(ResNet_encoder.state_dict(), f'{model_path}{encoder_type}_{date}_ResNetEncoder_state_dict{number_of_epochs+1}.pth')
torch.save(classification_head.state_dict(), f'{model_path}{classifier_type}_{date}_ClassifierHead_state_dict{number_of_epochs+1}.pth')