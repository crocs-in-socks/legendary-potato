import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ModelArchitecture.Encoders import VGG3D, ResNet3D, Classifier

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 4
number_of_epochs = 20
device = 'cuda:1'
model_type = 'ResNetClassifier'
date = '06_11_2023'

model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/proxy_models/'

trainset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TrainSet_5_11_23/*.npz'))
validationset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/ValidSet_5_11_23/*.npz'))

print(f'Trainset size: {len(trainset_paths)}')
print(f'Validationset size: {len(validationset_paths)}')

composed_transform = transforms.Compose([
        RandomRotation3D([10, 10], clean=True),
        RandomIntensityChanges(clean=True),
        ToTensor3D(True, clean=True)
    ])

trainset = ImageLoader3D(paths=trainset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)
validationset = ImageLoader3D(paths=validationset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)

ResNet_Model = ResNet3D(image_channels=1).to(device)
classification_head = Classifier(input_channels=32768, output_channels=2).to(device)

ResNet_trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
ResNet_validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=0)

optimizer = optim.Adam([*ResNet_Model.parameters(), *classification_head.parameters()], lr = 0.0001, eps = 0.0001)
criterion = nn.BCELoss().to(device)

train_losses = []
validation_losses = []

train_accuracies = []
validation_accuracies = []

print()
print('Training ResNet Model.')

for epoch in range(1, number_of_epochs+1):

    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    ResNet_Model.train()
    classification_head.train()

    epoch_train_loss = 0
    epoch_train_accuracy = 0

    for data in tqdm(ResNet_trainloader):

        # Anomalous data
        image = data['input'].to(device)
        labels = (torch.tensor([[1, 0]] * batch_size)).float().to(device)

        z_image = ResNet_Model.forward(image)
        z_image = torch.reshape(z_image, shape=(z_image.shape[0], -1))
        y_image = classification_head.forward(z_image)

        loss = criterion(y_image, labels)
        epoch_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_accuracy += determine_class_accuracy(y_image, labels).cpu()

        del image
        del z_image
        del y_image
        del loss

        # Clean data
        clean = data['clean'].to(device)
        labels = (torch.tensor([[0, 1]] * batch_size)).float().to(device)

        z_clean = ResNet_Model.forward(clean)
        z_clean = torch.reshape(z_clean, shape=(z_clean.shape[0], -1))
        y_clean = classification_head.forward(z_clean)

        loss = criterion(y_clean, labels)
        epoch_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_accuracy += determine_class_accuracy(y_clean, labels).cpu()

        del clean
        del z_clean
        del y_clean
        del loss
    
    train_losses.append(epoch_train_loss / (len(ResNet_trainloader)*2))
    train_accuracies.append(epoch_train_accuracy / (len(ResNet_trainloader)*2))

    print(f'Training loss at epoch #{epoch}: {train_losses[-1]}')
    print(f'Training accuracy at epoch #{epoch}: {train_accuracies[-1]}')

    print()
    torch.cuda.empty_cache()

    # Validation loop
    ResNet_Model.eval()
    classification_head.eval()

    epoch_validation_loss = 0
    epoch_validation_accuracy = 0

    for data in tqdm(ResNet_validationloader):
        with torch.no_grad():
            # Anomalous data
            image = data['input'].to(device)
            labels = (torch.tensor([[1, 0]] * batch_size)).float().to(device)

            z_image = ResNet_Model.forward(image)
            z_image = torch.reshape(z_image, shape=(z_image.shape[0], -1))
            y_image = classification_head.forward(z_image)

            loss = criterion(y_image, labels)
            epoch_validation_loss += loss.item()

            epoch_validation_accuracy += determine_class_accuracy(y_image, labels).cpu()

            del image
            del z_image
            del y_image
            del loss

            # Clean data
            clean = data['clean'].to(device)
            labels = (torch.tensor([[0, 1]] * batch_size)).float().to(device)

            z_clean = ResNet_Model.forward(clean)
            z_clean = torch.reshape(z_clean, shape=(z_clean.shape[0], -1))
            y_clean = classification_head.forward(z_clean)

            loss = criterion(y_clean, labels)
            epoch_validation_loss += loss.item()

            epoch_validation_accuracy += determine_class_accuracy(y_clean, labels).cpu()

            del clean
            del z_clean
            del y_clean
            del loss
    
    validation_losses.append(epoch_validation_loss / (len(ResNet_validationloader)*2))
    validation_accuracies.append(epoch_validation_accuracy / (len(ResNet_validationloader)*2))

    print(f'Validation loss at epoch #{epoch}: {validation_losses[-1]}')
    print(f'Validation accuracy at epoch #{epoch}: {validation_accuracies[-1]}')

    np.save(f'./results/{model_type}_{date}_losses.npy', [train_losses, validation_losses])
    np.save(f'./results/{model_type}_{date}_accuracies.npy', [train_accuracies, validation_accuracies])

    if epoch % 5 == 0:
        torch.save(ResNet_Model.state_dict(), f'{model_path}{model_type}_{date}_ResNet_state_dict{epoch}.pth')
        torch.save(classification_head.state_dict(), f'{model_path}{model_type}_{date}_ClassifierHead_state_dict{epoch}.pth')

    print()

torch.save(ResNet_Model.state_dict(), f'{model_path}{model_type}_{date}_ResNet_state_dict{number_of_epochs+1}.pth')
torch.save(classification_head.state_dict(), f'{model_path}{model_type}_{date}_ClassifierHead_state_dict{number_of_epochs+1}.pth')

print()
print('Script executed.')

# example = trainset[0]

# plt.figure(figsize=(20, 15))
# plt.subplot(1, 3, 1)
# plt.imshow(example['input'][0, :, :, 64], cmap='gray')
# plt.colorbar()
# plt.title('Input')

# plt.subplot(1, 3, 2)
# plt.imshow(example['gt'][0, :, :, 64], cmap='gray')
# plt.colorbar()
# plt.title('gt')

# plt.subplot(1, 3, 3)
# plt.imshow(example['clean'][0, :, :, 64], cmap='gray')
# plt.colorbar()
# plt.title('clean')

# plt.savefig('./temp')
# plt.close()