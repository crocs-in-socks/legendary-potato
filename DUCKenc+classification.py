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

batch_size = 2
patience = 15
device = 'cuda:1'
number_of_epochs = 100
date = '27_11_2023'
classifier_type = 'DUCK_WMHenc+ClassificationHead'
encoder_type = 'DUCK_WMHenc_sim_finetuned'

model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov27/'
DUCKmodel_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/Duck1wmh_focal + dice_state_dict_best_loss97.pth'

from_Sim1000_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*FLAIR.nii.gz'))
from_sim2211_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/brats/**/*FLAIR.nii.gz'))

from_Sim1000_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*mask.nii.gz'))
from_sim2211_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/brats/**/*mask.nii.gz'))

from_Sim1000_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Sim1000/Dark/all/**/*.json'))
from_sim2211_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Full_sim_22_11_23/Dark/brats/**/*.json'))

composed_transform = transforms.Compose([
        ToTensor3D(labeled=True)
    ])

from_Sim1000 = ImageLoader3D(paths=from_Sim1000_data_paths, gt_paths=from_Sim1000_gt_paths, json_paths=from_Sim1000_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)

from_sim2211 = ImageLoader3D(paths=from_sim2211_data_paths,gt_paths=from_sim2211_gt_paths, json_paths=from_sim2211_json_paths, image_size=128, type_of_imgs='nifty', transform=composed_transform)

fullset = ConcatDataset([from_Sim1000, from_sim2211])

train_size = int(0.8 * len(fullset))
# validation_size = int(0.1 * len(fullset))
validation_size = len(fullset) - train_size
# test_size = len(fullset) - (train_size + validation_size)

trainset, validationset = random_split(fullset, (train_size, validation_size))

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=0)

DUCKnet_encoder = DuckNet(input_channels=1, out_classes=2, starting_filters=17).to(device)
DUCKnet_encoder.load_state_dict(torch.load(DUCKmodel_path))
# classification_head = Classifier(input_channels=17408, output_channels=5).to(device)
classification_head = Classifier(input_channels=2176, output_channels=5).to(device)

# Freezing DUCKnet
# for param in DUCKnet_encoder.parameters():
#     param.requires_grad = False

DUCK_optimizer = optim.Adam(DUCKnet_encoder.parameters(), lr = 0.00001, eps = 0.0001)
optimizer = optim.Adam(classification_head.parameters(), lr = 0.0001, eps = 0.0001)
criterion = nn.BCELoss().to(device)

train_loss_list = []
validation_loss_list = []

train_accuracy_list = []
validation_accuracy_list = []

best_validation_loss = None

print()
print('Training Classifier.')
for epoch in range(1, number_of_epochs+1):

    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    DUCKnet_encoder.eval()
    classification_head.train()

    train_loss = 0
    train_accuracy = 0

    patience -= 1
    if patience == 0:
        print(f'Breaking at epoch #{epoch} due to lack of patience.')

    for data in tqdm(trainloader):
        image = data['input'].to(device)
        oneHot_label = data['lesion_labels'].float().to(device)

        final_out, t53_out = DUCKnet_encoder(image)
        # t53_out = torch.reshape(t53_out, shape=(t53_out.shape[0], -1))
        prediction = classification_head(t53_out)

        loss = criterion(prediction, oneHot_label)
        train_loss += loss.item()
        train_accuracy += determine_class_accuracy(prediction, oneHot_label).cpu()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del image
        del oneHot_label
        del final_out
        del t53_out
        del prediction
        del loss
    
    train_loss_list.append(train_loss / len(trainloader))
    train_accuracy_list.append(train_accuracy / len(trainloader))
    print(f'Train loss at epoch #{epoch}: {train_loss_list[-1]}')
    print(f'Train accuracy at epoch #{epoch}: {train_accuracy_list[-1]}')

    print()
    torch.cuda.empty_cache()

    # Validation loop
    DUCKnet_encoder.eval()
    classification_head.eval()

    validation_loss = 0
    validation_accuracy = 0

    for data in tqdm(validationloader):
        image = data['input'].to(device)
        oneHot_label = data['lesion_labels'].float().to(device)

        final_out, t53_out = DUCKnet_encoder(image)
        # t53_out = torch.reshape(t53_out, shape=(t53_out.shape[0], -1))
        prediction = classification_head(t53_out)

        loss = criterion(prediction, oneHot_label)
        validation_loss += loss.item()
        validation_accuracy += determine_class_accuracy(prediction, oneHot_label).cpu()

        del image
        del oneHot_label
        del final_out
        del t53_out
        del prediction
        del loss
    
    validation_loss_list.append(validation_loss / len(validationloader))
    validation_accuracy_list.append(validation_accuracy / len(validationloader))
    print(f'Train loss at epoch #{epoch}: {validation_loss_list[-1]}')
    print(f'Train accuracy at epoch #{epoch}: {validation_accuracy_list[-1]}')

    np.save(f'./results/{classifier_type}_{date}_accuracies.npy', [train_loss_list, validation_loss_list, train_accuracy_list, validation_accuracy_list])

    if best_validation_loss is None:
        best_validation_loss = validation_loss_list[-1]
    elif validation_loss_list[-1] < best_validation_loss:
        patience = 15
        best_validation_loss = validation_loss_list[-1]
        torch.save(DUCKnet_encoder.state_dict(), f'{model_path}{encoder_type}_{date}_state_dict_best_loss{epoch}.pth')
        torch.save(classification_head.state_dict(), f'{model_path}{classifier_type}_{date}_state_dict_best_loss{epoch}.pth')
        print(f'New best validation loss at epoch #{epoch}')

    if epoch % 5 == 0:
        torch.save(DUCKnet_encoder.state_dict(), f'{model_path}{encoder_type}_{date}_state_dict{epoch}.pth')
        torch.save(classification_head.state_dict(), f'{model_path}{classifier_type}_{date}_state_dict{epoch}.pth')

    print()

torch.save(DUCKnet_encoder.state_dict(), f'{model_path}{encoder_type}_{date}_state_dict{number_of_epochs+1}.pth')
torch.save(classification_head.state_dict(), f'{model_path}{classifier_type}_{date}_state_dict{number_of_epochs+1}.pth')

print()
print('Script executed.')