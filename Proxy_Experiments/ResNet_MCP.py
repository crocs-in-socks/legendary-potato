import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split

from ModelArchitecture.Encoders import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
import numpy as np
from tqdm import tqdm

batch_size = 8
number_of_epochs = 100
device = 'cuda:1'
encoder_type = 'MCP_moreData_ResNet_Encoder'
classifier_type = 'MCP_moreData_ResNet_ClassificationHead'
date = '24_11_2023'

model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov24/'

clean_data = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
clean_labels = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))
clean_masks = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*biascorr_brain_mask.nii.gz*'))

train_fraction = int(len(clean_data)*0.7)
validation_fraction = int(len(clean_data)*0.1)

clean_trainset_data = clean_data[:train_fraction]
clean_trainset_labels = clean_labels[:train_fraction]

clean_validationset_data = clean_data[train_fraction:train_fraction+validation_fraction]
clean_validationset_labels = clean_labels[train_fraction:train_fraction+validation_fraction]

# anomalous_trainset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TrainSet_5_11_23/*.npz'))
# anomalous_validationset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/ValidSet_5_11_23/*.npz'))

# anomalous_trainset_jsons = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TrainSet_5_11_23/*.json'))
# anomalous_validationset_jsons = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/ValidSet_5_11_23/*.json'))

# print(f'Anomalous Trainset size: {len(anomalous_trainset_paths)}')
# print(f'Anomalous Validationset size: {len(anomalous_validationset_paths)}')
# print(f'Clean Trainset size: {len(clean_trainset_data)}')
# print(f'Clean Validationset size: {len(clean_validationset_data)}')

# composed_transform = transforms.Compose([
#         ToTensor3D(labeled=True, clean=True)
#     ])

# anomalous_trainset = ImageLoader3D(paths=anomalous_trainset_paths, gt_paths=None, json_paths=anomalous_trainset_jsons,image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True)
# anomalous_validationset = ImageLoader3D(paths=anomalous_validationset_paths, gt_paths=None, json_paths=anomalous_validationset_jsons, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True)

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

train_size = int(0.7 * len(fullset))
validation_size = int(0.1 * len(fullset))
test_size = len(fullset) - (train_size + validation_size)

anomalous_trainset, anomalous_validationset, anomalous_testset = random_split(fullset, (train_size, validation_size, test_size))

clean_trainset = ImageLoader3D(paths=clean_trainset_data, gt_paths=clean_trainset_labels, type_of_imgs='nifty', transform=composed_transform)
clean_validationset = ImageLoader3D(paths=clean_validationset_data, gt_paths=clean_validationset_labels, type_of_imgs='nifty', transform=composed_transform)

trainset = ConcatDataset([anomalous_trainset, clean_trainset])
validationset = ConcatDataset([anomalous_validationset, clean_validationset])
# trainset = anomalous_trainset
# validationset = anomalous_validationset

ResNet_encoder = ResNet3D_Encoder(image_channels=1).to(device)
classification_head = Classifier(input_channels=32768, output_channels=5).to(device)

ResNet_trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
ResNet_validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=0)

optimizer = optim.Adam([*ResNet_encoder.parameters(), *classification_head.parameters()], lr = 0.0001, eps = 0.0001)

criterion = nn.BCELoss().to(device)

train_loss = []
validation_loss = []

train_accuracy = []
validation_accuracy = []

print(f'Total trainset size: {len(trainset)}')
print(f'Total validationset size: {len(validationset)}')

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
        # clean = data['clean'].to(device)
        # mixed = torch.cat([image, clean])
        mixed = image
            
        oneHot_labels = data['lesion_labels'].float().to(device)
        current_batch_size = image.shape[0]
        # extension = torch.tensor([[1, 0, 0, 0, 0]] * current_batch_size).float()
        # oneHot_labels = torch.cat([oneHot_labels, extension]).to(device)

        out_dict = ResNet_encoder.forward(mixed)
        z = out_dict['out4']

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
        del out_dict
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
            # clean = data['clean'].to(device)
            # mixed = torch.cat([image, clean])
            mixed = image

            oneHot_labels = data['lesion_labels'].float().to(device)
            current_batch_size = image.shape[0]

            # extension = torch.tensor([[1, 0, 0, 0, 0]] * current_batch_size).float()
            # oneHot_labels = torch.cat([oneHot_labels, extension]).to(device)

            out_dict = ResNet_encoder.forward(mixed)
            z = out_dict['out4']

            z = torch.reshape(z, shape=(z.shape[0], -1))
            y = classification_head.forward(z)

            loss = criterion(y, oneHot_labels)
            epoch_validation_loss += loss.item()

            epoch_validation_accuracy += determine_class_accuracy(y, oneHot_labels).cpu()

            del image
            del gt
            del out_dict
            del z
            del y
            del loss
    
    validation_loss.append(epoch_validation_loss / len(ResNet_validationloader))
    validation_accuracy.append(epoch_validation_accuracy / len(ResNet_validationloader))

    print(f'Validation loss at epoch #{epoch}: {validation_loss[-1]}')
    print(f'Validation accuracy at epoch #{epoch}: {validation_accuracy[-1]}')

    np.save(f'./results/{classifier_type}_{date}_accuracies.npy', [train_loss, validation_loss, train_accuracy, validation_accuracy])

    if epoch % 5 == 0:
        torch.save(ResNet_encoder.state_dict(), f'{model_path}{encoder_type}_{date}_state_dict{epoch}.pth')
        torch.save(classification_head.state_dict(), f'{model_path}{classifier_type}_{date}_state_dict{epoch}.pth')

    print()

torch.save(ResNet_encoder.state_dict(), f'{model_path}{encoder_type}_{date}_state_dict{number_of_epochs+1}.pth')
torch.save(classification_head.state_dict(), f'{model_path}{classifier_type}_{date}_state_dict{number_of_epochs+1}.pth')