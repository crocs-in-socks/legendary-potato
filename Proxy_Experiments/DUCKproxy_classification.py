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

batch_size = 1
patience = 15
num_workers = 16
device = 'cuda:1'
number_of_epochs = 100
date = '30_11_2023'
encoder_type = 'DUCK_WMH_proxy_encoder_simFinetuned_weightedBCE'
classifier_type = 'DUCK_WMH_proxy_classifier_simFinetuned_weightedBCE'
projector_type = 'DUCK_WMH_proxy_projector_simFinetuned_weightedBCE'

save_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov30/'

# DUCKmodel_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/Duck1wmh_focal + dice_state_dict_best_loss97.pth'
encoder_path = f'{save_model_path}DUCK_WMH_proxy_encoder_simFinetuned_weightedBCE_30_11_2023_state_dict_best_loss10.pth'
projection_head_path = f'{save_model_path}DUCK_WMH_proxy_projector_simFinetuned_weightedBCE_30_11_2023_state_dict_best_loss10.pth'
classification_head_path = f'{save_model_path}DUCK_WMH_proxy_classifier_simFinetuned_weightedBCE_30_11_2023_state_dict_best_loss10.pth'

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

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

DUCKnet_encoder = DuckNet(input_channels=1, out_classes=2, starting_filters=17).to(device)
DUCKnet_encoder.load_state_dict(torch.load(encoder_path))

# classification_head = Classifier(input_channels=17408, output_channels=5).to(device)
classification_head = Classifier(input_channels=2176, output_channels=5).to(device)
classification_head.load_state_dict(torch.load(classification_head_path))

projection_head = Projector(num_layers=5, layer_sizes=[17, 34, 68, 136, 272]).to(device)
projection_head.load_state_dict(torch.load(projection_head_path))

# Freezing DUCKnet
# for param in DUCKnet_encoder.parameters():
    # param.requires_grad = False

projector_optimizer = optim.Adam([*DUCKnet_encoder.parameters(), *projection_head.parameters()], lr = 0.0001, eps = 0.0001)
# projector_optimizer = optim.Adam(projection_head.parameters(), lr = 0.0001, eps = 0.0001)
classifier_optimizer = optim.Adam(classification_head.parameters(), lr = 0.0001, eps = 0.0001)

class_weights = torch.tensor([0.5, 1, 1, 0.5, 0.5]).float().to(device)
classification_criterion = nn.BCELoss(weight=class_weights).to(device)
projection_criterion = VoxelwiseSupConLoss_inImage(device=device).to(device)

projection_train_loss_list = []
projection_validation_loss_list = []
classification_train_loss_list = []
classification_validation_loss_list = []

train_accuracy_list = []
validation_accuracy_list = []

best_validation_loss = None

print()
print('Training Proxy.')
for epoch in range(11, number_of_epochs+1):

    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    DUCKnet_encoder.train()
    projection_head.train()
    classification_head.train()

    projection_train_loss = 0
    classification_train_loss = 0
    train_accuracy = 0

    # patience -= 1
    # if patience == 0:
    #     print()
    #     print(f'Breaking at epoch #{epoch} due to lack of patience. Best validation loss was: {best_validation_loss}')

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:

    for data in tqdm(trainloader):
        image = data['input'].to(device)
        gt = data['gt'].to(device)
        oneHot_label = data['lesion_labels'].float().to(device)

        to_projector, to_classifier = DUCKnet_encoder(image)
        prediction = classification_head(to_classifier)

        classification_loss = classification_criterion(prediction, oneHot_label)
        classification_train_loss += classification_loss.item()
        train_accuracy += determine_class_accuracy(prediction, oneHot_label).cpu()

        # classifier_optimizer.zero_grad()
        # classification_loss.backward()
        # classifier_optimizer.step()

        if torch.unique(gt[:, 1]).shape[0] == 2:
            projection = projection_head(to_projector)
            projection = F.interpolate(projection, size=(128, 128, 128))

            projection_loss = projection_criterion(projection, gt)
            projection_train_loss += projection_loss.item()

            # encoder_projector_optimizer.zero_grad()
            # projection_loss.backward()
            # encoder_projector_optimizer.step()
            # encoder_projector_optimizer.zero_grad()

            projector_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            loss = projection_loss + classification_loss
            loss.backward()
            # encoder_projector_optimizer.step()
            projector_optimizer.step()
            classifier_optimizer.step()

            del projection
            del projection_loss

        else:
            classifier_optimizer.zero_grad()
            loss = classification_loss
            loss.backward()
            classifier_optimizer.step()
            
        del image
        del gt
        del oneHot_label
        del to_projector
        del to_classifier
        del prediction
        del classification_loss
        del loss
    
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    # break
    
    projection_train_loss_list.append(projection_train_loss / len(trainloader))
    classification_train_loss_list.append(classification_train_loss / len(trainloader))
    train_accuracy_list.append(train_accuracy / len(trainloader))
    print(f'Projection train loss at epoch #{epoch}: {projection_train_loss_list[-1]}')
    print(f'Classification train loss at epoch #{epoch}: {classification_train_loss_list[-1]}')
    print(f'Train accuracy at epoch #{epoch}: {train_accuracy_list[-1]}')

    print()
    torch.cuda.empty_cache()

    # Validation loop
    DUCKnet_encoder.eval()
    projection_head.eval()
    classification_head.eval()

    projection_validation_loss = 0
    classification_validation_loss = 0
    validation_accuracy = 0

    for data in tqdm(validationloader):
        image = data['input'].to(device)
        gt = data['gt'].to(device)
        oneHot_label = data['lesion_labels'].float().to(device)

        to_projector, to_classifier = DUCKnet_encoder(image)
        prediction = classification_head(to_classifier)

        classification_loss = classification_criterion(prediction, oneHot_label)
        classification_validation_loss += classification_loss.item()
        validation_accuracy += determine_class_accuracy(prediction, oneHot_label).cpu()

        if torch.unique(gt[:, 1]).shape[0] == 2:
            projection = projection_head(to_projector)
            projection = F.interpolate(projection, size=(128, 128, 128))

            projection_loss = projection_criterion(projection, gt)
            projection_validation_loss += projection_loss.item()

            del projection
            del projection_loss

        del image
        del gt
        del oneHot_label
        del to_projector
        del to_classifier
        del prediction
        del classification_loss
    
    projection_validation_loss_list.append(projection_validation_loss / len(validationloader))
    classification_validation_loss_list.append(classification_validation_loss / len(validationloader))
    validation_accuracy_list.append(validation_accuracy / len(validationloader))
    print(f'Projection validation loss at epoch #{epoch}: {projection_validation_loss_list[-1]}')
    print(f'Classification validation loss at epoch #{epoch}: {classification_validation_loss_list[-1]}')
    print(f'Validation accuracy at epoch #{epoch}: {validation_accuracy_list[-1]}')

    np.save(f'./results/{projector_type}_{date}_losses.npy', [projection_train_loss_list, projection_validation_loss_list])
    np.save(f'./results/{classifier_type}_{date}_accuracies.npy', [classification_train_loss_list, classification_validation_loss_list, train_accuracy_list, validation_accuracy_list])

    if best_validation_loss is None:
        best_validation_loss = classification_validation_loss_list[-1]
    elif classification_validation_loss_list[-1] < best_validation_loss:
        patience = 15
        best_validation_loss = classification_validation_loss_list[-1]
        torch.save(DUCKnet_encoder.state_dict(), f'{save_model_path}{encoder_type}_{date}_state_dict_best_loss{epoch}.pth')
        torch.save(projection_head.state_dict(), f'{save_model_path}{projector_type}_{date}_state_dict_best_loss{epoch}.pth')
        torch.save(classification_head.state_dict(), f'{save_model_path}{classifier_type}_{date}_state_dict_best_loss{epoch}.pth')
        # torch.save(encoder_projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict_best_loss{epoch}.pth')
        torch.save(projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict_best_loss{epoch}.pth')
        torch.save(classifier_optimizer.state_dict(), f'{save_model_path}{classifier_type}_ optimizer_{date}_state_dict_best_loss{epoch}.pth')
        print(f'New best validation loss at epoch #{epoch}')

    if epoch % 10 == 0:
        torch.save(DUCKnet_encoder.state_dict(), f'{save_model_path}{encoder_type}_{date}_state_dict{epoch}.pth')
        torch.save(projection_head.state_dict(), f'{save_model_path}{projector_type}_{date}_state_dict{epoch}.pth')
        torch.save(classification_head.state_dict(), f'{save_model_path}{classifier_type}_{date}_state_dict{epoch}.pth')
        # torch.save(encoder_projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict{epoch}.pth')
        torch.save(projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict{epoch}.pth')
        torch.save(classifier_optimizer.state_dict(), f'{save_model_path}{classifier_type}_ optimizer_{date}_state_dict{epoch}.pth')

    print()

torch.save(DUCKnet_encoder.state_dict(), f'{save_model_path}{encoder_type}_{date}_state_dict{number_of_epochs+1}.pth')
torch.save(projection_head.state_dict(), f'{save_model_path}{projector_type}_{date}_state_dict{number_of_epochs+1}.pth')
torch.save(classification_head.state_dict(), f'{save_model_path}{classifier_type}_{date}_state_dict{number_of_epochs+1}.pth')
# torch.save(encoder_projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict{number_of_epochs+1}.pth')
torch.save(projector_optimizer.state_dict(), f'{save_model_path}{projector_type}_ optimizer_{date}_state_dict{number_of_epochs+1}.pth')
torch.save(classifier_optimizer.state_dict(), f'{save_model_path}{classifier_type}_ optimizer_{date}_state_dict{number_of_epochs+1}.pth')

print()
print('Script executed.')