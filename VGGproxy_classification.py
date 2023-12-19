import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import Constants, load_dataset

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import numpy as np
from tqdm import tqdm

c = Constants(
    batch_size = 1,
    patience = 15,
    num_workers = 8,
    number_of_epochs = 100,
    date = '01_12_2023',
    to_save_folder = 'Dec01',
    to_load_folder = None,
    device = 'cuda:0',
    proxy_type = 'VGGproxy',
    train_task = 'weightedBCE_&_VoxCFT_wNegativeRing',
    to_load_encoder_path = None,
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'simulated_lesions_on_brain_with_clean',
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
validationloader = DataLoader(validationset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

encoder = VGG3D_Encoder(input_channels=1).to(c.device)
projection_head = Projector(num_layers=5, layer_sizes=[32, 64, 128, 256, 512]).to(c.device)
classification_head = Classifier(input_channels=4096, output_channels=5, pooling_size=2).to(c.device)

projector_optimizer = optim.Adam(projection_head.parameters(), lr = 0.0001, eps = 0.0001)
classifier_optimizer = optim.Adam([*encoder.parameters(), *classification_head.parameters()], lr = 0.0001, eps = 0.0001)

class_weights = torch.tensor([1, 2, 2, 1, 1]).float().to(c.device)
classification_criterion = nn.BCELoss(weight=class_weights).to(c.device)
projection_criterion = VoxelwiseSupConLoss_inImage(device=c.device).to(c.device)

projection_train_loss_list = []
projection_validation_loss_list = []
classification_train_loss_list = []
classification_validation_loss_list = []

train_accuracy_list = []
validation_accuracy_list = []

best_validation_loss = None

print()
print('Training Proxy.')
for epoch in range(1, c.num_epochs+1):

    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    encoder.train()
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
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)
        oneHot_label = data['lesion_labels'].float().to(c.device)
        subtracted = data['subtracted'].to(c.device)

        to_projector, to_classifier = encoder(image)

        prediction = classification_head(to_classifier)
        classification_loss = classification_criterion(prediction, oneHot_label)
        classification_train_loss += classification_loss.item()
        train_accuracy += determine_multiclass_accuracy(prediction, oneHot_label).cpu()

        if torch.unique(gt[:, 1]).shape[0] == 2:
            # brain_mask = torch.zeros_like(image)
            # brain_mask[image != 0] = 1
            # brain_mask = brain_mask.float().to(device)

            projection = projection_head(to_projector)
            projection = F.interpolate(projection, size=(128, 128, 128))

            projection_loss = projection_criterion(projection, gt, subtracted)
            projection_train_loss += projection_loss.item()

            projector_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            loss = projection_loss + classification_loss
            loss.backward()
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
    encoder.eval()
    projection_head.eval()
    classification_head.eval()

    projection_validation_loss = 0
    classification_validation_loss = 0
    validation_accuracy = 0

    for data in tqdm(validationloader):
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)
        oneHot_label = data['lesion_labels'].float().to(c.device)
        subtracted = data['subtracted'].to(c.device)

        to_projector, to_classifier = encoder(image)
        prediction = classification_head(to_classifier)

        classification_loss = classification_criterion(prediction, oneHot_label)
        classification_validation_loss += classification_loss.item()
        validation_accuracy += determine_class_accuracy(prediction, oneHot_label).cpu()

        if torch.unique(gt[:, 1]).shape[0] == 2:
            # brain_mask = torch.zeros_like(image)
            # brain_mask[image != 0] = 1
            # brain_mask = brain_mask.float().to(device)

            projection = projection_head(to_projector)
            projection = F.interpolate(projection, size=(128, 128, 128))

            projection_loss = projection_criterion(projection, gt, subtracted)
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

    np.save(f'./results/{c.projector_type}_{c.date}_losses.npy', [projection_train_loss_list, projection_validation_loss_list])
    np.save(f'./results/{c.classifier_type}_{c.date}_accuracies.npy', [classification_train_loss_list, classification_validation_loss_list, train_accuracy_list, validation_accuracy_list])

    if best_validation_loss is None:
        best_validation_loss = classification_validation_loss_list[-1]
    elif classification_validation_loss_list[-1] < best_validation_loss:
        # patience = 15
        best_validation_loss = classification_validation_loss_list[-1]
        torch.save(encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict_best_loss{epoch}.pth')
        torch.save(projection_head.state_dict(), f'{c.to_save_folder}{c.projector_type}_{c.date}_state_dict_best_loss{epoch}.pth')
        torch.save(classification_head.state_dict(), f'{c.to_save_folder}{c.classifier_type}_{c.date}_state_dict_best_loss{epoch}.pth')

    if epoch % 10 == 0:
        torch.save(encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict{epoch}.pth')
        torch.save(projection_head.state_dict(), f'{c.to_save_folder}{c.projector_type}_{c.date}_state_dict{epoch}.pth')
        torch.save(classification_head.state_dict(), f'{c.to_save_folder}{c.classifier_type}_{c.date}_state_dict{epoch}.pth')

    print()

torch.save(encoder.state_dict(), f'{c.to_save_folder}{c.encoder_type}_{c.date}_state_dict{c.num_epochs+1}.pth')
torch.save(projection_head.state_dict(), f'{c.to_save_folder}{c.projector_type}_{c.date}_state_dict{c.num_epochs+1}.pth')
torch.save(classification_head.state_dict(), f'{c.to_save_folder}{c.classifier_type}_{c.date}_state_dict{c.num_epochs+1}.pth')

print()
print('Script executed.')