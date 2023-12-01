import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from ModelArchitecture.Encoders import VGG3D, Classifier
from ImageLoader.ContrastiveLoader3D import ContrastiveLoader3D

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.DUCK_Net import DuckNet
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
from tqdm import tqdm

DUCK_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/models_retrained/'
VGG_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/contrastive_models/'
ResNet_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/contrastive_models/'

model_type = 'VGG3D_encoder_samplelvl'
encoder_type = 'ResNet3D_encoder_samplelvl'
classifier_type = 'VGG3D_classifier_samplelvl'

batch_size = 1
number_of_epochs = 100

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

device = 'cuda:1'

DUCK_model = DuckNet(input_channels=1, out_classes=2, starting_filters=17).to(device)
DUCK_model.load_state_dict(torch.load(DUCK_model_path + '/DUCK_wmh_24_10_23_state_dict77.pth'))

t2_train = []
t2_validation = []

train_class_labels = []
validation_class_labels = []
oneHot_train_class_labels = []
oneHot_validation_class_labels = []

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

print(f'Number of sick train samples: {len(sick_trainset)}')
print(f'Number of healthy train samples: {len(healthy_trainset)}')
mixed_trainset = ConcatDataset([sick_trainset, healthy_trainset])
print(f'Number of mixed train samples: {len(mixed_trainset)}')

print(f'Number of sick validation samples: {len(sick_validationset)}')
print(f'Number of healthy validation samples: {len(healthy_validationset)}')
mixed_validationset = ConcatDataset([sick_validationset, healthy_validationset])
print(f'Number of mixed validation samples: {len(mixed_validationset)}')

DUCK_trainloader = DataLoader(mixed_trainset, batch_size=batch_size, shuffle=True, num_workers=0)
DUCK_validationloader = DataLoader(mixed_validationset, batch_size=batch_size, shuffle=False, num_workers=0)

print()
print('Runnning DUCK_Model for train set.')
for data in tqdm(DUCK_trainloader):
    if list(data['input'].size())[0] == batch_size:
        image = data['input'].to(device)
        label = data['gt'].to(device)

        if torch.unique(label[:,0]).shape[0] == 2:
            train_class_labels.append(0)
            oneHot_train_class_labels.append([0, 1])
        else:
            train_class_labels.append(1)
            oneHot_train_class_labels.append([1, 0])

        hook1 = DUCK_model.t2.register_forward_hook(getActivation('t2'))
        out = DUCK_model(image)
        t2_train.append(activations['t2'].cpu())
        torch.cuda.empty_cache()

        del image
        del label
        del out
print()
hook1.remove()

print()
print('Runnning DUCK_Model for validation set.')
for data in tqdm(DUCK_validationloader):
    if list(data['input'].size())[0] == batch_size:
        image = data['input'].to(device)
        label = data['gt'].to(device)

        if torch.unique(label[:,0]).shape[0] == 2:
            validation_class_labels.append(0)
            oneHot_validation_class_labels.append([0, 1])
        else:
            validation_class_labels.append(1)
            oneHot_validation_class_labels.append([1, 0])

        hook1 = DUCK_model.t2.register_forward_hook(getActivation('t2'))
        out = DUCK_model(image)
        t2_validation.append(activations['t2'].cpu())
        torch.cuda.empty_cache()

        del image
        del label
        del out
print()
hook1.remove()

train_class_labels = torch.tensor(train_class_labels).float().to(device)
oneHot_train_class_labels = torch.tensor(oneHot_train_class_labels).float().to(device)

validation_class_labels = torch.tensor(validation_class_labels).float().to(device)
oneHot_validation_class_labels = torch.tensor(oneHot_validation_class_labels).float().to(device)

t2_train = torch.stack(t2_train)
t2_train = t2_train.squeeze(1)
# t2_train = t2_train.to(device)

t2_validation = torch.stack(t2_validation)
t2_validation = t2_validation.squeeze(1)
# t2_validation = t2_validation.to(device)

number_of_features = t2_train.shape[1]

print(f't2_train shape: {t2_train.shape}')
print(f't2_validation shape: {t2_validation.shape}')
print(f'Number of features: {number_of_features}')

Contrastive_Transforms = transforms.Compose([
    RandomFlip3D(axis=0, p=0.5),
    RandomFlip3D(axis=1, p=0.5),
    RandomFlip3D(axis=2, p=0.5),
    RandomGaussianBlur3D(p=0.5),
    RandomIntensityChanges3D(p=0.5)
])

# transform1 = transforms.Compose([
#     RandomCropResize3D(p=1, scale_factor=2)
# ])

# transform2 = transforms.Compose([
#     RandomFlip3D(axis=0, p=0.5),
#     RandomFlip3D(axis=1, p=0.5),
#     RandomFlip3D(axis=2, p=0.5),
#     RandomGaussianBlur3D(p=0.5),
#     RandomIntensityChanges3D(p=1)
# ])

feature_map_dataset = ContrastiveLoader3D(t2_train, transform=Contrastive_Transforms)

VGG_Model = VGG3D(input_channels=number_of_features, output_classes=2).to(device)
classification_head = Classifier(input_channels=16384, output_channels=2).to(device)
# VGG_trainloader = DataLoader(t2_train, batch_size=batch_size, shuffle=True, num_workers=0)
VGG_trainloader = DataLoader(feature_map_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
VGG_validationloader = DataLoader(t2_validation, batch_size=batch_size, shuffle=False, num_workers=0)

encoder_optimizer = optim.Adam(VGG_Model.parameters(), lr = 0.0001, eps = 0.0001)
classifier_optimizer = optim.Adam(classification_head.parameters(), lr = 0.0001, eps = 0.0001)

encoder_criterion = SupervisedContrastiveLoss().to(device)
classifier_criterion = nn.CrossEntropyLoss().to(device)

encoder_train_losses = []
encoder_validation_losses = []

classifier_train_losses = []
classifier_validation_losses = []

best_encoder_train_loss = torch.inf
best_classifier_train_loss = torch.inf
best_encoder_validation_loss = torch.inf
best_classifier_validation_loss = torch.inf
best_epoch = 0

print()
print('Training VGG_Model.')

for epoch in range(number_of_epochs):
    
    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    VGG_Model.train()
    classification_head.train()

    Zs = []
    Ys = []

    print()
    for X in tqdm(VGG_trainloader):
        X = X.to(device)
        Z = VGG_Model(X)
        Z = torch.reshape(Z, shape=(-1,))
        Y = classification_head(Z.detach())
        Zs.append(Z)
        Ys.append(Y)
    print()
    Zs = torch.stack(Zs).to(device)
    Ys = torch.stack(Ys).to(device)

    encoder_loss = encoder_criterion(Zs, train_class_labels)
    encoder_train_losses.append(encoder_loss.item())
    VGG_Model.zero_grad()
    encoder_loss.backward()
    encoder_optimizer.step()

    del Zs

    print(f'Encoder train loss at epoch #{epoch}: {encoder_loss.item()}')

    classifier_loss = classifier_criterion(Ys, oneHot_train_class_labels)
    classifier_train_losses.append(classifier_loss.item())
    classification_head.zero_grad()
    classifier_loss.backward()
    classifier_optimizer.step()

    del Ys

    print(f'Classification train loss at epoch #{epoch}: {classifier_loss.item()}')

    if encoder_loss.item() < best_encoder_train_loss:
        best_encoder_train_loss = encoder_loss.item()
    if classifier_loss.item() < best_classifier_train_loss:
        best_classifier_train_loss = classifier_loss.item()

    # Validation loop
    VGG_Model.eval()
    classification_head.eval()
    with torch.no_grad():
        Zs = []
        Ys = []
        print()
        for X in tqdm(VGG_validationloader):
            Z = VGG_Model(X.to(device))
            Z = torch.reshape(Z, shape=(-1,))
            Y = classification_head(Z.detach())
            Zs.append(Z)
            Ys.append(Y)
        print()
        Zs = torch.stack(Zs).to(device)
        Ys = torch.stack(Ys).to(device)

        encoder_loss = encoder_criterion(Zs, validation_class_labels)
        encoder_validation_losses.append(encoder_loss.item())
        print(f'Encoder validation loss at epoch #{epoch}: {encoder_loss.item()}')

        classifier_loss = classifier_criterion(Ys, oneHot_validation_class_labels)
        classifier_validation_losses.append(classifier_loss.item())
        print(f'Classification validation loss at epoch #{epoch}: {classifier_loss.item()}')
    
    if encoder_loss.item() < best_encoder_validation_loss:
        best_encoder_validation_loss = encoder_loss.item()
    if classifier_loss.item() < best_classifier_validation_loss:
        best_classifier_validation_loss = classifier_loss.item()
        best_epoch = epoch

    np.save(f'./results/{model_type}_encoder_loss.npy', [encoder_train_losses, encoder_validation_losses])

    np.save(f'./results/{model_type}_classifier_loss.npy', [classifier_train_losses, classifier_validation_losses])

    print()

torch.save(VGG_Model.state_dict(), VGG_model_path+model_type+'_state_dict'+str(epoch)+'.pth')
torch.save(classification_head.state_dict(), VGG_model_path+classifier_type+'_state_dict'+str(epoch)+'.pth')

print(f'best encoder train loss: {best_encoder_train_loss}')
print(f'best classifier train loss: {best_classifier_train_loss}')
print(f'best encoder validation loss: {best_encoder_validation_loss}')
print(f'best classifier validation loss: {best_classifier_validation_loss}')
print(f'best epoch: {best_epoch}')

    # for outer_idx, (x1, x2) in tqdm(enumerate(feature_map_dataset)):

    #     x1 = x1.unsqueeze(0).to(device)
    #     x2 = x2.unsqueeze(0).to(device)
    #     x_label = class_labels[outer_idx]
        
    #     zx1 = VGG_Model.forward(x1)
    #     zx2 = VGG_Model.forward(x2)

    #     zx1 = torch.reshape(zx1, shape=(-1,))
    #     zx2 = torch.reshape(zx2, shape=(-1,))

    #     del x1
    #     del x2

    #     # calculating denominator
    #     denominatorx1 = denominatorx2 = 0
    #     for inner_idx, (y1, y2) in (enumerate(feature_map_dataset)):

    #         if inner_idx == outer_idx:
    #             continue

    #         y_label = class_labels[inner_idx]
    #         if x_label != y_label:

    #             y1 = y1.unsqueeze(0).to(device)
    #             y2 = y2.unsqueeze(0).to(device)
    #             zy1 = VGG_Model(y1)
    #             zy2 = VGG_Model(y2)

    #             zy1 = torch.reshape(zy1, shape=(-1,))
    #             zy2 = torch.reshape(zy2, shape=(-1,))

    #             del y1
    #             del y2

    #             denominatorx1 += torch.exp(torch.dot(zx1, zy1))
    #             denominatorx1 += torch.exp(torch.dot(zx1, zy2))
    #             denominatorx2 += torch.exp(torch.dot(zx2, zy1))
    #             denominatorx2 += torch.exp(torch.dot(zx2, zy2))

    #             del zy1
    #             del zy2
        
    #     # calculating rest of the term
    #     full_termx1 = full_termx2 = 0
    #     for inner_idx, (y1, y2) in (enumerate(feature_map_dataset)):
            
    #         if inner_idx == outer_idx:
    #             continue

    #         y_label = class_labels[inner_idx]
    #         if x_label == y_label:

    #             y1 = y1.unsqueeze(0).to(device)
    #             y2 = y2.unsqueeze(0).to(device)
    #             zy1 = VGG_Model(y1)
    #             zy2 = VGG_Model(y2)

    #             zy1 = torch.reshape(zy1, shape=(-1,))
    #             zy2 = torch.reshape(zy2, shape=(-1,))

    #             del y1
    #             del y2

    #             full_termx1 += torch.log(torch.exp(torch.dot(zx1, zy1)) / denominatorx1)
    #             full_termx1 += torch.log(torch.exp(torch.dot(zx1, zy1)) / denominatorx1)
    #             full_termx2 += torch.log(torch.exp(torch.dot(zx1, zy1)) / denominatorx2)
    #             full_termx2 += torch.log(torch.exp(torch.dot(zx1, zy1)) / denominatorx2)

    #             del zy1
    #             del zy2
        
    #     del denominatorx1
    #     del denominatorx2
        
    #     encoder_epoch_loss += (full_termx1 + full_termx2)
        
    #     del full_termx1
    #     del full_termx2
    
    # del zx1
    # del zx2

    # print(f'Training loss at epoch {epoch} is {encoder_epoch_loss}')
    
    # VGG_Model.zero_grad()
    # encoder_epoch_loss.backward()
    # encoder_optimizer.step()
    # print()

print()
print('Script executed.')