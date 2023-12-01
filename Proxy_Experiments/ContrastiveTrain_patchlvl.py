import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ModelArchitecture.Encoders import VGG3D, ResNet3D, Classifier
from ImageLoader.ContrastiveLoader3D import ContrastivePatchLoader3D

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

from tqdm import tqdm

VGG_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/contrastive_models/'
ResNet_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/contrastive_models/'
patches_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/patches/'

batch_size = 1
number_of_epochs = 100
number_of_train_samples = 67
number_of_validation_samples = 9
number_of_features = 0
patch_size = 32
device = 'cuda:1'

# model_type = f'VGG3D_patchlvl_{patch_size}'
# classifier_type = f'classifier_patchlvl_{patch_size}'
encoder_type = f'ResNet3D_encoder_patchlvl_{patch_size}'
classifier_type = f'ResNet3D_classifier_patchlvl_{patch_size}'

train_sample_paths = []
validation_sample_paths = []

train_patch_labels = []
validation_patch_labels = []

oneHot_train_patch_labels = []
oneHot_validation_patch_labels = []

print('Generating labels for trainset.')
for sample_idx in tqdm(range(number_of_train_samples)):
    patch_dict = np.load(f'{patches_path}/patch_size{patch_size}_train_patch_and_label_idx{sample_idx}.npy', allow_pickle=True).item()

    train_sample_paths.append(f'{patches_path}/patch_size{patch_size}_train_patch_and_label_idx{sample_idx}.npy')

    patches = patch_dict['patches']
    label_patches = patch_dict['labels']

    number_of_features = patches.shape[2]

    temp_labels = []
    oneHot_temp_labels = []

    number_of_patches = label_patches.shape[1]
    for patch_idx in range(number_of_patches):
        unique_values = torch.unique(label_patches[0, patch_idx, 0])
        
        if unique_values.shape[0] == 2 or unique_values == 0:
            temp_labels.append(1)
            oneHot_temp_labels.append([1, 0])
        else:
            temp_labels.append(0)
            oneHot_temp_labels.append([0, 1])
    train_patch_labels.append(temp_labels)
    oneHot_train_patch_labels.append(oneHot_temp_labels)
print()

print('Generating labels for validation set.')
for sample_idx in tqdm(range(number_of_validation_samples)):
    patch_dict = np.load(f'{patches_path}/patch_size{patch_size}_validation_patch_and_label_idx{sample_idx}.npy', allow_pickle=True).item()

    validation_sample_paths.append(f'{patches_path}/patch_size{patch_size}_validation_patch_and_label_idx{sample_idx}.npy')

    patches = patch_dict['patches']
    label_patches = patch_dict['labels']

    temp_labels = []
    oneHot_temp_labels = []

    number_of_patches = label_patches.shape[1]
    for patch_idx in range(number_of_patches):
        unique_values = torch.unique(label_patches[0, patch_idx, 0])
        
        if unique_values.shape[0] == 2 or unique_values == 0:
            temp_labels.append(1)
            oneHot_temp_labels.append([1, 0])
        else:
            temp_labels.append(0)
            oneHot_temp_labels.append([0, 1])
    validation_patch_labels.append(temp_labels)
    oneHot_validation_patch_labels.append(oneHot_temp_labels)
print()

Contrastive_Transforms = transforms.Compose([
    RandomFlip3D(axis=0, p=0.5),
    RandomFlip3D(axis=1, p=0.5),
    RandomFlip3D(axis=2, p=0.5),
    RandomGaussianBlur3D(p=0.5),
    RandomIntensityChanges3D(p=0.5)
])

train_patch_dataset = ContrastivePatchLoader3D(train_sample_paths, train_patch_labels, oneHot_train_patch_labels, device=device, transform=None)
validation_patch_dataset = ContrastivePatchLoader3D(validation_sample_paths, validation_patch_labels, oneHot_validation_patch_labels, device=device, transform=None)

# VGG_Model = VGG3D(input_channels=number_of_features, output_classes=2).to(device)
# classification_head = Classifier(input_channels=16384, output_channels=2).to(device)
ResNet_Model = ResNet3D(image_channels=number_of_features).to(device)
classification_head = Classifier(input_channels=2048, output_channels=2).to(device)

# VGG_trainloader = DataLoader(train_patch_dataset, batch_size=4, shuffle=True, num_workers=0)
# VGG_validationloader = DataLoader(validation_patch_dataset, batch_size=3, shuffle=True, num_workers=0)
ResNet_trainloader = DataLoader(train_patch_dataset, batch_size=1, shuffle=True, num_workers=0)
ResNet_validationloader = DataLoader(validation_patch_dataset, batch_size=1, shuffle=True, num_workers=0)

# encoder_optimizer = optim.Adam(VGG_Model.parameters(), lr = 0.0001, eps = 0.0001)
encoder_optimizer = optim.Adam(ResNet_Model.parameters(), lr = 0.0001, eps = 0.0001)
classifier_optimizer = optim.Adam(classification_head.parameters(), lr = 0.0001, eps = 0.0001)

encoder_criterion = SupervisedContrastiveLoss().to(device)
classifier_criterion = nn.BCELoss().to(device)

encoder_train_losses = []
classifier_train_losses = []

classifier_train_accuracies = []
classifier_validation_accuracies = []

encoder_validation_losses = []
classifier_validation_losses = []

# early_stop_counter = 15
# best_validation_loss = np.inf

print()
print('Training ResNet Model.')

for epoch in range(1, number_of_epochs+1):
    
    # if early_stop_counter == 0:
    #     print(f'Early stopped at epoch #{epoch}')
    #     break

    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    # VGG_Model.train()
    ResNet_Model.train()
    classification_head.train()

    skipped_batches = 0

    epoch_encoder_train_loss = 0
    epoch_classifier_train_loss = 0
    classifier_train_accuracy = 0

    # for (X, X_labels, oneHot_X_labels) in tqdm(VGG_trainloader):
    for (X, X_labels, oneHot_X_labels) in tqdm(ResNet_trainloader):

        X = X.to(device)
        X_labels = X_labels.to(device)
        oneHot_X_labels = oneHot_X_labels.to(device)

        if torch.unique(X_labels).shape[0] == 1:
            skipped_batches += 1
            continue

        X = torch.reshape(X, shape=(X.shape[0]*X.shape[1], number_of_features, patch_size, patch_size, patch_size))
        X_labels = torch.reshape(X_labels, shape=(-1,))
        oneHot_X_labels = torch.reshape(oneHot_X_labels, shape=(oneHot_X_labels.shape[0]*oneHot_X_labels.shape[1], 2))

        # Z = VGG_Model(X)
        Z = ResNet_Model(X)
        Z = torch.reshape(Z, shape=(Z.shape[0], -1))
        Y = classification_head(Z.detach())

        encoder_loss = encoder_criterion(Z, X_labels)
        epoch_encoder_train_loss += encoder_loss.item()
        # VGG_Model.zero_grad()
        ResNet_Model.zero_grad()
        encoder_loss.backward()
        encoder_optimizer.step()

        classifier_loss = classifier_criterion(Y, oneHot_X_labels)
        epoch_classifier_train_loss += classifier_loss.item()
        classification_head.zero_grad()
        classifier_loss.backward()
        classifier_optimizer.step()

        classifier_train_accuracy += determine_class_accuracy(Y, oneHot_X_labels).cpu()

        del X
        del X_labels
        del oneHot_X_labels
        del Y
        del Z

    # encoder_train_losses.append(epoch_encoder_train_loss / (len(VGG_trainloader) - skipped_batches))
    # classifier_train_losses.append(epoch_classifier_train_loss / (len(VGG_trainloader) - skipped_batches))
    # classifier_train_accuracies.append(classifier_train_accuracy / (len(VGG_trainloader) - skipped_batches))
    encoder_train_losses.append(epoch_encoder_train_loss / (len(ResNet_trainloader) - skipped_batches))
    classifier_train_losses.append(epoch_classifier_train_loss / (len(ResNet_trainloader) - skipped_batches))
    classifier_train_accuracies.append(classifier_train_accuracy / (len(ResNet_trainloader) - skipped_batches))
    print(f'Encoder Training Loss at epoch {epoch} is : {encoder_train_losses[-1]}')
    print(f'Classifier Training Loss at epoch {epoch} is : {classifier_train_losses[-1]}')
    print(f'Classifier Training Accuracy at epoch {epoch} is : {classifier_train_accuracies[-1]}')
    print(f'Skipped batches: {skipped_batches}')

    torch.cuda.empty_cache()

    # Validation loop
    # VGG_Model.train()
    ResNet_Model.eval()
    classification_head.eval()

    skipped_batches = 0

    epoch_encoder_validation_loss = 0
    epoch_classifier_validation_loss = 0
    classifier_validation_accuracy = 0

    # for (X, X_labels, oneHot_X_labels) in tqdm(VGG_validationloader):
    for (X, X_labels, oneHot_X_labels) in tqdm(ResNet_validationloader):
        with torch.no_grad():
            X = X.to(device)
            X_labels = X_labels.to(device)
            oneHot_X_labels = oneHot_X_labels.to(device)

            if torch.unique(X_labels).shape[0] == 1:
                skipped_batches += 1
                continue

            X = torch.reshape(X, shape=(X.shape[0]*X.shape[1], number_of_features, patch_size, patch_size, patch_size))
            X_labels = torch.reshape(X_labels, shape=(-1,))
            oneHot_X_labels = torch.reshape(oneHot_X_labels, shape=(oneHot_X_labels.shape[0]*oneHot_X_labels.shape[1], 2))

            # Z = VGG_Model(X)
            Z = ResNet_Model(X)
            Z = torch.reshape(Z, shape=(Z.shape[0], -1))
            Y = classification_head(Z.detach())

            encoder_loss = encoder_criterion(Z, X_labels)
            epoch_encoder_validation_loss += encoder_loss.item()

            classifier_loss = classifier_criterion(Y, oneHot_X_labels)
            epoch_classifier_validation_loss += classifier_loss.item()

            classifier_validation_accuracy += determine_class_accuracy(Y, X_labels).cpu()

            del X
            del X_labels
            del oneHot_X_labels
            del Y
            del Z
        
    # encoder_validation_losses.append(epoch_encoder_validation_loss / (len(VGG_validationloader) - skipped_batches))
    # classifier_validation_losses.append(epoch_classifier_validation_loss / (len(VGG_validationloader) - skipped_batches))
    # classifier_validation_accuracies.append(classifier_validation_accuracy / (len(VGG_validationloader) - skipped_batches))
    encoder_validation_losses.append(epoch_encoder_validation_loss / (len(ResNet_validationloader) - skipped_batches))
    classifier_validation_losses.append(epoch_classifier_validation_loss / (len(ResNet_validationloader) - skipped_batches))
    classifier_validation_accuracies.append(classifier_validation_accuracy / (len(ResNet_validationloader) - skipped_batches))
    print(f'Encoder Validation Loss at epoch {epoch} is : Total {encoder_validation_losses[-1]}')
    print(f'Classifier Validation Loss at epoch {epoch} is : Total {classifier_validation_losses[-1]}')
    print(f'Classifier Validation Accuracy at epoch {epoch} is : Total {classifier_validation_accuracies[-1]}')
    print(f'Skipped batches: {skipped_batches}')

    np.save(f'./results/{encoder_type}_encoder_results.npy', [encoder_train_losses, encoder_validation_losses])

    np.save(f'./results/{encoder_type}_classifier_results.npy', [classifier_train_losses, classifier_train_accuracies, classifier_validation_losses, classifier_validation_accuracies])

    # if classifier_validation_losses[-1] < best_validation_loss:
    #     best_validation_loss = classifier_validation_losses[-1]
    #     torch.save(ResNet_Model.state_dict(), VGG_model_path+encoder_type+'_state_dict_best_loss'+str(epoch)+'.pth')
    #     torch.save(classification_head.state_dict(), VGG_model_path+classifier_type+'_state_dict_best_loss'+str(epoch)+'.pth')
    #     print('New best validation loss! Model saved at this epoch.')
    #     early_stop_counter = 15
    # else:
    #     early_stop_counter -= 1
    
    if epoch % 5 == 0:
        # torch.save(VGG_Model.state_dict(), VGG_model_path+model_type+'_state_dict'+str(epoch)+'.pth')
        # torch.save(classification_head.state_dict(), VGG_model_path+classifier_type+'_state_dict'+str(epoch)+'.pth')
        torch.save(ResNet_Model.state_dict(), VGG_model_path+encoder_type+'_state_dict'+str(epoch)+'.pth')
        torch.save(classification_head.state_dict(), VGG_model_path+classifier_type+'_state_dict'+str(epoch)+'.pth')

    print()

# torch.save(VGG_Model.state_dict(), VGG_model_path+model_type+'_state_dict'+str(epoch)+'.pth')
# torch.save(classification_head.state_dict(), VGG_model_path+classifier_type+'_state_dict'+str(epoch)+'.pth')
torch.save(ResNet_Model.state_dict(), VGG_model_path+encoder_type+'_state_dict'+str(epoch)+'.pth')
torch.save(classification_head.state_dict(), VGG_model_path+classifier_type+'_state_dict'+str(epoch)+'.pth')

# ResNet_Model.load_state_dict(torch.load(f'{ResNet_model_path}ResNet3D_encoder_patchlvl_32_state_dict95.pth'))
# classification_head.load_state_dict(torch.load(f'{VGG_model_path}ResNet3D_classifier_patchlvl_32_state_dict145.pth'))

# print()
# print('Training only the classification head.')

# for epoch in range(number_of_epochs+1, number_of_epochs+51):
    
#     # if early_stop_counter == 0:
#     #     print(f'Early stopped at epoch #{epoch}')
#     #     break

#     print()
#     print(f'Epoch #{epoch}')
#     torch.cuda.empty_cache()

#     # Train loop
#     # VGG_Model.train()
#     ResNet_Model.eval()
#     classification_head.train()

#     skipped_batches = 0

#     epoch_encoder_train_loss = 0
#     epoch_classifier_train_loss = 0
#     classifier_train_accuracy = 0

#     # for (X, X_labels, oneHot_X_labels) in tqdm(VGG_trainloader):
#     for (X, X_labels, oneHot_X_labels) in tqdm(ResNet_trainloader):

#         X = X.to(device)
#         X_labels = X_labels.to(device)
#         oneHot_X_labels = oneHot_X_labels.to(device)

#         if torch.unique(X_labels).shape[0] == 1:
#             skipped_batches += 1
#             continue

#         X = torch.reshape(X, shape=(X.shape[0]*X.shape[1], number_of_features, patch_size, patch_size, patch_size))
#         X_labels = torch.reshape(X_labels, shape=(-1,))
#         oneHot_X_labels = torch.reshape(oneHot_X_labels, shape=(oneHot_X_labels.shape[0]*oneHot_X_labels.shape[1], 2))

#         # Z = VGG_Model(X)
#         Z = ResNet_Model(X)
#         Z = torch.reshape(Z, shape=(Z.shape[0], -1))
#         Y = classification_head(Z.detach())

#         classifier_loss = classifier_criterion(Y, oneHot_X_labels)
#         epoch_classifier_train_loss += classifier_loss.item()
#         classification_head.zero_grad()
#         classifier_loss.backward()
#         classifier_optimizer.step()


#         del X
#         del X_labels
#         del oneHot_X_labels
#         del Y
#         del Z

#     # encoder_train_losses.append(epoch_encoder_train_loss / (len(VGG_trainloader) - skipped_batches))
#     # classifier_train_losses.append(epoch_classifier_train_loss / (len(VGG_trainloader) - skipped_batches))
#     # classifier_train_accuracies.append(classifier_train_accuracy / (len(VGG_trainloader) - skipped_batches))
#     classifier_train_losses.append(epoch_classifier_train_loss / (len(ResNet_trainloader) - skipped_batches))
#     classifier_train_accuracies.append(classifier_train_accuracy / (len(ResNet_trainloader) - skipped_batches))
#     # print(f'Encoder Training Loss at epoch {epoch} is : {encoder_train_losses[-1]}')
#     print(f'Classifier Training Loss at epoch {epoch} is : {classifier_train_losses[-1]}')
#     print(f'Classifier Training Accuracy at epoch {epoch} is : {classifier_train_accuracies[-1]}')
#     print(f'Skipped batches: {skipped_batches}')

#     torch.cuda.empty_cache()

#     # Validation loop
#     # VGG_Model.train()
#     ResNet_Model.train()
#     classification_head.train()

#     skipped_batches = 0

#     # epoch_encoder_validation_loss = 0
#     epoch_classifier_validation_loss = 0
#     classifier_validation_accuracy = 0

#     # for (X, X_labels, oneHot_X_labels) in tqdm(VGG_validationloader):
#     for (X, X_labels, oneHot_X_labels) in tqdm(ResNet_validationloader):
#         with torch.no_grad():
#             X = X.to(device)
#             X_labels = X_labels.to(device)
#             oneHot_X_labels = oneHot_X_labels.to(device)

#             if torch.unique(X_labels).shape[0] == 1:
#                 skipped_batches += 1
#                 continue

#             X = torch.reshape(X, shape=(X.shape[0]*X.shape[1], number_of_features, patch_size, patch_size, patch_size))
#             X_labels = torch.reshape(X_labels, shape=(-1,))
#             oneHot_X_labels = torch.reshape(oneHot_X_labels, shape=(oneHot_X_labels.shape[0]*oneHot_X_labels.shape[1], 2))

#             # Z = VGG_Model(X)
#             Z = ResNet_Model(X)
#             Z = torch.reshape(Z, shape=(Z.shape[0], -1))
#             Y = classification_head(Z.detach())

#             classifier_loss = classifier_criterion(Y, oneHot_X_labels)
#             epoch_classifier_validation_loss += classifier_loss.item()

#             classifier_validation_accuracy += determine_class_accuracy(Y, X_labels).cpu()

#             del X
#             del X_labels
#             del oneHot_X_labels
#             del Y
#             del Z
        
#     # encoder_validation_losses.append(epoch_encoder_validation_loss / (len(VGG_validationloader) - skipped_batches))
#     # classifier_validation_losses.append(epoch_classifier_validation_loss / (len(VGG_validationloader) - skipped_batches))
#     # classifier_validation_accuracies.append(classifier_validation_accuracy / (len(VGG_validationloader) - skipped_batches))
#     # encoder_validation_losses.append(epoch_encoder_validation_loss / (len(ResNet_validationloader) - skipped_batches))
#     classifier_validation_losses.append(epoch_classifier_validation_loss / (len(ResNet_validationloader) - skipped_batches))
#     classifier_validation_accuracies.append(classifier_validation_accuracy / (len(ResNet_validationloader) - skipped_batches))
#     print(f'Classifier Validation Loss at epoch {epoch} is : Total {classifier_validation_losses[-1]}')
#     print(f'Classifier Validation Accuracy at epoch {epoch} is : Total {classifier_validation_accuracies[-1]}')
#     print(f'Skipped batches: {skipped_batches}')

#     np.save(f'./results/{encoder_type}_classifier_results.npy', [classifier_train_losses, classifier_train_accuracies, classifier_validation_losses, classifier_validation_accuracies])

#     if epoch % 5 == 0:
#         # torch.save(classification_head.state_dict(), VGG_model_path+classifier_type+'_state_dict'+str(epoch)+'.pth')
#         torch.save(classification_head.state_dict(), VGG_model_path+classifier_type+'_state_dict'+str(epoch)+'.pth')

#     print()

# torch.save(classification_head.state_dict(), VGG_model_path+classifier_type+'_state_dict'+str(epoch)+'.pth')

print()
print('Script executed.')