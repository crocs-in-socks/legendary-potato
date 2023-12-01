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
number_of_epochs = 50
number_of_test_samples = 20
number_of_features = 0
patch_size = 32
device = 'cuda:1'

# model_type = f'VGG3D_patchlvl_{patch_size}'
# classifier_type = f'classifier_patchlvl_{patch_size}'
encoder_type = f'ResNet3D_encoder_patchlvl_{patch_size}'
classifier_type = f'ResNet3D_classifier_patchlvl_{patch_size}'

test_sample_paths = []
test_patch_labels = []
oneHot_test_patch_labels = []

print('Generating labels for testset.')
for sample_idx in tqdm(range(number_of_test_samples)):
    patch_dict = np.load(f'{patches_path}/patch_size{patch_size}_test_patch_and_label_idx{sample_idx}.npy', allow_pickle=True).item()

    test_sample_paths.append(f'{patches_path}/patch_size{patch_size}_test_patch_and_label_idx{sample_idx}.npy')

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
    test_patch_labels.append(temp_labels)
    oneHot_test_patch_labels.append(oneHot_temp_labels)
print()

test_patch_dataset = ContrastivePatchLoader3D(test_sample_paths, test_patch_labels, oneHot_test_patch_labels, device=device, transform=None)

# VGG_testloader = DataLoader(test_patch_dataset, batch_size=4, shuffle=True, num_workers=0)
ResNet_testloader = DataLoader(test_patch_dataset, batch_size=4, shuffle=True, num_workers=0)

# VGG_Model = VGG3D(input_channels=number_of_features, output_classes=2).to(device)
# classification_head = Classifier(input_channels=16384, output_channels=2).to(device)
ResNet_Model = ResNet3D(image_channels=number_of_features).to(device)
classification_head = Classifier(input_channels=2048, output_channels=2).to(device)

# VGG_Model.load_state_dict(torch.load(f'{VGG_model_path}VGG3D_patchlvl_32_state_dict49.pth'))
ResNet_Model.load_state_dict(torch.load(f'{ResNet_model_path}ResNet3D_encoder_patchlvl_32_state_dict95.pth'))
classification_head.load_state_dict(torch.load(f'{VGG_model_path}ResNet3D_classifier_patchlvl_32_state_dict145.pth'))

classifier_criterion = nn.BCELoss().to(device)

test_loss = 0
test_accuracy = 0
test_recall = 0
test_precision = 0

tp = 0
tn = 0
fp = 0
fn = 0

skipped_batches = 0
number_of_positive = 0
number_of_negative = 0

print()
print('Testing ResNet Model.')

# VGG_Model.eval()
ResNet_Model.eval()
classification_head.eval()

# for (X, X_labels, oneHot_X_labels) in tqdm(VGG_testloader):
for (X, X_labels, oneHot_X_labels) in tqdm(ResNet_testloader):
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

        classifier_loss = classifier_criterion(Y, oneHot_X_labels)
        test_loss += classifier_loss.item()
        test_accuracy += (determine_class_accuracy(Y, X_labels).cpu())

        temp_recall, temp_precision, temp_tp, temp_tn, temp_fp, temp_fn = determine_accuracy_metrics(Y, X_labels)
        test_recall += temp_recall
        test_precision += temp_precision
        tp += temp_tp
        tn += temp_tn
        fp += temp_fp
        fn += temp_fn

        number_of_positive += torch.sum(X_labels.cpu() == 1)
        number_of_negative += torch.sum(X_labels.cpu() == 0)

        del X
        del X_labels
        del oneHot_X_labels
        del Y
        del Z

# test_loss = test_loss / len(VGG_testloader)
# test_accuracy = test_accuracy / len(VGG_testloader)

# test_loss = test_loss / (len(VGG_testloader) - skipped_batches)
# test_accuracy = test_accuracy / (len(VGG_testloader) - skipped_batches)
# test_recall = test_recall / (len(VGG_testloader) - skipped_batches)
# test_precision = test_precision / (len(VGG_testloader) - skipped_batches)

test_loss = test_loss / (len(ResNet_testloader) - skipped_batches)
test_accuracy = test_accuracy / (len(ResNet_testloader) - skipped_batches)
test_recall = test_recall / (len(ResNet_testloader) - skipped_batches)
test_precision = test_precision / (len(ResNet_testloader) - skipped_batches)

print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
print(f'Test recall: {test_recall}')
print(f'Test precision: {test_precision}')
print(f'Skipped batches: {skipped_batches}')
print(f'True positives: {tp}')
print(f'True negatives: {tn}')
print(f'False positives: {fp}')
print(f'False negatives: {fn}')
print(f'Number positive (1): {number_of_positive}')
print(f'Number negative (0): {number_of_negative}')