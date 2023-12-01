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
import numpy as np
from tqdm import tqdm

batch_size = 5
number_of_epochs = 100
device = 'cuda:1'
encoder_type = 'BCEpretrained'
classifier_type = 'BCEpretrained_ResNet'
date = '20_11_2023'

pretrained_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov21/'

clean_data = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
clean_labels = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))
clean_masks = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*biascorr_brain_mask.nii.gz*'))

anomalous_testset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TestSet_5_11_23/*.npz'))

print(f'Anomalous Testset size: {len(anomalous_testset_paths)}')
print(f'Clean Testset size: {len(clean_data)}')

composed_transform = transforms.Compose([
        ToTensor3D(labeled=True)
    ])

anomalous_testset = ImageLoader3D(paths=anomalous_testset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)

clean_testset = ImageLoader3D(paths=clean_data, gt_paths=clean_labels, type_of_imgs='nifty', transform=composed_transform)

testset = ConcatDataset([anomalous_testset, clean_testset])
# testset = anomalous_testset

ResNet_encoder = ResNet3D_Encoder(image_channels=1).to(device)
classification_head = Classifier(input_channels=32768, output_channels=2).to(device)

ResNet_encoder.load_state_dict(torch.load(f'{pretrained_model_path}BCEpretrained_21_11_2023_state_dict50.pth'))
classification_head.load_state_dict(torch.load(f'{pretrained_model_path}BCEpretrained_ResNet_21_11_2023_state_dict50.pth'))

ResNet_testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

criterion = nn.BCELoss().to(device)

ResNet_encoder.eval()
classification_head.eval()
epoch_train_loss = 0
epoch_train_accuracy = 0

test_loss = 0
test_accuracy = 0
test_recall = 0
test_precision = 0

tp = 0
tn = 0
fp = 0
fn = 0

for data in tqdm(ResNet_testloader):
    image = data['input'].to(device)
    gt = data['gt'].to(device)
    # clean = data['clean'].to(device)
    # mixed = torch.cat([image, clean])
    mixed = image
        
    oneHot_labels = []
    current_batch_size = image.shape[0]
        
    for sample_idx in range(current_batch_size):
        if torch.unique(gt[sample_idx, 1]).shape[0] == 2:
            # anomalous
            oneHot_labels.append([1, 0])
        else:
            # normal
            oneHot_labels.append([0, 1])

    # oneHot_labels.extend([[0, 1]] * current_batch_size)
    oneHot_labels = torch.tensor(oneHot_labels).float().to(device)

    out_dict = ResNet_encoder.forward(mixed)
    z = out_dict['out4']
    z = torch.reshape(z, shape=(z.shape[0], -1))
    y = classification_head.forward(z)

    loss = criterion(y, oneHot_labels)
    test_loss += loss.item()
    test_accuracy += determine_class_accuracy(y, oneHot_labels).cpu()

    temp_recall, temp_precision, temp_tp, temp_tn, temp_fp, temp_fn = determine_accuracy_metrics(y, oneHot_labels)
    test_recall += temp_recall
    test_precision += temp_precision
    tp += temp_tp
    tn += temp_tn
    fp += temp_fp
    fn += temp_fn
    
    del image
    del gt
    del oneHot_labels
    del z
    del y
    del loss

test_loss = test_loss / len(ResNet_testloader)
test_accuracy = test_accuracy / len(ResNet_testloader)
test_recall = test_recall / len(ResNet_testloader)
test_precision = test_precision / len(ResNet_testloader)

print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
print(f'Test recall: {test_recall}')
print(f'Test precision: {test_precision}')
print(f'True positives: {tp}')
print(f'True negatives: {tn}')
print(f'False positives: {fp}')
print(f'False negatives: {fn}')

print()
print('Script executed.')