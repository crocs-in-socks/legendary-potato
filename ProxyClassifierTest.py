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

testset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TestSet_5_11_23/*.npz'))
print(f'Testset size: {len(testset_paths)}')

composed_transform = transforms.Compose([
        ToTensor3D(True, True)
    ])

testset = ImageLoader3D(paths=testset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform)

ResNet_Model = ResNet3D(image_channels=1).to(device)
classification_head = Classifier(input_channels=32768, output_channels=2).to(device)

ResNet_Model.load_state_dict(torch.load(f'{model_path}ResNetClassifier_06_11_2023_ResNet_state_dict21.pth'))
classification_head.load_state_dict(torch.load(f'{model_path}ResNetClassifier_06_11_2023_ClassifierHead_state_dict21.pth'))

ResNet_testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

criterion = nn.BCELoss().to(device)

test_loss = 0
test_accuracy = 0
test_recall = 0
test_precision = 0

tp = 0
tn = 0
fp = 0
fn = 0

print()
print('Testing ResNet Model.')

ResNet_Model.eval()
classification_head.eval()

for data in tqdm(ResNet_testloader):
    with torch.no_grad():   
        # Anomalous data
        image = data['input'].to(device)
        labels = (torch.tensor([[1, 0]] * batch_size)).float().to(device)

        z_image = ResNet_Model.forward(image)
        z_image = torch.reshape(z_image, shape=(z_image.shape[0], -1))
        y_image = classification_head.forward(z_image)

        loss = criterion(y_image, labels)
        test_loss += loss.item()
        test_accuracy += determine_class_accuracy(y_image, labels).cpu()

        temp_recall, temp_precision, temp_tp, temp_tn, temp_fp, temp_fn = determine_accuracy_metrics(y_image, labels)
        test_recall += temp_recall
        test_precision += temp_precision
        tp += temp_tp
        tn += temp_tn
        fp += temp_fp
        fn += temp_fn

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
        test_loss += loss.item()
        test_accuracy += determine_class_accuracy(y_clean, labels).cpu()

        temp_recall, temp_precision, temp_tp, temp_tn, temp_fp, temp_fn = determine_accuracy_metrics(y_clean, labels)
        test_recall += temp_recall
        test_precision += temp_precision
        tp += temp_tp
        tn += temp_tn
        fp += temp_fp
        fn += temp_fn

        del clean
        del z_clean
        del y_clean
        del loss

test_loss = test_loss / (len(ResNet_testloader)*2)
test_accuracy = test_accuracy / (len(ResNet_testloader)*2)
test_recall = test_recall / (len(ResNet_testloader)*2)
test_precision = test_precision / (len(ResNet_testloader)*2)

print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
print(f'Test recall: {test_recall}')
print(f'Test precision: {test_precision}')
print(f'True positives: {tp}')
print(f'True negatives: {tn}')
print(f'False positives: {fp}')
print(f'False negatives: {fn}')