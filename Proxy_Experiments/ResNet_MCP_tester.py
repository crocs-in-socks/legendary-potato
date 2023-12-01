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
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

batch_size = 5
device = 'cuda:1'

pretrained_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov22/'

clean_data = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
clean_labels = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))
clean_masks = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*biascorr_brain_mask.nii.gz*'))

anomalous_testset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TestSet_5_11_23/*.npz'))
anomalous_testset_jsons = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TestSet_5_11_23/*.json'))

print(f'Anomalous Testset size: {len(anomalous_testset_paths)}')
print(f'Clean Testset size: {len(clean_data)}')

composed_transform = transforms.Compose([
        ToTensor3D(labeled=True)
    ])

anomalous_testset = ImageLoader3D(paths=anomalous_testset_paths, gt_paths=None, json_paths=anomalous_testset_jsons, image_size=128, type_of_imgs='numpy', transform=composed_transform)

clean_testset = ImageLoader3D(paths=clean_data, gt_paths=clean_labels, type_of_imgs='nifty', transform=composed_transform)

testset = ConcatDataset([anomalous_testset, clean_testset])
# testset = anomalous_testset

ResNet_encoder = ResNet3D_Encoder(image_channels=1).to(device)
classification_head = Classifier(input_channels=32768, output_channels=4).to(device)

def plot_confusion_matrix(epoch):
    ResNet_encoder.load_state_dict(torch.load(f'{pretrained_model_path}Multiclasspretrained_ResNet_Encoder_22_11_2023_state_dict{epoch}.pth'))
    classification_head.load_state_dict(torch.load(f'{pretrained_model_path}Multiclasspretrained_ResNet_ClassificationHead_22_11_2023_state_dict{epoch}.pth'))

    ResNet_testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    ResNet_encoder.eval()
    classification_head.eval()

    predictions = []
    ground_truths = []

    for data in tqdm(ResNet_testloader):
        image = data['input'].to(device)
        gt = data['gt'].to(device)
        mixed = image
        
        oneHot_labels = data['lesion_labels'].float().to(device)

        out_dict = ResNet_encoder.forward(mixed)
        z = out_dict['out4']
        z = torch.reshape(z, shape=(z.shape[0], -1))
        y = classification_head.forward(z)

        predictions.append(y)
        ground_truths.append(oneHot_labels)
    
        del image
        del gt
        del oneHot_labels
        del z
        del y

    predictions = torch.cat([*predictions]).detach().cpu()
    ground_truths = torch.cat([*ground_truths]).detach().cpu()

    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0

    confusion_matrix = multilabel_confusion_matrix(ground_truths, predictions)

    plot_labels = ['Healthy', 'Small lesions', 'Med lesions', 'Large lesions']

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for i in range(4):
        sns.heatmap(
            confusion_matrix[i],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'],
            ax=axes[i],
        )
        axes[i].set_title(plot_labels[i])

    plt.tight_layout()
    plt.savefig(f'./temporary/ep{epoch}')
    plt.close()

for epoch in range(5, 101, 5):
    print('Epoch#', epoch)
    plot_confusion_matrix(epoch)

print()
print('Script executed.')