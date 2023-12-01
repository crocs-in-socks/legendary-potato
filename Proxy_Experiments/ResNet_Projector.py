import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from ModelArchitecture.Encoders import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange

batch_size = 4
number_of_epochs = 20
device = 'cuda:1'
model_type = 'ResNetClassifier'
date = '08_11_2023'

model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov22/'

trainset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TrainSet_5_11_23/*.npz'))
validationset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/ValidSet_5_11_23/*.npz'))
testset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TestSet_5_11_23/*.npz'))
print(f'Trainset size: {len(trainset_paths)}')
print(f'Validationset size: {len(validationset_paths)}')
print(f'Testset size: {len(testset_paths)}')

clean_data = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
clean_labels = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))
clean_masks = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*biascorr_brain_mask.nii.gz*'))

composed_transform = transforms.Compose([
        ToTensor3D(True, clean=True)
    ])

trainset = ImageLoader3D(paths=trainset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True)
validationset = ImageLoader3D(paths=validationset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True)
testset = ImageLoader3D(paths=testset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True)

clean_set = ImageLoader3D(paths=clean_data, gt_paths=clean_labels, type_of_imgs='nifty', transform=composed_transform, clean=True)

allset = ConcatDataset([trainset, validationset, testset, clean_set])

ResNet_encoder= ResNet3D_Encoder(image_channels=1).to(device)
# projection_head = nn.Conv3d(960, 1, kernel_size=1).to(device)
# projection_head = nn.Conv3d(512, 1, kernel_size=1).to(device)

ResNet_allloader = DataLoader(allset, batch_size=batch_size, shuffle=True, num_workers=0)

ResNet_encoder.load_state_dict(torch.load(f'{model_path}Multiclasspretrained_ResNet_Encoder_22_11_2023_state_dict75.pth'))
# projection_head.load_state_dict(torch.load(f'{model_path}Contrastive_ProjectionHead_StackedOnlyAnomalous_wNIMH_09_11_2023_ProjectorHead_state_dict101.pth'))

for i in tqdm(range(len(allset))):

    torch.cuda.empty_cache()
    sample_dict = allset[i]
    sample_input = sample_dict['input'].to(device).unsqueeze(0)
    sample_clean = sample_dict['clean'].to(device).unsqueeze(0)
    sample_label = (sample_dict['gt'])[1].to(device).unsqueeze(0).unsqueeze(0)

    sample_mixed = torch.cat([sample_input, sample_clean])

    # z_mixed = ResNet_Model(sample_mixed)
    # feature_diff = z_mixed[0] - z_mixed[1]
    # feature_diff  = feature_diff.unsqueeze(0)
    # feature_diff = projection_head(feature_diff)
    # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128))
    # input_diff_with_label = sample_input * sample_label
    # input_diff_without_label = sample_input - sample_clean

    out_dict = ResNet_encoder(sample_mixed)
    layer1 = out_dict['out1']
    layer2 = out_dict['out2']
    layer3 = out_dict['out3']
    z_mixed = out_dict['out4']
    
    layer1 = F.interpolate(layer1, size=(32, 32, 32), mode='trilinear')
    layer2 = F.interpolate(layer2, size=(32, 32, 32), mode='trilinear')
    layer3 = F.interpolate(layer3, size=(32, 32, 32), mode='trilinear')
    layer4 = F.interpolate(z_mixed, size=(32, 32, 32), mode='trilinear')
    stacked_layers = torch.cat([layer1, layer2, layer3, layer4], dim=1)

    # layer1_feature_diff = layer1[0] - layer1[1]
    # layer2_feature_diff = layer2[0] - layer2[1]
    # layer3_feature_diff = layer3[0] - layer3[1]
    # layer4_feature_diff = layer4[0] - layer4[1]

    # layer1_feature_diff = layer1_feature_diff.unsqueeze(0)
    # layer2_feature_diff = layer2_feature_diff.unsqueeze(0)
    # layer3_feature_diff = layer3_feature_diff.unsqueeze(0)
    # layer4_feature_diff = layer4_feature_diff.unsqueeze(0)

    # layer1_feature_diff = torch.cat([
    #     layer1_feature_diff,
    #     torch.zeros(1, 960-64, 32, 32, 32).to(device)
    #     ],
    #     dim=1
    # )
    # layer1_feature_diff = projection_head.forward(layer1_feature_diff)

    # layer2_feature_diff = torch.cat([
    #     torch.zeros(1, 64, 32, 32, 32).to(device),
    #     layer2_feature_diff,
    #     torch.zeros(1, 960-192, 32, 32, 32).to(device)
    #     ],
    #     dim=1
    # )
    # layer2_feature_diff = projection_head.forward(layer2_feature_diff)

    # layer3_feature_diff = torch.cat([
    #     torch.zeros(1, 192, 32, 32, 32).to(device),
    #     layer3_feature_diff,
    #     torch.zeros(1, 960-448, 32, 32, 32).to(device)
    #     ],
    #     dim=1
    # )
    # layer3_feature_diff = projection_head.forward(layer3_feature_diff)

    # layer4_feature_diff = torch.cat([
    #     torch.zeros(1, 960-512, 32, 32, 32).to(device),
    #     layer4_feature_diff
    #     ],
    #     dim=1
    # )
    # layer4_feature_diff = projection_head.forward(layer4_feature_diff)

    # layer1_feature_diff = F.interpolate(layer1_feature_diff, size=(128, 128, 128), mode='trilinear')
    # layer2_feature_diff = F.interpolate(layer2_feature_diff, size=(128, 128, 128), mode='trilinear')
    # layer3_feature_diff = F.interpolate(layer3_feature_diff, size=(128, 128, 128), mode='trilinear')
    # layer4_feature_diff = F.interpolate(layer4_feature_diff, size=(128, 128, 128), mode='trilinear')

    feature_diff = stacked_layers[0]
    # feature_diff  = feature_diff.unsqueeze(0)

    conv_output = torch.mean(feature_diff, dim=0, keepdim=True)
    # conv_output = projection_head.forward(feature_diff)
    conv_output = conv_output.unsqueeze(0)
    conv_output = F.interpolate(conv_output, size=(128, 128, 128), mode='trilinear')

    del layer1
    del layer2
    del layer3
    del layer4
    del stacked_layers
    del out_dict
    del z_mixed

    # conv_output = torch.exp(conv_output)
    # print()
    # print(torch.min(conv_output))
    # conv_output -= torch.min(conv_output)
    # print(torch.max(conv_output))
    # print()
    # conv_output /= torch.max(conv_output)

    # conv_output = F.normalize(conv_output)
    input_diff = sample_mixed * sample_label
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(conv_output[0, 0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    plt.title(f'projection_output anomalous sample #{i+1}')
    plt.subplot(1, 3, 2)
    plt.imshow(input_diff[0, 0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    plt.title(f'input_diff anomalous sample #{i+1}')
    plt.subplot(1, 3, 3)
    plt.imshow(sample_input[0, 0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    plt.title(f'Sample input #{i+1}')

    # fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    # Plot the first row of subplots
    # axes[0, 0].imshow(conv_output[0, 0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    # axes[0, 0].set_title(f'conv_output anomalous sample #{i+1}')
    # axes[0, 0].axis('off')

    # axes[0, 1].imshow(conv_output[1, 0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    # axes[0, 1].set_title(f'conv_output clean sample #{i+1}')
    # axes[0, 1].axis('off')

    # axes[0, 1].imshow(input_diff[0, 0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    # axes[0, 1].set_title(f'input_diff anomalous sample #{i+1}')
    # axes[0, 1].axis('off')

    # axes[0, 2].imshow(input_diff[1, 0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    # axes[0, 2].set_title(f'input_diff clean sample #{i+1}')
    # axes[0, 2].axis('off')

    # Plot the second row of subplots
    # axes[1, 0].imshow(input_diff[0, :, :, 64].detach().cpu(), cmap='gray')
    # axes[1, 0].set_title(f'input_diff sample #{i+1}')
    # axes[1, 0].axis('off')

    # axes[0, 2].imshow(sample_input[0, 0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    # axes[0, 2].set_title(f'Sample input #{i+1}')
    # axes[0, 2].axis('off')

    # axes[1, 0].imshow(sample_clean[0, 0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    # axes[1, 0].set_title(f'Sample clean #{i+1}')
    # axes[1, 0].axis('off')

    # Remove the last subplot (if not needed)
    # fig.delaxes(axes[0, 3])
    # fig.delaxes(axes[1, 0])
    # fig.delaxes(axes[1, 1])
    # fig.delaxes(axes[1, 2])
    # fig.delaxes(axes[1, 3])

    # Adjust subplot spacing and save the figure
    # plt.tight_layout()
    plt.savefig(f'./temporary/sample#{i+1}')
    plt.close()

    # plt.figure(figsize=(20, 20))
    # plt.subplot(1, 4, 1)
    # plt.imshow(conv_output[0, 0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    # plt.colorbar()
    # plt.title(f'conv_output sample #{i+1}')
    
    # plt.subplot(1, 4, 2)
    # plt.imshow(input_diff[0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    # plt.colorbar()
    # plt.title(f'input_diff sample #{i+1}')
    # plt.subplot(1, 4, 3)
    # plt.imshow(sample_input[0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    # plt.colorbar()
    # plt.title(f'Sample input #{i+1}')
    # plt.subplot(1, 4, 4)
    # plt.imshow(sample_clean[0, :, :, 64].detach().cpu(), cmap='gray', vmin=0, vmax=1)
    # plt.colorbar()
    # plt.title(f'Sample clean #{i+1}')
    # plt.savefig(f'./temp/sample#{i+1}')
    # plt.close()