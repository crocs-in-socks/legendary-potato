import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import *

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *
from ModelArchitecture.SlimUNETR.SlimUNETR import *
from ModelArchitecture.SlimUNETR2D.SlimUNETR2D import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize
import matplotlib.pyplot as plt

c = Constants(
    batch_size = 1,
    patience = None,
    num_workers = 4,
    num_epochs = None,
    date = None,
    to_save_folder = 'Feb24',
    to_load_folder = 'Feb24',
    device = 'cuda:1',
    proxy_type = None,
    train_task = None,
    encoder_load_path = 'LiTS_Unet_preprocessing_-100_>_400_window_init_features_64_median_filter:kernel_size_3_zscore_segmentation_segmentor_24_02_2024_state_dict_best_score195.pth',
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = '3dpreprocessed_lits'
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))

trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)
validationloader = DataLoader(validationset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)
testloader = DataLoader(testset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)

model = UNet(out_channels=2, init_features=64).to(c.device)
model.load_state_dict(torch.load(c.encoder_load_path))

### WMH
### DA = 131, FT = 33, SS = 147, S = 135

### BraTS
### DA = 108, FT = 16, SS = 92, S = 163

# model = SlimUNETR(in_channels=1, out_channels=2).to(c.device)
# model.load_state_dict(torch.load('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/segmentation_models/OneDrive_1_1-2-2024/BraTS/Fine_Tuning/slimunetr_focal + dice_state_dict_best_loss16.pth')['model_state_dict'])

### BUSI
### DA = 147, FT = 138, SS = 65, S = 181

### STARE
### DA = 162, FT = , SS = , S =

# model = SlimUNETR2D(in_channels=1, out_channels=2).to(c.device)
# model.load_state_dict(torch.load('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/models_retrained/segmentation_models/OneDrive_1_1-2-2024/STARE/Data_Augmentation/slimunetr_focal + dice_state_dict_best_loss162.pth')['model_state_dict'])

data_type = '3D'

test_dice_loss_list = []
test_dice_score_list = []

print()
torch.cuda.empty_cache()

# Test loop
model.eval()

test_dice_scores = []
unique_gt_values = []

for idx, data in enumerate(tqdm(testloader), 0):
    with torch.no_grad():
        image = data['input'].to(c.device)

        if data_type == '3D':

            gt = data['og_gt']
            og_dims = data['og_dims']
            crop_para = data['crop_para']

            prediction = model(image)
            # prediction, _ = model(image)
            prediction = (prediction > 0.5).float()
            image = data['og_image'][0].cpu().numpy()
            prediction = prediction[:, 1].cpu().numpy()
            gt = gt[:, 0, 1].numpy()

            prediction = skiform.resize(prediction, (1, crop_para[1].item(), crop_para[3].item(), crop_para[5].item()), order=0, preserve_range=True)

            prediction_padded = np.zeros(shape=(1, *og_dims))

            prediction_padded[0, crop_para[0]:crop_para[0] + crop_para[1], crop_para[2]:crop_para[2] + crop_para[3], crop_para[4]:crop_para[4] + crop_para[5]] = prediction

            dice = Dice_Score(prediction_padded, gt)
            # dice = Dice_Score(prediction, gt)
            test_dice_scores.append(dice.item())

            # print(np.unique(prediction))
            unique_gt_values.append(np.unique(gt))

            plt.figure(figsize=(20, 15))
            plt.subplot(1, 3, 1)
            plt.imshow(prediction_padded[0, :, : , gt.shape[-1]//2])
            plt.colorbar()
            plt.title('Prediction')
            plt.subplot(1, 3, 2)
            plt.imshow(gt[0, :, :, gt.shape[-1]//2])
            plt.colorbar()
            plt.title('GT')
            plt.subplot(1, 3, 3)
            plt.imshow(image[0, :, :, gt.shape[-1]//2])
            plt.colorbar()
            plt.title('Image')
            plt.savefig(f'./temporary/{idx}')
            plt.close()
        
        elif data_type == '2D':

            gt = data['og_gt']
            og_dims = data['og_dims']

            prediction = model(image)
            prediction = (prediction > 0.5).float()
            image = image[0].detach().cpu().numpy()
            prediction = prediction[:, 1].cpu().numpy()
            gt = gt[:, 1].numpy()

            # prediction = torchvision.transforms.functional.resize(prediction, size=gt.shape[1:],interpolation=torchvision.transforms.InterpolationMode.NEAREST,antialias=True)
            prediction = skiform.resize(prediction, gt.shape, order=0, preserve_range=True)

            dice = Dice_Score(prediction, gt)
            test_dice_scores.append(dice.item())

            # print(np.unique(prediction))
            unique_gt_values.append(np.unique(gt))
        
            plt.figure(figsize=(20, 15))
            plt.subplot(1, 3, 1)
            plt.imshow(prediction[0])
            plt.colorbar()
            plt.title('Prediction')
            plt.subplot(1, 3, 2)
            plt.imshow(gt[0])
            plt.colorbar()
            plt.title('GT')
            plt.subplot(1, 3, 3)
            plt.imshow(image[0])
            plt.colorbar()
            plt.title('Image')
            plt.savefig(f'./temporary/{idx}')
            plt.close()

print()

for idx, (score, unique_values) in enumerate(zip(test_dice_scores, unique_gt_values)):
    print(f'Sample #{idx+1}:\t{score}\tNo. unique:\t{len(unique_values)}')

avg_dice_score = np.mean(test_dice_scores)
std_dice_score = np.std(test_dice_scores)
print()

print(f'Average dice score: {avg_dice_score}\tStd of dice scores: {std_dice_score}')

print()
print('Script executed.')