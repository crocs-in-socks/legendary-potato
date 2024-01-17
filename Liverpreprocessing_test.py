import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import *

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

c = Constants(
    batch_size = 1,
    patience = 5,
    num_workers = 4,
    num_epochs = 100,
    date = '09_01_2024',
    to_save_folder = 'Jan09',
    to_load_folder = 'Jan09',
    device = 'cuda:1',
    proxy_type = 'LiTS_Unet_preprocessing_0_>_200_window_init_features_64',
    train_task = 'segmentation',
    encoder_load_path = 'LiTS_Unet_preprocessing_0_>_200_window_init_features_64_segmentation_segmentor_09_01_2024_state_dict_best_score46.pth',
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'lits:window:0_200'
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))

testloader = DataLoader(testset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)

model = UNet(out_channels=2, init_features=64).to(c.device)
model.load_state_dict(torch.load(c.encoder_load_path))
criterion = DiceLoss().to(c.device)
optimizer = optim.Adam(model.parameters(), lr = 0.001, eps = 0.0001)

test_dice_loss_list = []
test_dice_score_list = []

print()
torch.cuda.empty_cache()

# Test loop
model.eval()

highest_dice_scores = [17, 21, 22]
medium_dice_scores = [2, 10, 12]
lowest_dice_scores = [1, 6, 16, 15, 18]

for idx in tqdm(highest_dice_scores):
    data = testset[idx]

    with torch.no_grad():
        image = data['input'].to(c.device).unsqueeze(0)
        gt = data['gt'].unsqueeze(0)
        prediction = model(image)
        dice = Dice_Score(prediction[:, 1].cpu().numpy(), gt[:, 1].detach().cpu().numpy())
        dice = dice.item()

    image = data['input'].numpy()
    gt = data['gt'].numpy()
    prediction = prediction.detach().cpu().numpy()

    nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
    nifti_gt = nib.Nifti1Image(gt, affine=np.eye(4))
    nifti_prediction = nib.Nifti1Image(prediction, affine=np.eye(4))

    nib.save(nifti_image, f'./highest_dice_scores/image_{idx}_dice_{dice}.nii.gz')
    nib.save(nifti_gt, f'./highest_dice_scores/gt_{idx}_dice_{dice}.nii.gz')
    nib.save(nifti_prediction, f'./highest_dice_scores/prediction_{idx}_dice_{dice}.nii.gz')

for idx in tqdm(medium_dice_scores):
    data = testset[idx]

    with torch.no_grad():
        image = data['input'].to(c.device).unsqueeze(0)
        gt = data['gt'].unsqueeze(0)
        prediction = model(image)
        dice = Dice_Score(prediction[:, 1].cpu().numpy(), gt[:, 1].detach().cpu().numpy())
        dice = dice.item()

    image = data['input'].numpy()
    gt = data['gt'].numpy()
    prediction = prediction.detach().cpu().numpy()

    nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
    nifti_gt = nib.Nifti1Image(gt, affine=np.eye(4))
    nifti_prediction = nib.Nifti1Image(prediction, affine=np.eye(4))

    nib.save(nifti_image, f'./medium_dice_scores/image_{idx}_dice_{dice}.nii.gz')
    nib.save(nifti_gt, f'./medium_dice_scores/gt_{idx}_dice_{dice}.nii.gz')
    nib.save(nifti_prediction, f'./medium_dice_scores/prediction_{idx}_dice_{dice}.nii.gz')

for idx in tqdm(lowest_dice_scores):
    data = testset[idx]

    with torch.no_grad():
        image = data['input'].to(c.device).unsqueeze(0)
        gt = data['gt'].unsqueeze(0)
        prediction = model(image)
        dice = Dice_Score(prediction[:, 1].cpu().numpy(), gt[:, 1].detach().cpu().numpy())
        dice = dice.item()

    image = data['input'].numpy()
    gt = data['gt'].numpy()
    prediction = prediction.detach().cpu().numpy()

    nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
    nifti_gt = nib.Nifti1Image(gt, affine=np.eye(4))
    nifti_prediction = nib.Nifti1Image(prediction, affine=np.eye(4))

    nib.save(nifti_image, f'./lowest_dice_scores/image_{idx}_dice_{dice}.nii.gz')
    nib.save(nifti_gt, f'./lowest_dice_scores/gt_{idx}_dice_{dice}.nii.gz')
    nib.save(nifti_prediction, f'./lowest_dice_scores/prediction_{idx}_dice_{dice}.nii.gz')

# for idx, data in enumerate(tqdm(testloader), 0):
#     with torch.no_grad():
#         image = data['input'].to(c.device)
#         gt = data['gt'].to(c.device)
#         print(torch.unique(gt[:, 1]))

#         dice = Dice_Score(segmentation[:, 1].cpu().numpy(), gt[:, 1].detach().cpu().numpy())
#         dice = dice.item()
#         print(dice)

        # if lowest_dice_score is None:
        #     lowest_dice_score = dice
        # if highest_dice_score is None:
        #     highest_dice_score = dice
        
        # if dice > highest_dice_score:
        #     highest_dice_score = dice
        #     highest_dice_score_image = image.detach().cpu()
        #     highest_dice_score_gt = gt.detach().cpu()

        # if dice < lowest_dice_score:
        #     lowest_dice_score = dice
        #     lowest_dice_score_image = image.detach().cpu()
        #     lowest_dice_score_gt = gt.detach().cpu()

# print(f'Highest dice score: {highest_dice_score}')
# print(f'Lowest dice score: {lowest_dice_score}')
# print(type(highest_dice_score_image))

print()
print('Script executed.')