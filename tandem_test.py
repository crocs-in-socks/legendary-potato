import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import Constants, load_dataset

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

import numpy as np
from tqdm import tqdm

c = Constants(
    batch_size = 1,
    patience = None,
    num_workers = 8,
    num_epochs = None,
    date = None,
    to_save_folder = None,
    to_load_folder = 'Dec13',
    device = 'cuda:1',
    proxy_type = None,
    train_task = None,
    encoder_load_path = None,
    projector_load_path = None,
    classifier_load_path = None,
    proxy_load_path = None,
    dataset = 'wmh'
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
testloader = DataLoader(testset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

proxy_encoder = VGG3D_Encoder(input_channels=1).to(c.device)
proxy_encoder.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/experiments/Dec14/Integrated_Unet_&_VGGproxy_pat5_dice_stepped4_multichannel_projection_1e-4_>_1e-5_lr_seg(decoder_step_per_epoch)_&_proxy(classifier)_simulated_brain_bg_>_real_wmh_ratiod_wrt_wmh_simulated_brain_bg_encoder_14_12_2023_state_dict_best_score73.pth'))

proxy_projector = IntegratedSpatialProjector(num_layers=4, layer_sizes=[64, 128, 256, 512]).to(c.device)
proxy_projector.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/experiments/Dec14/Integrated_Unet_&_VGGproxy_pat5_dice_stepped4_multichannel_projection_1e-4_>_1e-5_lr_seg(decoder_step_per_epoch)_&_proxy(classifier)_simulated_brain_bg_>_real_wmh_ratiod_wrt_wmh_simulated_brain_bg_projector_14_12_2023_state_dict_best_score73.pth'))

segmentation_model = SA_UNet(out_channels=2).to(c.device)
segmentation_model.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/experiments/Dec14/Integrated_Unet_&_VGGproxy_pat5_dice_stepped4_multichannel_projection_1e-4_>_1e-5_lr_seg(decoder_step_per_epoch)_&_proxy(classifier)_simulated_brain_bg_>_real_wmh_ratiod_wrt_wmh_simulated_brain_bg_segmentor_14_12_2023_state_dict_best_score73.pth'))

print()
print('Tandem testing segmentation and proxy models.')

# Validation loop
proxy_encoder.eval()
proxy_projector.eval()
segmentation_model.eval()

test_dice_score = 0

for idx, data in enumerate(tqdm(testloader), 0):
    with torch.no_grad():
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)

        to_projector, _ = proxy_encoder(image)
        projection_maps = proxy_projector(to_projector)
        segmentation = segmentation_model(image, projection_maps)

        dice = Dice_Score(segmentation[:, 1].cpu().numpy(), gt[:, 1].detach().cpu().numpy())
        test_dice_score += dice.item()

        del image
        del gt
        del to_projector
        del _
        del projection_maps
        del segmentation
        del dice

print(f'Test dice score: {test_dice_score / len(testloader)}')

print()
print('Script executed.')