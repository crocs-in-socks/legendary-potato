import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import Constants, load_dataset

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

c = Constants(
    batch_size = 1,
    patience = None,
    num_workers = 16,
    num_epochs = None,
    date = None,
    to_save_folder = None,
    to_load_folder = 'Dec11',
    device = 'cuda:1',
    proxy_type = None,
    train_task = None,
    encoder_load_path = 'Integrated_Unet_&_VGGproxy_tandem_seg_&_proxy(classifier)_simulated_brain_bg_>_realwmh_ratiod_wrt_wmh_simulated_brain_bg_encoder_11_12_2023_state_dict_best_score47.pth',
    projector_load_path = 'Integrated_Unet_&_VGGproxy_tandem_seg_&_proxy(classifier)_simulated_brain_bg_>_realwmh_ratiod_wrt_wmh_simulated_brain_bg_projector_11_12_2023_state_dict_best_score47.pth',
    classifier_load_path = None,
    proxy_load_path = None,
    dataset = 'wmh'
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
testloader = DataLoader(testset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

# encoder = DuckNet(input_channels=1, out_classes=2, starting_filters=17).to(device)
# encoder = ResNet3D_Encoder(image_channels=1).to(device)
encoder = VGG3D_Encoder(input_channels=1).to(c.device)
# encoder = SA_UNet_Encoder(out_channels=2).to(c.device)
encoder.load_state_dict(torch.load(c.encoder_load_path))

# classification_head = Classifier(input_channels=17408, output_channels=5).to(device)
# classification_head = Classifier(input_channels=2176, output_channels=5).to(device)

# projection_head = Projector(num_layers=5, layer_sizes=[17, 34, 68, 136, 272]).to(device)
# projection_head = Projector(num_layers=4, layer_sizes=[64, 128, 256, 512]).to(device)
# projection_head = Projector(num_layers=5, layer_sizes=[32, 64, 128, 256, 512]).to(device)
projection_head = IntegratedChannelProjector(num_layers=4, layer_sizes=[64, 128, 256, 512]).to(c.device)
# projection_head = Projector(num_layers=4, layer_sizes=[64, 128, 256, 512], test=True).to(c.device)
projection_head.load_state_dict(torch.load(c.projector_load_path))

encoder.eval()
projection_head.eval()

for idx, data in enumerate(tqdm(testloader), 0):
    image = data['input'].to(c.device)
    gt = data['gt'].to(c.device)

    to_projector, to_classifier = encoder(image)
    projection_maps = projection_head(to_projector)

    for map_idx, map in enumerate(projection_maps):
        projection_maps[map_idx] = F.interpolate(projection_maps[map_idx], size=(128, 128, 128))

    plt.figure(figsize=(40, 15))
    plt.subplot(1, 6, 1)
    plt.imshow(image[0, 0, :, : , 64].detach().cpu())
    plt.colorbar()
    plt.title('Image')
    plt.subplot(1, 6, 2)
    plt.imshow(gt[0, 1, :, : , 64].detach().cpu())
    plt.colorbar()
    plt.title('gt')
    plt.subplot(1, 6, 3)
    plt.imshow(projection_maps[0][0, 0, :, : , 64].detach().cpu())
    plt.colorbar()
    plt.title('projection 1')
    plt.subplot(1, 6, 4)
    plt.imshow(projection_maps[1][0, 0, :, : , 64].detach().cpu())
    plt.colorbar()
    plt.title('projection 2')
    plt.subplot(1, 6, 5)
    plt.imshow(projection_maps[2][0, 0, :, : , 64].detach().cpu())
    plt.colorbar()
    plt.title('projection 3')
    plt.subplot(1, 6, 6)
    plt.imshow(projection_maps[3][0, 0, :, : , 64].detach().cpu())
    plt.colorbar()
    plt.title('projection 4')

    plt.savefig(f'./temporary/testset{idx}')
    plt.close()
            
    del image
    del gt
    del to_projector
    del to_classifier