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
    to_load_folder = 'Dec06',
    device = 'cuda:0',
    proxy_type = None,
    train_task = None,
    to_load_encoder_path = 'UNETcopy_encoder_VoxCFT18000_randomBG_06_12_2023_state_dict50.pth',
    to_load_projector_path = 'UNETcopy_projector_VoxCFT18000_randomBG_06_12_2023_state_dict50.pth',
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'wmh'
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
testloader = DataLoader(testset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

# encoder = DuckNet(input_channels=1, out_classes=2, starting_filters=17).to(device)
# encoder = ResNet3D_Encoder(image_channels=1).to(device)
# encoder = VGG3D_Encoder(input_channels=1).to(device)
encoder = SA_UNet_Encoder(out_channels=2).to(c.device)
encoder.load_state_dict(torch.load(c.to_load_encoder_path))

# classification_head = Classifier(input_channels=17408, output_channels=5).to(device)
# classification_head = Classifier(input_channels=2176, output_channels=5).to(device)

# projection_head = Projector(num_layers=5, layer_sizes=[17, 34, 68, 136, 272]).to(device)
# projection_head = Projector(num_layers=4, layer_sizes=[64, 128, 256, 512]).to(device)
# projection_head = Projector(num_layers=5, layer_sizes=[32, 64, 128, 256, 512]).to(device)
projection_head = Projector(num_layers=4, layer_sizes=[64, 128, 256, 512], test=True).to(c.device)
projection_head.load_state_dict(torch.load(c.to_load_projector_path))

encoder.eval()
projection_head.eval()

for idx, data in enumerate(tqdm(testloader), 0):
    image = data['input'].to(c.device)
    gt = data['gt'].to(c.device)
    oneHot_label = data['lesion_labels'].float().to(c.device)

    to_projector, to_classifier = encoder(image)

    if torch.unique(gt[:, 1]).shape[0] == 2:
        combined, stacked_projections = projection_head(to_projector)
        projection = F.interpolate(stacked_projections, size=(128, 128, 128))
        combined = F.interpolate(combined, size=(128, 128, 128))

        plt.figure(figsize=(40, 15))
        plt.subplot(1, 7, 1)
        plt.imshow(image[0, 0, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.title(f'Input Sample #{idx+1}')
        plt.subplot(1, 7, 2)    
        plt.imshow(gt[0, 1, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.title(f'Input gt #{idx+1}')
        plt.subplot(1, 7, 3)
        plt.imshow(projection[0, 0, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.title(f'Projection1 #{idx+1}')
        plt.subplot(1, 7, 4)
        plt.imshow(projection[0, 1, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.title(f'Projection2 #{idx+1}')
        plt.subplot(1, 7, 5)
        plt.imshow(projection[0, 2, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.title(f'Projection3 #{idx+1}')
        plt.subplot(1, 7, 6)
        plt.imshow(projection[0, 3, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.title(f'Projection4 #{idx+1}')
        plt.subplot(1, 7, 7)
        plt.imshow(combined[0, 0, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.title(f'Combined projection #{idx+1}')


        plt.savefig(f'./temporary/valset#{idx+1}')
        plt.close()

        del projection
            
    del image
    del gt
    del oneHot_label
    del to_projector
    del to_classifier