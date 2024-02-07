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
from tqdm import tqdm
import matplotlib.pyplot as plt

c = Constants(
    batch_size = 1,
    patience = None,
    num_workers = 8,
    num_epochs = None,
    date = '14_12_2023',
    to_save_folder = 'Dec14',
    to_load_folder = None,
    device = 'cuda:1',
    proxy_type = None,
    train_task = None,
    to_load_encoder_path = None,
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = 'brats+wmh'
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
testloader = DataLoader(testset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

encoder = VGG3D_Encoder(input_channels=1).to(c.device)
encoder.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/experiments/Dec08/VGGproxy_weightedBCE_wLRScheduler_simulated_lesions_on_brain_encoder_08_12_2023_state_dict_best_loss80.pth'))

classification_head = Classifier(input_channels=4096, output_channels=5, pooling_size=2).to(c.device)
classification_head.load_state_dict(torch.load(f'/mnt/{c.drive}/LabData/models_retrained/experiments/Dec08/VGGproxy_weightedBCE_wLRScheduler_simulated_lesions_on_brain_classifier_08_12_2023_state_dict_best_loss80.pth'))

encoder.eval()
classification_head.eval()

for idx, data in tqdm(enumerate(testloader, 0)):
    image = data['input'].to(c.device)
    image.requires_grad = True
    gt = data['gt'].to(c.device)

    to_projector, to_classifier = encoder(image)
    prediction = classification_head(to_classifier)

    fig = plt.figure(figsize=(40, 15))
    for score_idx in range(5):
        prediction[0, score_idx].backward(retain_graph=True)
        map, _ = torch.max(image.grad.data.abs(), dim=1)

        ax = fig.add_subplot(1, 7, score_idx + 1)
        ax.imshow(map[0, :, :, 64].detach().cpu())
        ax.set_title(f"saliency map {score_idx + 1}: {prediction[0, score_idx].data}")

    ax = fig.add_subplot(1, 7, 6)
    ax.imshow(image[0, 0, :, :, 64].detach().cpu())
    ax.set_title(f"Input image")

    ax = fig.add_subplot(1, 7, 7)
    ax.imshow(gt[0, 1, :, :, 64].detach().cpu())
    ax.set_title(f"gt")
    
    plt.tight_layout()
    plt.savefig(f'./temporary/testset{idx}')
    plt.close()
            
    del image
    del gt
    del to_projector
    del to_classifier
    del prediction