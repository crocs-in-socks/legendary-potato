import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import Constants, load_dataset

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

from tqdm import tqdm

c = Constants(
    batch_size = 1,
    patience = None,
    num_workers = 8,
    num_epochs = None,
    date = None,
    to_save_folder = None,
    to_load_folder = 'Dec06',
    device = 'cuda:0',
    proxy_type = None,
    train_task = None,
    to_load_encoder_path = 'UNETcopy_encoder_VoxCFT18000_randomBG_06_12_2023_state_dict40.pth',
    to_load_projector_path = 'UNETcopy_projector_VoxCFT18000_randomBG_06_12_2023_state_dict40.pth',
    to_load_classifier_path = 'UNETcopy_projector_VoxCFT18000_randomBG_06_12_2023_state_dict40.pth',
    to_load_proxy_path = None,
    dataset = 'wmh'
)

segmentation_path = './ModelArchitecture/unet_focal + dice_state_dict_best_loss28.pth'

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
testloader = DataLoader(testset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

proxy_encoder = SA_UNet_Encoder(out_channels=2).to(c.device)
proxy_encoder.load_state_dict(torch.load(c.to_load_encoder_path))
proxy_projector = Projector(num_layers=4, layer_sizes=[ 64, 128, 256, 512], test=True).to(c.device)
proxy_projector.load_state_dict(torch.load(c.to_load_projector_path))

segmentation_model = SA_UNet(out_channels=2).to(c.device)
segmentation_model.load_state_dict(torch.load(segmentation_path)['model_state_dict'], strict=False)

test_loss = 0
test_dice = 0
test_f1_accuracy = 0

proxy_encoder.eval()
proxy_projector.eval()
segmentation_model.eval()

print()
print('Testing Integrated model.')

for data_idx, data in enumerate(tqdm(testloader), 0):
    with torch.no_grad():
        image = data['input'].to(c.device)
        gt = data['gt'].to(c.device)

        to_projector, _ = proxy_encoder(image)
        combined_projection, projection_maps = proxy_projector(to_projector)
        
        segmentation = segmentation_model(image, projection_maps)

        dice = Dice_Score(segmentation[:, 1].cpu().numpy(), gt[:,1].cpu().numpy())
        f1_acc = F1_score(segmentation[:, 1].cpu().numpy(), gt[:,1].cpu().numpy())
        test_dice += dice.item()
        test_f1_accuracy += f1_acc.item()

        combined_projection = F.interpolate(combined_projection, size=(128, 128, 128))

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 4, 1)
        plt.imshow(image[0, 0, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.subplot(1, 4, 2)
        plt.imshow(gt[0, 1, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.subplot(1, 4, 3)
        plt.imshow(segmentation[0, 1, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.subplot(1, 4, 4)
        plt.imshow(combined_projection[0, 0, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.savefig(f'./temporary/sample#{data_idx}')
        plt.close()

        for map_idx, map in enumerate(projection_maps):
            projection_maps[map_idx] = F.interpolate(map, size=(128, 128, 128))
        plt.figure(figsize=(20, 15))
        plt.subplot(1, 4, 1)
        plt.imshow(projection_maps[0][0, 0, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.subplot(1, 4, 2)
        plt.imshow(projection_maps[1][0, 0, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.subplot(1, 4, 3)
        plt.imshow(projection_maps[2][0, 0, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.subplot(1, 4, 4)
        plt.imshow(projection_maps[3][0, 0, :, :, 64].detach().cpu())
        plt.colorbar()
        plt.savefig(f'./temporary/maps#{data_idx}')
        plt.close()
    
test_dice /= len(testloader)
test_f1_accuracy /= len(testloader)
print(f'Test dice score: {test_dice}')
print(f'Test f1 accuracy: {test_f1_accuracy}')

print()
print('Script executed.')