import glob
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ModelArchitecture.Losses import DiceLoss
from ModelArchitecture.DUCK_Net import DuckNet,DuckNet_smaller
from ModelArchitecture.metrics import F1_score, Dice_Score
from ModelArchitecture.Transformations import *
from ImageLoader.ImageLoader3D import ImageLoader3D

import matplotlib.pyplot as plt

# Brats or WMH
which_data = 'wmh'

if(which_data=='brats'):
    # Brain Tumor Segmentation Challenge
    brats_indexes = np.load('../brats_2020_indexes.npy', allow_pickle=True).item()
    datadict_test = ImageLoader3D(brats_indexes['test_names_flair'],brats_indexes['test_names_seg'],type_of_imgs='nifty', transform = ToTensor3D(True))

else:
    # White Matter Hyperintensities
    wmh_indexes = np.load('../wmh_indexes.npy', allow_pickle=True).item()
    datadict_test = ImageLoader3D(wmh_indexes['test_names'],None,type_of_imgs='numpy', transform = ToTensor3D(True))


testloader = DataLoader(datadict_test, batch_size=1, shuffle=False, num_workers=0)

device = 'cuda:0'

model = DuckNet(input_channels = 1,out_classes = 2,starting_filters = 17).to(device)
model.load_state_dict(torch.load('./models/DUCK_WMH_16_10_23_state_dict93.pth'))
criterion = nn.BCELoss().to(device)

test_loss = 0
test_dice = 0
test_f1_acc = 0

with tqdm(range(len(testloader))) as pbar:
    for i, data in zip(pbar, testloader):
        with torch.no_grad():
            torch.cuda.empty_cache()
            err = 0
            image = Variable(data['input']).to(device)
            output = model.forward(image) 
            label = data['gt'].to(device)

            err = criterion(output,label)
            dice = Dice_Score(output[:,1].cpu().numpy(),label[:,1].cpu().numpy())
            f1_acc = F1_score(output[:,1].cpu().numpy(),label[:,1].cpu().numpy())

            pbar.set_postfix(Test_Loss = np.round(err.cpu().detach().numpy().item(), 5),
                             Test_dice =  np.round(dice, 5), 
                             Test_f1_acc = np.round(f1_acc, 5),)
            pbar.update(0)
            test_dice += dice.item()
            test_loss += err.item()
            test_f1_acc += f1_acc.item()

            figure = plt.figure(figsize=(20, 5))
            plt.subplot(1, 5, 1)
            plt.imshow(output.detach().cpu().squeeze(0)[0, :, :, 50])
            plt.title('Output idx: 0, Slice: 50')

            plt.subplot(1, 5, 2)
            plt.imshow(output.detach().cpu().squeeze(0)[1, :, :, 50])
            plt.title('Output idx: 1, Slice: 50')

            plt.subplot(1, 5, 3)
            plt.imshow(label.detach().cpu().squeeze(0)[0, :, :, 50])
            plt.title('Label idx: 0, Slice: 50')

            plt.subplot(1, 5, 4)
            plt.imshow(label.detach().cpu().squeeze(0)[1, :, :, 50])
            plt.title('Label idx: 1, Slice: 50')

            plt.subplot(1, 5, 5)
            plt.imshow(image.detach().cpu().squeeze(0)[0, :, :, 50])
            plt.title('Label idx: 0, Slice: 50')

            plt.savefig(f'./temp/Iteration_{i+1}')
            plt.close()

            del image
            del label
            del err

    print('Testing Loss, Dice Score and F1 Score is : {}, {} and {}'.format(test_loss/len(testloader),test_dice/len(testloader),test_f1_acc/len(testloader)))