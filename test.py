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

wmh_indexes = np.load('../OOD/data/wmh_indexes.npy', allow_pickle=True).item()
wmh_files = glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/MICCAI_Challenge_data/*npz')

datadict_test = ImageLoader3D(wmh_indexes['test_names'],None,type_of_imgs='numpy', transform = ToTensor3D(True))
testloader = DataLoader(datadict_test, batch_size=1, shuffle=False, num_workers=0)

device = 'cuda:0'

model = DuckNet_smaller(input_channels = 1,out_classes = 2,starting_filters = 17).to(device)
model.load_state_dict(torch.load('./models/DUCK_smaller_11_10_23_state_dict_best_loss38.pth'))
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
            del image
            del label
            del err

    print('Testing Loss, Dice Score and F1 Score is : {}, {} and {}'.format(test_loss/len(testloader),test_dice/len(testloader),test_f1_acc/len(testloader)))