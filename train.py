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
from ModelArchitecture.DUCK_Net import DuckNet
from ModelArchitecture.Transformations import *
from ImageLoader.ImageLoader3D import ImageLoader3D


composed_transform = transforms.Compose([
            RandomRotation2Dslice([10,10]),
            RandomIntensityChangesslice(),
            ToTensor2Dslice(True)])

wmh_indexes = np.load('../OOD/data/wmh_indexes.npy', allow_pickle=True).item()
wmh_files = glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/MICCAI_Challenge_data/*npz')

datadict_train = ImageLoader3D(wmh_indexes['train_names'],None,type_of_imgs='numpy',transform = composed_transform)
datadict_val = ImageLoader3D(wmh_indexes['val_names'],None,type_of_imgs='numpy', transform = ToTensor2Dslice(True))
datadict_test = ImageLoader3D(wmh_indexes['test_names'],None,type_of_imgs='numpy', transform = ToTensor2Dslice(True))

trainloader = DataLoader(datadict_train, batch_size=4, shuffle=True, num_workers=0)
valloader = DataLoader(datadict_val, batch_size=1, shuffle=True, num_workers=0)

device = 'cuda:0'

model = DuckNet(input_channels = 1,out_classes = 2,starting_filters = 34).to(device)
model_name = 'DUCK'
model_type = 'DUCK_10_10_23'

criterion = DiceLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.001, eps = 0.0001)

train_losses = []
val_losses = []
best_loss = np.inf

iters = 0
index = 0
num_epochs = 100

early_stopping_counter = 15

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    if(early_stopping_counter == 0):
        break

    epoch_loss = 0

    model.train()

    with tqdm(range(len(trainloader))) as pbar:
        for i, data in zip(pbar, trainloader):
            torch.cuda.empty_cache()
            err = 0
            image = Variable(data['input']).to(device)
            output = model.forward(image)
            label = data['gt'].to(device)
            err = criterion(output,label)
            model.zero_grad()
            err.backward()
            optimizer.step()
            pbar.set_postfix(Train_Loss = np.round(err.cpu().detach().numpy().item(), 5))
            pbar.update(0)
            epoch_loss += err.item()

        train_losses.append([epoch_loss/len(trainloader)])
        print('Training Loss at epoch {} is : Total {}'.format(epoch,*train_losses[-1]))

    epoch_loss = 0
    model.eval()
    with tqdm(range(len(valloader))) as pbar:
        for i, data in zip(pbar, trainloader):
            torch.cuda.empty_cache()
            err = 0
            with torch.no_grad():
                image = Variable(data['input']).to(device)
                output = model.forward(image)
                label = data['gt'].to(device)
                err = criterion(output,label)

            pbar.set_postfix(Val_Loss = np.round(err.cpu().detach().numpy().item(), 5))
            pbar.update(0)
            epoch_loss += err.item()

        val_losses.append([epoch_loss/len(valloader)])
        print('Validation Loss at epoch {} is : Total {}'.format(epoch,*val_losses[-1]))
    
    if(epoch_loss<best_loss):
            best_loss = epoch_loss
            torch.save(model.state_dict(),'../models/'+model_type+'_state_dict_best_loss'+str(epoch)+'.pth')
            early_stopping_counter=15
    else:
            early_stopping_counter-=1

    np.save('../results/'+model_type+'_loss' + '.npy', [train_losses,val_losses])
    
    if(epoch%10==0):
        torch.save(model.state_dict(),'../models/'+model_type+'_state_dict'+str(epoch)+'.pth')


torch.save(model.state_dict(), '../models/' + model_type + '_state_dict' + str(epoch) + '.pth')
