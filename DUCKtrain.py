import glob
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from ModelArchitecture.Losses import DiceLoss
from ModelArchitecture.DUCK_Net import DuckNet, DuckNet_smaller
from ModelArchitecture.Transformations import *
from ImageLoader.ImageLoader3D import ImageLoader3D
from ImageLoader.Ablation import SphereGeneration, LesionGeneration
import matplotlib.pyplot as plt

def commence(test, hs, hn, hsm, epochs):
    composed_transform = transforms.Compose([
                RandomRotation3D([10,10]),
                RandomIntensityChanges(),
                ToTensor3D(True)])

    # Brats or WMH
    which_data = test

    models_folder_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/Ablation-tests-round#2/'

    if which_data=='brats':
        # Brain Tumor Segmentation Challenge
        brats_indexes = np.load('../brats_2020_indexes.npy', allow_pickle=True).item()
        brats_flair_files = glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/**/*flair*[:-7]*')
        brats_seg_files = glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/**/*seg*[:-7]*')

        datadict_train = ImageLoader3D(brats_indexes['train_names_flair'],brats_indexes['train_names_seg'],type_of_imgs='nifty',transform = composed_transform)
        datadict_val = ImageLoader3D(brats_indexes['val_names_flair'],brats_indexes['val_names_seg'],type_of_imgs='nifty', transform = ToTensor3D(True))
        datadict_test = ImageLoader3D(brats_indexes['test_names_flair'],brats_indexes['test_names_seg'],type_of_imgs='nifty', transform = ToTensor3D(True))

    elif which_data == 'wmh':
        # White Matter Hyperintensities
        wmh_indexes = np.load('../wmh_indexes.npy', allow_pickle=True).item()
        wmh_files = glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/MICCAI_Challenge_data/*npz')

        datadict_train = ImageLoader3D(wmh_indexes['train_names'],None,type_of_imgs='numpy',transform = composed_transform)
        datadict_val = ImageLoader3D(wmh_indexes['val_names'],None,type_of_imgs='numpy', transform = ToTensor3D(True))
        datadict_test = ImageLoader3D(wmh_indexes['test_names'],None,type_of_imgs='numpy', transform = ToTensor3D(True))

    elif which_data == 'sphere_generation':
        healthy_data = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
        healthy_seg = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))
        healthy_masks = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*biascorr_brain_mask.nii.gz*'))

        train_fraction = int(0.7 * len(healthy_data))
        val_fraction = int(0.1 * len(healthy_data))
        test_fraction = int(0.2 * len(healthy_data))

        generated_train = SphereGeneration(
            paths=healthy_data[:train_fraction],
            gt_paths=healthy_masks[:train_fraction],
            transform=composed_transform
        )

        generated_val = SphereGeneration(
            paths=healthy_data[train_fraction:train_fraction+val_fraction],
            gt_paths=healthy_masks[train_fraction:train_fraction+val_fraction],
            transform=composed_transform
        )

        generated_test = SphereGeneration(
            paths=healthy_data[train_fraction+val_fraction:],
            gt_paths=healthy_masks[train_fraction+val_fraction:],
            transform=composed_transform
        )

        healthy_train = ImageLoader3D(
            paths=healthy_data[:train_fraction],
            gt_paths=healthy_seg[:train_fraction],
            image_size=128,
            type_of_imgs='nifty',
            transform=composed_transform
        )

        healthy_val = ImageLoader3D(
            paths=healthy_data[train_fraction:train_fraction+val_fraction],
            gt_paths=healthy_seg[train_fraction:train_fraction+val_fraction],
            image_size=128,
            type_of_imgs='nifty',
            transform=composed_transform
        )

        healthy_test = ImageLoader3D(
            paths=healthy_data[train_fraction+val_fraction:],
            gt_paths=healthy_seg[train_fraction+val_fraction:],
            image_size=128,
            type_of_imgs='nifty',
            transform=composed_transform
        )

        datadict_train = ConcatDataset([generated_train, healthy_train])
        datadict_val = ConcatDataset([generated_val, healthy_val])
        datadict_test = ConcatDataset([generated_test, healthy_test])

    elif which_data == 'lesion_generation':
        healthy_data = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
        healthy_seg = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))
        healthy_masks = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*biascorr_brain_mask.nii.gz*'))

        train_fraction = int(0.7 * len(healthy_data))
        val_fraction = int(0.1 * len(healthy_data))
        test_fraction = int(0.2 * len(healthy_data))

        generated_train = LesionGeneration(
            paths=healthy_data[:train_fraction],
            gt_paths=healthy_masks[:train_fraction],
            transform=composed_transform,
            have_smoothing=hs,
            have_noise=hn,
            have_small=hsm
        )

        generated_val = LesionGeneration(
            paths=healthy_data[train_fraction:train_fraction+val_fraction],
            gt_paths=healthy_masks[train_fraction:train_fraction+val_fraction],
            transform=composed_transform,
            have_smoothing=hs,
            have_noise=hn,
            have_small=hsm
        )

        generated_test = LesionGeneration(
            paths=healthy_data[train_fraction+val_fraction:],
            gt_paths=healthy_masks[train_fraction+val_fraction:],
            transform=composed_transform,
            have_smoothing=hs,
            have_noise=hn,
            have_small=hsm
        )

        healthy_train = ImageLoader3D(
            paths=healthy_data[:train_fraction],
            gt_paths=healthy_seg[:train_fraction],
            image_size=128,
            type_of_imgs='nifty',
            transform=composed_transform
        )

        healthy_val = ImageLoader3D(
            paths=healthy_data[train_fraction:train_fraction+val_fraction],
            gt_paths=healthy_seg[train_fraction:train_fraction+val_fraction],
            image_size=128,
            type_of_imgs='nifty',
            transform=composed_transform
        )

        healthy_test = ImageLoader3D(
            paths=healthy_data[train_fraction+val_fraction:],
            gt_paths=healthy_seg[train_fraction+val_fraction:],
            image_size=128,
            type_of_imgs='nifty',
            transform=composed_transform
        )

        datadict_train = ConcatDataset([generated_train, healthy_train])
        datadict_val = ConcatDataset([generated_val, healthy_val])
        datadict_test = ConcatDataset([generated_test, healthy_test])

    trainloader = DataLoader(datadict_train, batch_size=2, shuffle=True, num_workers=0)
    valloader = DataLoader(datadict_val, batch_size=1, shuffle=False, num_workers=0)

    device = 'cuda:1'

    model = DuckNet(input_channels = 1,out_classes = 2,starting_filters = 17).to(device)
    model_name = 'DUCK'
    model_type = f'DUCK_{test}_25_10_23'

    if test == 'lesion_generation':
        model_type = f'DUCK_{test}_smooth:{hs}_noise:{hn}_small:{hsm}_25_10_23'

    criterion = nn.BCELoss().to(device)
    # criterion = DiceLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001, eps = 0.0001)

    train_losses = []
    val_losses = []
    best_loss = np.inf

    iters = 0
    index = 0
    num_epochs = epochs

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

                # seg_map = (output > 0.5).float()
                # seg_map.requires_grad=True

                # err = criterion(output[:,1], label[:,1])
                err = criterion(output, label)
                model.zero_grad()
                err.backward()
                optimizer.step()
                pbar.set_postfix(Train_Loss = np.round(err.cpu().detach().numpy().item(), 5))
                pbar.update(0)
                epoch_loss += err.item()
                del image
                del label
                del err
                # del seg_map
                

            train_losses.append([epoch_loss/len(trainloader)])
            print('Training Loss at epoch {} is : Total {}'.format(epoch,*train_losses[-1]))

        epoch_loss = 0
        model.eval()
        with tqdm(range(len(valloader))) as pbar:
            for i, data in zip(pbar, valloader):
                torch.cuda.empty_cache()
                err = 0
                with torch.no_grad():
                    image = Variable(data['input']).to(device)
                    output = model.forward(image)
                    label = data['gt'].to(device)
                    err = criterion(output, label)
                    # err = criterion((output>0.5).float(),label)
                    del image
                    del label

                pbar.set_postfix(Val_Loss = np.round(err.cpu().detach().numpy().item(), 5))
                pbar.update(0)
                epoch_loss += err.item()
                del err

            val_losses.append([epoch_loss/len(valloader)])
            print('Validation Loss at epoch {} is : Total {}'.format(epoch,*val_losses[-1]))

        if(epoch_loss<best_loss):
                best_loss = epoch_loss
                torch.save(model.state_dict(),models_folder_path+model_type+'_state_dict_best_loss'+str(epoch)+'.pth')
                early_stopping_counter=15
        else:
                early_stopping_counter-=1

        np.save('./results/' + model_type + '_loss' + '.npy', [train_losses, val_losses])

        if(epoch%10==0):
            torch.save(model.state_dict(),models_folder_path + model_type + '_state_dict' + str(epoch) + '.pth')

    torch.save(model.state_dict(), models_folder_path + model_type + '_state_dict' + str(epoch) + '.pth')

tests = [
#    ('sphere_generation', None, None, None, 20),
#    ('lesion_generation', False, False, False, 20),
#    ('lesion_generation', True, False, False, 20),
#    ('lesion_generation', True, True, False, 20),
#    ('lesion_generation', True, True, True, 20),
    ('wmh', None, None, None, 100)
]

for test, hs, hn, hsm, epochs in tests:

    print(test, hs, hn, hsm)
    commence(test, hs, hn, hsm, epochs)
