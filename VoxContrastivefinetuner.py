import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ModelArchitecture.Encoders import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import os
import glob
import numpy as np
from tqdm import tqdm

batch_size = 1
number_of_epochs = 100
device = 'cuda:1'
encoder_type = 'VoxContrastivefinetuned_MulticlasspretrainResNet18Encoder'
# decoder_type = 'VoxContrastivefinetuned_ResNet18Decoder'
projection_type = 'VoxContrastivefinetuned_MulticlasspretrainResNet18ProjectionHead'
classifier_type = 'VoxContrastivefinetuned_MulticlasspretrainResNet18ClassificationHead'
date = '22_11_2023'

pretrained_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov21/'
finetuned_model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov21/'

anomalous_trainset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TrainSet_5_11_23/*.npz'))
anomalous_validationset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/ValidSet_5_11_23/*.npz'))

print(f'Anomalous Trainset size: {len(anomalous_trainset_paths)}')
print(f'Anomalous Validationset size: {len(anomalous_validationset_paths)}')

composed_transform = transforms.Compose([
        ToTensor3D(True, clean=True, subtracted=True)
    ])

anomalous_trainset = ImageLoader3D(paths=anomalous_trainset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True, subtracted=True)
anomalous_validationset = ImageLoader3D(paths=anomalous_validationset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True, subtracted=True)

ResNet_encoder = ResNet3D_Encoder(image_channels=1).to(device)
# ResNet_decoder = ResNet3D_Decoder(feature_channels=512).to(device)
# projection_head = nn.Conv3d(960, 1, kernel_size=1).to(device)
projection_head = Projector(num_layers=4, layer_sizes=[64, 128, 256, 512]).to(device)
classification_head = Classifier(input_channels=32768, output_channels=2).to(device)

ResNet_encoder.load_state_dict(torch.load(f'{pretrained_model_path}BCEpretrained_21_11_2023_state_dict50.pth'))
classification_head.load_state_dict(torch.load(f'{pretrained_model_path}BCEpretrained_ResNet_21_11_2023_state_dict50.pth'))

trainset = anomalous_trainset
validationset = anomalous_validationset

print(f'Trainset size: {len(trainset)}')
print(f'Validationset size: {len(validationset)}')

ResNet_trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)
ResNet_validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=0)

# encoder_optimizer = optim.Adam([*ResNet_encoder.parameters(), *ResNet_decoder.parameters(), *projection_head.parameters()], lr = 0.0001, eps = 0.0001)
encoder_optimizer = optim.Adam([*ResNet_encoder.parameters(), *projection_head.parameters()], lr = 0.0001, eps = 0.0001)
# encoder_optimizer = optim.Adam(ResNet_encoder.parameters(), lr = 0.0001, eps = 0.0001)
classifier_optimizer = optim.Adam(classification_head.parameters(), lr = 0.00003, eps = 0.0001)

encoder_criterion = VoxelwiseSupConLoss_inImage(device=device).to(device)
classifier_criterion = nn.BCELoss().to(device)

encoder_train_losses = []
encoder_validation_losses = []

classifier_train_losses = []
classifier_validation_losses = []

classifier_train_accuracies = []
classifier_validation_accuracies = []

print()
print('Training ResNet Model.')
for epoch in range(1, number_of_epochs+1):

    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    ResNet_encoder.train()
    # ResNet_decoder.train()
    projection_head.train()

    classification_head.train()

    epoch_encoder_train_loss = 0
    epoch_classifier_train_loss = 0
    epoch_classifier_train_accuracy = 0

    skipped_batches = 0
    idx = 1

    for data in tqdm(ResNet_trainloader):

        image = data['input'].to(device)
        gt = data['gt'].to(device)
        clean = data['clean'].to(device)
        subtracted = data['subtracted'].to(device)
        mixed = torch.cat([image, clean])

        current_batch_size = image.shape[0]

        if torch.unique(gt[:, 1]).shape[0] == 1:
            skipped_batches += 1
            continue

        oneHot_labels = []
            
        for sample_idx in range(current_batch_size):
            if torch.unique(gt[sample_idx, 1]).shape[0] == 2:
                # anomalous
                oneHot_labels.append([1, 0])
            else:
                # normal
                oneHot_labels.append([0, 1])

        oneHot_labels.extend([[0, 1]] * current_batch_size)
        oneHot_labels = torch.tensor(oneHot_labels).float().to(device)

        out_dict = ResNet_encoder.forward(mixed)
        layer1 = out_dict['out1']
        layer2 = out_dict['out2']
        layer3 = out_dict['out3']
        z_mixed = out_dict['out4']

        layer1 = F.interpolate(layer1, size=(32, 32, 32), mode='trilinear')
        layer2 = F.interpolate(layer2, size=(32, 32, 32), mode='trilinear')
        layer3 = F.interpolate(layer3, size=(32, 32, 32), mode='trilinear')
        layer4 = F.interpolate(z_mixed, size=(32, 32, 32), mode='trilinear')

        stacked_layers = torch.cat([layer1, layer2, layer3, layer4], dim=1)
        feature_diff = stacked_layers[:current_batch_size]

        feature_diff = projection_head.forward(feature_diff)
        feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')

        if not os.path.exists(f'./temporary/epoch#{epoch}'):
            os.makedirs(f'./temporary/epoch#{epoch}')

        plt.figure(figsize=(20, 15))
        plt.subplot(1, 4, 1)
        plt.imshow(feature_diff[0, 0, :, :, 64].detach().cpu(), cmap='gray')
        plt.title(f'Epoch #{epoch} Feature diff #{idx+1}')
        plt.colorbar()
        plt.subplot(1, 4, 2)
        plt.imshow(image[0, 0, :, :, 64].detach().cpu(), cmap='gray')
        plt.title(f'Epoch #{epoch} Input #{idx+1}')
        plt.colorbar()
        plt.subplot(1, 4, 3)
        plt.imshow(gt[0, 1, :, :, 64].detach().cpu(), cmap='gray')
        plt.title(f'Epoch #{epoch} GT #{idx+1}')
        plt.colorbar()
        plt.subplot(1, 4, 4)
        plt.imshow(subtracted[0, 0, :, :, 64].detach().cpu(), cmap='gray')
        plt.title(f'Epoch #{epoch} Subtracted GT #{idx+1}')
        plt.colorbar()
        plt.savefig(f'./temporary/epoch#{epoch}/sample#{idx}')
        plt.close()

        idx += 1

        # decoder_output = ResNet_decoder.forward(z_mixed[:current_batch_size])

        z_mixed = torch.reshape(z_mixed, shape=(z_mixed.shape[0], -1))
        y_mixed = classification_head.forward(z_mixed.detach())

        # encoder_loss = encoder_criterion(decoder_output, gt, subtracted)
        encoder_loss = encoder_criterion(feature_diff, gt, subtracted)
        epoch_encoder_train_loss += encoder_loss.item()

        encoder_optimizer.zero_grad()
        encoder_loss.backward()
        encoder_optimizer.step()

        classifier_loss = classifier_criterion(y_mixed, oneHot_labels)
        epoch_classifier_train_loss += classifier_loss.item()
        
        classifier_optimizer.zero_grad()
        classifier_loss.backward()
        classifier_optimizer.step()

        epoch_classifier_train_accuracy += determine_class_accuracy(y_mixed, oneHot_labels).cpu()

        del image
        del clean
        del gt
        del subtracted
        del mixed
        del oneHot_labels
        del out_dict
        del layer1
        del layer2
        del layer3
        del layer4
        del stacked_layers
        del feature_diff
        del z_mixed
        del y_mixed
        del encoder_loss
        del classifier_loss
    
    encoder_train_losses.append(epoch_encoder_train_loss / ((len(ResNet_trainloader) - skipped_batches)))
    classifier_train_losses.append(epoch_classifier_train_loss / ((len(ResNet_trainloader) - skipped_batches)))
    classifier_train_accuracies.append(epoch_classifier_train_accuracy / ((len(ResNet_trainloader) - skipped_batches)))

    print(f'Encoder Training loss at epoch #{epoch}: {encoder_train_losses[-1]}')
    print(f'Classifier Training loss at epoch #{epoch}: {classifier_train_losses[-1]}')
    print(f'Classifier Training accuracy at epoch #{epoch}: {classifier_train_accuracies[-1]}')
    print(f'Skipped batches: {skipped_batches}')

    print()
    torch.cuda.empty_cache()

    # Validation loop
    ResNet_encoder.eval()
    # ResNet_decoder.eval()
    projection_head.eval()
    classification_head.eval()

    epoch_encoder_validation_loss = 0
    epoch_classifier_validation_loss = 0
    epoch_classifier_validation_accuracy = 0

    skipped_batches = 0

    for data in tqdm(ResNet_validationloader):
        with torch.no_grad():

            image = data['input'].to(device)
            gt = data['gt'].to(device)
            clean = data['clean'].to(device)
            subtracted = data['subtracted'].to(device)
            mixed = torch.cat([image, clean])

            current_batch_size = image.shape[0]

            if torch.unique(gt[:, 1]).shape[0] == 1:
                skipped_batches += 1
                continue

            oneHot_labels = []
            
            for sample_idx in range(current_batch_size):
                if torch.unique(gt[sample_idx, 1]).shape[0] == 2:
                    # anomalous
                    oneHot_labels.append([1, 0])
                else:
                    # normal
                    oneHot_labels.append([0, 1])

            oneHot_labels.extend([[0, 1]] * current_batch_size)
            oneHot_labels = torch.tensor(oneHot_labels).float().to(device)

            out_dict = ResNet_encoder.forward(mixed)
            layer1 = out_dict['out1']
            layer2 = out_dict['out2']
            layer3 = out_dict['out3']
            z_mixed = out_dict['out4']

            layer1 = F.interpolate(layer1, size=(32, 32, 32), mode='trilinear')
            layer2 = F.interpolate(layer2, size=(32, 32, 32), mode='trilinear')
            layer3 = F.interpolate(layer3, size=(32, 32, 32), mode='trilinear')
            layer4 = F.interpolate(z_mixed, size=(32, 32, 32), mode='trilinear')

            stacked_layers = torch.cat([layer1, layer2, layer3, layer4], dim=1)
            feature_diff = stacked_layers[:current_batch_size]

            feature_diff = projection_head.forward(feature_diff)
            feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')

            # decoder_output = ResNet_decoder.forward(z_mixed[:current_batch_size])

            z_mixed = torch.reshape(z_mixed, shape=(z_mixed.shape[0], -1))
            y_mixed = classification_head.forward(z_mixed.detach())

            # encoder_loss = encoder_criterion(decoder_output, gt, subtracted)
            encoder_loss = encoder_criterion(feature_diff, gt, subtracted)
            epoch_encoder_validation_loss += encoder_loss.item()

            classifier_loss = classifier_criterion(y_mixed, oneHot_labels)
            epoch_classifier_validation_loss += classifier_loss.item()

            epoch_classifier_validation_accuracy += determine_class_accuracy(y_mixed, oneHot_labels).cpu()

            del image
            del clean
            del gt
            del mixed
            del out_dict
            del layer1
            del layer2
            del layer3
            del layer4
            del stacked_layers
            del feature_diff
            del z_mixed
            del y_mixed
            del encoder_loss
            del classifier_loss
    
    encoder_validation_losses.append(epoch_encoder_validation_loss / ((len(ResNet_validationloader) - skipped_batches)))
    classifier_validation_losses.append(epoch_classifier_validation_loss / ((len(ResNet_validationloader) - skipped_batches)))
    classifier_validation_accuracies.append(epoch_classifier_validation_accuracy / ((len(ResNet_validationloader) - skipped_batches)))

    print(f'Encoder Validation loss at epoch #{epoch}: {encoder_validation_losses[-1]}')
    print(f'Classifier Validation loss at epoch #{epoch}: {classifier_validation_losses[-1]}')
    print(f'Classifier Validation accuracy at epoch #{epoch}: {classifier_validation_accuracies[-1]}')
    print(f'Skipped batches: {skipped_batches}')

    np.save(f'./results/{encoder_type}_{date}_losses.npy', [encoder_train_losses, encoder_validation_losses])
    np.save(f'./results/{classifier_type}_{date}_accuracies.npy', [classifier_train_losses, classifier_validation_losses, classifier_train_accuracies, classifier_validation_accuracies])

    if epoch % 5 == 0:
        torch.save(ResNet_encoder.state_dict(), f'{finetuned_model_path}{encoder_type}_{date}_state_dict{epoch}.pth')
        # torch.save(ResNet_decoder.state_dict(), f'{finetuned_model_path}{decoder_type}_{date}_state_dict{epoch}.pth')
        torch.save(projection_head.state_dict(), f'{finetuned_model_path}{projection_type}_{date}_state_dict{epoch}.pth')
        torch.save(classification_head.state_dict(), f'{finetuned_model_path}{classifier_type}_{date}_state_dict{epoch}.pth')

    print()

torch.save(ResNet_encoder.state_dict(), f'{finetuned_model_path}{encoder_type}_{date}_state_dict{number_of_epochs+1}.pth')
# torch.save(ResNet_decoder.state_dict(), f'{finetuned_model_path}{encoder_type}_{date}_ResNetDecoder_state_dict{number_of_epochs+1}.pth')
torch.save(projection_head.state_dict(), f'{finetuned_model_path}{projection_type}_{date}_state_dict{number_of_epochs+1}.pth')
torch.save(classification_head.state_dict(), f'{finetuned_model_path}{classifier_type}_{date}_state_dict{number_of_epochs+1}.pth')

print()
print('Script executed.')
