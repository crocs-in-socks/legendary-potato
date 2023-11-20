import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from ModelArchitecture.Encoders import *

from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

import glob
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 1
number_of_epochs = 100
device = 'cuda:0'
# model_type = 'ResNetClassifier'
encoder_type = 'Contrastive_ResNetEncoder_Pixelwise_withoutProjection'
decoder_type = 'Contrastive_ResNetDecoder_Pixelwise_withoutProjection'
projection_type = 'Contrastive_ProjectionHead_Pixelwise_withoutProjection'
classifier_type = 'Contrastive_ClassificationHead_Pixelwise_withoutProjection'
date = '15_11_2023'

model_path = '/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/models_retrained/experiments/Nov15/'

# clean_data = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
# clean_labels = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))
# clean_masks = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/LabData/NIMH/3d/*.anat*/*biascorr_brain_mask.nii.gz*'))

# train_fraction = int(len(clean_data)*0.7)
# validation_fraction = int(len(clean_data)*0.1)

# clean_trainset_data = clean_data[:train_fraction]
# clean_trainset_labels = clean_labels[:train_fraction]

# clean_validationset_data = clean_data[train_fraction:train_fraction+validation_fraction]
# clean_validationset_labels = clean_labels[train_fraction:train_fraction+validation_fraction]

anomalous_trainset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TrainSet_5_11_23/*.npz'))
anomalous_validationset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/ValidSet_5_11_23/*.npz'))

print(f'Anomalous Trainset size: {len(anomalous_trainset_paths)}')
print(f'Anomalous Validationset size: {len(anomalous_validationset_paths)}')
# print(f'Clean Trainset size: {len(clean_trainset_data)}')
# print(f'Clean Validationset size: {len(clean_validationset_data)}')

# composed_transform = transforms.Compose([
#         RandomRotation3D([10, 10], clean=True),
#         RandomIntensityChanges(clean=True),
#         ToTensor3D(True, clean=True)
#     ])

composed_transform = transforms.Compose([
        ToTensor3D(True, clean=True, subtracted=True)
    ])

anomalous_trainset = ImageLoader3D(paths=anomalous_trainset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True, subtracted=True)
anomalous_validationset = ImageLoader3D(paths=anomalous_validationset_paths, gt_paths=None, image_size=128, type_of_imgs='numpy', transform=composed_transform, clean=True, subtracted=True)

# clean_trainset = ImageLoader3D(paths=clean_trainset_data, gt_paths=clean_trainset_labels, type_of_imgs='nifty', transform=composed_transform, clean=True, subtracted=True)
# clean_validationset = ImageLoader3D(paths=clean_validationset_data, gt_paths=clean_validationset_labels, type_of_imgs='nifty', transform=composed_transform, clean=True, subtracted=True)

ResNet_encoder = ResNet3D_Encoder(image_channels=1).to(device)
ResNet_decoder = ResNet3D_Decoder(feature_channels=512).to(device)
classification_head = Classifier(input_channels=32768, output_channels=2).to(device)

# projection_head = nn.Conv3d(512, 1, kernel_size=1).to(device)
# projection_head = nn.Conv3d(960, 1, kernel_size=1).to(device)

# trainset = ConcatDataset([anomalous_trainset, clean_trainset])
# validationset = ConcatDataset([anomalous_validationset, clean_validationset])
trainset = anomalous_trainset
validationset = anomalous_validationset

del anomalous_trainset
del anomalous_validationset

print(f'Trainset size: {len(trainset)}')
print(f'Validationset size: {len(validationset)}')

ResNet_trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
ResNet_validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=0)

# encoder_optimizer = optim.Adam([*ResNet_encoder.parameters(), *ResNet_decoder.parameters(), *projection_head.parameters()], lr = 0.0001, eps = 0.0001)
encoder_optimizer = optim.Adam([*ResNet_encoder.parameters(), *ResNet_decoder.parameters()], lr = 0.0001, eps = 0.0001)
classifier_optimizer = optim.Adam(classification_head.parameters(), lr = 0.0001, eps = 0.0001)

# encoder_criterion = SupervisedContrastiveLoss().to(device)
# encoder_criterion = VoxelwiseSupConMSELoss(device=device).to(device)
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

def get_model_memory_usage(model, input_size):

    def get_tensor_memory(tensor):
        return tensor.element_size() * tensor.nelement()

    total_params = 0
    total_buffers = 0

    for param in model.parameters():
        total_params += get_tensor_memory(param)

    for buffer in model.buffers():
        total_buffers += get_tensor_memory(buffer)

    dummy_input = torch.randn(*input_size).to(device)

    # Compute memory usage for the model's forward pass
    with torch.no_grad():
        output = model(dummy_input)
        # print('Output shape:', output.shape)
    total_output = get_tensor_memory(output)

    # Total memory usage is the sum of parameters, buffers, and output tensors
    total_memory = total_params + total_buffers + total_output

    return total_memory / (1024 ** 2)  # Convert to MB

# Usage example:
# Assuming `your_model` is your PyTorch model and `input_size` is the size of the input tensor
model_memory = get_model_memory_usage(ResNet_encoder, [2, 1, 128, 128, 128])
print(f"ResNet Encoder memory usage: {model_memory:.2f} MB")
model_memory = get_model_memory_usage(ResNet_decoder, [2, 512, 4, 4, 4])
print(f"ResNet Decoder memory usage: {model_memory:.2f} MB")

for epoch in range(1, number_of_epochs+1):

    print()
    print(f'Epoch #{epoch}')
    torch.cuda.empty_cache()

    # Train loop
    ResNet_encoder.train()
    ResNet_decoder.train()
    # projection_head.train()
    classification_head.train()

    epoch_encoder_train_loss = 0
    epoch_classifier_train_loss = 0
    epoch_classifier_train_accuracy = 0

    skipped_batches = 0

    for data in tqdm(ResNet_trainloader):
    # for data in (ResNet_trainloader):

        # torch.cuda.empty_cache()
        # print("GPU memory allocated:", torch.cuda.memory_allocated())
        # print("GPU memory cached:", torch.cuda.memory_cached())

        image = data['input'].to(device)
        clean = data['clean'].to(device)
        #gt = data['gt'][:, 1].to(device)
        gt = data['gt'].to(device)
        subtracted = data['subtracted'].to(device)

        #gt = gt.unsqueeze(1)
        mixed = torch.cat([image, clean])
        # mixed = image

        current_batch_size = image.shape[0]

        if torch.unique(gt[:, 1]).shape[0] == 1:
            skipped_batches += 1
            continue
            
        labels = []
        oneHot_labels = []
            
        for sample_idx in range(current_batch_size):
            if torch.unique(gt[sample_idx, 1]).shape[0] == 2:
                # anomalous
                labels.append(1)
                oneHot_labels.append([1, 0])
            else:
                # normal
                labels.append(0)
                oneHot_labels.append([0, 1])

        labels.extend([0] * current_batch_size)
        oneHot_labels.extend([[0, 1]] * current_batch_size)
        
        labels = torch.tensor(labels).float().to(device)
        oneHot_labels = torch.tensor(oneHot_labels).float().to(device)

        # labels = (torch.tensor([1]*current_batch_size + [0]*current_batch_size)).float().to(device)
        # oneHot_labels = (torch.tensor([[1, 0]]*current_batch_size + [[0, 1]]*current_batch_size)).float().to(device)

        # z_mixed = ResNet_Model.forward(mixed)
        # out_dict = ResNet_encoder.forward(mixed)
        z_mixed = ResNet_encoder.forward(mixed)
        # layer1 = out_dict['out1']
        # layer2 = out_dict['out2']
        # layer3 = out_dict['out3']
        # z_mixed = out_dict['out4']

        # layer1 = F.interpolate(layer1, size=(32, 32, 32), mode='trilinear')
        # layer2 = F.interpolate(layer2, size=(32, 32, 32), mode='trilinear')
        # layer3 = F.interpolate(layer3, size=(32, 32, 32), mode='trilinear')
        # layer4 = F.interpolate(z_mixed, size=(32, 32, 32), mode='trilinear')
        # stacked_layers = torch.cat([layer1, layer2, layer3, layer4], dim=1)

        # feature_diff = stacked_layers[:current_batch_size]
        # # feature_diff = stacked_layers
        # feature_diff = projection_head.forward(feature_diff)
        # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')

        # feature_diff = stacked_layers[:batch_size] - stacked_layers[batch_size:]
        # feature_diff = projection_head.forward(feature_diff)
        # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')
        
        # feature_diff = z_mixed[:batch_size] - z_mixed[batch_size:]
        # feature_diff = projection_head.forward(feature_diff)
        # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')

        # feature_diff = F.interpolate(feature_diff, size=(64, 64, 64), mode='trilinear')
        # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')
        # feature_diff = projection_head.forward(feature_diff)
        
        # positive_mask = gt[:, 1].unsqueeze(1)
        # input_diff = image * positive_mask
        # input_diff = mixed * torch.cat([gt, gt])
        # input_diff = image - clean

        # decoder_output = ResNet_decoder.forward(z_mixed[:current_batch_size])
        decoder_output = ResNet_decoder.forward(z_mixed)

        z_mixed = torch.reshape(z_mixed, shape=(z_mixed.shape[0], -1))
        y_mixed = classification_head.forward(z_mixed.detach())

        #encoder_loss = encoder_criterion(z_mixed, labels, feature_diff, input_diff)
        # encoder_loss = encoder_criterion(decoder_output, gt, subtracted, feature_diff, input_diff)
        encoder_loss = encoder_criterion(decoder_output, gt, subtracted)
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
        del labels
        del oneHot_labels
        del z_mixed
        # del feature_diff
        # del input_diff
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
    ResNet_decoder.eval()
    # projection_head.eval()
    classification_head.eval()

    epoch_encoder_validation_loss = 0
    epoch_classifier_validation_loss = 0
    epoch_classifier_validation_accuracy = 0

    skipped_batches = 0

    for data in tqdm(ResNet_validationloader):
    # for data in (ResNet_validationloader):
        with torch.no_grad():

            image = data['input'].to(device)
            clean = data['clean'].to(device)
            #gt = data['gt'][:, 1].to(device)

            #gt = gt.unsqueeze(1)
            gt = data['gt'].to(device)
            subtracted = data['subtracted'].to(device)
            mixed = torch.cat([image, clean])
            # mixed = image

            current_batch_size = image.shape[0]

            if torch.unique(gt[:, 1]).shape[0] == 1:
                skipped_batches += 1
                continue

            labels = []
            oneHot_labels = []
            
            for sample_idx in range(current_batch_size):
                if torch.unique(gt[sample_idx, 1]).shape[0] == 2:
                    # anomalous
                    labels.append(1)
                    oneHot_labels.append([1, 0])
                else:
                    # normal
                    labels.append(0)
                    oneHot_labels.append([0, 1])

            labels.extend([0] * current_batch_size)
            oneHot_labels.extend([[0, 1]] * current_batch_size)
        
            labels = torch.tensor(labels).float().to(device)
            oneHot_labels = torch.tensor(oneHot_labels).float().to(device)

            # labels = (torch.tensor([1] * current_batch_size + [0] * current_batch_size)).float().to(device)
            # oneHot_labels = (torch.tensor([[1, 0]] * current_batch_size + [[0, 1]] * current_batch_size)).float().to(device)

            z_mixed = ResNet_encoder.forward(mixed)
            # out_dict = ResNet_encoder.forward(mixed)
            # layer1 = out_dict['out1']
            # layer2 = out_dict['out2']
            # layer3 = out_dict['out3']
            # z_mixed = out_dict['out4']

            # layer1 = F.interpolate(layer1, size=(32, 32, 32), mode='trilinear')
            # layer2 = F.interpolate(layer2, size=(32, 32, 32), mode='trilinear')
            # layer3 = F.interpolate(layer3, size=(32, 32, 32), mode='trilinear')
            # layer4 = F.interpolate(z_mixed, size=(32, 32, 32), mode='trilinear')
            # stacked_layers = torch.cat([layer1, layer2, layer3, layer4], dim=1)

            # # feature_diff = stacked_layers[:current_batch_size]
            # feature_diff = stacked_layers
            # feature_diff = projection_head.forward(feature_diff)
            # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')

            # feature_diff = stacked_layers[:batch_size] - stacked_layers[batch_size:]
            # feature_diff = projection_head.forward(feature_diff)
            # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')

            # feature_diff = z_mixed[:batch_size] - z_mixed[batch_size:]
            # feature_diff = projection_head(feature_diff)
            # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128))
            
            # positive_mask = gt[:, 1].unsqueeze(1)
            # input_diff = image * gt
            # input_diff = mixed * torch.cat([gt, gt])
            # input_diff = image - clean

            # decoder_output = ResNet_decoder.forward(z_mixed[:current_batch_size])
            decoder_output = ResNet_decoder.forward(z_mixed)

            z_mixed = torch.reshape(z_mixed, shape=(z_mixed.shape[0], -1))
            y_mixed = classification_head.forward(z_mixed.detach())

            #encoder_loss = encoder_criterion(z_mixed, labels, feature_diff, input_diff)
            # encoder_loss = encoder_criterion(decoder_output, gt, feature_diff, input_diff)
            encoder_loss = encoder_criterion(decoder_output, gt, subtracted)
            epoch_encoder_validation_loss += encoder_loss.item()

            classifier_loss = classifier_criterion(y_mixed, oneHot_labels)
            epoch_classifier_validation_loss += classifier_loss.item()

            # print(f'Classifier validation loss: {classifier_loss.item()}')
            # print(f'Classifier validation accuracy: {determine_class_accuracy(y_mixed, oneHot_labels).cpu()}')

            epoch_classifier_validation_accuracy += determine_class_accuracy(y_mixed, oneHot_labels).cpu()

            del image
            del clean
            del gt
            del mixed
            del labels
            del z_mixed
            # del feature_diff
            # del input_diff
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
        torch.save(ResNet_encoder.state_dict(), f'{model_path}{encoder_type}_{date}_ResNetEncoder_state_dict{epoch}.pth')
        torch.save(ResNet_decoder.state_dict(), f'{model_path}{decoder_type}_{date}_ResNetDecoder_state_dict{epoch}.pth')
        # torch.save(projection_head.state_dict(), f'{model_path}{projection_type}_{date}_ProjectorHead_state_dict{epoch}.pth')
        torch.save(classification_head.state_dict(), f'{model_path}{classifier_type}_{date}_ClassifierHead_state_dict{epoch}.pth')

    print()

torch.save(ResNet_encoder.state_dict(), f'{model_path}{encoder_type}_{date}_ResNetEncoder_state_dict{number_of_epochs+1}.pth')
torch.save(ResNet_decoder.state_dict(), f'{model_path}{encoder_type}_{date}_ResNetDecoder_state_dict{number_of_epochs+1}.pth')
# torch.save(projection_head.state_dict(), f'{model_path}{projection_type}_{date}_ProjectorHead_state_dict{number_of_epochs+1}.pth')
torch.save(classification_head.state_dict(), f'{model_path}{classifier_type}_{date}_ClassifierHead_state_dict{number_of_epochs+1}.pth')

# ResNet_Model.load_state_dict(torch.load(f'{model_path}{encoder_type}_{date}_ResNet_state_dict101.pth'))
# classification_head.load_state_dict(torch.load(f'{model_path}{classifier_type}_{date}_ClassifierHead_state_dict101.pth'))

# print()
# print('Training only the classification head.')

# for epoch in range(number_of_epochs+1, number_of_epochs+51):

#     print()
#     print(f'Epoch #{epoch}')
#     torch.cuda.empty_cache()

#     # Train loop
#     ResNet_encoder.eval()
#     # projection_head.train()
#     classification_head.train()

#     epoch_classifier_train_loss = 0
#     epoch_classifier_train_accuracy = 0

#     skipped_batches = 0

#     for data in tqdm(ResNet_trainloader):
#     # for data in (ResNet_trainloader):

#         image = data['input'].to(device)
#         clean = data['clean'].to(device)
#         gt = data['gt'][:, 1].to(device)
#         # gt = data['gt'].to(device)
#         gt = gt.unsqueeze(1)
#         mixed = torch.cat([image, clean])
#         # mixed = image

#         current_batch_size = image.shape[0]

#         if torch.unique(gt).shape[0] == 1:
#             skipped_batches += 1
#             continue

#         labels = (torch.tensor([1]*current_batch_size + [0]*current_batch_size)).float().to(device)
#         oneHot_labels = (torch.tensor([[1, 0]]*current_batch_size + [[0, 1]]*current_batch_size)).float().to(device)

#         # z_mixed = ResNet_Model.forward(mixed)
#         out_dict = ResNet_encoder.forward(mixed)
#         # layer1 = out_dict['out1']
#         # layer2 = out_dict['out2']
#         # layer3 = out_dict['out3']
#         z_mixed = out_dict['out4']

#         # layer1 = F.interpolate(layer1, size=(32, 32, 32), mode='trilinear')
#         # layer2 = F.interpolate(layer2, size=(32, 32, 32), mode='trilinear')
#         # layer3 = F.interpolate(layer3, size=(32, 32, 32), mode='trilinear')
#         # layer4 = F.interpolate(z_mixed, size=(32, 32, 32), mode='trilinear')
#         # stacked_layers = torch.cat([layer1, layer2, layer3, layer4], dim=1)

#         # feature_diff = projection_head.forward(stacked_layers)
#         # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')

#         # feature_diff = stacked_layers[:batch_size] - stacked_layers[batch_size:]
#         # feature_diff = projection_head.forward(feature_diff)
#         # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')
        
#         # feature_diff = z_mixed[:batch_size] - z_mixed[batch_size:]
#         # feature_diff = projection_head.forward(feature_diff)
#         # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')

#         # feature_diff = F.interpolate(feature_diff, size=(64, 64, 64), mode='trilinear')
#         # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')
#         # feature_diff = projection_head.forward(feature_diff)

#         # input_diff = mixed * torch.cat([gt, gt])
#         # input_diff = image * gt
#         # input_diff = image - clean

#         z_mixed = torch.reshape(z_mixed, shape=(z_mixed.shape[0], -1))
#         y_mixed = classification_head.forward(z_mixed.detach())

#         classifier_loss = classifier_criterion(y_mixed, oneHot_labels)
#         epoch_classifier_train_loss += classifier_loss.item()
#         classifier_optimizer.zero_grad()
#         classifier_loss.backward()
#         classifier_optimizer.step()

#         epoch_classifier_train_accuracy += determine_class_accuracy(y_mixed, oneHot_labels).cpu()

#         del image
#         del clean
#         del gt
#         del mixed
#         del labels
#         del z_mixed
#         # del feature_diff
#         # del input_diff
#         del y_mixed
#         del classifier_loss
    
#     classifier_train_losses.append(epoch_classifier_train_loss / ((len(ResNet_trainloader) - skipped_batches)))
#     classifier_train_accuracies.append(epoch_classifier_train_accuracy / ((len(ResNet_trainloader) - skipped_batches)))

#     print(f'Classifier Training loss at epoch #{epoch}: {classifier_train_losses[-1]}')
#     print(f'Classifier Training accuracy at epoch #{epoch}: {classifier_train_accuracies[-1]}')
#     print(f'Skipped batches: {skipped_batches}')

#     print()
#     torch.cuda.empty_cache()

#     # Validation loop
#     ResNet_encoder.eval()
#     # projection_head.eval()
#     classification_head.eval()

#     epoch_classifier_validation_loss = 0
#     epoch_classifier_validation_accuracy = 0

#     skipped_batches = 0

#     for data in tqdm(ResNet_validationloader):
#     # for data in (ResNet_validationloader):
#         with torch.no_grad():

#             image = data['input'].to(device)
#             clean = data['clean'].to(device)
#             gt = data['gt'][:, 1].to(device)

#             gt = gt.unsqueeze(1)
#             mixed = torch.cat([image, clean])

#             current_batch_size = image.shape[0]

#             if torch.unique(gt).shape[0] == 1:
#                 skipped_batches += 1
#                 continue

#             labels = (torch.tensor([1] * current_batch_size + [0] * current_batch_size)).float().to(device)
#             oneHot_labels = (torch.tensor([[1, 0]] * current_batch_size + [[0, 1]] * current_batch_size)).float().to(device)

#             # z_mixed = ResNet_Model.forward(mixed)
#             out_dict = ResNet_encoder.forward(mixed)
#             # layer1 = out_dict['out1']
#             # layer2 = out_dict['out2']
#             # layer3 = out_dict['out3']
#             z_mixed = out_dict['out4']

#             # layer1 = F.interpolate(layer1, size=(32, 32, 32), mode='trilinear')
#             # layer2 = F.interpolate(layer2, size=(32, 32, 32), mode='trilinear')
#             # layer3 = F.interpolate(layer3, size=(32, 32, 32), mode='trilinear')
#             # layer4 = F.interpolate(z_mixed, size=(32, 32, 32), mode='trilinear')
#             # stacked_layers = torch.cat([layer1, layer2, layer3, layer4], dim=1)

#             # feature_diff = projection_head.forward(stacked_layers)
#             # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')

#             # feature_diff = stacked_layers[:batch_size] - stacked_layers[batch_size:]
#             # feature_diff = projection_head.forward(feature_diff)
#             # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128), mode='trilinear')

#             # feature_diff = z_mixed[:batch_size] - z_mixed[batch_size:]
#             # feature_diff = projection_head(feature_diff)
#             # feature_diff = F.interpolate(feature_diff, size=(128, 128, 128))
            
#             # input_diff = mixed * torch.cat([gt, gt])
#             # input_diff = image * gt
#             # input_diff = image - clean

#             z_mixed = torch.reshape(z_mixed, shape=(z_mixed.shape[0], -1))
#             y_mixed = classification_head.forward(z_mixed.detach())

#             classifier_loss = classifier_criterion(y_mixed, oneHot_labels)
#             epoch_classifier_validation_loss += classifier_loss.item()

#             epoch_classifier_validation_accuracy += determine_class_accuracy(y_mixed, oneHot_labels).cpu()

#             del image
#             del clean
#             del gt
#             del mixed
#             del labels
#             del z_mixed
#             # del feature_diff
#             # del input_diff
#             del y_mixed
#             del classifier_loss
    
#     classifier_validation_losses.append(epoch_classifier_validation_loss / ((len(ResNet_validationloader) - skipped_batches)))
#     classifier_validation_accuracies.append(epoch_classifier_validation_accuracy / ((len(ResNet_validationloader) - skipped_batches)))

#     print(f'Classifier Validation loss at epoch #{epoch}: {classifier_validation_losses[-1]}')
#     print(f'Classifier Validation accuracy at epoch #{epoch}: {classifier_validation_accuracies[-1]}')
#     print(f'Skipped batches: {skipped_batches}')

#     np.save(f'./results/{classifier_type}_{date}_accuracies.npy', [classifier_train_losses, classifier_validation_losses, classifier_train_accuracies, classifier_validation_accuracies])

#     if epoch % 5 == 0:
#         torch.save(classification_head.state_dict(), f'{model_path}{classifier_type}_{date}_ClassifierHead_state_dict{epoch}.pth')

#     print()

# torch.save(classification_head.state_dict(), f'{model_path}{classifier_type}_{date}_ClassifierHead_state_dict{number_of_epochs+51}.pth')

print()
print('Script executed.')
