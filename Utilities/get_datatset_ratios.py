import torch
from torch.utils.data import DataLoader, ConcatDataset

from Utilities.Generic import Constants, load_dataset

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *

from tqdm import tqdm

c = Constants(
    batch_size = 8,
    patience = None,
    num_workers = 8,
    number_of_epochs = None,
    date = '08_12_2023',
    to_save_folder = None,
    to_load_folder = 'Dec08',
    device = 'cuda:1',
    proxy_type = None,
    train_task = None,
    encoder_load_path = 'VGGproxy_weightedBCE_wLRScheduler_simulated_lesions_on_brain_encoder_08_12_2023_state_dict_best_loss80.pth',
    projector_load_path = None,
    classifier_load_path = 'VGGproxy_weightedBCE_wLRScheduler_simulated_lesions_on_brain_classifier_08_12_2023_state_dict_best_loss80.pth',
    proxy_load_path = None,
    dataset = 'wmh',
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
allset = ConcatDataset([trainset, validationset, testset])

allloader = DataLoader(allset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

encoder = VGG3D_Encoder(input_channels=1).to(c.device)
classification_head = Classifier(input_channels=4096, output_channels=5, pooling_size=2).to(c.device)

encoder.load_state_dict(torch.load(c.encoder_load_path))
classification_head.load_state_dict(torch.load(c.classifier_load_path))

print()
print(f'Generating ratios for {c.dataset}.')
torch.cuda.empty_cache()

encoder.eval()
classification_head.eval()

prediction_list = []

for data in tqdm(allloader):
    image = data['input'].to(c.device)

    to_projector, to_classifier = encoder(image)
    prediction = classification_head(to_classifier)

    prediction_list.append(prediction.detach().cpu())

    del image
    del to_projector
    del to_classifier
    del prediction

prediction_list = torch.cat(prediction_list)
total_samples = prediction_list.shape[0]

healthy_samples = 0
very_small_lesions = 0
small_lesions = 0
medium_lesions = 0
large_lesions = 0

prediction_list[prediction_list > 0.5] = 1
prediction_list[prediction_list <= 0.5] = 0

for idx in range(total_samples):
    if prediction_list[idx, 0]:
        healthy_samples += 1
    if prediction_list[idx, 1]:
        very_small_lesions += 1
    if prediction_list[idx, 2]:
        small_lesions += 1
    if prediction_list[idx, 3]:
        medium_lesions += 1
    if prediction_list[idx, 4]:
        large_lesions += 1

print()
print(f'healthy sample ratio: {healthy_samples / total_samples}')
print(f'very small lesion ratio: {very_small_lesions / total_samples}')
print(f'small lesion ratio: {small_lesions / total_samples}')
print(f'medium lesion ratio: {medium_lesions / total_samples}')
print(f'large lesion ratio: {large_lesions / total_samples}')
print(f'total number of samples: {total_samples}')

print()
print('Script executed.')