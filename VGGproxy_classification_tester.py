import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    num_epochs = None,
    date = '08_12_2023',
    to_save_folder = None,
    to_load_folder = 'Dec08',
    device = 'cuda:1',
    proxy_type = 'VGGproxy',
    train_task = None,
    to_load_encoder_path = 'VGGproxy_weightedBCE_wLRScheduler_simulated_lesions_on_brain_encoder_08_12_2023_state_dict_best_loss80.pth',
    to_load_projector_path = None,
    to_load_classifier_path = 'VGGproxy_weightedBCE_wLRScheduler_simulated_lesions_on_brain_classifier_08_12_2023_state_dict_best_loss80.pth',
    to_load_proxy_path = None,
    dataset = 'sim_1000+sim_2211',
)

trainset, validationset, testset = load_dataset(c.dataset, c.drive, ToTensor3D(labeled=True))
testloader = DataLoader(testset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

encoder = VGG3D_Encoder(input_channels=1).to(c.device)
classification_head = Classifier(input_channels=4096, output_channels=5, pooling_size=2).to(c.device)

if c.to_load_encoder_path is not None:
    encoder.load_state_dict(torch.load(c.encoder_load_path))
if c.to_load_classifier_path is not None:
    classification_head.load_state_dict(torch.load(c.classifier_load_path))

class_weights = torch.tensor([1, 2, 2, 1, 1]).float().to(c.device)
classification_criterion = nn.BCELoss(weight=class_weights).to(c.device)

print()
print('Testing Proxy.')

# Test loop
encoder.eval()
classification_head.eval()

classification_test_loss = 0
test_accuracy = 0

for data in tqdm(testloader):
    image = data['input'].to(c.device)
    oneHot_label = data['lesion_labels'].float().to(c.device)

    to_projector, to_classifier = encoder(image)
    prediction = classification_head(to_classifier)

    classification_loss = classification_criterion(prediction, oneHot_label)
    classification_test_loss += classification_loss.item()
    test_accuracy += determine_multiclass_accuracy(prediction, oneHot_label).cpu()

    del image
    del oneHot_label
    del to_projector
    del to_classifier
    del prediction
    del classification_loss

classification_test_loss = (classification_test_loss / len(testloader))
test_accuracy = (test_accuracy / len(testloader))
print(f'Classification validation loss: {classification_test_loss}')
print(f'Classification validation accuracy: {test_accuracy}')

print()
print('Script executed.')