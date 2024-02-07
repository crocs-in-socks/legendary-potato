import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from Utilities.Generic import *

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *
from ModelArchitecture.GANs import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

c = Constants(
    batch_size = 1,
    patience = 5,
    num_workers = 0,
    num_epochs = 200,
    date = '18_01_2024',
    to_save_folder = 'Jan18',
    to_load_folder = None,
    device = 'cuda:1',
    proxy_type = 'LiTS_Unet_preprocessing_0_>_200_window_init_features_64_with_median_filtering_cl_0_5',
    train_task = 'segmentation',
    to_load_encoder_path = None,
    to_load_projector_path = None,
    to_load_classifier_path = None,
    to_load_proxy_path = None,
    dataset = None
)

Gen_sim2real = CycleGAN_Generator(1, 1).to(c.device)
Gen_real2sim = CycleGAN_Generator(1, 1).to(c.device)
Dis_sim = CycleGAN_Discriminator(1).to(c.device)
Dis_real = CycleGAN_Discriminator(1).to(c.device)

Gen_sim2real.apply(init_weights)
Gen_real2sim.apply(init_weights)
Dis_sim.apply(init_weights)
Dis_real.apply(init_weights)

GAN_criterion = nn.MSELoss().to(c.device)
cycle_criterion = nn.L1Loss().to(c.device)
identity_criterion = nn.L1Loss().to(c.device)

Gen_optimizer = optim.Adam([*Gen_sim2real.parameters(), *Gen_real2sim.parameters()], lr=0.0002, betas=(0.5, 0.999))
Dis_sim_optimizer = optim.Adam(Dis_sim.parameters(), lr=0.0002, betas=(0.5, 0.999))
Dis_real_optimizer = optim.Adam(Dis_real.parameters(), lr=0.0002, betas=(0.5, 0.999))

Gen_scheduler = optim.lr_scheduler.LambdaLR(Gen_optimizer, lr_lambda=LambdaLR(c.num_epochs, 0, 100).step)
Dis_sim_scheduler = optim.lr_scheduler.LambdaLR(Dis_sim_optimizer, lr_lambda=LambdaLR(c.num_epochs, 0, 100).step)
Dis_real_scheduler = optim.lr_scheduler.LambdaLR(Dis_real_optimizer, lr_lambda=LambdaLR(c.num_epochs, 0, 100).step)

GAN_transforms = transforms.Compose([
    transforms.Normalize((0.5), (0.5))
])

simulated_trainset, simulated_validationset, simulated_testset = load_dataset('sim_2211_wmh', c.drive, transform=None)
real_trainset, real_validationset, real_testset = load_dataset('wmh', c.drive, transform=None)

simulated_trainset = Subset(simulated_trainset, list(range(len(real_trainset))))
# simulated_validationset = Subset(simulated_validationset, list(range(len(real_validationset))))
# simulated_testset = Subset(simulated_testset, list(range(len(real_testset))))

trainset = Translation_Dataset(simulated_trainset, real_trainset, GAN_transforms)
# validationset = Translation_Dataset(simulated_validationset, real_validationset, GAN_transforms)
# testset = Translation_Dataset(simulated_testset, real_testset, GAN_transforms)

trainloader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
# validationloader = DataLoader(validationset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)

train_loss_list = []
best_validation_loss = None
validation_loss_list = []

target_real = torch.ones(size=(c.batch_size, 1), requires_grad=False).float().to(c.device)
target_fake = torch.zeros(size=(c.batch_size, 1), requires_grad=False).float().to(c.device)
fake_sim_buffer = ReplayBuffer()
fake_real_buffer = ReplayBuffer()

for epoch in range(1, c.num_epochs+1):
    for idx, (sim, real) in enumerate(tqdm(trainloader)):
        sim = sim.to(c.device)
        real = real.to(c.device)

        Gen_optimizer.zero_grad()

        ### Identity loss
        # Gen_sim2real(real) should be equal to real
        same_real = Gen_sim2real(real)
        real_identity_loss = identity_criterion(same_real, real) * 5.0
        # Gen_real2sim(sim) should be equal to sim
        same_sim = Gen_real2sim(sim)
        sim_identity_loss = identity_criterion(same_sim, sim) * 5.0

        ### GAN loss
        fake_real = Gen_sim2real(sim)
        pred_fake = Dis_real(fake_real)
        sim2real_GAN_loss = GAN_criterion(pred_fake, target_real)

        fake_sim = Gen_real2sim(real)
        pred_fake = Dis_sim(fake_sim)
        real2sim_GAN_loss = GAN_criterion(pred_fake, target_real)

        ### Cycle loss
        recovered_sim = Gen_real2sim(fake_real)
        sim2real2sim_cycle_loss = cycle_criterion(recovered_sim, sim) * 10.0
        recovered_real = Gen_sim2real(fake_sim)
        real2sim2real_cycle_loss = cycle_criterion(recovered_real, real) * 10.0

        Gen_loss = sim_identity_loss + real_identity_loss + sim2real_GAN_loss + real2sim_GAN_loss + sim2real2sim_cycle_loss + real2sim2real_cycle_loss
        Gen_loss.backward()

        Gen_optimizer.step()

        ### Dis_sim
        Dis_sim_optimizer.zero_grad()
        pred_real = Dis_sim(sim)
        Dis_sim_real_loss = GAN_criterion(pred_real, target_real)
        
        fake_sim = fake_sim_buffer.push_and_pop(fake_sim)
        pred_fake = Dis_sim(fake_sim.detach())
        Dis_sim_fake_loss = GAN_criterion(pred_fake, target_fake)

        Dis_sim_loss = (Dis_sim_real_loss + Dis_sim_fake_loss) * 0.5
        Dis_sim_loss.backward()

        Dis_sim_optimizer.step()

        ### Dis_real
        Dis_real_optimizer.zero_grad()
        pred_real = Dis_real(real)
        Dis_real_real_loss = GAN_criterion(pred_real, target_real)
        
        fake_real = fake_real_buffer.push_and_pop(fake_real)
        pred_fake = Dis_real(fake_real.detach())
        Dis_real_fake_loss = GAN_criterion(pred_fake, target_fake)

        Dis_real_loss = (Dis_real_real_loss + Dis_real_fake_loss) * 0.5
        Dis_real_loss.backward()

        Dis_real_optimizer.step()

        print()
        print('Gen loss:', Gen_loss)
        print('Gen identity loss:', sim_identity_loss+real_identity_loss)
        print('Gen GAN loss:', sim2real_GAN_loss+real2sim_GAN_loss)
        print('Gen cycle loss:', sim2real2sim_cycle_loss+real2sim2real_cycle_loss)
        print('Dis loss:', Dis_sim_loss+Dis_real_loss)
        print()

        plt.figure(figsize=(20, 16))
        plt.subplot(1, 4, 1)
        plt.imshow(sim[0, 0, :, :, 64].detach().cpu())
        plt.title('Simulated')
        plt.subplot(1, 4, 2)
        plt.imshow(real[0, 0, :, :, 64].detach().cpu())
        plt.title('Real')
        plt.subplot(1, 4, 3)
        plt.imshow(fake_sim[0, 0, :, :, 64].detach().cpu())
        plt.title('Fake sim')
        plt.subplot(1, 4, 4)
        plt.imshow(fake_real[0, 0, :, :, 64].detach().cpu())
        plt.title('Fake real')

        plt.savefig('./temp')
        plt.close()

    Gen_scheduler.step()
    Dis_sim_scheduler.step()
    Dis_real_scheduler.step()

    torch.save(Gen_sim2real.state_dict(), )