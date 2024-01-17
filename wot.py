import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Utilities.Generic import *

from ModelArchitecture.DUCK_Net import *
from ModelArchitecture.Encoders import *
from ModelArchitecture.UNet import *

from ModelArchitecture.Transformations import *
from ModelArchitecture.Losses import *
from ModelArchitecture.metrics import *

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

model = UNet(out_channels=2, init_features=16)
print(sum(p.numel() for p in model.parameters()) / 10e6)

from hypnettorch.mnets import UNet