import torch
import numpy as np
from torch.utils.data import Dataset

class ContrastiveLoader3D(Dataset):
    def __init__(self, feature_maps, transform=None):
        self.feature_maps = feature_maps
        self.transform = transform
    
    def __len__(self):
        return self.feature_maps.shape[0]

    def __getitem__(self, index):
        data = self.feature_maps[index]
        if self.transform:
            data = self.transform(data)
        return data
    
class ContrastivePatchLoader3D(Dataset):
    def __init__(self, sample_paths, oneHot_sample_labels, device='cuda:1', transform=None):
        self.sample_paths = sample_paths
        self.oneHot_sample_labels = oneHot_sample_labels
        self.device = device
        self.transform = transform
    
    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        data_dict = np.load(self.sample_paths[index], allow_pickle=True).item()
        data = data_dict['patches']

        transformed_patches = []
        number_of_patches = data.shape[1]

        if self.transform:
            for patch_idx in range(number_of_patches):
                transformed_patches.append(self.transform(data[0, patch_idx].cpu()))
        else:
            for patch_idx in range(number_of_patches):
                transformed_patches.append(data[0, patch_idx])

        transformed_patches = torch.stack(transformed_patches).to(self.device)
        oneHot_patch_labels = torch.tensor(self.oneHot_sample_labels[index]).to(self.device).float()

        return (transformed_patches, oneHot_patch_labels)

# class ContrastiveLoader3D(Dataset):
#     def __init__(self, feature_maps, transform1=None, transform2=None):
#         self.feature_maps = feature_maps
#         self.transform1 = transform1
#         self.transform2 = transform2
    
#     def __len__(self):
#         return self.feature_maps.shape[0]

#     def __getitem__(self, index):
#         data = self.feature_maps[index]
#         x1 = self.transform1(data)
#         x2 = self.transform2(data)

#         return x1, x2