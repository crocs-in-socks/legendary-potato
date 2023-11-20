import numpy as np
from torch.utils.data import ConcatDataset
import nibabel as nib
from ImageLoader.Ablation import SphereGeneration, LesionGeneration

np.random.seed(0)

clean_indexes = np.load('../NIMH_indexes.npy', allow_pickle=True).item()

test = LesionGeneration(
    paths=clean_indexes['data'],
    gt_paths=clean_indexes['brain_mask']
)
data, params = test[0]

for key, item in params.items():
    print(key, item)

# We want to fix the Validation set so we are having around 36 samples generated from the Lesion Generation which is available in 
# Abalation. We need to know the parameter.

# clean_indexes = np.load('../nimh_indexes.npy', allow_pickle=True).item()

# validation = ConcatDataset([LesionGeneration(clean_indexes['val_names_flair'],clean_indexes['val_masks_flair'],return_param=True) for i in range(12)])

# for i in range(36):
#     dict = {}
#     out,param = validation[i]
#     dict['data'] = out['input']
#     dict['label'] = out['gt']
#     dict['param_dict'] = param
#     np.savez('./ValidationSet/Validation_'+str(i)+'.npz', dict)