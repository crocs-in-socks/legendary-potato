import glob
import torch
import numpy as np
from tqdm import tqdm
from skimage.morphology import binary_dilation
import matplotlib.pyplot as plt

# anomalous_trainset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TrainSet_5_11_23/*.npz'))
# anomalous_validationset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/ValidSet_5_11_23/*.npz'))
# anomalous_testset_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/TestSet_5_11_23/*.npz'))

data = np.load('../wmh_indexes.npy', allow_pickle=True).item()
anomalous_trainset_paths = data['train_names']
anomalous_validationset_paths = data['val_names']
anomalous_testset_paths = data['test_names']

print(f'Anomalous Trainset size: {len(anomalous_trainset_paths)}')
print(f'Anomalous Validationset size: {len(anomalous_validationset_paths)}')
print(f'Anomalous Testset size: {len(anomalous_testset_paths)}')

allset = anomalous_trainset_paths + anomalous_validationset_paths + anomalous_testset_paths

outer_struct_element = np.ones((7, 7, 7), dtype=bool)
inner_struct_element = np.ones((3, 3, 3), dtype=bool)

for idx, path in tqdm(enumerate(allset)):
    full_f = dict(np.load(path, allow_pickle=True))
    # image = full_f['data']
    # # clean = full_f['data_clean']
    # gt = full_f['label']

    # outer_dilated_gt = torch.from_numpy(binary_dilation(gt, outer_struct_element)).int()
    # inner_dilated_gt = torch.from_numpy(binary_dilation(gt, inner_struct_element)).int()
    # subtracted_gt = outer_dilated_gt - inner_dilated_gt

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(image[:, :, 64], cmap='gray')
    # plt.title(f'Sample #{idx+1}')
    # plt.subplot(1, 3, 2)
    # plt.imshow(gt[:, :, 64], cmap='gray')
    # plt.title(f'Original ground truth #{idx+1}')
    # plt.subplot(1, 3, 3)
    # plt.imshow(subtracted_gt[:, :, 64], cmap='gray')
    # plt.title(f'Dilated then subtracted mask #{idx+1}')
    # plt.savefig(f'./temporary/sample#{idx+1}')
    # plt.close()

    full_f['dilated_subtracted'] = None
    np.savez(path, **full_f)