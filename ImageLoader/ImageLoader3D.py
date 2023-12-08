import json
import glob
import torch
from torch.utils.data import Dataset
import nibabel as nib
import skimage.transform as skiform
from skimage.morphology import binary_dilation

import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
import matplotlib.pyplot as plt


class ImageLoader3D(Dataset):
    def __init__(self, paths, gt_paths, json_paths=None, image_size=128, type_of_imgs = 'numpy', transform=None, clean=False, subtracted=False, is_clean=False):
        self.paths = paths
        self.gt_paths = gt_paths
        self.json_paths = json_paths
        self.transform = transform
        self.image_size = image_size
        self.type_of_imgs = type_of_imgs
        self.clean = clean
        self.subtracted=subtracted
        self.is_clean = is_clean

    def __len__(self,):
        return len(self.paths)
    
    def __getitem__(self, index):

        if(self.type_of_imgs == 'nifty'):
            image = nib.load(self.paths[index]).get_fdata()
            gt = nib.load(self.gt_paths[index]).get_fdata()
            if self.clean:
                clean = np.copy(image)
            if self.subtracted:
                if self.is_clean:
                    subtracted = np.copy(gt)
                else:
                    outer_struct_element = np.ones((7, 7, 7), dtype=bool)
                    inner_struct_element = np.ones((3, 3, 3), dtype=bool)
                    outer_dilated_gt = binary_dilation(gt, outer_struct_element).astype(int)
                    inner_dilated_gt = binary_dilation(gt, inner_struct_element).astype(int)
                    subtracted = outer_dilated_gt - inner_dilated_gt
            if self.json_paths:
                with open(self.json_paths[index], 'r') as file:
                    metadata = json.load(file)
                file.close()

        elif(self.type_of_imgs == 'numpy'):

            try:
                full_f = np.load(self.paths[index])
                image = full_f['data']
                gt = full_f['label']
                if self.clean:
                    clean = full_f['data_clean']
                if self.subtracted:
                    subtracted = full_f['dilated_subtracted']
                if self.json_paths:
                    with open(self.json_paths[index], 'r') as file:
                        metadata = json.load(file)
                    file.close()
            except Exception as e:
                print(e)
                print(self.paths[index])
                exit(0)

        image, img_crop_para = self.tight_crop_data(image)
        if self.clean:
            clean = clean[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
        gt = gt[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
        if self.subtracted:
            subtracted = subtracted[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]

        image = skiform.resize(image, (self.image_size, self.image_size, self.image_size), order = 1, preserve_range=True)
        if self.clean:
            clean = skiform.resize(clean, (self.image_size, self.image_size, self.image_size), order = 1, preserve_range=True)
        gt = skiform.resize(gt, (self.image_size, self.image_size, self.image_size), order = 0, preserve_range=True)
        gt = gt > 0
        if self.subtracted:
            subtracted = skiform.resize(subtracted, (self.image_size, self.image_size, self.image_size), order = 0, preserve_range=True)
            subtracted = subtracted > 0

        image -= np.min(image)
        image /= np.max(image)

        if self.clean:
            masked_image = image * ~gt

            masked_min = np.min(masked_image)
            masked_max = np.max(masked_image)
            clean_min = np.min(clean)
            clean_max = np.max(clean)

            clean = ((clean - clean_min) / (clean_max - clean_min)) * (masked_max - masked_min) + masked_min

        image = np.expand_dims(image, -1).astype(np.single)
        if self.clean:
            clean = np.expand_dims(clean, -1).astype(np.single)
        gt = np.stack([gt==0, gt>0], -1).astype(np.single)
        if self.subtracted:
            subtracted = np.expand_dims(subtracted, -1).astype(np.single)

        data_dict = {}
        
        data_dict['input'] = image
        data_dict['gt'] = gt

        if self.clean:
            data_dict['clean'] = clean
        if self.subtracted:
            data_dict['subtracted'] = subtracted

        if self.transform:
            data_dict = self.transform(data_dict)

        data_dict['lesion_labels'] = torch.tensor([1, 0, 0, 0, 0])
        if self.json_paths:
            for lesion_idx in range(metadata['num_lesions']):
                if metadata[f'{lesion_idx}_semi_axes_range'] == [2, 5]:
                    data_dict['lesion_labels'][0] = 0
                    data_dict['lesion_labels'][1] = 1
                if metadata[f'{lesion_idx}_semi_axes_range'] == [3, 5]:
                    data_dict['lesion_labels'][0] = 0
                    data_dict['lesion_labels'][2] = 1
                if metadata[f'{lesion_idx}_semi_axes_range'] == [5, 10]:
                    data_dict['lesion_labels'][0] = 0
                    data_dict['lesion_labels'][3] = 1
                if metadata[f'{lesion_idx}_semi_axes_range'] == [10, 15]:
                    data_dict['lesion_labels'][0] = 0
                    data_dict['lesion_labels'][4] = 1
        
        return data_dict
    
    def cut_zeros1d(self, im_array):
        '''
     Find the window for cropping the data closer to the brain
     :param im_array: input array
     :return: starting and end indices, and length of non-zero intensity values
        '''

        im_list = list(im_array > 0)
        start_index = im_list.index(1)
        end_index = im_list[::-1].index(1)
        length = len(im_array[start_index:]) - end_index
        return start_index, end_index, length

    def tight_crop_data(self, img_data):
        '''
     Crop the data tighter to the brain
     :param img_data: input array
     :return: cropped image and the bounding box coordinates and dimensions.
        '''

        row_sum = np.sum(np.sum(img_data, axis=1), axis=1)
        col_sum = np.sum(np.sum(img_data, axis=0), axis=1)
        stack_sum = np.sum(np.sum(img_data, axis=1), axis=0)
        rsid, reid, rlen = self.cut_zeros1d(row_sum)
        csid, ceid, clen = self.cut_zeros1d(col_sum)
        ssid, seid, slen = self.cut_zeros1d(stack_sum)
        return img_data[rsid:rsid + rlen, csid:csid + clen, ssid:ssid + slen], [rsid, rlen, csid, clen, ssid, slen]