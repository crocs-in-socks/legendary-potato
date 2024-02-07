import json
import glob
import torch
import torchvision
from torch.utils.data import Dataset
import nibabel as nib
import skimage.transform as skiform
from skimage.morphology import binary_dilation
from skimage.restoration import denoise_bilateral

import cv2
import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import exposure
from scipy.signal import medfilt
from skimage.filters import unsharp_mask

class PreprocessedImageLoader3D(Dataset):
    def __init__(self, paths, gt_paths, np_data_paths=None, type_of_imgs='numpy', transform=None):
        self.paths = paths
        self.gt_paths = gt_paths
        self.np_data_paths = np_data_paths
        self.type_of_imgs = type_of_imgs
        self.transform = transform

    def __len__(self,):
        return len(self.paths)
    
    def __getitem__(self, index):
        # if self.type_of_imgs == 'nifty':
        #     image = nib.load(self.paths[index]).get_fdata()
        #     gt = nib.load(self.gt_paths[index]).get_fdata()

        data_dict = (np.load(self.np_data_paths[index], allow_pickle=True)).item()
        # print(type(data_dict))
        # og_dims = data_dict[-2]
        # print(og_dims)
        
        # image = torch.from_numpy(image).float()
        # image = image.squeeze(0)
        # gt = torch.from_numpy(gt).float()
        # gt = gt.squeeze(0)

        # data_dict = {}
        data_dict['input'] = data_dict['input'].squeeze(0)
        data_dict['gt'] = data_dict['gt'].squeeze(0)

        # if self.transform:
        #     data_dict = self.transform(data_dict)
        
        return data_dict

class Translation_Dataset(Dataset):
    def __init__(self, domainA_dataset, domainB_dataset, transform=None):
        self.domainA_dataset = domainA_dataset
        self.domainB_dataset = domainB_dataset
        self.transform = transform

        assert len(domainA_dataset) == len(domainB_dataset), "Datasets must have the same length"
    
    def __len__(self,):
        return len(self.domainA_dataset)
    
    def __getitem__(self, index):
        domainA_sample = self.domainA_dataset[index]
        domainB_sample = self.domainB_dataset[index]

        if self.transform:
            
            domainA_sample = torch.permute(domainA_sample, dims=(3, 0, 1, 2))
            domainB_sample = torch.permute(domainB_sample, dims=(3, 0, 1, 2))
            domainA_sample = self.transform(domainA_sample)
            domainB_sample = self.transform(domainB_sample)

        return domainA_sample, domainB_sample

class VolumeSliceDataset(Dataset):
    def __init__(self, original_datatset, slice_axis=-1):
        self.original_datatset = original_datatset
        self.slice_axis = slice_axis
    
    def __len__(self):
        num_volumes = len(self.original_datatset)
        num_slices_per_volume = self.original_datatset[0]['input'].shape[self.slice_axis]
        return num_volumes*num_slices_per_volume
    
    def __getitem__(self, index):
        volume_index = index // self.original_datatset[0]['input'].shape[self.slice_axis]
        slice_index = index % self.original_datatset[0]['input'].shape[self.slice_axis]

        image_volume = self.original_datatset[volume_index]['input']
        image_slice = torch.index_select(image_volume, dim=self.slice_axis, index=torch.tensor(slice_index)).squeeze(-1)

        gt_volume = self.original_datatset[volume_index]['gt']
        gt_slice = torch.index_select(gt_volume, dim=self.slice_axis, index=torch.tensor(slice_index)).squeeze(-1)

        data_dict = {}
        data_dict['input'] = image_slice
        data_dict['gt'] = gt_slice

        return data_dict

class ImageLoader2D(Dataset):
    def __init__(self, paths, gt_paths, json_paths=None, image_size=128, type_of_imgs='png', recon=False, no_crop=False, transform=None, data='busi'):
        self.paths = paths
        self.gt_paths = gt_paths
        self.json_paths = json_paths
        self.transform = transform
        self.image_size=  image_size
        self.type_of_imgs = type_of_imgs

    def __len__(self,):
        return len(self.paths)
    
    def __getitem__(self, index):
        image = torchvision.io.read_image(self.paths[index], torchvision.io.ImageReadMode.GRAY).float()
        gt = torchvision.io.read_image(self.gt_paths[index], torchvision.io.ImageReadMode.GRAY).float()

        og_dims = image.shape
        og_gt = torch.concat([gt==0, gt>0], 0).float()

        image = torchvision.transforms.functional.resize(image, size=(self.image_size, self.image_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        gt = torchvision.transforms.functional.resize(gt, size=(self.image_size, self.image_size),interpolation=torchvision.transforms.InterpolationMode.NEAREST,antialias=True)

        gt = gt > 0

        image -= image.min()
        image /= image.max()

        gt = torch.concat([gt==0, gt>0], -1).float()

        data_dict = {}
        data_dict['input'] = image
        data_dict['gt'] = gt

        data_dict['og_dims'] = og_dims
        data_dict['og_gt'] = og_gt

        return data_dict

class ImageLoader3D(Dataset):
    def __init__(self, paths, gt_paths, json_paths=None, image_size=128, type_of_imgs = 'numpy', transform=None, clean=False, subtracted=False, is_clean=False, window=None, ahe=False, median_filter=None):
        self.paths = paths
        self.gt_paths = gt_paths
        self.json_paths = json_paths
        self.transform = transform
        self.image_size = image_size
        self.type_of_imgs = type_of_imgs
        self.clean = clean
        self.subtracted=subtracted
        self.is_clean = is_clean
        self.window = window
        self.liver_preprocess=median_filter

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

        og_dims = image.shape
        og_gt = np.stack([gt==0, gt>0], 0).astype(np.single)
        
        if self.window is not None:
            image[gt == 0] = self.window[0]
            image[image < self.window[0]] = self.window[0]
            image[image > self.window[1]] = self.window[0]
        
            # image -= np.min(image)
            # image /= np.max(image)

        image, img_crop_para = self.tight_crop_data(image)
        if self.clean:
            clean = clean[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]

        gt = gt[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]

        if self.liver_preprocess:

            # image = medfilt(image, kernel_size=5)

            # for z in range(image.shape[-1]):
            #     image[:, :, z] = denoise_bilateral(image[:, :, z])

            # image = unsharp_mask(image, preserve_range=True)
            # image = np.clip(image, 0, 1)

            # non_zero_indices = np.ndarray.nonzero(image)
            # image[non_zero_indices] = exposure.equalize_adapthist(image[non_zero_indices], clip_limit=0.005)

            # image = image.astype(np.float32)
            # image = (image * 255).astype(np.uint8)
            image = (image).astype(np.uint8)
            # uint8_image = (image * 255).astype(np.uint8)

            image = cv2.medianBlur(image, ksize=5)
            # for z in range(image.shape[-1]):
                # image[:, :, z] = cv2.bilateralFilter(image[:, :, z], d=7, sigmaColor=50, sigmaSpace=50)

            # uint8_image = (image * 255).astype(np.uint8)

            clahe = cv2.createCLAHE(clipLimit=0.05)
            for z in range(image.shape[-1]):
                # image[:, :, z] = clahe.apply(uint8_image[:, :, z])
                image[:, :, z] = clahe.apply(image[:, :, z])
                
            # image = image.astype(np.float32) / 255.0
            image = image.astype(float)
            gt = (gt==2).astype(float)

        if self.subtracted:
            subtracted = subtracted[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]

        image = skiform.resize(image, (self.image_size, self.image_size, self.image_size), order = 1, preserve_range=True)
        if self.clean:
            clean = skiform.resize(clean, (self.image_size, self.image_size, self.image_size), order = 1, preserve_range=True)
        gt = skiform.resize(gt, (self.image_size, self.image_size, self.image_size), order = 0, preserve_range=True)

        if self.subtracted:
            subtracted = skiform.resize(subtracted, (self.image_size, self.image_size, self.image_size), order = 0, preserve_range=True)
            subtracted = subtracted > 0

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

        image -= np.min(image)
        image /= np.max(image)

        data_dict = {}
        
        data_dict['input'] = image
        data_dict['gt'] = gt

        data_dict['crop_para'] = img_crop_para
        data_dict['og_dims'] = og_dims
        data_dict['og_gt'] = og_gt

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