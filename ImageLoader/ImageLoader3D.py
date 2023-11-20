import glob
import torch
from torch.utils.data import Dataset
import nibabel as nib
import skimage.transform as skiform

import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm


class ImageLoader3D(Dataset):
    def __init__(self, paths, gt_paths, image_size=128, type_of_imgs = 'numpy', transform=None, clean=False, subtracted=False):
        self.paths = paths
        self.gt_paths = gt_paths
        self.transform = transform
        self.image_size = image_size
        self.type_of_imgs = type_of_imgs
        self.clean = clean
        self.subtracted=subtracted

    def __len__(self,):
        return len(self.paths)
    
    def __getitem__(self, index):

        if(self.type_of_imgs == 'nifty'):
            image = nib.load(self.paths[index]).get_fdata()
            gt = nib.load(self.gt_paths[index]).get_fdata()
            if self.clean:
                clean = np.copy(image)
            if self.subtracted:
                subtracted = np.copy(gt)

        elif(self.type_of_imgs == 'numpy'):
            full_f = np.load(self.paths[index])
            image = full_f['data']
            gt = full_f['label']
            if self.clean:
                clean = full_f['data_clean']
            if self.subtracted:
                subtracted = full_f['dilated_subtracted']

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
            clean -= np.min(image)
            clean /= np.max(image)

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

        if(self.transform):
            data_dict = self.transform(data_dict)
        
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