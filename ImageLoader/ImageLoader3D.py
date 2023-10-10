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
    def __init__(self,paths,gt_paths,image_size =128,type_of_imgs = 'numpy', transform=None):
        self.paths = paths
        self.gt_paths = gt_paths
        self.transform = transform
        self.image_size = image_size
        self.type_of_imgs = type_of_imgs

    def __len__(self,):
        return len(self.paths)
    def __get__(self,index):

        if(self.type_of_imgs == 'nifty'):
            image = nib.load(self.paths[index]).get_fdata()
            gt = nib.load(self.gt_paths[index]).get_fdata()
        elif(self.type_of_imgs == 'numpy'):
            full_f = np.load(self.paths[index])
            image = full_f['data']
            gt = full_f['label']

        image,img_crop_para = self.tight_crop_data(image)
        gt = gt[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]

        image = skiform.resize(image, (self.image_size, self.image_size, self.image_size), order = 1, preserve_range=True )
        gt = skiform.resize(gt, (self.image_size, self.image_size, self.image_size), order = 0, preserve_range=True )
        gt = gt > 0

        image -= np.min(image)
        image /= np.max(image)


        data_dict = {}
        data_dict['input'] = image
        data_dict['gt'] = gt

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