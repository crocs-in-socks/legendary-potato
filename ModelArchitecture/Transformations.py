import torch
from scipy.ndimage import rotate
from skimage.exposure import rescale_intensity
import numpy as np
from torchvision import transforms

class RandomRotation(object):
    """Make a rotation of the volume's values.
    :param degrees: Maximum rotation's degrees.
    """

    def __init__(self, degrees, axis=0, labeled=True, segment=True):
        self.degrees = degrees
        self.labeled = labeled
        self.segment = segment
        self.order = 0 if self.segment == True else 5

    @staticmethod
    def get_params(degrees):  # Get random theta value for rotation
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        #print(sample['input'].shape)
        if len(sample['input'].shape) != 4:  # C x X_dim x Y_dim x Z_dim
            raise ValueError("Input of RandomRotation3D should be a 4 dimensionnal tensor.")

        angle = self.get_params(self.degrees)
        input_rotated = np.zeros(input_data.shape, dtype=input_data.dtype)
        
        gt_data = sample['gt'] if self.labeled else None
        gt_rotated = np.zeros(gt_data.shape, dtype=gt_data.dtype) if self.labeled else None

        # Rotation angle chosen at random and rotation happens only on XY plane for both image and label.
        for sh in range(input_data.shape[2]):
            input_rotated[:, :, sh, 0] = rotate(input_data[:, :, sh, 0], float(angle), reshape=False, order=self.order,
                                                mode='nearest')

            if self.labeled:
                gt_rotated[:, :, sh, 0] = rotate(gt_data[:, :, sh, 0], float(angle), reshape=False, order=self.order,
                                                 mode='nearest')
                gt_rotated[:, :, sh, 1] = rotate(gt_data[:, :, sh, 1], float(angle), reshape=False, order=self.order,
                                                 mode='nearest')
                gt_rotated = (gt_rotated > 0.6).astype(float)

        # Update the dictionary with transformed image and labels
        rdict['input'] = input_rotated
       
        if self.labeled:
            rdict['gt'] = gt_rotated
        sample.update(rdict)
        return sample

class RandomContrastMatching(object):
    def __init__(self):
        pass
    def __call__(self,sample):
        
        rdict = {}
        input_data = sample['input']
        input_data_t = sample['input_t']

        if(1):
            a,b = np.random.uniform(0.2,0.6),np.random.uniform(0.2,0.6)
            if(a>b):
                low = a
                high = b
            else:    
                low = b
                high = a

            rdict['input'] = rescale_intensity(input_data,(-0.1,1.2))
        sample.update(rdict)
        return sample


class ToTensor2Dslice(object):

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}


        input_data = sample['input']
        ret_input = input_data.transpose(2,0,1)
        ret_input = torch.from_numpy(ret_input).float()

        rdict['input'] = ret_input.float()


        if('input_t' in list(sample.keys())):
            input_data_t = sample['input_t']
            ret_input_t = input_data_t.transpose(2,0,1)
            ret_input_t = torch.from_numpy(ret_input_t).float()

            rdict['input_t'] = ret_input_t.float()


        if self.labeled:
            gt_data = sample['gt']
            if gt_data is not None:
                ret_gt = gt_data.transpose(2,0,1)
                ret_gt = torch.from_numpy(ret_gt).float()

                rdict['gt'] = ret_gt
            

            if('orig_gt' in list(sample.keys())):
                gt_data = sample['orig_gt']
                ret_gt = gt_data.transpose(2,0,1)
                ret_gt = torch.from_numpy(ret_gt).float()

                rdict['orig_gt'] = ret_gt




        sample.update(rdict)
        return sample

    


class RandomColorJitterslice(object):
    def __init__(self,):
        pass
    def __call__(self,sample):
        rdict = {}

        input_data = sample['input']

        jitter = transforms.ColorJitter(brightness=.2, hue=.2)

        rdict['input'] = jitter(input_data)

        sample.update(rdict)
        return sample


class RandomGaussianBlurslice(object):
    def __init__(self,):
        pass
    def __call__(self,sample):
        rdict = {}

        input_data = sample['input']

        blur = transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 3))

        rdict['input'] = blur(input_data)

        sample.update(rdict)
        return sample


class RandomNoiseslice(object):
    def __init__(self,):
        pass
    def __call__(self,sample):
        rdict = {}

        input_data = sample['input']

        noise = 0
        if(np.random.choice([0,1])):
            noise = np.random.randn(*(input_data.shape))
            noise = noise-noise.min()
            noise /= noise.max()


        noisy_image = input_data+np.random.choice([0,0.1,0.2,0.3,0.4,0.5])*(input_data>0.1)*noise
        
        rdict['input'] = noisy_image

        sample.update(rdict)
        return sample


class RandomIntensityChangesslice(object):
    def __init__(self,):
        pass
    def __call__(self,sample,p=0.5):
        rdict = {}
        input_data = sample['input']

        if(np.random.choice(2,1,p=[p,1-p])):
            mid = np.random.uniform(0,1)
            a,b = np.random.uniform(0,mid),np.random.uniform(mid,1)

            rdict['input'] = rescale_intensity(input_data,(a,b),(0.0,1.0))
        
        sample.update(rdict)
        return sample


class RandomRotation2Dslice(object):
    def __init__(self, degrees, axis=0, labeled=True, segment=True):
        self.degrees = degrees
        self.labeled = labeled
        self.segment = segment
        self.order = 0 if self.segment == True else 5

    @staticmethod
    def get_params(degrees):  # Get random theta value for rotation
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        angle = self.get_params(self.degrees)

        input_rotated = np.zeros(input_data.shape, dtype=input_data.dtype)

        gt_data = sample['gt'] if self.labeled else None
        gt_rotated = np.zeros(gt_data.shape, dtype=gt_data.dtype) if self.labeled else None


        # Rotation angle chosen at random and rotation happens only on XY plane for both image and label.

        if(np.random.choice([0,1])):
            input_rotated[:, :, 0] = rotate(input_data[:, :, 0], float(angle), reshape=False, order=self.order,
                                                    mode='nearest')


            if self.labeled:
                gt_rotated[:, :,0] = rotate(gt_data[:, :,0], float(angle), reshape=False, order=self.order,
                                                    mode='nearest')
                gt_rotated[:, :,1] = rotate(gt_data[:, :,1], float(angle), reshape=False, order=self.order,
                                                    mode='nearest')
                gt_rotated = (gt_rotated > 0.6).astype(float)
        else:
            input_rotated = input_data
            if(self.labeled):
                gt_rotated = gt_data


        # Update the dictionary with transformed image and labels
        rdict['input'] = input_rotated


        if self.labeled:
            rdict['gt'] = gt_rotated
        sample.update(rdict)
        return sample