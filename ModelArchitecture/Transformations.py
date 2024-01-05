from typing import Any
import torch
from scipy.ndimage import rotate, gaussian_filter, zoom
from skimage.exposure import rescale_intensity
import numpy as np
from torchvision import transforms

################################################ FOR 2D Transformations ##########################################################################3##############

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


class RandomIntensityChanges(object):
    def __init__(self, clean=False):
        self.clean = clean
    
    def __call__(self,sample,p=0.5):
        rdict = {}
        input_data = sample['input']
        if self.clean:
            clean_data = sample['clean']

        if(np.random.choice(2,1,p=[p,1-p])):
            mid = np.random.uniform(0,1)
            a,b = np.random.uniform(0,mid),np.random.uniform(mid,1)

            rdict['input'] = rescale_intensity(input_data, (a, b), (0.0, 1.0))
            if self.clean:
                rdict['clean'] = rescale_intensity(clean_data, (a, b), (0.0, 1.0))
        
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
    

################################################################## For 3D transformations #####################################################################

class ToTensor3D(object):
    """Convert a PIL image or numpy array to a PyTorch tensor."""

    def __init__(self, labeled=True, clean=False, subtracted=False):
        self.labeled = labeled
        self.clean = clean
        self.subtracted = subtracted

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        ret_input = input_data.transpose(3, 0, 1, 2)  # Pytorch supports N x C x X_dim x Y_dim x Z_dim
        ret_input = torch.from_numpy(ret_input).float()

        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']
            if gt_data is not None:
                ret_gt = gt_data.transpose(3, 0, 1, 2)  # Pytorch supports N x C x X_dim x Y_dim x Z_dim
                ret_gt = torch.from_numpy(ret_gt).float()

                rdict['gt'] = ret_gt

        if self.clean:
            clean_data = sample['clean']
            if clean_data is not None:
                ret_clean = clean_data.transpose(3, 0, 1, 2) # Pytorch supports N x C x X_dim x Y_dim x Z_dim
                ret_clean = torch.from_numpy(ret_clean).float()

                rdict['clean'] = ret_clean
        
        if self.subtracted:
            subtracted_data = sample['subtracted']
            if subtracted_data is not None:
                ret_subtracted = subtracted_data.transpose(3, 0, 1, 2) # Pytorch supports N x C x X_dim x Y_dim x Z_dim
                ret_subtracted = torch.from_numpy(ret_subtracted).float()

                rdict['subtracted'] = ret_subtracted
        
        sample.update(rdict)
        return sample

class RandomRotation3D(object):
    """Make a rotation of the volume's values.
    :param degrees: Maximum rotation's degrees.
    """

    def __init__(self, degrees, axis=0, labeled=True, segment=True, clean=False, subtracted=False):
        self.degrees = degrees
        self.labeled = labeled
        self.segment = segment
        self.clean = clean
        self.subtracted = subtracted
        self.order = 0 if self.segment == True else 5

    @staticmethod
    def get_params(degrees):  # Get random theta value for rotation
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        if len(sample['input'].shape) != 4:  # C x X_dim x Y_dim x Z_dim
            raise ValueError("Input of RandomRotation3D should be a 4 dimensionnal tensor.")

        angle = self.get_params(self.degrees)

        input_rotated = np.zeros(input_data.shape, dtype=input_data.dtype)

        gt_data = sample['gt'] if self.labeled else None
        gt_rotated = np.zeros(gt_data.shape, dtype=gt_data.dtype) if self.labeled else None

        clean_data = sample['clean'] if self.clean else None
        clean_rotated = np.zeros(clean_data.shape, dtype=clean_data.dtype) if self.clean else None

        subtracted_data = sample['subtracted'] if self.subtracted else None
        subtracted_rotated = np.zeros(subtracted_data.shape, dtype=subtracted_data.dtype) if self.subtracted else None

        # Rotation angle chosen at random and rotation happens only on XY plane for both image and label.
        for sh in range(input_data.shape[2]):
            input_rotated[:, :, sh, 0] = rotate(input_data[:, :, sh, 0], float(angle), reshape=False, order=self.order, mode='nearest')

            if self.labeled:
                gt_rotated[:, :, sh, 0] = rotate(gt_data[:, :, sh, 0], float(angle), reshape=False, order=self.order, mode='nearest')
                gt_rotated[:, :, sh, 1] = rotate(gt_data[:, :, sh, 1], float(angle), reshape=False, order=self.order, mode='nearest')
                gt_rotated = (gt_rotated > 0.6).astype(float)
            
            if self.clean:
                clean_rotated[:, :, sh, 0] = rotate(clean_data[:, :, sh, 0], float(angle), reshape=False, order=self.order, mode='nearest')
            
            if self.subtracted:
                subtracted_rotated[:, :, sh, 0] = rotate(subtracted_data[:, :, sh, 0], float(angle), reshape=False, order=self.order, mode='nearest')

        # Update the dictionary with transformed image and labels
        rdict['input'] = input_rotated

        if self.labeled:
            rdict['gt'] = gt_rotated
        if self.clean:
            rdict['clean'] = clean_rotated
        if self.subtracted:
            rdict['subtracted'] = subtracted_rotated

        sample.update(rdict)
        return sample

class RandomFlip3D(object):

    def __init__(self, axis=0, labeled=True, p=0.5):
        self.axis = axis
        self.labeled = labeled
        self.p = p
    
    @staticmethod
    def get_params(p):
        probability = np.random.uniform(0, 1)
        return probability < p

    def __call__(self, sample):
        if self.get_params(self.p):
            # print(f'Flipped axis {self.axis}')
            # print(f'Flip sample type: {type(sample)}')
            # print(f'Flip sample shape: {sample.shape}')
            if self.axis == 0:
                input_rotated = rotate(sample, angle=180, axes=(1, 2), reshape=False)
            elif self.axis == 1:
                input_rotated = rotate(sample, angle=180, axes=(0, 2), reshape=False)
            elif self.axis == 2:
                input_rotated = rotate(sample, angle=180, axes=(0, 1), reshape=False)
            input_rotated = torch.from_numpy(input_rotated)
            return input_rotated
        else:
            return sample

class RandomGaussianBlur3D(object):

    def __init__(self, p=0.5):
        self.p = p
        pass

    @staticmethod
    def get_params(p):
        probability = np.random.uniform(0, 1)
        return probability < p

    def __call__(self, sample):
        if self.get_params(self.p):

            # print('Blurred')

            number_of_channels = sample.shape[0]
            blurred_sample = []
            for channel in range(number_of_channels):
                blur = gaussian_filter(sample[channel], sigma=(1.5))
                blurred_sample.append(blur)
            blurred_sample = np.stack(blurred_sample)
            blurred_sample = torch.from_numpy(blurred_sample)
            return blurred_sample
        else:
            return sample

class RandomCropResize3D(object):

    def __init__(self, p=0.5, scale_factor=2):
        self.p = p
        self.scale_factor = scale_factor

    @staticmethod
    def get_params(p):
        probability = np.random.uniform(0, 1)
        return probability < p
    
    def __call__(self, sample):
        if self.get_params(self.p):
            sample_size = sample.shape
            number_of_channels = sample.shape[0]
            crop_size = [sample.shape[-1] // self.scale_factor] * 3

            x = np.random.randint(0, sample_size[1] - crop_size[0] + 1)
            y = np.random.randint(0, sample_size[2] - crop_size[1] + 1)
            z = np.random.randint(0, sample_size[3] - crop_size[2] + 1)

            # print(x, y, z)

            cropped_sample = sample[:, x:x+crop_size[0], y:y+crop_size[1], z:z+crop_size[2]]
            # print(f'Cropped sample shape: {cropped_sample.shape}')
            resized_sample = []
            for channel in range(number_of_channels):
                resize = zoom(cropped_sample[channel], zoom=self.scale_factor)
                resized_sample.append(resize)
            resized_sample = np.stack(resized_sample)
            resized_sample = torch.from_numpy(resized_sample)
            # print(f'Resized sample shape: {resized_sample.shape}')
            return resized_sample
        else:
            return sample
    
class RandomIntensityChanges3D(object):

    def __init__(self, p=0.5):
        self.p = p
    
    @staticmethod
    def get_params(p):
        probability = np.random.uniform(0, 1)
        return probability < p

    def __call__(self, sample):
        if self.get_params(self.p):
            mid = np.random.uniform(0, 1)
            a, b = np.random.uniform(0, mid), np.random.uniform(mid, 1)

            if not isinstance(sample, np.ndarray):
                # print(f'RandomIntensity sample type: {type(sample)}')
                sample = sample.cpu()
                sample = sample.numpy()

            rescaled_sample = rescale_intensity(sample, (a, b), (0.0, 1.0))
            rescaled_sample = torch.from_numpy(rescaled_sample)

            return rescaled_sample
        else:
            return sample