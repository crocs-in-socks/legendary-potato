import skimage
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import skimage.morphology
import skimage.transform as skiform
from scipy.ndimage import gaussian_filter

# For SpehereGeneration:
# - Use seg_label
# 1. Just run it.

# For LesionGeneration:
# - Use brain_mask
# 2. self.have_smoothing = False, have_noise = False, have_small = False
# 3. self.have_smoothing = True, have_noise = False, have_small = False
# 4. self.have_smoothing = True, have_noise = True, have_small = False
# 5. self.have_smoothing = True, have_noise = True, have_small = True

# For Simple Sphere
class SphereGeneration(Dataset):
    def __init__(self, paths, gt_paths=None, transform=None):
        super().__init__()
        self.paths = paths
        self.mask_path = gt_paths
        self.transform = transform

    def sphere(self, centroid, size=64, radius=10):
        xx, yy, zz = np.mgrid[-size:size, -size:size, -size:size]
        circle = (xx - centroid[0] + size) ** 2 + (yy - centroid[1] + size) ** 2 + (zz - centroid[2] + size) ** 2 - radius**2
        mask = (circle < 0)
        return mask

    def lesion_simulation(self, image, brain_mask_img, num_les=3):

        roi = skimage.morphology.binary_erosion(brain_mask_img, skimage.morphology.ball(10))*(image>0.1)

        # Generating centroids within the roi generated above
        x_corr, y_corr, z_corr = np.nonzero(roi[:,:,:])
        centroid_list = []
        for d in range(num_les):
            random_coord_index = np.random.choice(len(x_corr), 1)
            centroid_main = np.array([x_corr[random_coord_index], y_corr[random_coord_index], z_corr[random_coord_index]])
            centroid_list.append(centroid_main)
        
        # Generating spheres and combining the masks
        mask_total = np.zeros_like(image)
        for i in range(num_les):
            radius = np.random.randint(5, 18)
            mask = self.sphere(centroid_list[i], 64, radius)
            mask_total = np.logical_or(mask,mask_total)

        # sumout = np.sum(np.sum(mask_total, axis=0), axis=0)
        # slide_no = np.where(sumout == np.amax(sumout))[0][0]

        alpha = np.random.uniform(0.5, 0.8)
        beta = 1-alpha

        image = alpha*image*(1-mask_total) + beta*mask_total
        image -= image.min()
        image /= image.max()
        
        # plt.imshow(image[:,:,slide_no])
        # plt.show()
    
        return image, mask_total
        

    def __getitem__(self, index):

        # Reading the nifty image and brain mask
        nii_img = nib.load(self.paths[index]).get_fdata()
        image, img_crop_para = self.tight_crop_data(nii_img)
        image = skiform.resize(image, (128, 128, 128), order=1, preserve_range=True)
        image -= image.min()
        image /= image.max() + 1e-7

        if self.mask_path :
            brain_mask_img = nib.load(self.mask_path[index]).get_fdata()
            brain_mask_img = brain_mask_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1],img_crop_para[2]:img_crop_para[2] + img_crop_para[3],img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            brain_mask_img = skiform.resize(brain_mask_img, (128, 128, 128), order=0, preserve_range=True)

        else:
            # In case we don't have brain mask, we can close out the holes in the mask (the ventricles)
            brain_mask_img = skimage.morphology.binary_closing(image>0.05,skimage.morphology.ball(3))

        # Random number of lesions generated 
        number_les = np.random.randint(1, 5)
        image,label = self.lesion_simulation(image,brain_mask_img, number_les)

        image = np.expand_dims(image, -1).astype(np.single)
        gt = np.stack([label==0, label>0], -1).astype(np.single)

        data_dict = {}
        data_dict['input'] = image
        data_dict['gt'] = gt

        if(self.transform):
            data_dict = self.transform(data_dict)
        
        return data_dict

    def __len__(self):
        return len(self.paths)

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

# For Shapes(L+M) + flat
class LesionGeneration(Dataset):
    def __init__(self, paths, gt_paths = None, type_of_imgs='nifty', have_noise=True, have_smoothing=True, have_small=True, have_edema=False, return_param=True, transform=None, size=128):
        self.paths = paths
        self.mask_path = gt_paths
        self.transform = transform
        self.size = size
        self.have_noise = have_noise
        self.have_smoothing = have_smoothing
        self.have_small = have_small
        self.have_edema = have_edema
        self.img_type = type_of_imgs
        self.return_param = return_param
        

    def ellipsoid(self,coord=(1,2,1),semi_a = 4, semi_b = 34, semi_c = 34, alpha=np.pi/4, beta=np.pi/4, gamma=np.pi/4,img_dim=64):
        x = y = z = np.linspace(-img_dim,img_dim,img_dim*2)
        x,y,z = np.meshgrid(x,y,z)

        # Take the centering into effect   
        x=(x - coord[0] + img_dim)
        y=(y - coord[1] + img_dim)
        z=(z - coord[2] + img_dim)

        ellipsoid_std_axes = np.stack([x,y,z],0)

        alpha = -alpha
        beta = -beta
        gamma = -gamma    

        rotation_x = np.array([[1, 0, 0],
                                [0, np.cos(alpha), -np.sin(alpha)],
                                [0, np.sin(alpha), np.cos(alpha)]])
        
        rotation_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                                [0, 1, 0],
                                [-np.sin(beta), 0, np.cos(beta)]])
        
        rotation_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                                [np.sin(gamma), np.cos(gamma), 0],
                                [0, 0, 1]])

        rot_matrix = rotation_x@rotation_y@rotation_z
        ellipsoid_rot_axes = np.tensordot(rot_matrix,ellipsoid_std_axes,axes=([1,0]))

        x,y,z = ellipsoid_rot_axes
        x**=2
        y**=2
        z**=2

        a = semi_a**2
        b = semi_b**2
        c = semi_c**2

        ellipsoid = x/a + y/b + z/c - 1
        ellipsoid = ellipsoid < 0 
        return ellipsoid
        
    def gaussian_noise(self, sigma=1.0, size = 128, range_min=-0.3, range_max=1.0):
        noise = np.random.random((size,size,size))
        gaussian_noise = gaussian_filter(noise,sigma)
        gaussian_noise_min = gaussian_noise.min()
        gaussian_noise_max = gaussian_noise.max()

        # Normalizing  (a - amin)*(tmax-tmin)/(a_max - a_min) + tmin
        tex_noise = ((gaussian_noise - gaussian_noise_min)*(range_max-range_min)/(gaussian_noise_max-gaussian_noise_min)) + range_min 

        return tex_noise
    
    def shape_generation(self,scale_centroids,centroid_main,num_ellipses,semi_axes_range,image_mask):
        # For the number of ellipsoids generate centroids in the local cloud
        random_centroids = centroid_main.T + (np.random.random((num_ellipses,3))*scale_centroids)


        # Selection of the semi axis length
        # 1. Select the length of the major axis length 
        # 2. Others just need to be a random ratio over the major axis length
        random_major_axes = np.random.randint(semi_axes_range[0],semi_axes_range[1],(num_ellipses,1))
        random_minor_axes = np.concatenate([np.ones((num_ellipses,1)),np.random.uniform(0.5,1,size = (num_ellipses,2))],1)
        random_semi_axes = random_major_axes*random_minor_axes


        # Permuting the axes so that one axes doesn't end up being the major every time.
        rng = np.random.default_rng()
        random_semi_axes = rng.permuted(random_semi_axes, axis=1)

        
        # Random rotation angles for the ellipsoids
        random_rot_angles = np.random.uniform(size = (num_ellipses,3))*np.pi
        #random_rot_angles = np.zeros(shape = (num_ellipses,3))*np.pi
        
        out = []
        for i in range(num_ellipses):
            out.append(self.ellipsoid(random_centroids[i],*random_semi_axes[i],*random_rot_angles[i],img_dim=64) )

        out = np.logical_or.reduce(out)*image_mask

        return out

    def simulation(self, image, image_mask,num_lesions=3):
        param_dict = {}

        roi = skimage.morphology.binary_erosion(image_mask,skimage.morphology.ball(10))*(image>0.1)
        roi_with_masks = roi
        output_image = image
        output_mask = np.zeros_like(image_mask)
        
        total_param_list = ['scale_centroid','num_ellipses','semi_axes_range','alpha','beta','gamma','smoothing_mask',
                            'tex_sigma','range_min','range_max','tex_sigma_edema']
        for i in range(num_lesions):
            gamma = 0
            tex_sigma_edema = 0

            x_corr,y_corr,z_corr = np.nonzero(roi_with_masks[:,:,:])
            random_coord_index = np.random.choice(len(x_corr),1)
            centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])

            # We need a loop and random choices tailored here for multiple lesions 

            scale_centroid = np.random.randint(2,15)
            num_ellipses = 15
            ranges = [(5,10),(10,15),(15,20)]
            semi_axes_range = ranges[int(np.random.choice(3,p=[0.5,0.3,0.2]))]
            
            if(self.have_small):
                ranges = [(2,5),(5,10),(10,15),(15,20)]
                semi_axes_range = ranges[int(np.random.choice(4,p=[0.5,0.25,0.15,0.1]))]
                if(semi_axes_range == (2,5)):
                    scale_centroid = np.random.randint(2,20)
                num_ellipses = 15

            alpha = np.random.uniform(0.5,0.8)
            beta = 1-alpha
            smoothing_mask = np.random.uniform(0.5,1)
            smoothing_image = 0
            tex_sigma = np.random.uniform(0.5,1)
            range_min = np.random.uniform(-0.5,0.5)
            range_max = np.random.uniform(0.7,1)


            
            out = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,image_mask)

            if(semi_axes_range !=(2,5)):
                semi_axes_range_edema = (semi_axes_range[0]+5,semi_axes_range[1]+5)
                tex_sigma_edema = 1.5*tex_sigma
                gamma = (1-beta)/2
                out_edema = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,image_mask)

            output_mask = np.logical_or(output_mask,out)

            if(self.have_noise):
                tex_noise = self.gaussian_noise(tex_sigma,self.size,range_min,range_max)
            else:
                tex_noise = 1.0
            
            if(self.have_edema and semi_axes_range !=(2,5)):
                tex_noise_edema = self.gaussian_noise(tex_sigma_edema,self.size,range_min,range_max)

                smoothed_les = gaussian_filter(gamma*out_edema*(1-out)*tex_noise_edema, sigma=smoothing_mask)
                image1 = alpha*output_image*(1-out_edema*(1-out)) + smoothed_les
                image2 = alpha*output_image + smoothed_les

                image1[out_edema*(1-out)>0]=image2[out_edema*(1-out)>0]
                image1[image1<0] = 0

                output_mask = np.logical_or(output_mask,out_edema)
                output_image = image1
                output_image -= output_image.min()
                output_image /= output_image.max()
            if(self.have_smoothing):
                smoothed_les = gaussian_filter(beta*out*tex_noise, sigma=smoothing_mask)
                image1 = alpha*output_image*(1-out) + smoothed_les
                image2 = alpha*output_image + smoothed_les

                image1[out>0]=image2[out>0]
                image1[image1<0] = 0
            else:
                image1 = alpha*output_image*(1-out) + beta*out*tex_noise
                image1[image1<0] = 0
            
            output_image = image1
            output_image -= output_image.min()
            output_image /= output_image.max()

            roi_with_masks *= (1-output_mask)>0
            
            total_params = [scale_centroid,num_ellipses,semi_axes_range,alpha,beta,gamma,smoothing_mask,
                            tex_sigma,range_min,range_max,tex_sigma_edema]
            
            for j in range(len(total_params)):
                param_dict[str(i)+'_'+total_param_list[j]] = total_params[j]

        if(self.return_param):
            return output_image, output_mask, param_dict
        else:
            return output_image, output_mask

    
    def __getitem__(self, index):

        nii_img = nib.load(self.paths[index]).get_fdata()
        image, img_crop_para = self.tight_crop_data(nii_img)
        image = skiform.resize(image, (128, 128, 128), order=1, preserve_range=True)
        image -= image.min()
        image /= image.max() + 1e-7

        if(self.mask_path):
            brain_mask_img = nib.load(self.mask_path[index]).get_fdata()
            brain_mask_img = brain_mask_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1],img_crop_para[2]:img_crop_para[2] + img_crop_para[3],img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            brain_mask_img = skiform.resize(brain_mask_img, (128, 128, 128), order=0, preserve_range=True)

        else:
            brain_mask_img = skimage.morphology.binary_closing(image>0.05,skimage.morphology.ball(3))

        param_dict = {}

        num_lesions = np.random.randint(1,5)
        image, label, param_dict = self.simulation(image, brain_mask_img, num_lesions)

        if(self.return_param):
            param_dict['num_lesions'] = num_lesions

        # else:
        #     num_lesions = np.random.randint(1,5)
        #     image, label = self.simulation(image, brain_mask_img,num_lesions)


        # sumout = np.sum(np.sum(label, axis=0), axis=0)
        # slide_no = np.where(sumout == np.amax(sumout))[0][0]
        # plt.subplot(1,2,1)
        # plt.imshow(image[:,:,slide_no])
        # plt.colorbar()
        # plt.subplot(1,2,2)
        # plt.imshow(label[:,:,slide_no])
        # plt.colorbar()
        # plt.show()

        image = np.expand_dims(image, -1).astype(np.single)
        gt = np.stack([label==0, label>0], -1).astype(np.single)

        data_dict = {}
        data_dict['input'] = image
        data_dict['gt'] = gt

        if(self.transform):
            data_dict = self.transform(data_dict)
        
        if(self.return_param):
            return data_dict, param_dict
        return data_dict

    def __len__(self):
        """Return the dataset size."""
        return len(self.paths)

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
