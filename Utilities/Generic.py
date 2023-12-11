import re
import glob
import numpy as np
from torchvision import transforms
from ImageLoader.ImageLoader3D import ImageLoader3D
from torch.utils.data import ConcatDataset, random_split

class Constants:
    def __init__(self, **kwargs):
        self.batch_size                 = kwargs.get('batch_size', 1)
        self.patience                   = kwargs.get('patience', 15)
        self.num_workers                = kwargs.get('num_workers', 0)
        self.num_epochs                 = kwargs.get('num_epochs', 100)
        self.date                       = kwargs.get('date')
        self.to_save_folder             = kwargs.get('to_save_folder')
        self.to_load_folder             = kwargs.get('to_load_folder', None)
        self.device                     = kwargs.get('device')
        self.proxy_type                 = kwargs.get('proxy_type')
        self.train_task                 = kwargs.get('train_task')
        self.encoder_load_path          = kwargs.get('encoder_load_path', None)
        self.projector_load_path        = kwargs.get('projector_load_path', None)
        self.classifier_load_path       = kwargs.get('classifier_load_path', None)
        self.proxy_load_path            = kwargs.get('proxy_load_path', None)
        self.dataset                    = kwargs.get('dataset')

        drive_names = {
            '62': 'fd67a3c7-ac13-4329-bdbb-bdad39a33bf1',
            '64': '70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a',
        }
        self.drive = drive_names[input('Server IP: ')]

        self.encoder_type = f'{self.proxy_type}_{self.train_task}_encoder'
        self.projector_type = f'{self.proxy_type}_{self.train_task}_projector'
        self.segmentor_type = f'{self.proxy_type}_{self.train_task}_segmentor'
        self.classifier_type = f'{self.proxy_type}_{self.train_task}_classifier'

        self.to_save_folder = f'/mnt/{self.drive}/LabData/models_retrained/experiments/{self.to_save_folder}/' if self.to_save_folder else None

        self.encoder_load_path = f'/mnt/{self.drive}/LabData/models_retrained/experiments/{self.to_load_folder}/{self.encoder_load_path}' if self.encoder_load_path else None

        self.projector_load_path = f'/mnt/{self.drive}/LabData/models_retrained/experiments/{self.to_load_folder}/{self.projector_load_path}' if self.projector_load_path else None

        self.classifier_load_path = f'/mnt/{self.drive}/LabData/models_retrained/experiments/{self.to_load_folder}/{self.classifier_load_path}' if self.classifier_load_path else None

        self.proxy_load_path = f'/mnt/{self.drive}/LabData/models_retrained/experiments/{self.to_load_folder}/{self.proxy_load_path}' if self.proxy_load_path else None

        print()
        print('Constant Name'.ljust(30) + 'Constant Value')
        for key, item in vars(self).items():
            print(f'{key.ljust(30)}{item}')
        print()

        double_check = input('Please check if the above is correct, (y/n): ')
        if double_check != 'y':
            exit(0)

def load_dataset(name: str, drive: str, transform: transforms.Compose) -> (ImageLoader3D, ImageLoader3D, ImageLoader3D):

    dataset_names = {
        'wmh' : _load_wmh,
        'brats': _load_brats,
        'clean': _load_clean,
        'sim_1000': _load_sim_1000,
        'sim_2211': _load_sim_2211,
        'sim_2211_wmh': _load_sim_2211,
        'sim_2211_size': _load_sim_2211,
        'sim_2211_brats': _load_sim_2211,
        'sim_2211_ratios': _load_sim_2211,
        'simulated_lesions_on_noise_background': _load_simulated_lesions_on_noise_background,
    }

    trainset_list, validationset_list, testset_list = [], [], []

    name_list = name.split('+')
    for name in name_list:
        # Assert doesn't work for examples like sim_2211_ratios:10_10_10_10
        # assert name in dataset_names.keys(), f'name should be in {list(dataset_names.keys())}'
        if name in ['sim_2211_size', 'sim_2211_wmh', 'sim_2211_brats']:
            specification = name.split('_')[-1]
            temp_trainset, temp_validationset, temp_testset = dataset_names[name](drive, transform, specification)
        elif 'ratios' in name:
            size_ratios = name.split(':')[1].split('_')
            size_ratios = [int(ratio) for ratio in size_ratios]
            temp_trainset, temp_validationset, temp_testset = dataset_names[name.split(':')[0]](drive, transform, ratios=size_ratios)
        else:
            temp_trainset, temp_validationset, temp_testset = dataset_names[name](drive, transform)
        
        trainset_list.append(temp_trainset)
        validationset_list.append(temp_validationset)
        testset_list.append(temp_testset)

    trainset = ConcatDataset(trainset_list)
    validationset = ConcatDataset(validationset_list)
    testset = ConcatDataset(testset_list)

    return trainset, validationset, testset

def _load_clean(drive, transform):

    clean_data_paths = sorted(glob.glob(f'/mnt/{drive}/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
    clean_gt_paths = sorted(glob.glob(f'/mnt/{drive}/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))

    clean = ImageLoader3D(paths=clean_data_paths, gt_paths=clean_gt_paths, json_paths=None, image_size=128, type_of_imgs='nifty', transform=transform)

    train_size = int(0.7 * len(clean))
    validation_size = int(0.1 * len(clean))
    test_size = len(clean) - (train_size + validation_size)

    clean_trainset, clean_validationset, clean_testset = random_split(clean, (train_size, validation_size, test_size))

    return clean_trainset, clean_validationset, clean_testset

def _load_wmh(drive, transform):

    data = np.load('../wmh_indexes.npy', allow_pickle=True).item()
    trainset = ImageLoader3D(paths=data['train_names'], gt_paths=None, json_paths=None, image_size=128, type_of_imgs='numpy', transform=transform)

    validationset = ImageLoader3D(paths=data['val_names'], gt_paths=None, json_paths=None, image_size=128, type_of_imgs='numpy', transform=transform)

    testset = ImageLoader3D(paths=data['test_names'], gt_paths=None, json_paths=None, image_size=128, type_of_imgs='numpy', transform=transform)

    return trainset, validationset, testset

def _load_brats(drive, transform):

    data = np.load('../brats_2020_indexes.npy', allow_pickle=True).item()
    trainset = ImageLoader3D(paths=data['train_names_flair'], gt_paths=data['train_names_seg'], json_paths=None, image_size=128, type_of_imgs='nifty', transform=transform)

    validationset = ImageLoader3D(paths=data['val_names_flair'], gt_paths=data['val_names_seg'], json_paths=None, image_size=128, type_of_imgs='nifty', transform=transform)

    testset = ImageLoader3D(paths=data['test_names_flair'], gt_paths=data['test_names_seg'], json_paths=None, image_size=128, type_of_imgs='nifty', transform=transform)

    return trainset, validationset, testset

def _load_simulated_lesions_on_noise_background(drive, transform):
    Noise_train_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Noise_sim_30_11_23/all/TrainSet/*FLAIR.nii.gz'))
    Noise_train_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Noise_sim_30_11_23/all/TrainSet/*mask.nii.gz'))
    Noise_train_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Noise_sim_30_11_23/all/TrainSet/*.json'))

    Noise_validation_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Noise_sim_30_11_23/all/ValSet/*FLAIR.nii.gz'))
    Noise_validation_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Noise_sim_30_11_23/all/ValSet/*mask.nii.gz'))
    Noise_validation_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Noise_sim_30_11_23/all/ValSet/*.json'))

    Noise_test_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Noise_sim_30_11_23/all/TestSet/*FLAIR.nii.gz'))
    Noise_test_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Noise_sim_30_11_23/all/TestSet/*mask.nii.gz'))
    Noise_test_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Noise_sim_30_11_23/all/TestSet/*.json'))    

    trainset = ImageLoader3D(paths=Noise_train_data_paths, gt_paths=Noise_train_gt_paths, json_paths=Noise_train_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)
    validationset = ImageLoader3D(paths=Noise_validation_data_paths, gt_paths=Noise_validation_gt_paths, json_paths=Noise_validation_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)
    testset = ImageLoader3D(paths=Noise_test_data_paths, gt_paths=Noise_test_gt_paths, json_paths=Noise_test_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    return trainset, validationset, testset

def _load_sim_1000(drive, transform):

    Sim1000_train_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*FLAIR.nii.gz'))
    Sim1000_train_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*mask.nii.gz'))
    Sim1000_train_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*.json'))

    Sim1000_validation_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*FLAIR.nii.gz'))
    Sim1000_validation_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*mask.nii.gz'))
    Sim1000_validation_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*.json'))

    Sim1000_test_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TestSet/*FLAIR.nii.gz'))
    Sim1000_test_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TestSet/*mask.nii.gz'))
    Sim1000_test_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TestSet/*.json'))

    Sim1000_trainset = ImageLoader3D(paths=Sim1000_train_data_paths, gt_paths=Sim1000_train_gt_paths, json_paths=Sim1000_train_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    Sim1000_validationset = ImageLoader3D(paths=Sim1000_validation_data_paths, gt_paths=Sim1000_validation_gt_paths, json_paths=Sim1000_validation_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    Sim1000_testset = ImageLoader3D(paths=Sim1000_test_data_paths, gt_paths=Sim1000_test_gt_paths, json_paths=Sim1000_test_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    return Sim1000_trainset, Sim1000_validationset, Sim1000_testset

def _load_sim_2211(drive, transform, specification='', ratios=None):

    specification_list = [specification]
    if ratios is not None:
        specification_list = ['size1', 'size2', 'size3', 'size4']

    trainset_list, validationset_list, testset_list = [], [], []

    for idx, specification in enumerate(specification_list):

        sim2211_train_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*{specification}*/TrainSet/*FLAIR.nii.gz'))
        sim2211_train_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*{specification}*/TrainSet/*mask.nii.gz'))
        sim2211_train_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*{specification}*/TrainSet/*.json'))

        sim2211_validation_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*{specification}*/ValSet/*FLAIR.nii.gz'))
        sim2211_validation_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*{specification}*/ValSet/*mask.nii.gz'))
        sim2211_validation_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*{specification}*/ValSet/*.json'))

        sim2211_test_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*{specification}*/TestSet/*FLAIR.nii.gz'))
        sim2211_test_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*{specification}*/TestSet/*mask.nii.gz'))
        sim2211_test_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*{specification}*/TestSet/*.json'))

        if ratios is not None:
            sim2211_train_data_paths = sim2211_train_data_paths[:ratios[idx]]
            sim2211_train_gt_paths = sim2211_train_gt_paths[:ratios[idx]]
            sim2211_train_json_paths = sim2211_train_json_paths[:ratios[idx]]

            sim2211_validation_data_paths = sim2211_validation_data_paths[:ratios[idx]]
            sim2211_validation_gt_paths = sim2211_validation_gt_paths[:ratios[idx]]
            sim2211_validation_json_paths = sim2211_validation_json_paths[:ratios[idx]]

            sim2211_test_data_paths = sim2211_test_data_paths[:ratios[idx]]
            sim2211_test_gt_paths = sim2211_test_gt_paths[:ratios[idx]]
            sim2211_test_json_paths = sim2211_test_json_paths[:ratios[idx]]

        sim2211_trainset = ImageLoader3D(paths=sim2211_train_data_paths, gt_paths=sim2211_train_gt_paths, json_paths=sim2211_train_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

        sim2211_validationset = ImageLoader3D(paths=sim2211_validation_data_paths, gt_paths=sim2211_validation_gt_paths, json_paths=sim2211_validation_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

        sim2211_testset = ImageLoader3D(paths=sim2211_test_data_paths, gt_paths=sim2211_test_gt_paths, json_paths=sim2211_test_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

        trainset_list.append(sim2211_trainset)
        validationset_list.append(sim2211_validationset)
        testset_list.append(sim2211_testset)

    trainset = ConcatDataset(trainset_list)
    validationset = ConcatDataset(validationset_list)
    testset = ConcatDataset(testset_list)

    return trainset, validationset, testset