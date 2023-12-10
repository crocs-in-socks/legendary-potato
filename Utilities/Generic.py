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
        self.to_load_encoder_path       = kwargs.get('to_load_encoder_path', None)
        self.to_load_projector_path     = kwargs.get('to_load_projector_path', None)
        self.to_load_classifier_path    = kwargs.get('to_load_classifier_path', None)
        self.to_load_proxy_path         = kwargs.get('to_load_proxy_path', None)
        self.dataset                    = kwargs.get('dataset')

        drive_names = {
            '62': 'fd67a3c7-ac13-4329-bdbb-bdad39a33bf1',
            '64': '70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a',
        }
        self.drive = drive_names[input('Server IP: ')]

        self.encoder_type = f'{self.proxy_type}_{self.train_task}_encoder'
        self.projector_type = f'{self.proxy_type}_{self.train_task}_projector'
        self.classifier_type = f'{self.proxy_type}_{self.train_task}_classifier'

        self.model_save_path = f'/mnt/{self.drive}/LabData/models_retrained/experiments/{self.to_save_folder}/'

        self.encoder_load_path = f'/mnt/{self.drive}/LabData/models_retrained/experiments/{self.to_load_folder}/{self.to_load_encoder_path}' if self.to_load_encoder_path else None

        self.projector_load_path = f'/mnt/{self.drive}/LabData/models_retrained/experiments/{self.to_load_folder}/{self.to_load_projector_path}' if self.to_load_projector_path else None

        self.classifier_load_path = f'/mnt/{self.drive}/LabData/models_retrained/experiments/{self.to_load_folder}/{self.to_load_classifier_path}' if self.to_load_classifier_path else None

        self.proxy_load_path = f'/mnt/{self.drive}/LabData/models_retrained/experiments/{self.to_load_folder}/{self.to_load_proxy_path}' if self.to_load_proxy_path else None

        print()
        print('Constant Name'.ljust(30) + 'Constant Value')
        for key, item in vars(self).items():
            print(f'{key.ljust(30)}{item}')
        print()

        double_check = input('Please check if the above is correct, (y/n): ')
        if double_check != 'y':
            exit(0)

def load_dataset(name: str, drive: str, transform: transforms.Compose, verbose: bool =False) -> (ImageLoader3D, ImageLoader3D, ImageLoader3D):

    dataset_names = {
        'wmh' : _load_wmh,
        'simulated_lesions_on_brain': _load_simulated_lesions_on_brain,
        'simulated_lesions_on_noise_background': _load_simulated_lesions_on_noise_background,
        'simulated_lesions_on_brain_with_clean': _load_simulated_lesions_on_brain_with_clean,
    }

    assert name in dataset_names.keys(), f'name should be in {list(dataset_names.keys())}'
    if verbose:
        print('\nDataset loaded.\n')
    return dataset_names[name](drive, transform)

def _load_simulated_lesions_on_brain(drive, transform):

    sim2211_train_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*size*/TrainSet/*FLAIR.nii.gz'))
    sim2211_train_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*size*/TrainSet/*mask.nii.gz'))
    sim2211_train_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*size*/TrainSet/*.json'))

    sim2211_validation_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*size*/ValSet/*FLAIR.nii.gz'))
    sim2211_validation_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/ValSet/*mask.nii.gz'))
    sim2211_validation_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*size*/ValSet/*.json'))

    sim2211_test_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*size*/TestSet/*FLAIR.nii.gz'))
    sim2211_test_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*size*/TestSet/*mask.nii.gz'))
    sim2211_test_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/*size*/TestSet/*.json'))

    sim2211_trainset = ImageLoader3D(paths=sim2211_train_data_paths, gt_paths=sim2211_train_gt_paths, json_paths=sim2211_train_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    sim2211_validationset = ImageLoader3D(paths=sim2211_validation_data_paths, gt_paths=sim2211_validation_gt_paths, json_paths=sim2211_validation_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    sim2211_testset = ImageLoader3D(paths=sim2211_test_data_paths, gt_paths=sim2211_test_gt_paths, json_paths=sim2211_test_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    return sim2211_trainset, sim2211_validationset, sim2211_testset

def _load_wmh(drive, transform):

    data = np.load('../wmh_indexes.npy', allow_pickle=True).item()
    trainset = ImageLoader3D(paths=data['train_names'], gt_paths=None, json_paths=None, image_size=128, type_of_imgs='numpy', transform=transform)

    validationset = ImageLoader3D(paths=data['val_names'], gt_paths=None, json_paths=None, image_size=128, type_of_imgs='numpy', transform=transform)

    testset = ImageLoader3D(paths=data['test_names'], gt_paths=None, json_paths=None, image_size=128, type_of_imgs='numpy', transform=transform)

    return trainset, validationset, testset

def _load_simulated_lesions_on_noise_background(drive, transform):
    Noise_train_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Noise_sim_30_11_23/all/TrainSet/*FLAIR.nii.gz'))
    Noise_train_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Noise_sim_30_11_23/all/TrainSet/*mask.nii.gz'))
    Noise_train_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Noise_sim_30_11_23/all/TrainSet/*.json'))

    Noise_validation_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Noise_sim_30_11_23/all/ValSet/*FLAIR.nii.gz'))
    Noise_validation_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Noise_sim_30_11_23/all/ValSet/*mask.nii.gz'))
    Noise_validation_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Noise_sim_30_11_23/all/ValSet/*.json'))

    Noise_test_data_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Noise_sim_30_11_23/all/TestSet/*FLAIR.nii.gz'))
    Noise_test_gt_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Noise_sim_30_11_23/all/TestSet/*mask.nii.gz'))
    Noise_test_json_paths = sorted(glob.glob('/mnt/fd67a3c7-ac13-4329-bdbb-bdad39a33bf1/Gouri/simulation_data/Noise_sim_30_11_23/all/TestSet/*.json'))    

    trainset = ImageLoader3D(paths=Noise_train_data_paths, gt_paths=Noise_train_gt_paths, json_paths=Noise_train_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)
    validationset = ImageLoader3D(paths=Noise_validation_data_paths, gt_paths=Noise_validation_gt_paths, json_paths=Noise_validation_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)
    testset = ImageLoader3D(paths=Noise_test_data_paths, gt_paths=Noise_test_gt_paths, json_paths=Noise_test_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    return trainset, validationset, testset


def _load_simulated_lesions_on_brain_with_clean(drive, transform):

    Sim1000_train_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*FLAIR.nii.gz'))
    Sim1000_train_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*mask.nii.gz'))
    Sim1000_train_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TrainSet/*.json'))

    Sim1000_validation_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*FLAIR.nii.gz'))
    Sim1000_validation_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*mask.nii.gz'))
    Sim1000_validation_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/ValSet/*.json'))

    Sim1000_test_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TestSet/*FLAIR.nii.gz'))
    Sim1000_test_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TestSet/*mask.nii.gz'))
    Sim1000_test_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Sim1000/Dark/all/TestSet/*.json'))

    sim2211_train_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TrainSet/*FLAIR.nii.gz'))
    sim2211_train_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TrainSet/*mask.nii.gz'))
    sim2211_train_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TrainSet/*.json'))

    sim2211_validation_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/ValSet/*FLAIR.nii.gz'))
    sim2211_validation_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/ValSet/*mask.nii.gz'))
    sim2211_validation_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/ValSet/*.json'))

    sim2211_test_data_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TestSet/*FLAIR.nii.gz'))
    sim2211_test_gt_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TestSet/*mask.nii.gz'))
    sim2211_test_json_paths = sorted(glob.glob(f'/mnt/{drive}/Gouri/simulation_data/Full_sim_22_11_23/Dark/**/TestSet/*.json'))

    clean_data_paths = sorted(glob.glob(f'/mnt/{drive}/LabData/NIMH/3d/*.anat*/*fast_restore.nii.gz*'))
    clean_gt_paths = sorted(glob.glob(f'/mnt/{drive}/LabData/NIMH/3d/*.anat*/*seg_label.nii.gz*'))

    Sim1000_trainset = ImageLoader3D(paths=Sim1000_train_data_paths, gt_paths=Sim1000_train_gt_paths, json_paths=Sim1000_train_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    Sim1000_validationset = ImageLoader3D(paths=Sim1000_validation_data_paths, gt_paths=Sim1000_validation_gt_paths, json_paths=Sim1000_validation_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    Sim1000_testset = ImageLoader3D(paths=Sim1000_test_data_paths, gt_paths=Sim1000_test_gt_paths, json_paths=Sim1000_test_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)


    sim2211_trainset = ImageLoader3D(paths=sim2211_train_data_paths, gt_paths=sim2211_train_gt_paths, json_paths=sim2211_train_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    sim2211_validationset = ImageLoader3D(paths=sim2211_validation_data_paths, gt_paths=sim2211_validation_gt_paths, json_paths=sim2211_validation_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)

    sim2211_testset = ImageLoader3D(paths=sim2211_test_data_paths, gt_paths=sim2211_test_gt_paths, json_paths=sim2211_test_json_paths, image_size=128, type_of_imgs='nifty', transform=transform)


    clean = ImageLoader3D(paths=clean_data_paths, gt_paths=clean_gt_paths, json_paths=None, image_size=128, type_of_imgs='nifty', transform=transform)

    train_size = int(0.7 * len(clean))
    validation_size = int(0.1 * len(clean))
    test_size = len(clean) - (train_size + validation_size)

    clean_trainset, clean_validationset, clean_testset = random_split(clean, (train_size, validation_size, test_size))

    trainset = ConcatDataset([Sim1000_trainset, sim2211_trainset, clean_trainset])
    validationset = ConcatDataset([Sim1000_validationset, sim2211_validationset, clean_validationset])
    testset = ConcatDataset([Sim1000_testset, sim2211_testset, clean_testset])

    return trainset, validationset, testset