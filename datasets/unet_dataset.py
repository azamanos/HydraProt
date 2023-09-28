import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import coo_to_dense_ones, load_training_coordinates, load_validation_coordinates, transform_coordinates, prepare_train_validation_unet_data

class Training_coordinates():
    '''
    Class object that loads training coordinates for the 3D Unet

    Parameters
    ----------
    dataset_path : str
        Path of the h5 dataset.

    dataset_list_path : str
        Path of pdb dataset list.

    Attributes
    ----------
    dataset_path : str
        Path of the h5 dataset.

    dataset_list_path : str
        Path of pdb dataset list.

    dataset_list : numpy array
        Numpy array that contains pdb ids for the training.

    dataset_coordinates : list
        List that contains training coordinates information.
    '''
    def __init__(self, dataset_path, dataset_list_path):
        #Load Data
        self.dataset_path = dataset_path
        self.dataset_list_path = dataset_list_path
        self.dataset_list = np.load(self.dataset_list_path, allow_pickle=True)
        self.dataset_coordinates = []
        for pdbid in self.dataset_list:
            self.dataset_coordinates.append(load_training_coordinates(pdbid, self.dataset_path))

class Training_transformed_submaps(Dataset):
    '''
    Dataset class that transforms training coordinates for the 3D Unet

    Parameters
    ----------
    train_data : class
        Training_coordinates class object.

    vs : float
        Length of the voxel side.

    pad : int
        Lenght of the padding for the 3D array.

    Attributes
    ----------
    dataset : list
        List that contains dataset information in coo matrix format.

    n_samples : int
        Length of dataset instances.
    '''
    def __init__(self, train_data, vs, pad, flip=False):
        self.dataset = []
        a = time.time()
        dataset_len = len(train_data.dataset_coordinates)
        for i, info in enumerate(train_data.dataset_coordinates):
            #Load water coordinates, atom coordinates and atom types arrays
            water_coords, atom_hetatm_coords, atom_hetatm_atomtype = info
            #Transform water and atom coordinates
            water_coords, atom_hetatm_coords = transform_coordinates(water_coords, atom_hetatm_coords, flip = flip)
            #Create input submaps in coo represantation and add them into the dataset
            self.dataset += prepare_train_validation_unet_data(water_coords, atom_hetatm_coords, atom_hetatm_atomtype, vs, pad, validation=False)
            print(f'Training dataset creation step {i+1}/{dataset_len}.',end='\r')
        print(f'Training dataset was created in {round(time.time()-a,2)} seconds.')
        self.n_samples = len(self.dataset)

    def __getitem__(self,index):
        return coo_to_dense_ones(self.dataset[index])

    def __len__(self):
        #Length of your dataset
        return self.n_samples

class Validation_coordinates():
    '''
    Class object that loads validation coordinates for the 3D Unet

    Parameters
    ----------
    dataset_path : str
        Path of the h5 dataset.

    dataset_list_path : str
        Path of pdb dataset list.

    Attributes
    ----------
    dataset_path : str
        Path of the h5 dataset.

    dataset_list_path : str
        Path of pdb dataset list.

    dataset_list : numpy array
        Numpy array that contains pdb ids for the training.

    dataset_coordinates : list
        List that contains training coordinates information.
    '''
    def __init__(self, dataset_path, dataset_list_path):
        #Load Data
        self.dataset_path = dataset_path
        self.dataset_list_path = dataset_list_path
        self.dataset_list = np.load(self.dataset_list_path, allow_pickle=True)
        self.dataset_coordinates = []
        for pdbid in self.dataset_list:
            self.dataset_coordinates.append(load_validation_coordinates(pdbid, self.dataset_path))
        self.n_samples = len(self.dataset_list)

class Validation_transformed_submaps(Dataset):
    '''
    Dataset class that transforms validation coordinates for the 3D Unet

    Parameters
    ----------
    train_data : class
        Training_coordinates class object.

    vs : float
        Length of the voxel side.

    pad : int
        Lenght of the padding for the 3D array.

    Attributes
    ----------
    dataset : list
        List that contains dataset information in coo matrix format.

    n_samples : int
        Length of dataset instances.
    '''
    def __init__(self, validation_data, vs, pad, flip=False):
        self.dataset = []
        a = time.time()
        dataset_len = len(validation_data.dataset_coordinates)
        for i, info in enumerate(validation_data.dataset_coordinates):
            #Load water coordinates, atom coordinates and atom types arrays
            water_coords, atom_hetatm_coords, atom_hetatm_atomtype = info
            #Transform water and atom coordinates
            water_coords, atom_hetatm_coords = transform_coordinates(water_coords, atom_hetatm_coords, flip = flip)
            #Create input submaps in coo represantation and add them into the dataset
            self.dataset.append(prepare_train_validation_unet_data(water_coords, atom_hetatm_coords, atom_hetatm_atomtype, vs, pad, validation=True))
            print(f'Validation dataset creation step {i+1}/{dataset_len}.',end='\r')
        print(f'Validation dataset was created in {round(time.time()-a,2)} seconds.')
        self.n_samples = len(self.dataset)

    def __getitem__(self,index):
        return self.dataset[index]

    def __len__(self):
        #Length of your dataset
        return self.n_samples
