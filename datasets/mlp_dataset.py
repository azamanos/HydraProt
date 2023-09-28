import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import load_embedding_dataset, shuffle_along_axis

class TrainingEmbeddingLoader(Dataset):
    '''
    Dataset class that loads training embedding dataset for the MLP.

    Parameters
    ----------
    train_dataset_path : str
        Path of the h5 dataset.

    train_dataset_list : numpy array
        Numpy array with training pdb ids.

    Attributes
    ----------
    x : numpy array
        Numpy array of initial shape '(20000000,10,5)' that contains training embeddings.

    y : numpy array
        Numpy array of initial shape '(20000000,3)' that contains targets for the training embedding.

    n_samples : int
        Length of dataset instances.
    '''
    def __init__(self, train_dataset_path, train_dataset_list):
        #Load Data
        self.x, self.y = np.zeros((30000000,10,5)), np.zeros((30000000))
        ind = 0
        for pdb_id in train_dataset_list:
            #Load embedding and target
            x_emb, y_target = load_embedding_dataset(pdb_id, train_dataset_path, range(2))
            #Keep length of embedding
            size = len(y_target)
            #Append data
            self.x[ind:ind+size], self.y[ind:ind+size] = x_emb, y_target
            #Update index
            ind += size
        #Finally keep only the filled arrays
        self.x, self.y = self.x[:ind], self.y[:ind]
        #Keep size of data
        self.n_samples = len(self.x)

    def __getitem__(self,index):
        return self.x[index], self.y[index]

    def __len__(self):
        #Length of your dataset
        return self.n_samples

class ValidationEmbeddingLoader(Dataset):
    '''
    Dataset class that loads validation embedding dataset for the MLP.

    Parameters
    ----------
    validation_dataset_path : str
        Path of the h5 validation files.

    validation_dataset_list : numpy array
        Numpy array with training pdb ids.

    Attributes
    ----------
    data : list
        List that contains validation embeddings per pdb structure.

    n_samples : int
        Length of dataset instances.
    '''
    def __init__(self, validation_dataset_path, validation_dataset_list, shuffle=False):
        #Load Data
        self.data = []
        ind = 0
        for pdb_id in validation_dataset_list:
            #Load embedding and target
            self.data.append([load_embedding_dataset(pdb_id, validation_dataset_path, range(9))])
            if shuffle:
                self.data[-1][0][0] = shuffle_along_axis(np.array(self.data[-1][0][0]), 1)
        #Keep size of data
        self.n_samples = len(self.data)

    def __getitem__(self,index):
        return self.data[index]

    def __len__(self):
        #Length of your dataset
        return self.n_samples

class PredictionEmbeddingLoader(Dataset):
    '''
    Dataset class that loads validation embedding dataset for the MLP.

    Parameters
    ----------
    prediction_dataset_path : str
        Path of the h5 prediction file.

    prediction_dataset_list : list or numpy array
        List or numpy array with training pdb ids.

    Attributes
    ----------
    data : list
        List that contains validation embeddings per pdb structure.

    n_samples : int
        Length of dataset instances.
    '''
    def __init__(self, prediction_dataset_path, prediction_dataset_list, shuffle=False):
        #Load Data
        self.data = []
        ind = 0
        for pdb_id in prediction_dataset_list:
            #Load embedding and target
            self.data.append([load_embedding_dataset(pdb_id, prediction_dataset_path, range(3))])
            if shuffle:
                self.data[-1][0][0] = shuffle_along_axis(np.array(self.data[-1][0][0]), 1)
        #Keep size of data
        self.n_samples = len(self.data)

    def __getitem__(self,index):
        return self.data[index]

    def __len__(self):
        #Length of your dataset
        return self.n_samples
