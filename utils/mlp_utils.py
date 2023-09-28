import torch
import numpy as np
import torch.nn.functional as F

class revive_pdb(object):
    """Create pdb object"""
    def __init__(self, coords, atomname, resname, resnum, watercoords):
        self.coords = coords.squeeze(0)
        self.atomname = atomname.squeeze(0)
        self.resname = resname.squeeze(0)
        self.resnum = resnum.squeeze(0)
        self.water_coords = watercoords.squeeze(0)

def sigmoid_function(target, scale=2):
    '''
    Sigmoid function that normalizes input distance.

    Parameters
    ----------
    target : torch.Tensor
        torch.Tensor with your targets, of length (N).

    scale : float
        Number of scaling for the sigmoid function, default 2.

    Returns
    -------
    torch.Tensor of shape (N) set to [0,1] interval by the sigmoid function
    '''
    return (2/(1+torch.exp(-(scale*target)/torch.log(torch.tensor(2))))-1)**2

def loss_function(prediction, y, norm_scale=2):
    '''
    Computes loss for the mlp model.

    Parameters
    ----------
    prediction : torch.Tensor
        torch.Tensor with the prediction of the model, of length (N).

    y : torch.Tensor
        torch.Tensor with the respective targets, of length (N).

    norm_scale : float
        Number of scaling for the sigmoid function, default 2.

    Returns
    -------
    torch.Tensor mean squared error (MSE) of the predictions and targets.
    '''
    batch_len = len(y)
    return torch.sum((prediction-sigmoid_function(y,norm_scale))**2)/batch_len

def transform_embedding_torch(emb):
    '''
    Transforms embedding of atomname to one hot vector, and concatenates it with the rest information.

    Parameters
    ----------
    emb: torch.Tensor
        embedding of dimension (N,6), float datatype.

    Returns
    -------
    Transformed torch.Tensor of shape (N,40) that contains information ready to input in the mlp model.
    '''
    # Isolate Pair Atom Type to compute one hot encoding and remove first column.
    patoh = F.one_hot(emb[:,:,0].to(torch.int64), num_classes=37)[:,:,1:]
    return torch.cat((patoh,(emb[:,:,1:5])), dim=-1).float()
        
def coordinates_to_dmatrix_torch(a_coords, b_coords):
    '''
    Creates distance matrix for torch.Tensor input.

    Parameters
    ----------
    a_coords : torch.Tensor
        torch.Tensor of shape (N,3) that contains coordinates information.

    b_coords : torch.Tensor
        torch.Tensor of shape (M,3) that contains coordinates information.

    Returns
    -------
    torch.Tensor of shape (N,M), the distance matrix of a_coords and b_coords.
    '''
    return torch.cdist(a_coords,b_coords)

def statistical_reduction_torch3D(m):
    '''
    Compute statistical reduction of input embedding from 3D arrays.
    3D array of [number of waters x 10 x feature length output of the first part]
    
    Parameters
    ----------
    m : torch.Tensor
        torch.Tensor of shape (N,10,M) output matrix of the first part of the model.

    Returns
    -------
    torch.Tensor of shape (N,10,M) output matrix of the aggregation function.
    '''
    #Calculate the statistical reduction only for the rows that correspond to a protein atom.
    return torch.mean(m, axis=1)

def model_pass_GPU(embedding, model, batch_size, device):
    '''
    Function that organizes embedding through the two models, outputs score for all candidate water points.

    Parameters
    ----------
    embedding : torch.Tensor
        torch tensor of shape '(N,10,40)' containes the embeddings for candidate water points.

    model : torch.model
        HydrationNN model

    batch_size : int
        Size of batch that you want to pass from your architecture, depends on GPU/system RAM capabilities, int.

    device : str
        Device in which you wish to run the model/data, str 'cpu' or 'cuda'.

    Returns
    -------
    torch.Tensor predictions of length (N) for the given water embedding of shape (N,10,40).
    '''
    emb_len = embedding.shape[0]
    #Create output array.
    out = torch.ones((emb_len))
    #Number of batches
    # Calculate the total number of batches
    batches = (emb_len + batch_size - 1) // batch_size
    #Loop through your batches
    for batch_index in range(batches):
        # Calculate the start and end index for the current batch
        start_index = batch_index * batch_size
        slicer = slice(start_index, min(start_index + batch_size, emb_len))
        #Pass your data through your model
        pred = model(embedding[slicer].to(device)).squeeze(1)
        #Keep prediction
        out[slicer] = pred.to('cpu')
    return out
