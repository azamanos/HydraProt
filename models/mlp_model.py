import torch
import torch.nn as nn
from models.modules import MLP
from utils.mlp_utils import transform_embedding_torch, statistical_reduction_torch3D

#Features example
first_part_features = [40, 200, 400, 800]
second_part_features = [800, 400, 200, 100]

#Model
class HydrationNN(nn.Module):
    '''
    HydrationNN architecture.

    Parameters
    ----------
    first_part_features : list
        List of length 4 with integer elements that define the size of layers for the first part of HydrationNN.

    second_part_features : list
        List of length 4 with integer elements that define the size of layers for the second part of HydrationNN.

    dropout_p : float
        Dropout probability applied after each hidden layer.

    Attributes
    ----------
    first_part : MLP
        MLP block with 3 hidden layers.

    second_part : MLP
        MLP block with 3 hidden layers.

    final_layer : nn.Linear
        Pytorch linear layer, the output linear of the model.
    '''
    def __init__(self, first_part_features, second_part_features, dropout_p):
        super(HydrationNN, self).__init__()
        self.first_part = MLP(first_part_features,dropout_p)
        self.second_part = MLP(second_part_features,dropout_p)
        self.final_layer = nn.Linear(second_part_features[-1],1)

    def forward(self, x):
        '''
        Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_water_candidates, 10, 40)`.

        Returns
        -------
        torch.Tensor
            Shape '(n_water_candidates, 1)'.
        '''
        x = self.first_part(transform_embedding_torch(x))
        x = self.second_part(statistical_reduction_torch3D(x))
        return self.final_layer(x)
