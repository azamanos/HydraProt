import torch
import torch.nn as nn

class Conv3DBlock(nn.Module):
    '''
    3D convolution block for 3D U-net.

    Parameters
    ----------
    in_ch : int
        Number of input channels.

    out_ch : int
        Number of output channels.

    drop_p : float
        Dropout probability applied after the two 3D convolutions.

    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    block_dropout : nn.Dropout
        Dropout layers.

    ReLU : nn.ReLU
        Activation function.

    conv_1 : nn.Conv3d
        First 3D convolution.

    conv_2 : nn.Conv3d
        Second 3D convolution.
    '''
    def __init__(self, in_ch, out_ch, drop_p=0.):
        super().__init__()
        self.block_dropout = nn.Dropout(p=drop_p)
        self.ReLU = nn.ReLU()
        self.conv_1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm3d(out_ch)
        self.conv_2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        '''
        Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_channels, 64, 64, 64)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, out_channels, 64, 64, 64)`.
        '''
        #Pass through the first convolution
        x = self.conv_1(x)
        #Pass through the first batch normalization
        x = self.bn_1(x)
        #Pass through activation function and apply dropout
        x = self.block_dropout(self.ReLU(x))
        x = self.conv_2(x)
        x = self.bn_2(x)
        #Pass through activation function and apply dropout
        x = self.block_dropout(self.ReLU(x))
        #Return x
        return x


#MLP
class MLP(nn.Module):
    '''
    HydrationNN architecture.

    Parameters
    ----------
    features : list
        List of length 4 with integer elements that defines the size of layers

    dropout_p : float
        Dropout probability applied after each hidden layer

    Attributes
    ----------
    relu : nn.ReLU
        Activation function

    layer_1 : nn.Linear
        First/Input linear layer

    layer_2 : nn.Linear
        Second linear layer

    layer_3 : nn.Linear
        Last/Output linear layer

    dropout : nn.Dropout
        Dropout layer
    '''
    def __init__(self, features, dropout_p):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.layer_1 = nn.Linear(features[0], features[1], bias=True)
        self.layer_2 = nn.Linear(features[1], features[2], bias=True)
        self.layer_3 = nn.Linear(features[2], features[3], bias=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        '''
        Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, ... , features[0])`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, ... , features[4])`.
        '''
        x = self.dropout(self.relu(self.layer_1(x)))
        x = self.dropout(self.relu(self.layer_2(x)))
        return self.dropout(self.relu(self.layer_3(x)))
