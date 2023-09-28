import torch
import torch.nn as nn
from models.modules import Conv3DBlock

#Unet model parameters example
in_channels = 3
out_channels = 1
intermediate_channels = [16,32,64,128]

class Unet3D(nn.Module):
    '''
    3D Unet full architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.

    out_channels : int
        Number of output channels.

    channels : list
        List with number of intermediate channels.

    drop_p : float
        Dropout probability applied after the two 3D convolutions.

    Attributes
    ----------
    channels : list
        List with number of intermediate channels.

    blocks_len : int
        Number of up and down blocks.

    conv_dropout : nn.Dropout
        Dropout layers.

    max_pool3d : nn.MaxPool3d
        3D max pooling function.

    down_blocks : nn.ModuleList
        Module list with down blocks.

    conv_transpose3d : nn.ModuleList
        Module list with up convolution 3D transpose.

    up_blocks : nn.ModuleList
        Module list with up blocks.

    bottleneck : Conv3DBlock
        Conv3DBlock of the U-net bottleneck.

    fconv : nn.Conv3d
        Final 3D convolution of the network.
    '''
    def __init__(self, in_channels, out_channels, channels, drop_p=0.):
        super().__init__()
        #Keep channel's dimensions
        self.channels = channels
        #Keep also the length of blocks
        self.blocks_len = len(channels)
        #Dropout
        self.conv_dropout = nn.Dropout(p=drop_p)
        #Max Pooling Layer
        self.max_pool3d= nn.MaxPool3d((2,2,2))
        #Initilize the blocks of down, transpose and up convolutions
        self.down_blocks, self.conv_transpose3d, self.up_blocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        #Down blocks of U-net
        for channel in channels:
            self.down_blocks.append(Conv3DBlock(in_channels,channel, drop_p))
            in_channels = channel

        #Create the bottleneck
        self.bottleneck = Conv3DBlock(channels[-1], channels[-1]*2, drop_p)

        #Up blocks of U-net
        for channel in reversed(channels):
            #Append first the ConvTranspose3d layer
            self.conv_transpose3d.append(nn.ConvTranspose3d(channel*2, channel*2, kernel_size=2, stride=2))
            #Append then the Conv3DBlock
            self.up_blocks.append(Conv3DBlock(channel+channel*2, channel, drop_p))

        #The final convolution
        self.fconv = nn.Conv3d(channels[0], out_channels, kernel_size=1, stride=1)

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
        skip_connections = []

        #Pass through the down convolution blocks.
        for down in self.down_blocks:
            #Pass through the down block.
            x = down(x)
            #Keep the output.
            skip_connections.append(x)
            #Pool the output.
            x = self.max_pool3d(x)

        #Then comes the bottleneck.
        x = self.bottleneck(x)
        #Remeber to reverse skip skip_connections.
        skip_connections.reverse()

        #Pass through the transpose convolutions and up blocks.
        for i in range(self.blocks_len):
            #Pass through the 3D transpose convolution.
            x = self.conv_transpose3d[i](x)
            #Pass the up block.
            x = self.up_blocks[i](torch.cat([skip_connections[i], x], dim=1))

        #Then pass through the final convolution
        x = self.conv_dropout(self.fconv(x))
        #and return
        return x
