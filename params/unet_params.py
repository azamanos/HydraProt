import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.utils import Config
from utils.unet_utils import weighted_BCELoss

#Initialize config class
config = Config()

## Paths ##
config.train_dataset_path = './datasets/unet_dataset/unet_train_dataset'
config.validation_dataset_path = './datasets/unet_dataset/unet_validation_dataset/'
config.train_list_path = './datasets/data_lists/unet_train_data.npy'
config.validation_list_path = './datasets/data_lists/unet_validation_data.npy'

## Unet3D Model Parameters ##
config.in_channels = 3
config.out_channels = 1
config.intermediate_channels = [16,32,64,128]

## Dataset Parameters ##
config.vs = 0.8
config.pad = 5
config.flip = False

## Training Parameters ##
config.training_batch_size = 6
config.learning_rate = 1e-5
config.weight_dec = 1e-5
config.dropout_p = 0.2
#Choose if you want to load a previous epoch, also defines starting epoch.
config.load_model = False#'./checkpoints/3D_Unet/Unet3D_epoch_299.pth.tar'
config.load_and_validate = False
if not config.load_model:
    config.starting_epoch = 0
else:
    config.starting_epoch = int(config.load_model.split('_')[-1].split('.')[0]) + 1
#Number of epochs
config.num_epochs = 400
#Define loss function and the target weights
config.loss_function = weighted_BCELoss
config.weights  = torch.FloatTensor([0.3333333333333333,1.6666666666666665]) #adds to 2 1:5
#Optimizer
config.optim_algorithm = optim.Adam
#Trace training parameters
config.shuffle = True
config.num_workers = 6
config.pin_memory = True

## Validation Parameters ##
config.cap_values = np.linspace(0.1,0.3,21)
config.cap = config.cap_values[0]
config.radius = 4
config.prediction_pad = 0
config.prediction_iterations= 3
config.sigmoid = nn.Sigmoid()
config.validation_batch_size = 20
config.thresholds = [0.5, 1.0, 1.5]
#Threads to use multiprocessing for cpu
config.threads = 8

## Choose device ##
config.device = 'cuda:0'
#If you want to use all available gpus turn parallel into True
config.parallel = False
