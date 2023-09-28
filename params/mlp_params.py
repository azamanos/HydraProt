import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.utils import Config

#Initialize config class
config = Config()

## Paths ##
config.train_dataset_path = './datasets/mlp_dataset/embedding_mlp_train_36_374_dataset.h5'
config.validation_dataset_path = './datasets/mlp_dataset/embedding_mlp_validation_36_374_dataset.h5'
config.train_dataset_list = np.load('./datasets/data_lists/mlp_train_data.npy', allow_pickle=True)
config.validation_dataset_list = np.load('./datasets/data_lists/mlp_validation_data.npy', allow_pickle=True)

## MLP Model Parameters ##
config.first_part_features = [40, 200, 400, 800]
config.second_part_features = [800, 400, 200, 100]

## Training Parameters ##
config.training_batch_size = 20000
config.learning_rate = 1e-4
config.weight_dec = 0
config.dropout_p = 0.2
config.norm_scale = 2
#Choose if you want to load a previous epoch, also defines starting epoch.
config.load_model = False#'./checkpoints/MLP/HydrationNN_epoch_349.pth.tar'
config.load_and_validate = False
if not config.load_model:
    config.starting_epoch = 0
else:
    config.starting_epoch = int(config.load_model.split('_')[-1].split('.')[0]) + 1
#Number of epochs
config.num_epochs = 400
#Optimizer
config.optim_algorithm = optim.Adam
#Trace training parameters
config.shuffle = True
config.num_workers = 6
config.pin_memory = True


## Validation Parameters ##
config.cap_values = np.linspace(0.05,0.35,31)
config.validation_batch_size = 32000
#Threads to use multiprocessing for cpu
config.threads = 8
#Radius from protein for embedding calculations.
config.radius = 4
config.thresholds = (0.5, 1.0, 1.5) #distance threshold to match predicted and ground truth waters, float array, list, or tuple.
#Post-Processing Parameters
config.initial_cap = 0.05
config.final_cap = 0.05
#Distances below are reported in Ångstrom.
config.initial_distance_to_water = 1.4
config.final_distance_to_water = 2.25
config.distance_to_protein = 2.25

## Choose device ##
config.device = 'cuda:0'
#If you want to use all available gpus turn parallel into True
config.parallel = False
