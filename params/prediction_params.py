import torch
import numpy as np
from utils.utils import Config

#Initialize config class
config = Config()

## Paths ##
config.pdb_path = 'PATH_TO_PDB'
#Here give npy, text file, or config.pdb_path text file should have comma or next line delimeter.
config.pdb_list_path = config.pdb_path

## Unet3D Model Parameters ##
config.in_channels = 3
config.out_channels = 1
config.intermediate_channels = [16,32,64,128]
config.unet_checkpoint = './checkpoints/unet/Unet3D_36_epoch_374.pth.tar'

## Unet3D Prediction Parameters
config.unet_results_dir = './temp/unet_prediction_waters/' #The directory which you will save the results from the 3D unet.
config.radius = 4
config.vs = 0.8
config.pad = 5
config.unet_batch_size = 20 #Here adjust unet batch size according to your system's RAM.
config.device = 'cuda:0' #Here complete the device you wish to use for the deep learning prediction.
config.include_hetatm = True
config.sigmoid = torch.nn.Sigmoid()
config.cap = 0.12
config.prediction_pad = 0
config.prediction_iterations = 3

## MLP Model Parameters ##
config.first_part = [40, 200, 400, 800]
config.second_part = [800, 400, 200, 100]
config.mlp_checkpoint = './checkpoints/mlp/HydrationNN_epoch_344_92.pth.tar'

## MLP Prediction Parameters
config.mlp_embedding_dir = './temp/mlp_embedding/' #The directory which you will save the embedding for the MLP.
config.threads = 8 #Adjust the number of threads according to your system's capabilities.
config.mlp_batch_size = 32000 #Here adjust mlp batch size according to your system's RAM.
config.initial_cap = 0.05
config.final_cap = 0.05
#Distances below are reported in Ångstrom.
config.initial_distance_to_water = 1.4
config.final_distance_to_water = 2.25
config.distance_to_protein = 2.25
#Directory to save results
config.results_dir = './prediction_results/HydraProt_Predictions/'
#Loader Parameter
config.num_workers = 6
config.pin_memory = True
