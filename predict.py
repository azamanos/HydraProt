import os
import time
import h5py
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet_model import Unet3D
from models.mlp_model import HydrationNN
from params.prediction_params import config
from prediction.unet_prediction import unet_prediction
from prediction.mlp_prediction import prepare_mlp_prediction_embedding, mlp_prediction
from datasets.mlp_dataset import PredictionEmbeddingLoader
from utils.utils import load_txt_pdb_ids, load_checkpoint

def main():
    #Keep starting time
    initial_time = time.time()
    #First load pdb ids you want to predict
    if config.pdb_list_path == config.pdb_path:
        config.pdb_list = []
        for i in os.listdir(config.pdb_path):
            config.pdb_list.append(i.split('.')[0])
    elif config.pdb_list_path[-3:] == 'npy':
        config.pdb_list = np.load(config.pdb_list_path, allow_pickle=True)
    else:
        config.pdb_list = load_txt_pdb_ids(config.pdb_list_path)
    config.len_pdb_list = len(config.pdb_list)
    if config.len_pdb_list-1:
        config.structure_str = 'structures'
    else:
        config.structure_str = 'structure'
    #Then remove every remaining file from temp directory.
    os.system('rm ./temp/* -rf')
    #Initialize 3D Unet model and load data.
    model = Unet3D(config.in_channels, config.out_channels, config.intermediate_channels)
    model.to(config.device)
    load_checkpoint(torch.load(config.unet_checkpoint, map_location=config.device), model, 'for 3D Unet model', print_load = False)
    #Create the directory which you will save the results from the 3D unet.
    os.system(f'mkdir {config.unet_results_dir}')
    #Compute and save unet prediction of water coordinates for your set.
    unet_prediction(config,model)
    #Delete the 3D Unet model.
    del model
    # Prediction of the 3D Unet has finished ##
    torch.set_num_threads(1)
    #Create the directory which you will save the embedding for the MLP.
    os.system(f'mkdir {config.mlp_embedding_dir}')
    #Now prepare the data for the final prediction from MLP.
    prepare_mlp_prediction_embedding(config)
    #Now predict with MLP model, initialize mlp model and load data.
    model = HydrationNN(config.first_part, config.second_part, 0)
    model.to(config.device)
    load_checkpoint(torch.load(config.mlp_checkpoint, map_location=config.device), model, 'for MLP model', print_load = False)
    #Load data
    prediction_set = PredictionEmbeddingLoader(f'{config.mlp_embedding_dir}/embeddings.h5', config.pdb_list, shuffle=True)
    #Define loader
    prediction_loader = DataLoader(prediction_set, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory = config.pin_memory)
    #Before making the prediction create the results directory
    if not os.path.exists(config.results_dir):
        os.system(f'mkdir {config.results_dir}')
    #Make the predictions
    mlp_prediction(prediction_loader, model, config)
    print(f'The total time taken to complete the water prediction for {config.len_pdb_list} {config.structure_str} was {"%3.2f" % round((time.time()-initial_time)/60,2)} minutes.')
    #Finally remove every remaining file from temp directory.
    os.system('rm ./temp/* -rf')
    return

if __name__ == '__main__':
    main()
