import os
import h5py
import time
import torch
import numpy as np
from utils.mlp_utils import model_pass_GPU
from utils.utils import PDB, embedding_multiprocess, reform_dataset, save_embedding_dataset, remove_and_refine_duplicates, remove_duplicates, batchify_dmatrix_computation_for_dmin, write_pdb_waters

def prepare_mlp_prediction_embedding(config):
    '''
    Function that prepares and saves the mlp embedding into h5 file.

    Parameters
    ----------
    config : class object
            contains the following parameters

        config.pdb_list : numpy.array
            numpy array containing pdb ids of structures to be hydrated.

        config.pdb_path : str
            directory path to pdb structures.

        config.unet_results_dir : str
            directory path of the save 3D Unet prediction.

        config.threads : int
            number of processes to start concurrently, depends mainly on memory capacity of your machine and CPU capabilities.

        config.radius : float
            distance to look for protein atom pair and neighbors.

        config.include_hetatm : bool
            True or False, to include or not HETATM molecules in the pdb structures.

        config.mlp_embedding_dir : str
            directory path to save embedding for MLP.

        config.len_pdb_list : int
            length of given pdb ids list.

        config.structure_str : str
            if you have one structure the equals to 'structure', else equals to 'structures'.
    '''
    m_st = time.time()
    for i,pdb_id in enumerate(config.pdb_list):
        #Load PDB files.
        if os.path.exists(f'{config.pdb_path}{pdb_id}.pdb.gz'):
            pdb = PDB(f'{config.pdb_path}{pdb_id}.pdb.gz')
        else:
            pdb = PDB(f'{config.pdb_path}{pdb_id}.pdb')
        #Load predicted water candidates.
        water_coords = np.load(f'{config.unet_results_dir}{pdb_id}_waters.npy')
        #Compute embedding
        pdb, emb = embedding_multiprocess(pdb, water_coords, 3000, threads=config.threads, distance=config.radius, need_labels=False)
        #First keep all atom coordinates of protein and HETATM molecules
        if len(pdb.HETATM_coords) and config.include_hetatm:
            pdb_all_coords = np.concatenate((pdb.coords,pdb.HETATM_coords),axis=0)
        else:
            pdb_all_coords = np.copy(pdb.coords)
        #Normalize embedding
        emb[:,1] /= 4
        emb[:,2:5] /= np.pi
        #Reform dataset
        emb_reformed, batch_unique_waters = reform_dataset(emb, 10)
        #Keep also the coordinates of the waters that have embedding.
        embedded_water_coordinates = water_coords[batch_unique_waters]
        #Save_Data
        with h5py.File(f'{config.mlp_embedding_dir}/embeddings.h5', 'a') as f:
            save_embedding_dataset(f, pdb_id,(emb_reformed, embedded_water_coordinates, pdb_all_coords), range(3))
        print(f'Embedding was generated for {i+1} out of {config.len_pdb_list} {config.structure_str}, approximately {"%3.2f" % round((time.time()-m_st)/60,2)} minutes have passed.', end='\r')
    print(f'{" " * 200}', end='\r')
    print(f'The preparation of the MLP embedding was completed in {"%3.2f" % round((time.time()-m_st)/60,2)} minutes after generating the embeddings for the {config.len_pdb_list} given {config.structure_str}.')
    return

def mlp_prediction(loader, model, mlp_params):
    '''
    Validation loop, evaluates the recall, precision and F1 of the prediction on validation set.

    Parameters
    ----------
    loader : torch.nn.DataLoader
        Pytorch dataloader that loads validation data.

    model : torch.nn.Model
        Pytorch MLP model that process embedding information.

    mlp_params : class object
            contains the following parameters

        mlp_params.device : str
            device in which you wish to run the model/data. str 'cpu' or 'cuda'.


        mlp_params.mlp_batch_size : int
            size of batch during inference, int.

        mlp_params.results_dir : str
            directory to save results.

        mlp_params.pdb_list : numpy.array
            numpy array containing pdb ids of structures to be hydrated.

        mlp_params.initial_cap : float
            initial value to cap predictions

        mlp_params.initial_distance_to_water : float
            initial minimum distance to remove closeby waters and at the same time refine them.

        mlp_params.final_cap : float
            final value to cap predictions

        mlp_params.final_distance_to_water : float
            final minimum distance to remove closeby waters.

        mlp_params.distance_to_protein : float
            minimum acceptable distance for a water to be found close to protein's heavy atoms.
    '''
    torch.cuda.empty_cache()
    model.to(mlp_params.device).eval()
    loader_len = len(loader)
    with torch.no_grad():
        st = time.time()
        for v, data in enumerate(loader):
            #ct = time.time()
            #Allocate data to tensors and device.
            x, water_coordinates, pdb_coords = data[0][0][0].float(), np.array(data[0][1][0]), np.array(data[0][2][0])
            #Make prediction
            prediction_scores = np.array(model_pass_GPU(x, model, mlp_params.mlp_batch_size, mlp_params.device))
            #Cap to 1 and inverse predictions.
            prediction_scores[prediction_scores>1] = 1
            prediction_scores = 1-prediction_scores
            #Make the first initial cap of prediction scores
            initial_cap_indexes = np.where(prediction_scores<mlp_params.initial_cap)[0]
            water_coordinates_i, prediction_scores_i = np.delete(water_coordinates, initial_cap_indexes, 0), np.delete(prediction_scores, initial_cap_indexes, 0)
            #First removal of duplicates and refining at mlp_params.initial_distance_to_water (default, 1.4 Angstrom).
            water_coordinates, prediction_scores, todelete = remove_and_refine_duplicates(water_coordinates_i, prediction_scores_i, mlp_params.initial_distance_to_water)
            if len(todelete):
                water_coordinates, prediction_scores = np.delete(water_coordinates, todelete, 0), np.delete(prediction_scores, todelete, 0)
            #Μake the final capping.
            final_cap_indexes = np.where(prediction_scores<mlp_params.final_cap)[0]
            if len(final_cap_indexes):
                water_coordinates, prediction_scores = np.delete(water_coordinates, final_cap_indexes, 0), np.delete(prediction_scores, final_cap_indexes, 0)
            #Remove duplicates at 0.5*mlp_params.initial_distance_to_water+0.5*mlp_params.final_distance_to_water
            todelete = remove_duplicates(water_coordinates, prediction_scores, 0.5*mlp_params.initial_distance_to_water+0.5*mlp_params.final_distance_to_water)
            if len(todelete):
                water_coordinates, prediction_scores = np.delete(water_coordinates, todelete, 0), np.delete(prediction_scores, todelete, 0)
            #Then remove duplicates at mlp_params.final_distance_to_water
            todelete = remove_duplicates(water_coordinates, prediction_scores, mlp_params.final_distance_to_water)
            if len(todelete):
                water_coordinates, prediction_scores = np.delete(water_coordinates, todelete, 0), np.delete(prediction_scores, todelete, 0)
            #Fill gaps with initial water coordinates where is needed
            #Μake the final capping for initial predictions.
            final_cap_indexes = np.where(prediction_scores_i<mlp_params.final_cap)[0]
            if len(final_cap_indexes):
                water_coordinates_i, prediction_scores_i = np.delete(water_coordinates_i, final_cap_indexes, 0), np.delete(prediction_scores_i, final_cap_indexes, 0)
            fill_gaps = np.where(batchify_dmatrix_computation_for_dmin(water_coordinates, water_coordinates_i)>=mlp_params.final_distance_to_water)[0]
            if len(fill_gaps):
                water_coordinates = np.concatenate((water_coordinates, water_coordinates_i[fill_gaps]))
                prediction_scores = np.concatenate((prediction_scores, prediction_scores_i[fill_gaps]))
                #First removal of duplicates and refining at mlp_params.initial_distance_to_water (default, 1.4 Angstrom).
                water_coordinates, prediction_scores, todelete = remove_and_refine_duplicates(water_coordinates, prediction_scores, mlp_params.initial_distance_to_water)
                if len(todelete):
                    water_coordinates, prediction_scores = np.delete(water_coordinates, todelete, 0), np.delete(prediction_scores, todelete, 0)
                #Then remove again duplicates at mlp_params.final_distance_to_water
                todelete = remove_duplicates(water_coordinates, prediction_scores, mlp_params.final_distance_to_water)
                if len(todelete):
                    water_coordinates, prediction_scores = np.delete(water_coordinates, todelete, 0), np.delete(prediction_scores, todelete, 0)
            #Finally, after removing water duplicates, remove waters closer than mlp_params.distance_to_protein (default, 2.25 Angstrom) to protein structure.
            to_delete_close_to_protein = np.where(batchify_dmatrix_computation_for_dmin(pdb_coords, water_coordinates)<mlp_params.distance_to_protein)[0]
            if len(to_delete_close_to_protein):
                water_coordinates, prediction_scores = np.delete(water_coordinates, to_delete_close_to_protein, 0), np.delete(prediction_scores, to_delete_close_to_protein, 0)
            #Finally write to pdb file
            write_pdb_waters(water_coordinates, prediction_scores, f'{mlp_params.results_dir}{mlp_params.pdb_list[v]}_waters.pdb')
            print(f'The MLP water prediction completed for {v+1} out of {loader_len} structures, approximately {"%3.2f" % round((time.time()-st)/60, 2)} minutes have passed.', end='\r')
        print(f'{" " * 200}', end='\r')
        print(f'The MLP prediction task for {loader_len} structures was completed in {"%3.2f" % round((time.time()-st)/60, 2)} minutes.')
    return
