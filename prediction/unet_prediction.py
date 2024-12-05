import os
import time
import torch
import numpy as np
from utils.utils import random_transformation_matrix, prepare_prediction_unet_data, reconstruct_map, indexes_to_coordinates, batchify_dmatrix_computation_for_dmin, PDB, preprocess_pdb_for_unet

def unet_model_pass(data, submaps_length, submaps_indexes, original_map_shape, model, origin, unet_params):
    '''
    Unet model pass

    Parameters
    ----------
    data : array
        numpy array of shape (N,4,64,64,64) with prepared submaps to be inserted in the unet model.

    submaps_length : int
        initial lenght of submaps K, where K>=N.

    submaps_indexes : array
        array of shape (N) with the indexes of submaps thar are non zero, to be inserted in the future K lenght array.
        
    original_map_shape : tuple
        shape of the original map

    model : torch model
        3D unet model

    origin : array
        numpy array with length 3 contains the origin of the protein coordinates.

    unet_params : class object
        contains the following parameters

    unet_params.vs: float
        cubic voxel size of the 3D array.

    unet_params.unet_batch_size : int
        size of batch during inference, int.

    unet_params.device : str
        device in which you wish to run the model/data. str 'cpu' or 'cuda'.

    unet_params.sigmoid : torch sigmoid
        sigmoid function by pytorch.

    unet_params.cap : float
        value to cap your predictions, it affects the prediction

    unet_params.prediction_pad: int
        padding lenght for the prediction.

    Returns
    -------
    refined_coords : array
        array of shape (L,3) with coordinates of predicted waters.

    weights_of_refined_indexes : list
        list of shape (L) with the scores for the predicted water coordinates.
    '''
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(data).to(unet_params.device).float()
        pred = np.zeros((data.shape[0],64,64,64))
        # Calculate the total number of batches
        batches = (data.shape[0] + unet_params.unet_batch_size - 1) // unet_params.unet_batch_size
        for batch_index in range(batches):
            # Calculate the start and end index for the current batch
            start_index = batch_index * unet_params.unet_batch_size
            interval = slice(start_index, min(start_index + unet_params.unet_batch_size, data.shape[0]))
            #Send batch to device and run batch through the model, then apply sigmoid, squeeze the 2nd dimension = N, send to cpu and finally transform it to numpy array.
            pred[interval]= np.array(unet_params.sigmoid(model(inp[interval])).squeeze(1).to('cpu'))
    #initiallize prediction submaps array given the dimensions of original submaps
    pred_submaps = np.zeros((submaps_length,64,64,64))
    #Then for the indexes that you do get prediction fill the pred_submaps array
    pred_submaps[submaps_indexes] = pred
    #Finally reconstruct your original map
    pred = reconstruct_map(pred_submaps, o_dim=original_map_shape)
    #delete pred_submaps
    del pred_submaps
    #Find prediction indexes
    refined_indexes = np.where(pred>=unet_params.cap)
    weights_of_refined_indexes = pred[refined_indexes].tolist()
    refined_indexes = np.array(refined_indexes).T
    if not len(refined_indexes):
        return [], 0
    #Transform them to coordinates
    refined_coords = indexes_to_coordinates(refined_indexes, origin, unet_params.vs)
    return refined_coords, weights_of_refined_indexes


def unet_prediction_pass(atom_hetatm_coords, atom_hetatm_atomtype, model, unet_params):
    '''
    Prediction loop, predicts water coordinates

    Parameters
    ----------
    atom_hetatm_coords : array
        numpy array of shape (N,3) with coordinate information.

    atom_hetatm_atomtype : array
        numpy array of shape (N,1) with atomtype information.

    model : torch model
        3D unet model

    unet_params : class object
        contains the following parameters

    unet_params.vs: float
        cubic voxel size of the 3D array.

    unet_params.pad : int
        padding lenght for calculations on the map, it depends on the voxel size.

    unet_params.unet_batch_size : int
        size of batch during inference, int.

    unet_params.device : str
        device in which you wish to run the model/data. str 'cpu' or 'cuda'.

    unet_params.sigmoid : torch sigmoid
        sigmoid function by pytorch.

    unet_params.cap : float
        value to cap your predictions, it affects the prediction

    unet_params.prediciton pad: int
        padding lenght for the prediction.

    unet_params.prediction_iterations : int
        default 3, number of iteration to pass your transformed coordinates from the unet.

    Returns
    -------
    pred_coords : array
        array of shape (K,3) with coordinates of predicted waters.

    scores : array
        array of shape (K) with the scores for the predicted water coordinates.
    '''
    if unet_params.device[:4] == 'cuda':
        with torch.cuda.device(unet_params.device):
            torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        #Initialize matrix transformation list and refined_coords_transformed
        refined_coords, refined_weights = [], []
        atom_hetatm_coords_to_transform = np.concatenate((atom_hetatm_coords,np.ones((len(atom_hetatm_coords),1))), axis=1).T
        for iteration in range(unet_params.prediction_iterations):
            #Compute a random transformation (translation and rotation)
            matrix_transformation = random_transformation_matrix()
            #Transform accordingly atom_hetatm_coords
            atom_hetatm_coords_transformed = (matrix_transformation@atom_hetatm_coords_to_transform).T[:,:3]
            #Compute your data
            submaps, submaps_length, submaps_indexes, original_map_shape, origin = prepare_prediction_unet_data(atom_hetatm_coords_transformed, atom_hetatm_atomtype, unet_params)
            #Make the prediction
            transformed_refined_coords, weights_of_transformed_refined_coords = unet_model_pass(submaps, submaps_length, submaps_indexes, original_map_shape, model, origin, unet_params)
            if not len(transformed_refined_coords):
                continue
            #Add a column of ones to transformed_refined_coords
            transformed_refined_coords = np.concatenate((transformed_refined_coords, np.ones((len(transformed_refined_coords),1))),axis=1).T
            #Then return to original basis and add predicted coords and their weights
            currently_predicted = (np.linalg.inv(matrix_transformation)@transformed_refined_coords).T[:,:3]
            if not iteration:
                refined_coords += currently_predicted.tolist()
                refined_weights += weights_of_transformed_refined_coords
            else:
                d_min = batchify_dmatrix_computation_for_dmin(np.array(refined_coords), currently_predicted)
                newly_predicted_to_keep = np.where(d_min>0.3)[0]
                refined_coords += currently_predicted[newly_predicted_to_keep].tolist()
                refined_weights += np.array(weights_of_transformed_refined_coords)[newly_predicted_to_keep].tolist()
        pred_coords, scores = np.array(refined_coords), np.array(refined_weights)
        return pred_coords, scores

def unet_prediction(config,model):
    '''
    Complete function that prepares and predicts via the pretrained 3D Unet the initally sampled candidate water coordinates.

    Parameters
    ----------
    config : class object
            contains the following parameters

        config.pdb_list : numpy.array
            numpy array containing pdb ids of structures to be hydrated.

        config.pdb_path : str
            directory path to pdb structures.

        config.radius : float
            distance to look for protein atom pair and neighbors.

        config.include_hetatm : bool
            True or False, to include or not HETATM molecules in the pdb structures.

        config.unet_results_dir : str
            directory path of the save 3D Unet prediction.

        config.len_pdb_list : int
            length of given pdb ids list.

        config.structure_str : str
            if you have one structure the equals to 'structure', else equals to 'structures'.

    model : torch model
        3D unet model
    '''
    #Compute and save unet prediction of water coordinates for your set.
    u_st = time.time()
    for i, pdb_id in enumerate(config.pdb_list):
        #Load pdb
        if os.path.exists(f'{config.pdb_path}{pdb_id}.pdb.gz'):
            pdb = PDB(f'{config.pdb_path}{pdb_id}.pdb.gz')
        else:
            pdb = PDB(f'{config.pdb_path}{pdb_id}.pdb')
        #Preprocess pdb class object
        pdb, atom_hetatm_coords, atom_hetatm_atomtype = preprocess_pdb_for_unet(pdb, config.radius, include_hetatm=config.include_hetatm)
        #Predict
        pred_coords, scores = unet_prediction_pass(atom_hetatm_coords, atom_hetatm_atomtype, model, config)
        np.save(f'{config.unet_results_dir}{pdb_id}_waters.npy', pred_coords)
        print(f'The Unet 3D prediction has been completed for {i+1} out of {config.len_pdb_list} {config.structure_str}, approximately {"%3.2f" % round((time.time()-u_st)/60, 2)} minutes have passed.', end='\r')
    print(f'{" " * 200}', end='\r')
    print(f'The Unet 3D prediction task was completed in {"%3.2f" % round((time.time()-u_st)/60, 2)} minutes, for {config.len_pdb_list} {config.structure_str}.')
