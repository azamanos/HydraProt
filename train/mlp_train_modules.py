import time
import torch
import h5py
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.mlp_utils import loss_function, model_pass_GPU, revive_pdb
from utils.utils import remove_and_refine_duplicates, remove_duplicates, plot_metrics_per_cap_value, batchify_dmatrix_computation_for_dmin, allocate_multiprocess_validation_match_waters

def train_loop(loader, model, config):
    '''
    Train loop, trains the model with the embedding from the predicted water coordinates of the 3D Unet.

    Parameters
    ----------
    loader : torch.nn.DataLoader
        Pytorch dataloader that loads training data.

    model : torch.nn.Model
        Pytorch MLP model that process embedding information.

    config : class
        Config class containing all the parameters needed for the training.

    Returns
    -------
    Total normalized epoch loss.
    '''
    with torch.cuda.device(config.device):
        torch.cuda.empty_cache()
    model.to(config.device).train()
    batches = tqdm(loader)
    epoch_loss = 0
    for i, data in enumerate(batches):
        x, y = data[0].to(config.device), data[1].to(config.device)
        #Pass your data through your model
        pred = model(x).squeeze(1)
        #Then compute the loss, remember |scores, neighbor_aa, neighbor_waters = targets[:,2], targets[:,0], targets[:,1]|
        loss = loss_function(pred, y, config.norm_scale)
        #Back propagate
        loss.backward()
        #Optimize
        config.optimizer.step()
        #Zero the gradients
        config.optimizer.zero_grad()
        #Compute batch loss
        batch_loss = loss.item()
        #Keep epoch loss normalized by size of batch
        epoch_loss += batch_loss
        #Print epoch loss so far
        batches.set_postfix(loss=epoch_loss/(i+1))
    return epoch_loss/(i+1)


def validation_loop(loader, model, epoch, config):
    '''
    Validation loop, evaluates the validation set on the trained MLP based on the recall, precision and F1 of the prediction.

    Parameters
    ----------
    loader : torch.nn.DataLoader
        Pytorch dataloader that loads validation data.

    model : torch.nn.Model
        Pytorch MLP model that process embedding information.

    epoch : int
        Number of epoch that is going to be validated.

    config : class
        Config class containing all the parameters needed for the validation.

    Returns
    -------
    mean_recall : float
        mean recall computed for current epoch of the model and for the high F1 selected cap value, correspons to the whole ground truth and predicted waters of validation set.

    mean_precision : float
        mean precision computed for current epoch of the model and for the high F1 selected cap value, correspons to the whole ground truth and predicted waters of validation set.

    mean_F1 : float
        mean F1 computed for current epoch of the model and for the high F1 selected cap value, correspons to the whole ground truth and predicted waters of validation set.

    selected_cap_value : float
        selected cap value based on the highest F1 for the selected threshold.

    epoch_loss : float
        total normalized epoch loss for the validation set.
    '''
    model.to(config.device).eval()
    with torch.no_grad():
        epoch_loss = 0
        st = time.time()
        loaderlen = len(loader)
        #Initiallize your data variables
        capped_match_results = np.zeros((loaderlen, len(config.cap_values), 3*len(config.thresholds)))
        for v, data in enumerate(loader):
            #Allocate data to tensors and device.
            x, water_coordinates, y = data[0][0][0].float(), np.array(data[0][1][0]), data[0][-1][0]
            #Create pdb object, with coords, water_coords, atomname, resname, resnum.
            pdb = revive_pdb(*[np.array(e) for e in data[0][2:7]])
            #Make prediction
            prediction_scores = np.array(model_pass_GPU(x, model, config.validation_batch_size, config.device))
            #Then compute the loss
            loss = loss_function(torch.from_numpy(prediction_scores), y, config.norm_scale)
            #Compute batch loss
            batch_loss = loss
            #Keep epoch loss normalized by size of batch
            epoch_loss += batch_loss
            #Cap to 1 and inverse predictions.
            prediction_scores[prediction_scores>1] = 1
            prediction_scores = 1-prediction_scores
            #Make the first initial cap of prediction scores
            initial_cap_indexes = np.where(prediction_scores<config.initial_cap)[0]
            water_coordinates_i, prediction_scores_i = np.delete(water_coordinates, initial_cap_indexes, 0), np.delete(prediction_scores, initial_cap_indexes, 0)
            #First removal of duplicates and refining at config.initial_distance_to_water (default, 1.4 Angstrom).
            water_coordinates, prediction_scores, todelete = remove_and_refine_duplicates(water_coordinates_i, prediction_scores_i, config.initial_distance_to_water)
            if len(todelete):
                water_coordinates, prediction_scores = np.delete(water_coordinates, todelete, 0), np.delete(prediction_scores, todelete, 0)
            #Μake the final capping.
            final_cap_indexes = np.where(prediction_scores<config.final_cap)[0]
            if len(final_cap_indexes):
                water_coordinates, prediction_scores = np.delete(water_coordinates, final_cap_indexes, 0), np.delete(prediction_scores, final_cap_indexes, 0)
            #Remove duplicates at 0.5*config.initial_distance_to_water+0.5*config.final_distance_to_water
            todelete = remove_duplicates(water_coordinates, prediction_scores, 0.5*config.initial_distance_to_water+0.5*config.final_distance_to_water)
            if len(todelete):
                water_coordinates, prediction_scores = np.delete(water_coordinates, todelete, 0), np.delete(prediction_scores, todelete, 0)
            #Then remove duplicates at config.final_distance_to_water
            todelete = remove_duplicates(water_coordinates, prediction_scores, config.final_distance_to_water)
            if len(todelete):
                water_coordinates, prediction_scores = np.delete(water_coordinates, todelete, 0), np.delete(prediction_scores, todelete, 0)
            #Fill gaps with initial water coordinates where is needed
            #Μake the final capping for initial predictions.
            final_cap_indexes = np.where(prediction_scores_i<config.final_cap)[0]
            if len(final_cap_indexes):
                water_coordinates_i, prediction_scores_i = np.delete(water_coordinates_i, final_cap_indexes, 0), np.delete(prediction_scores_i, final_cap_indexes, 0)
            fill_gaps = np.where(batchify_dmatrix_computation_for_dmin(water_coordinates, water_coordinates_i)>=config.final_distance_to_water)[0]
            if len(fill_gaps):
                water_coordinates = np.concatenate((water_coordinates, water_coordinates_i[fill_gaps]))
                prediction_scores = np.concatenate((prediction_scores, prediction_scores_i[fill_gaps]))
                #First removal of duplicates and refining at config.initial_distance_to_water (default, 1.4 Angstrom).
                water_coordinates, prediction_scores, todelete = remove_and_refine_duplicates(water_coordinates, prediction_scores, config.initial_distance_to_water)
                if len(todelete):
                    water_coordinates, prediction_scores = np.delete(water_coordinates, todelete, 0), np.delete(prediction_scores, todelete, 0)
                #Then remove again duplicates at config.final_distance_to_water
                todelete = remove_duplicates(water_coordinates, prediction_scores, config.final_distance_to_water)
                if len(todelete):
                    water_coordinates, prediction_scores = np.delete(water_coordinates, todelete, 0), np.delete(prediction_scores, todelete, 0)
            #Finally, after removing water duplicates, remove waters closer than config.distance_to_protein (default, 2.25 Angstrom) to protein structure.
            to_delete_close_to_protein = np.where(batchify_dmatrix_computation_for_dmin(pdb.coords, water_coordinates)<config.distance_to_protein)[0]
            if len(to_delete_close_to_protein):
                water_coordinates, prediction_scores = np.delete(water_coordinates, to_delete_close_to_protein, 0), np.delete(prediction_scores, to_delete_close_to_protein, 0)
            #Compute metrics for different cap values
            capped_match_results[v] = allocate_multiprocess_validation_match_waters(pdb.water_coords, water_coordinates, prediction_scores, config.cap_values, config.thresholds, config.threads)
            print(f'For {(v+1)}/{loaderlen} structures, validation have been completed, {round((time.time()-st)/60, 2)} minutes have passed.', end='\r')
    epoch_loss = epoch_loss.to('cpu')/(v+1)
    print(f'For {(v+1)}/{loaderlen} structures, validation have been completed, {round((time.time()-st)/60, 2)} minutes have passed, and validation epoch loss {round(float(epoch_loss), 4)}.')
    sum_match_results = np.sum(capped_match_results, 0)
    for c in range(len(sum_match_results)):
        for t_ind, t in enumerate(config.thresholds):
            precision, recall = sum_match_results[c,3*t_ind]/max(1e-3,sum_match_results[c,(3*t_ind)+1]), sum_match_results[c,3*t_ind]/max(1e-3,sum_match_results[c,(3*t_ind)+2])
            F1 = (2*precision*recall)/(max(1e-3,precision+recall))
            sum_match_results[c,3*t_ind:3+3*t_ind] = recall, precision, F1
    #Find best F1 value for largest threshold value.
    argmax_F1_selected = np.argmax(sum_match_results[:,-1], 0)
    #Compute mean metrics for best F1 value for largest threhsold value, print results.
    mean_recall, mean_precision, mean_F1 = sum_match_results[argmax_F1_selected].reshape((3, 3)).T
    print(f'Threshold  Recall  Precision   F1')
    for thresh_ind, threshold in enumerate(config.thresholds):
        #Plot figures in results directory
        plot_metrics_per_cap_value(config.cap_values, sum_match_results, thresh_ind, threshold, epoch,'./training_results/mlp/')
        print(f'{threshold:^10}{round(mean_recall[thresh_ind],3):^8}{round(mean_precision[thresh_ind],3):^12}{round(mean_F1[thresh_ind],3)}')
    print(f'With selected cap value at {config.thresholds[-1]} Å distance, {config.cap_values[argmax_F1_selected]}.')
    #Finally save the match results array and return,  match metrics for best F1 value for largest threhsold value, and the selected cap value.
    np.save(f'./training_results/mlp/match_results_epoch_{epoch}.npy', capped_match_results)
    return mean_recall, mean_precision, mean_F1, config.cap_values[argmax_F1_selected], epoch_loss
