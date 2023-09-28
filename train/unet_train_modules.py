import torch
import time
import numpy as np
from tqdm import tqdm
from utils.utils import reconstruct_map, indexes_to_coordinates, remove_duplicates, allocate_multiprocess_validation_match_waters, plot_metrics_per_cap_value

def train_loop(loader, model, config):
    '''
    Train loop, trains the model with the embedding from the predicted water coordinates of the 3D Unet.

    Parameters
    ----------
    loader : torch.nn.DataLoader
        Pytorch dataloader that loads training data.

    model : torch.nn.Model
        Pytorch 3D Unet model that process embedding information.

    config : class
        Config class containing all the parameters needed for the training.

    Returns
    -------
    Total normalized epoch loss.
    '''
    epoch_loss = 0
    with torch.cuda.device(config.device):
        torch.cuda.empty_cache()
    model.train()
    batches = tqdm(loader)
    for i, data in enumerate(batches):
        x, y = data[:,:-1].to(config.device).float(), data[:,-1].to(config.device).float()
        prediction = model(x)
        loss = config.loss_function(config.sigmoid(prediction.squeeze(1)), y, config.weights)
        #Backwards
        loss.backward()
        #Optimize
        config.optimizer.step()
        #Zero Gradients
        config.optimizer.zero_grad()
        batches.set_postfix(loss=loss.item())
        #Keep epoch loss normalized by size of batch
        epoch_loss += loss.item()/x.shape[0]
    return epoch_loss/(i+1)

def validation_loop(loader, model, epoch, config):
    '''
    Validation loop, evaluates the validation set on the trained 3D Unet based on the recall, precision and F1 of the prediction.

    Parameters
    ----------
    loader : torch.nn.DataLoader
        Pytorch dataloader that loads validation data.

    model : torch.nn.Model
        Pytorch 3D Unet model that process embedding information.

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
    with torch.cuda.device(config.device):
        torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        st = time.time()
        loaderlen = len(loader)
        epoch_loss = []
        #Initiallize your data variables
        capped_match_results = np.zeros((loaderlen, len(config.cap_values), 3*len(config.thresholds)))
        for v, data in enumerate(loader):
            #Assign data
            x, y = data[0][0][:,:-1].float(), data[0][0][:,-1].float()
            #Prepare output array
            pred = np.zeros((x.shape[0],64,64,64))
            #Compute number of batches
            num_batches = int(np.ceil(x.shape[0]/config.validation_batch_size))+1
            #Create batch loop
            batch_loop = np.linspace(0, x.shape[0], num_batches, dtype=int)
            for i, batch in enumerate(batch_loop[:-1]):
                interval = slice(batch, batch_loop[i+1])
                #Send batch to device and run batch through the model, then apply sigmoid, squeeze the 2nd dimension = N, send to cpu and finally transform it to numpy array.
                batch_pred = config.sigmoid(model(x[interval].to(config.device))).squeeze(1).to('cpu')
                loss = config.loss_function(batch_pred, y[interval], config.weights)
                #Compute batch loss
                batch_loss = loss/(batch_loop[i+1]-batch)
                #Keep epoch loss normalized by size of batch
                epoch_loss.append(batch_loss)
                pred[interval]= np.array(batch_pred)
            #initiallize prediction submaps array given the dimensions of original submaps
            submaps_num = int(data[2][0][0])
            pred_submaps = np.zeros((submaps_num,64,64,64))
            #Then for the indexes that you do get prediction fill the pred_submaps array
            pred_submaps[data[1][0]] = pred
            #Finally reconstruct your original map
            pred = reconstruct_map(pred_submaps, o_dim=tuple(data[5][0].tolist()))
            #delete pred_submaps
            del pred_submaps
            #Find prediction indexes
            refined_indexes = np.where(pred>=config.cap)
            weights_of_refined_indexes = pred[refined_indexes]
            refined_indexes = np.array(refined_indexes).T
            if len(refined_indexes):
                #Transform them to coordinates
                refined_coords = indexes_to_coordinates(np.array(refined_indexes), np.array(data[4][0]), config.vs)
                #Remove duplicates
                todelete = remove_duplicates(refined_coords, weights_of_refined_indexes, 1.8)
                if len(todelete):
                    pred_coords, scores = np.delete(refined_coords, todelete, 0), np.delete(weights_of_refined_indexes, todelete, 0)
                capped_match_results[v] = allocate_multiprocess_validation_match_waters(np.array(data[3][0]), pred_coords, scores, config.cap_values, config.thresholds, config.threads)
                #Print status
            #if not (v+1)%10:
            print(f'For {(v+1)}/{loaderlen} structures, validation have been completed, {round((time.time()-st)/60, 2)} minutes have passed.', end='\r')
    epoch_loss = np.mean(epoch_loss)
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
        plot_metrics_per_cap_value(config.cap_values, sum_match_results, thresh_ind, threshold, epoch,'./training_results/unet/')
        print(f'{threshold:^10}{round(mean_recall[thresh_ind],3):^8}{round(mean_precision[thresh_ind],3):^12}{round(mean_F1[thresh_ind],3)}')
    print(f'With selected cap value at {config.thresholds[-1]} Å distance, {config.cap_values[argmax_F1_selected]}.')
    #Finally save the match results array and return,  match metrics for best F1 value for largest threhsold value, and the selected cap value.
    np.save(f'./training_results/unet/match_results_epoch_{epoch}.npy', capped_match_results)
    return mean_recall, mean_precision, mean_F1, config.cap_values[argmax_F1_selected], epoch_loss
