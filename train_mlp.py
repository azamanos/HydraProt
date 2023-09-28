import time
import torch
from params.mlp_params import config
from models.mlp_model import HydrationNN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train.mlp_train_modules import train_loop, validation_loop
from datasets.mlp_dataset import TrainingEmbeddingLoader, ValidationEmbeddingLoader
from utils.utils import shuffle_along_axis, write_loss, save_checkpoint, load_checkpoint, save_metrics


def main():
    torch.set_num_threads(1)
    #To save accuracy metrics
    writer = SummaryWriter(f'./checkpoints/mlp/report/')
    #Load training data
    train_dataset = TrainingEmbeddingLoader(config.train_dataset_path, config.train_dataset_list)
    #Load validation data and shuffle once
    validation_dataset = ValidationEmbeddingLoader(config.validation_dataset_path, config.validation_dataset_list, shuffle=True)
    #Shuffle
    shuffle_time = time.time()
    train_dataset.x = shuffle_along_axis(train_dataset.x, 1)
    #Define loaders
    train_loader = DataLoader(train_dataset, batch_size=config.training_batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory = config.pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory = config.pin_memory)
    print(f'Train dataset shuffling took {round(time.time()-shuffle_time,1)} seconds.')
    #Initialize model
    model = HydrationNN(config.first_part_features, config.second_part_features, config.dropout_p)
    #Define optimizer
    model_params = list(model.parameters())
    config.optimizer = config.optim_algorithm(model_params, lr=config.learning_rate, weight_decay=config.weight_dec)
    #Use all available GPUs
    #Load model
    if config.load_model:
        epoch = config.starting_epoch-1
        load_checkpoint(torch.load(config.load_model, map_location=config.device), model, epoch)
    #Use all available GPUs if you want
    if config.parallel:
        if torch.cuda.device_count() > 1:
            print(f"Let's use, {torch.cuda.device_count()}, GPUs!")
            model = nn.DataParallel(model)
    model.to(config.device)
    for epoch in range(config.starting_epoch, config.starting_epoch+config.num_epochs):
        #Train you models
        epoch_loss = train_loop(train_loader, model, config)
        #Write loss to tensorboard report
        write_loss(writer, epoch_loss, epoch)
        #Save models every 5 epochs.
        if not (epoch+1)%5:
            #Save model checkpoints according to training devices.
            if config.parallel:
                model_checkpoint = {"state_dict": model.module.state_dict()}
            else:
                model_checkpoint = {"state_dict": model.state_dict()}
            #Save it
            save_checkpoint(model_checkpoint, epoch, filename=f'./checkpoints/mlp/HydrationNN_epoch_{epoch}.pth.tar')
            #After epoch 20 start evaluation
            if (epoch+1) >= 20:
                #Evaluate models with validation set.
                recall, precision, F1, selected_cap_value, validation_epoch_loss = validation_loop(validation_loader, model, epoch, config)
                #And save metrics
                save_metrics(config.thresholds, writer, recall, precision, F1, selected_cap_value, 'validation', epoch)
            #Every 5 epochs also shuffle training embedding
            shuffle_time = time.time()
            train_dataset.x = shuffle_along_axis(train_dataset.x, 1)
            #Redefine loaders
            train_loader = DataLoader(train_dataset, batch_size=config.training_batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory = config.pin_memory)
            print(f'Train dataset shuffling took {round(time.time()-shuffle_time,1)} seconds.')
    writer.close()
    return

if __name__ == '__main__':
    main()
