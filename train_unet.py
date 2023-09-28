import torch
from params.unet_params import config
from models.unet_model import Unet3D
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.unet_dataset import Training_coordinates, Validation_coordinates, Training_transformed_submaps, Validation_transformed_submaps
from train.unet_train_modules import train_loop, validation_loop
from utils.utils import load_checkpoint, save_checkpoint, write_loss, save_metrics


def main():
    torch.set_num_threads(1)
    #To save accuracy metrics
    writer = SummaryWriter(f'./checkpoints/unet/report/')
    #Load training coordinates data
    train_coordinates = Training_coordinates(config.train_dataset_path, config.train_list_path)
    #Load validation coordinates data
    validation_coordinates = Validation_coordinates(config.validation_dataset_path, config.validation_list_path)
    #Initialize model
    model = Unet3D(config.in_channels, config.out_channels, config.intermediate_channels, config.dropout_p)
    #Define optimizer
    model_params = list(model.parameters())
    config.optimizer = config.optim_algorithm(model_params, lr=config.learning_rate, weight_decay=config.weight_dec)
    #Load model
    if config.load_model:
        epoch = config.starting_epoch-1
        load_checkpoint(torch.load(config.load_model, map_location=config.device), model, epoch)
        if load_and_validate:
            #Evaluate models with validation set.
            recall, precision, F1, selected_cap_value, epoch_loss = validation_loop(validation_loader, model, epoch, config)
            #And save metrics
            save_metrics(config.thresholds, writer, recall, precision, F1, selected_cap_value, 'validation', epoch)
    #Use all available GPUs if you want
    if config.parallel:
        if torch.cuda.device_count() > 1:
            print(f"Let's use, {torch.cuda.device_count()}, GPUs!")
            model = nn.DataParallel(model)
    model.to(config.device)
    #Transform coordinates and create training dataset
    train_dataset = Training_transformed_submaps(train_coordinates, config.vs, config.pad, config.flip)
    #Define training loader
    train_loader = DataLoader(train_dataset, batch_size=config.training_batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory=config.pin_memory)
    #Transform validation coordinates
    validation_dataset = Validation_transformed_submaps(validation_coordinates, config.vs, config.pad)
    #Define validation loader
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
    #Start Training
    for epoch in range(config.starting_epoch, config.starting_epoch+config.num_epochs):
        #Train you model
        epoch_loss = train_loop(train_loader, model, config)
        #Write loss to tensorboard report
        write_loss(writer, epoch_loss, epoch)
        #Save model every five epochs.
        if not (1+epoch)%5:
            #Save model checkpoints according to training devices.
            if config.parallel:
                model_checkpoint = {"state_dict": model.module.state_dict()}
            else:
                model_checkpoint = {"state_dict": model.state_dict()}
            #Save it
            save_checkpoint(model_checkpoint, epoch, filename=f'./checkpoints/unet/Unet3D_epoch_{epoch}.pth.tar')
            #After epoch 20 start evaluation
            if (epoch+1) >= 20:
                #Evaluate models with validation set.
                recall, precision, F1, selected_cap_value, epoch_loss = validation_loop(validation_loader, model, epoch, config)
                #And save metrics
                save_metrics(config.thresholds, writer, recall, precision, F1, selected_cap_value, 'validation', epoch)
            #After epoch 150 start transforming input data.
            if (epoch+1) >= 150:
                #Transform coordinates and create training dataset
                train_dataset = Training_transformed_submaps(train_coordinates, config.vs, config.pad, config.flip)
                #Define anew training loader
                train_loader = DataLoader(train_dataset, batch_size=config.training_batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory=config.pin_memory)
    writer.close()
    return

if __name__ == '__main__':
    main()
