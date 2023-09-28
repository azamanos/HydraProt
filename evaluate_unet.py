import torch
from params.params import config
from models.unet_model import Unet3D
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.unet_dataset import Validation_coordinates, Validation_transformed_submaps
from train.unet_train_modules import validation_loop
from utils.utils import load_checkpoint, save_checkpoint, write_loss, save_metrics

def main():
    #Load validation coordinates data
    validation_coordinates = Validation_coordinates(config.validation_dataset_path, config.validation_list_path)
    #Transform validation coordinates
    validation_dataset = Validation_transformed_submaps(validation_coordinates, config.vs, config.pad)
    #Define validation loader
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
    #To save accuracy metrics
    writer = SummaryWriter(f'./checkpoints/unet/report/evaluation')
    torch.cuda.empty_cache()
    #Initialize model
    model = Unet3D(config.in_channels, config.out_channels, config.intermediate_channels, config.dropout_p)
    model.to(config.device)
    #Load model
    for epoch in np.linspace(19,399,77).astype(int):
        #First load the checkpoint
        load_checkpoint(torch.load(f'./checkpoints/unet/Unet3D_epoch_{epoch}.pth.tar', map_location=config.device), model, epoch)
        #Evaluate models with validation set.
        recall, precision, F1, selected_cap_value, epoch_loss = validation_loop(validation_loader, model, epoch, config)
        #And save metrics
        save_metrics(config.thresholds, writer, recall, precision, F1, selected_cap_value, 'validation', epoch)
        #Write loss to tensorboard report
        write_loss(writer, epoch_loss, epoch)
    writer.close()
    return

if __name__ == '__main__':
    main()
