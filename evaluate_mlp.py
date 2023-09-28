import torch
from params.params import config
from models.mlp_model import HydrationNN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train.mlp_train_modules import validation_loop
from datasets.mlp_dataset import ValidationEmbeddingLoader
from utils.utils import write_loss, save_checkpoint, load_checkpoint, save_metrics

def main():
    #Load validation data and shuffle once
    validation_dataset = ValidationEmbeddingLoader(config.validation_dataset_path, config.validation_dataset_list, shuffle=True)
    #Define loaders
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory = config.pin_memory)
    #To save accuracy metrics
    writer = SummaryWriter(f'./checkpoints/mlp/report/evaluation/')
    torch.cuda.empty_cache()
    #Initialize model
    model = HydrationNN(config.first_part_features, config.second_part_features, config.dropout_p)
    model.to(device)
    #Load model
    for epoch in np.linspace(19,399,77).astype(int):
        load_checkpoint(torch.load(f'./checkpoints/mlp/HydrationNN_epoch_{epoch}.pth.tar', map_location=device), model, epoch)
        #Evaluate models with validation set.
        recall, precision, F1, selected_cap_value, validation_epoch_loss = validation_loop(validation_loader, model, epoch, config)
        #And save metrics
        save_metrics(config.thresholds, writer, recall, precision, F1, selected_cap_value, 'validation', epoch)
        #Write loss to tensorboard report
        write_loss(writer, validation_epoch_loss, epoch)
    writer.close()
    return

if __name__ == '__main__':
    main()
