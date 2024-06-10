import argparse

import os
from time import time
from nettcr_archs import NetTCRGlobalMax
import numpy as np
import pandas as pd 

#Imports the util module and network architectures for NetTCR
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from constants import blosum50_20aa
from dataset import CustomDataset
from utils import EarlyStopper, make_sequence_ds

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", default="../../NetTCR-2.2/data/examples/train_example.csv", help = "Dataset used for training", type = str)
parser.add_argument("--val_data", default="../../NetTCR-2.2/data/examples/validation_example.csv", help = "Dataset used for validation", type = str)
parser.add_argument("--outdir", default = "models", help = "Folder to save the model in", type = str)
parser.add_argument("--model_name", default = "peptide_model", help = "Prefix of the saved model", type = str)
parser.add_argument("--dropout_rate", "-dr", default = 0.6, help = "Fraction of concatenated max-pooling values set to 0. Used for preventing overfitting", type = float)
parser.add_argument("--learning_rate", "-lr", default = 0.001, type = float)
parser.add_argument("--patience", "-p", default = 100, type = int)
parser.add_argument("--batch_size", "-bs", default = 64, type = int)
parser.add_argument("--epochs", "-e", default = 200, type = int)
parser.add_argument("--verbose", default = 2, choices = [0,1,2], type = int)
parser.add_argument("--seed", default = 15, type = int)
parser.add_argument("--inter_threads", default = 1, type = int)
    
args = parser.parse_args()

### Model training parameters ###
train_data = str(args.train_data)
val_data = str(args.val_data)
outdir = str(args.outdir)
model_name = str(args.model_name)
patience = int(args.patience) #Patience for Early Stopping
dropout_rate = float(args.dropout_rate) #Dropout Rate
lr = float(args.learning_rate)
batch_size = int(args.batch_size) #Default batch size. Might be changed slightly due to the adjust_batch_size function
EPOCHS = int(args.epochs) #Number of epochs in the training
verbose = int(args.verbose) #Determines how often metrics are reported during training
seed = int(args.seed)

encoding = blosum50_20aa #Encoding for amino acid sequences


def train(train_data_dir, val_data_dir, outdir, model_name, dropout_rate, lr, patience, batch_size, EPOCHS, seed):
    #Creation of output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Set random seed for reproducibility
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # Load the data
    train_data = CustomDataset(train_data_dir, encoding)
    val_data = CustomDataset(val_data_dir, encoding)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training on GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Training on MPS")
    else:
        device = torch.device("cpu")
        print("Training on CPU")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # initialize model and optimizer
    model = NetTCRGlobalMax(num_filters=16, embed_dim=20, dropout_rate=dropout_rate)
    model = model.to(device)
    loss_fn = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

    # early stopper
    early_stopper = EarlyStopper(patience=patience, min_delta=0)

    # keep track of losses
    train_losses = []
    val_losses = [] 

    # Training loop
    start_time = time()
    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        for batch_idx, (data, target, weight) in enumerate(train_loader):
            data = [i.to(device) for i in data]
            target = target.to(device)          
            weight = weight.to(device)
            optimizer.zero_grad()
            pred = model(data)
            target = target.to(torch.float32)
            loss = loss_fn(pred, target)
            weighted_loss = (loss * weight).mean()  # Apply sample weights
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item()
            print(f"Time passed: {time() - start_time} seconds")
            print(f'Epoch [{epoch+1}/{EPOCHS}], batch_ids {batch_idx+1}/{len(train_loader)}')
            

        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target, weight in val_loader:
                data = [i.to(device) for i in data]
                target = target.to(device)
                weight = weight.to(device)

                pred = model(data)
                target = target.to(torch.float32)
                loss = loss_fn(pred, target)
                val_loss += loss.item()

        # Print training progress
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save model checkpoint every 100 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'{outdir}/checkpoint/model_{model_name}_checkpoint_epoch_{epoch+1}.pt')

        # Early stop
        if early_stopper.early_stop(val_losses): 
            print("Early stopping at epoch {0}".format(epoch))            
            break

    print("Training finished!")
        
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train(train_data, val_data, outdir, model_name, dropout_rate, lr, patience, batch_size, EPOCHS, seed)