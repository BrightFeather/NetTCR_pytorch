# -*- coding: utf-8 -*-
#!/usr/bin/env python
import argparse

import onnxruntime as ort

import onnx
import torch

import numpy as np
import pandas as pd 
import random
import time

from torch.utils.data import DataLoader
from dataset import CustomDataset

from constants import blosum50_20aa
from utils import make_sequence_ds, roc_auc_function
from sklearn.metrics import roc_auc_score

ENCODING = blosum50_20aa #Encoding for amino acid sequences

def load_model(path):
    # Create an inference session with onnxruntime
    return ort.InferenceSession(path)

def compute_metrics(y_true, y_pred):
    accuracy = (np.array(y_pred) == np.array(y_true)).mean()
    auc = roc_auc_score(y_true, y_pred)
    return accuracy, auc

def eval(test_data, outdir, model_name, model_type, seed=15, batch_size=64):
    # Read in data
    test_data = CustomDataset(test_data, ENCODING, isEval=True)
    data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # cuda setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Eval on GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Eval on MPS")
    else:
        device = torch.device("cpu")
        print("Eval on CPU")

    time_start = time.time()

    # Load the TFLite model and allocate tensors.
    try:
        # Load ONNX model using onnxruntime
        session = load_model("models/onnx/t.0.v.1.onnx")

        y_pred = []
        y_true = []

        # Iterate over the data_loader
        for batch_idx, (data, labels, weight) in enumerate(data_loader):
            # Convert data to a format suitable for ONNX runtime (numpy)
            
            input_data = {k:v.to(device).numpy() for k,v in data.items()}
            labels = labels.to(device)
            # Run inference
            output = session.run(None, input_data)

            # Convert outputs to tensor and get predictions
            output_tensor = torch.tensor(output[0])
            preds = (output_tensor > 0.5).to(torch.float32)
            preds.squeeze()

            # Store predictions and true labels
            y_pred.extend(preds.numpy())
            y_true.extend(labels.numpy())

            print(f"Batch id: {batch_idx}")

        # Compute accuracy and AUC
        accuracy, auc = compute_metrics(y_true, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
        print(f'AUC (One-vs-Rest): {auc:.4f}')

        # Save prediction
        pd.DataFrame(y_pred).to_csv(outdir + '/{}_prediction.csv'.format(model_name), index=False)
        
        #Report time spent for prediction
        print(str(round(time.time()-time_start, 3))+" seconds")
    except ValueError:
        print("A model for pan does not exist. Skipping predictions for this peptide")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", help = "Dataset to perform the predictions on", type = str)
    parser.add_argument("--outdir", default = "data/eval", help = "Folder where the models are stored. The prediction file is saved to this location", type = str)
    parser.add_argument("--model_name", default = "pan", help = "Prefix for the saved models. This prefix is also used for the prediction file", type = str)
    parser.add_argument("--model_type", default = "pan", help = "Type of NetTCR 2.2 model", choices = ["pan", "peptide", "pretrained"], type = str)
    parser.add_argument("--seed", default = 15, type = int)
    parser.add_argument("--batch_size", "-bs", default = 64, type = int)
    parser.add_argument("--inter_threads", default=1)    

    args = parser.parse_args()

    test_data = str(args.test_data)
    outdir = str(args.outdir)
    model_name = str(args.model_name)
    model_type = str(args.model_type)
    seed = int(args.seed)

    # for debug
    # test_data = "/Users/chenweijia/Documents/code/nettcr_pytorch/data/train_example.csv"

    if model_type == "pan":
        eval(test_data, outdir, model_name, model_type, seed=15, batch_size=64)
    


