from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from utils import make_sequence_ds, make_sequence_ds_eval

class CustomDataset(Dataset):
    def __init__(self, dir, encoding, isEval=False):
        self.df = pd.read_csv(dir)
        self.encoding = encoding
        self.isEval = isEval
        self.__validate__()

        # calculate sample weight
        weight_dict = np.log2(self.df.shape[0]/(self.df.peptide.value_counts()))/np.log2(len(self.df.peptide.unique()))
        #Normalize, so that loss is comparable
        weight_dict = weight_dict*(self.df.shape[0]/np.sum(weight_dict*self.df.peptide.value_counts()))
        self.df["sample_weight"] = self.df["peptide"].map(weight_dict)

    def __validate__(self):
        # From https://github.com/mnielLab/NetTCR-2.1
        assert "A1" and "A2" and "A3" in self.df.columns, "Make sure the input files contains all the CDRs"
        assert "B1" and "B2" and "B3" in self.df.columns, "Make sure the input files contains all the CDRs"
        assert "peptide" in self.df.columns, "Couldn't find peptide in the input data"
        assert "binder" in self.df.columns, "Couldn't find target labels in the input data, which is required for training"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not self.isEval:
            X, y, weights = make_sequence_ds(self.df, encoding = self.encoding) # X_train, targets_train, weights_train
            return [xx[idx,:,:] for xx in X], y[idx], weights[idx]
        else:
            X, y, weights = make_sequence_ds_eval(self.df, encoding = self.encoding) # X_train, targets_train, weights_train
            return {k: v[idx] for k, v in X.items()}, y[idx], weights[idx]