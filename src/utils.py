####!/usr/bin/env python

"""
Functions for data IO for neural network training.
"""

from __future__ import print_function
import argparse
import sys
import os
import time
import matplotlib.pyplot as plt
from operator import add
import math
from sklearn.metrics import roc_auc_score
import torch
from constants import *

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def mkdir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

def enc_list_bl_max_len(aa_seqs, blosum, max_seq_len, padding = "right", pad_value = -5):
    '''
    blosum encoding of a list of amino acid sequences with padding 
    to a max length

    parameters:
        - aa_seqs : list with AA sequences
        - blosum : dictionnary: key= AA, value= blosum encoding
        - max_seq_len: common length for padding
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''

    # encode sequences:
    sequences=[]
    for seq in aa_seqs:
        e_seq= np.zeros((len(seq),len(blosum["A"])))
        count=0
        for aa in seq:
            if aa in blosum:
                e_seq[count]=blosum[aa]
                count+=1
            else:
                sys.stderr.write("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
                sys.exit(2)
                
        sequences.append(e_seq)

    # pad sequences:
    #max_seq_len = max([len(x) for x in aa_seqs])
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = pad_value*np.ones((n_seqs, max_seq_len, n_features))
    if padding == "right":
        for i in range(0,n_seqs):
            enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]
            
    elif padding == "left":
        for i in range(0,n_seqs):
            enc_aa_seq[i, max_seq_len-sequences[i].shape[0]:max_seq_len, :n_features] = sequences[i]
    
    else:
        print("Error: No valid padding has been chosen.\nValid options: 'right', 'left'")
        

    return enc_aa_seq

def enc_list_bl_max_len_pos(aa_seqs, blosum, max_seq_len, padding = "right"):
    '''
    blosum encoding of a list of amino acid sequences with padding 
    to a max length

    parameters:
        - aa_seqs : list with AA sequences
        - blosum : dictionnary: key= AA, value= blosum encoding
        - max_seq_len: common length for padding
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''

    # encode sequences:
    sequences=[]
    for seq in aa_seqs:
        e_seq= np.zeros((len(seq),len(blosum["A"])+1)).astype(float)
        count=0
        for aa in seq:
            if aa in blosum:
                pos_enc = ((count+1)/len(seq))*5
                #print(blosum[aa])
                e_seq[count]= np.insert(blosum[aa].astype(float), 0, float(pos_enc), axis = 0)
                count+=1
            else:
                sys.stderr.write("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
                sys.exit(2)
                
        sequences.append(e_seq)

    # pad sequences:
    #max_seq_len = max([len(x) for x in aa_seqs])
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = -5*np.ones((n_seqs, max_seq_len, n_features))
    if padding == "right":
        for i in range(0,n_seqs):
            enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]
            
    elif padding == "left":
        for i in range(0,n_seqs):
            enc_aa_seq[i, max_seq_len-sequences[i].shape[0]:max_seq_len, :n_features] = sequences[i]
    
    else:
        print("Error: No valid padding has been chosen.\nValid options: 'right', 'left'")
        

    return enc_aa_seq
            
def adjust_batch_size(obs, batch_size, threshold = 0.5):
    if obs/batch_size < threshold:
        pass
    
    else:
        if (obs/batch_size % 1) >= threshold:
            pass
        else:
            while (obs/batch_size % 1) < threshold and (obs/batch_size % 1) != 0:
                batch_size += 1
    return batch_size

def make_sequence_ds(df, encoding):
    """Encodes amino acid sequences using a BLOSUM50 matrix with a normalization factor of 5.
    Sequences are right-padded with [-1x20] for each AA missing, compared to the maximum embedding 
    length for that given feature
    
    Additionally, the input is prepared for predictions, by loading the data into a list of numpy arrays"""
    encoded_pep = enc_list_bl_max_len(df.peptide, encoding, pep_max)/5
    encoded_a1 = enc_list_bl_max_len(df.A1, encoding, a1_max)/5
    encoded_a2 = enc_list_bl_max_len(df.A2, encoding, a2_max)/5
    encoded_a3 = enc_list_bl_max_len(df.A3, encoding, a3_max)/5
    encoded_b1 = enc_list_bl_max_len(df.B1, encoding, b1_max)/5
    encoded_b2 = enc_list_bl_max_len(df.B2, encoding, b2_max)/5
    encoded_b3 = enc_list_bl_max_len(df.B3, encoding, b3_max)/5
    X = [np.float32(t) for t in [encoded_pep, encoded_a1, encoded_a2, encoded_a3, encoded_b1, encoded_b2, encoded_b3]]
    X = [np.transpose(t, (0,2,1)) for t in X]
    targets = df.binder.values
    sample_weights = df.sample_weight
    return X, np.float32(targets), np.float32(sample_weights)

def my_roc_auc_function(y_true, y_pred):
    """Implementation of AUC 0.1 metric for Tensorflow"""
    try:
        auc = roc_auc_score(y_true, y_pred, max_fpr = 0.1)
    except ValueError:
        auc = np.array([float(0)])
    return auc

#Custom metric for AUC 0.1
# def auc_01(y_true, y_pred):
#     """Converts function to optimised tensorflow numpy function"""
#     auc_01 = tf.numpy_function(my_roc_auc_function, [y_true, y_pred], tf.float64)
#     return auc_01