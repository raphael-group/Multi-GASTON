#################################################################
############################ IMPORTS ############################
#################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import sys
import os
import argparse

########################################################################
# MAIN
########################################################################
def main():

    # ######## Arguments Parser ########
    description="Multi-GASTON"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-i', '--input', type=str, required=True, 
                        help="filename for the input, a N x (2+M) numpy array with the first 2 columns being coordinates (x,y), and each of the remaining column the metabolite abundance of one metabolite")
    parser.add_argument('-dir', '--output_dir', type=str, required=False, default="./", help="The directory to save the output files")

    parser.add_argument('-s', '--hiddenS', type=str, required=True, 
                        help="architecture of the spatial mapping NN transforming spatial coordinates (x,y) to isodepths, in the form 'a1,a2,..,an', with ai hidden nodes in the i-th hidden layer")
    parser.add_argument('-a', '--hiddenA', type=str, required=True, 
                        help="architecture of the abundnace mapping NN transforming isodepths to metabolite abundance, in the form 'a1,a2,..,an' with ai hidden nodes in the i-th hidden layer")
    parser.add_argument('-e', '--epochs', type=int, required=False, default=10000, help="number of epochs to train the neural network")
    parser.add_argument('-c', '--checkpoint', type=int, required=False, default=1000, help="save model every checkpoint epochs")
    parser.add_argument('-t', '--trial', type=int, required=False, default=0, help="randan trial seed for reproducibility")
    parser.add_argument('-o', '--optimizer', type=str, required=False, default="adam", help="optimizer for fitting the neural network")

    parser.add_argument('-l', '--lasso', type=int, required=False, default=0, help="if positive, apply lasso regularization with given lasso coefficient")
    parser.add_argument('-k', '--K', type=int, required=False, default=1, help="number of isodepths to learn")

    parser.add_argument('-conv', '--conv_thd', type=float, required=False, default=1e-9, help="NN loss convergence check threshold")
    parser.add_argument('-lr', '--lr', type=float, required=False, default=1e-3, help="learning rate")
    parser.add_argument('-scl', '--scheduler', type=str, required=False, default='linear', help="type of learning rate decay, currently accepts linear/expo")
    parser.add_argument('-spara', '--scheduler_para', type=str, required=False, default='0.05,12000', 
                        help="learning rate decay parameters. If linear, scheduler_para[0]=end_factor, scheduler_para[1]=total_iter; if expo,scheduler_para[0]=multiplicative factor gamma")

    parser.add_argument('-pos', '--positional_encoding', action='store_true', help="positional encoding option")
    parser.add_argument('-emb', '--embedding_size', type=int, required=False, default=4, help="positional encoding embedding size")
    parser.add_argument('-sig', '--sigma', type=float, required=False, default=0.1, help="positional encoding sigma hyperparameter")
    
    args=parser.parse_args(sys.argv[1:])

    out_dir_seed=f"{args.output_dir}/trial{args.trial}" # save in rep{seed}
    os.makedirs(out_dir_seed, exist_ok=True) 
    nhS = [int(i) for i in str(args.hiddenS).split(',')]
    nhA = [int(i) for i in str(args.hiddenA).split(',')]
    scheduler_para = [float(i) for i in str(args.scheduler_para).split(',')]
    # We set the expression function to be linear when learning multiple 
    # isodepths
    if args.K > 1: A_linear = True
    else: A_linear = False

    # ######## Load data ######## 
    # rescale as torch tensors S_torch and A_torch, representing the spatial
    # and abundance matrix respectively.    
    data=np.load(args.input)
    S = data[:,:2]
    A = data[:,2:]

    scaler = preprocessing.StandardScaler().fit(A)
    A_scaled = scaler.transform(A)

    scaler = preprocessing.StandardScaler().fit(S)
    S_scaled = scaler.transform(S)

    S_torch=torch.tensor(S_scaled,dtype=torch.float32)
    A_torch=torch.tensor(A_scaled,dtype=torch.float32)

    # ######## Train neural net ########
    from multi_gaston import train
    mod, loss_list, lasso_loss = train(S_torch, A_torch, 
            S_hidden_list=nhS, A_hidden_list=nhA, 
            epochs=args.epochs, checkpoint=args.checkpoint,
            SAVE_PATH=out_dir_seed, optim=args.optimizer, seed=args.trial, 
            A_linear=A_linear, lasso_lambda=args.lasso, K = args.K, loss_threshold=args.conv_thd,
            schedule=args.scheduler, schedule_para=scheduler_para, lr=args.lr,
            pos_encoding=args.positional_encoding, embed_size=args.embedding_size, sigma=args.sigma)

if __name__ == '__main__':
    main()
