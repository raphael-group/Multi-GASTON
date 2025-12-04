import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from sklearn import preprocessing

# Inputs:
#   file_dir: path to data, as a matrix of size N x (2+1+M), with the first two
#       columns being the spatial coordinates (x,y), the 3rd column TIC, and the 
#       remaining columns as individual metabolite abundance.
#   plot: option to plot the TIC heatmap
#   save_dir: directory to save the files
# 
# Output:
#   X,Y: the 2d size of the input data
#   S: spatial coordinates, N x 2
#   A1: normalized metabolite abundance, N x M
def load_slices(file_dir, slice_names):
    M = len(slice_names)
    if M == 0: 
        print('Need at least 1 sample input!')
        return
    slices,S,A = [],[],[]
    for s in range(M):
        if len(slice_names[s]) == 0: continue
        filename = f'{slice_names[s]}.csv'
        slice = pd.read_csv(folder+filename,header=None).to_numpy()
        A.append(slice[:,2:])
        # We make sure the 2D coordinates in each slice start with zeros (for 
        # downstream plotting purposes)
        S_slice = slice[:,:2]
        S_slice[:,0] -= S_slice[:,0].min()
        S_slice[:,1] -= S_slice[:,1].min()
        S.append(S_slice)
        slices.append(S_slice.shape[0])
        print(f'for the {s}th slice, S.shape = {S[-1].shape}, A.shape = {A[-1].shape}')
    S = np.vstack(S)
    A = np.vstack(A)
    return slices,S,A

def process_visiumHD(slices,S,A,min_counts=100, min_cells=50,n_top_genes=1000,plot = False, save_dir = None):
    adata = ad.AnnData(A)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, inplace=True)
    logA = sc.pp.log1p(adata,copy=True)
    sc.pp.highly_variable_genes(logA, flavor="seurat", inplace=True, n_top_genes=n_top_genes)
    A = logA.X[:,logA.var['highly_variable']]
    filter_spot_index=np.array(logA.obs['n_counts'].index,dtype=int)
    S = S[filter_spot_index]
    counter,slices_filtered = 0,slices.copy()
    for l in range(len(slices)):
        slices_filtered[l] = len(np.intersect1d(filter_spot_index,range(counter,counter+slices[l])))
        counter += slices[l]
    slices = slices_filtered
    return slices,S,A