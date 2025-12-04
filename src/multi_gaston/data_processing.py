import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from sklearn import preprocessing
import anndata as ad
import scanpy as sc
from zipfile import ZipFile

# Inputs:
#   file_dir: path to data, as a matrix of size N x (2+G), with the first two
#       columns being the spatial coordinates (x,y), and the remaining columns
#       as individual gene expression.
#   slice_names: names of all slices
# 
# Output:
#   slices: a list of number of spots in each slice as [N1,..., N_M]
#   S: array of spatial coordinates, (N_1+...+N_M) x 2
#   A: array of expression matrix, (N_1+...+N_M) x G
def load_slices(file_dir, slice_names, if_compressed = False):
    M = len(slice_names)
    if M == 0: 
        print('Need at least 1 sample input!')
        return
    slices,S,A = [],[],[]
    for s in range(M):
        if len(slice_names[s]) == 0: continue
        filename = f'{slice_names[s]}.csv'
        if if_compressed:
            with ZipFile(file_dir+filename+'.zip', 'r') as zip_file:
                with zip_file.open(filename) as csv_file:
                    slice = pd.read_csv(csv_file,header=None).to_numpy()
        else:
            slice = pd.read_csv(file_dir+filename,header=None).to_numpy()
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


# Inputs:
#   slices: a list of number of spots in each slice as [N1,..., N_M]
#   S: array of spatial coordinates, (N_1+...+N_M) x 2
#   A: array of raw expression matrix, (N_1+...+N_M) x G
#   names: 1D array of feature names
#   min_counts,min_cells: parameter for filterring cells and genes respectively
#   n_top_genes: number of top HVGs to find
# 
# Output:
#   slices,S,A (and optionally feature names) after filtering, as well as raw
#   gene expression matrix A_all_genes with filtered spots
def process_visiumHD(slices,S,A,names=None,min_counts=100,min_cells=50,n_top_genes=1000):
    adata = ad.AnnData(A)
    if names is not None:
        assert len(names) == A.shape[1]
        adata.var['names'] = names
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, inplace=True)
    logA = sc.pp.log1p(adata,copy=True)
    sc.pp.highly_variable_genes(logA, flavor="seurat", inplace=True, n_top_genes=n_top_genes)
    filter_spot_index=np.array(logA.obs['n_counts'].index,dtype=int)
    A_all_genes = A[filter_spot_index]
    A = logA.X[:,logA.var['highly_variable']]
    if names is not None:
        new_names = logA.var['names'][logA.var['highly_variable']]
    else: 
        new_names = None
    S = S[filter_spot_index]
    counter,slices_filtered = 0,slices.copy()
    for l in range(len(slices)):
        slices_filtered[l] = len(np.intersect1d(filter_spot_index,range(counter,counter+slices[l])))
        counter += slices[l]
    slices = slices_filtered
    return slices,S,A_all_genes,A,new_names


# Inputs:
#   file_dir: path to spatial metabolomics data, as a matrix of size N x (2+1+M), 
#       with the first twocolumns being the spatial coordinates (x,y), the 3rd column 
#       TIC, and the remaining columns as individual metabolite abundance.
#   plot: option to plot the TIC heatmap
#   save_dir: directory to save the files
# 
# Output:
#   X,Y: the 2d size of the input data
#   S: spatial coordinates, N x 2
#   A1: normalized metabolite abundance, N x M
def process_metabolite(file_dir, plot = True, save_dir = None):
    # Read in data
    data=pd.read_csv(file_dir,header=None).to_numpy()
    S = data[:,:2]
    A = data[:,2:]
    # Make sure both x and y coordinate start with 0
    S[:,0] = S[:,0] - min(S[:,0])
    S[:,1] = S[:,1] - min(S[:,1]) 
    # Normalize by TIC and then log-transform
    A1 = (A[:,1:] / A[:,0][:,np.newaxis]) * A[:,0].mean()
    A1 = np.log(A1+1)

    X = int(max(S[:,0])-min(S[:,0]))+1
    Y = int(max(S[:,1])-min(S[:,1]))+1
    # Visiualize the tissue by plotting the TIC heatmap
    if plot == True and save_dir is not None: 
        os.makedirs(save_dir, exist_ok=True) 
        plt.imshow(A[:,0].reshape((X,Y)), interpolation='nearest')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Total Ion Count")
        plt.savefig(save_dir)
        plt.close()

    return X,Y,S,A1


# Plot the i-th metabolite abundance
# 
# Inputs:
#   data: spatially resolved data matrix. The first two columns 
#       should bespatial coordinates (x,y), the 3rd column TIC, 
#       and the remaining columns as individual metabolite abundance.
#   id: i-th metabolite index
#   names: metabolite names
#   save_dir: directory to save the files
#   threshold: when > 0, only show metabolite abundance at spots with 
#       abundance over threshold
def visualize_metabolite(data,id,names,save_dir = '', show_plot=True, threshold = -1):
    S = data[:,:2]
    X = int(max(S[:,0])-min(S[:,0]))+1
    Y = int(max(S[:,1])-min(S[:,1]))+1
    mat = np.zeros((X,Y))
    for i in range(0,S.shape[0]):
        row = S[i,:]
        x = int(row[0])
        y = int(row[1])
        if threshold < 0:
            mat[x,y] = data[i,2+id]
        else:
            if data[i,2+id]>= threshold: mat[x,y] = data[i,2+id]
    plt.imshow(mat, interpolation='nearest')
    plt.xlabel("X",fontsize=13)
    plt.ylabel("Y",fontsize=13)
    plt.title(f"Metabolite {id}: {names[id]}",fontsize=15,fontweight='demi')
    if len(save_dir)>0:
        plt.savefig(save_dir+f'M{id}.png')
    if show_plot:
        plt.show()
    plt.close()


# Rescale (z-score normalize) the input matrices as torch tensors
def tensor_transform_inputs(S, A):
  assert S.shape[0] == A.shape[0], 'Spatial matrix and feature matrix do not have same number of spots!'
  
  scaler = preprocessing.StandardScaler().fit(A)
  A_scaled = scaler.transform(A)
  A_torch = torch.tensor(A_scaled,dtype=torch.float32)

  scaler = preprocessing.StandardScaler().fit(S)
  S_scaled = scaler.transform(S)
  S_torch = torch.tensor(S_scaled,dtype=torch.float32)
  return S_torch, A_torch