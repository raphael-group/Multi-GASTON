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
def process(file_dir, plot = True, save_dir = None):
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
def plot_mat(data,id,names,save_dir = '', show_plot=True, threshold = -1):
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