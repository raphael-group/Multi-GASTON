import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Inputs:
#   file_dir: path to data, as a matrix of size N x (2+1+M), with the first two
#       columns being the spatial coordinates (x,y), the 3rd column TIC, and the 
#       remaining columns as individual metabolite abudacnce
#   plot: whether to plot TIC heatmap
#   save_dir: directory to save the files
# Output:
#   X,Y: the 2d size of the input data
#   S: the spatial coordinates, N x 2
#   A1: the scaled metabolite abundance, N x M
def process(file_dir, plot = True, save_dir = None):
    # Read in data
    data=pd.read_csv(file_dir,header=None).to_numpy()
    S = data[:,:2]
    A = data[:,2:]
    # Make sure both x and y coordinate start with 0
    S[:,0] = S[:,0] - min(S[:,0])
    S[:,1] = S[:,1] - min(S[:,1]) 
    # Normalize by TIC and then log-transform
    A1 = (A[:,3:] / A[:,2][:,np.newaxis]) * A[:,2].mean()
    A1 = np.log(A1+1)

    X = int(max(S[:,0])-min(S[:,0]))+1
    Y = int(max(S[:,1])-min(S[:,1]))+1

    if plot == True and save_dir is not None: 
        os.makedirs(save_dir, exist_ok=True) 
        plt.imshow(A[:,2].reshape((X,Y)), interpolation='nearest')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Total Ion Count")
        plt.savefig(save_dir)
        plt.close()

    return X,Y,S,A1

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
            mat[x,y] = data[i,3+id]
        else:
            if data[i,3+id]>= threshold: mat[x,y] = data[i,3+id]
    plt.imshow(mat, interpolation='nearest')
    plt.xlabel("X",fontsize=13)
    plt.ylabel("Y",fontsize=13)
    plt.title(f"Metabolite {id}: {names[id,0]}",fontsize=15,fontweight='demi')
    if len(save_dir)>0:
        plt.savefig(save_dir+f'M{id}.png')
    if show_plot:
        plt.show()
    plt.close()
