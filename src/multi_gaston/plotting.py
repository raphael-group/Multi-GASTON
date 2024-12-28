import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

import numpy as np
import matplotlib.pyplot as plt
import os
from multi_gaston import Multi_GASTON

# Inputs: 
#   save_dir: directory to save the image to
#   loss_list: training loss list over all epochs
#   lasso_loss: training lasso loss
def plot_loss(save_dir, loss_list, lasso_loss = None):
    os.makedirs(save_dir, exist_ok=True) 
    if lasso_loss is None:
        plt.plot(range(500,len(loss_list)),loss_list[500:len(loss_list)])
        plt.yscale('log')
        plt.title(f'Training Loss with final loss {loss_list[-1]}',fontsize=13,weight='demi')
        plt.xlabel('Epoch',fontsize=13)
        plt.ylabel('Log(loss)',fontsize=13)
        plt.savefig(save_dir+'/loss.png')
        plt.close()
    else:
        actual_loss = loss_list - lasso_loss
        plt.subplot(121)
        plt.plot(range(500,len(actual_loss)),actual_loss[500:len(actual_loss)])
        plt.yscale('log')
        plt.title(f'Training Loss with final loss {actual_loss[-1]}',fontsize=13,weight='demi')
        plt.xlabel('Epoch',fontsize=13)
        plt.ylabel('Log(loss)',fontsize=13)

        plt.subplot(122)
        plt.plot(range(500,len(lasso_loss)),lasso_loss[500:len(lasso_loss)])
        plt.yscale('log')
        plt.title(f'Lasso Loss',fontsize=13,weight='demi')
        plt.xlabel('Epoch',fontsize=13)
        plt.ylabel('Log(loss)',fontsize=13)

        plt.savefig(save_dir+'/loss.png')
        plt.close()

# Inputs: 
#   save_dir: directory to save the image to
#   S: original unscaled spatial coordinates, (N x 2) numpy array
#   isodepth: the learned isodepth(s), (N,) or (N x K) numpy array 
#   percentile_plot: whether to plot the percentile heatmap, useful for small or 
#       samples with strong spatial gradients e.g. within intestinal villi. only for
#       multiple isodepth plotting
def plot_liver_isodepth(save_dir, S, isodepth, percentile_plot = False):
    os.makedirs(save_dir, exist_ok=True) 
    X,Y = int(max(S[:,0])-min(S[:,0])+1),int(max(S[:,1])-min(S[:,1])+1)
    # single isodepth
    if len(isodepth.shape) == 1:
        plt.imshow(isodepth.reshape((X,Y)), cmap='hot', interpolation='nearest')
        plt.contour(isodepth.reshape((X,Y)))
        plt.xlabel("X",fontsize=13)
        plt.ylabel("Y",fontsize=13)
        plt.title(f'Isodepth',fontsize=15,fontweight='demi')
        plt.savefig(save_dir+f'/heatmap.png')
        plt.close()
    # multiple isodepths
    else:
        for k in range(isodepth.shape[1]):
            isodepth1 = isodepth[:,k]
            plt.imshow(isodepth1.reshape((X,Y)), cmap='hot', interpolation='nearest')
            plt.contour(isodepth1.reshape((X,Y)))
            plt.xlabel("X",fontsize=13)
            plt.ylabel("Y",fontsize=13)
            plt.title(f'Isodepth {k}',fontsize=15,fontweight='demi')
            plt.savefig(save_dir+f'/heatmap{k}.png')
            plt.close()

    if percentile_plot:
        for k in range(isodepth.shape[1]):
            isodepth1 = isodepth[:,k]
            t1,t2,t3 = np.percentile(isodepth1, 25),np.percentile(isodepth1, 50),np.percentile(isodepth1, 75)
            vein_mat = np.zeros((X,Y))
            for i in range(0,S.shape[0]):
                row = S[i,:]
                x = int(row[0])
                y = int(row[1])
                if isodepth1[i] >= t3:vein_mat[x,y] = 1
                elif isodepth1[i] >= t2:vein_mat[x,y] = 0.75
                elif isodepth1[i] >= t1:vein_mat[x,y] = 0.5
                else:vein_mat[x,y] = 0.25
            plt.imshow(vein_mat, cmap='hot', interpolation='nearest')
            plt.contour(vein_mat)
            plt.xlabel("X",fontsize=13)
            plt.ylabel("Y",fontsize=13)
            plt.title(f'Isodepth {k}',fontsize=15,fontweight='demi')
            plt.savefig(save_dir+f'/heatmap{k}_perc.png')
            plt.close()

# Plot the (linear) abundance mapping weights after training multiple isodpeths
# Inputs: 
#   model: multi GASTON object
#   save_dir: directory to save the image to
#   isodepths: learned isodepths (only for multiple-isodepth)
def plot_weights(model, save_dir, isodepths):
    os.makedirs(save_dir, exist_ok=True) 
    weights = model.expression_function[0].weight.data.detach().numpy()
    plt.figure(figsize=(8,5))
    for k in range(weights.shape[1]):
        # first scale the weithts by the corresponding isodepths
        weights[:,k] *= np.linalg.norm(isodepths[:,k])
        w = [x for _, x in sorted(zip(weights[:,0], weights[:,k]),reverse=False)]
        plt.scatter(range(len(w)),w,label=f'Isodepth {k}',alpha=0.5)
    plt.title(f'Scaled Weights of Mean-1 Isodepths',fontsize=13,weight='demi')
    plt.xlabel('Metabolite index, sorted by the first-isodepth weight',fontsize=13)
    plt.ylabel('Weight',fontsize=13)
    plt.legend(loc='best',fontsize=13)
    plt.savefig(save_dir+'/weights.png', bbox_inches='tight')
    plt.close()
