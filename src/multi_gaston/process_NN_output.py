import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# Inputs: 
#   save_dir: directory to save the image to
#   S: original unscaled spatial coordinates, (N x 2) numpy array
#   isodepth: the learned isodepth(s), (N,) or (N x K) numpy array 
#   percentile_plot: whether to plot the percentile heatmap, useful for small or 
#       samples with strong spatial gradients e.g. within intestinal villi. only for
#       multiple isodepth plotting
def plot_isodepth(isodepth, S, save_dir = '', percentile_plot = False ,show_plot=False,n_contour=7):
    X,Y = int(max(S[:,0])-min(S[:,0])+1),int(max(S[:,1])-min(S[:,1])+1)
    # Single isodepth
    if len(isodepth.shape) == 1:
        plt.figure(figsize=(6/Y*X,6.3))
        plt.tricontour(S[:,0],S[:,1], isodepth, n_contour, linewidths=0.5, colors='k')
        plt.tricontourf(S[:,0],S[:,1], isodepth, n_contour,cmap='coolwarm')
        plt.title('Isodepth',fontsize=15,fontweight='demi')
        if len(save_dir)>0: plt.savefig(save_dir+f'/isodepth.png')
        if show_plot: plt.show()
        plt.close()
    # Multiple isodepths
    else:
        # For fine tissues, e.g. intestinal villi, it's easier to visualize the isodepth by plotting the , 
        # percentiles, i.e. for spots with the lowest 25% isodepth values, we plot them with color 1; for spots  
        # with the 25%-50% lowest isodepth values, plot with color 2, etc.
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
                plt.imshow(vein_mat, cmap='coolwarm', interpolation='nearest')
                plt.title(f'Isodepth {k}',fontsize=15,fontweight='demi')
                if len(save_dir)>0: plt.savefig(save_dir+f'/isodepth{k}.png')
                if show_plot: plt.show()
                plt.close()
        else:
            for k in range(isodepth.shape[1]):
                isodepth1 = isodepth[:,k].copy()
                isodepth1 -= isodepth1.min()
                isodepth1 *= 1/isodepth1.max()
                vein_mat = np.ones((X,Y))*-0.1
                for i in range(0,S.shape[0]):
                    row = S[i,:]
                    x = int(row[0])
                    y = int(row[1])
                    vein_mat[x,y] = isodepth1[i]
                plt.imshow(vein_mat, cmap='coolwarm', interpolation='nearest')
                # plt.contour(vein_mat)
                plt.title(f'Isodepth {k}',fontsize=15,fontweight='demi')
                if len(save_dir)>0: plt.savefig(save_dir+f'/isodepth{k}.png')
                if show_plot: plt.show()
                plt.close()

# Plot the isodepth result from trials, for K=1 isodepth
def plot_trials(output_dir, save_dir, S, trial_id='unknown'):
    X,Y = int(max(S[:,0])-min(S[:,0])+1),int(max(S[:,1])-min(S[:,1])+1)
    loss = np.load(output_dir+'loss_list.npy')[-1]
    isodepth = np.loadtxt(output_dir+'isodepth.txt')
    title=f'Trial {trial_id}: Isodepth'
    save_dir=save_dir+f'/loss{loss:.4g}_trial{trial_id}_isodepth.png'
    plt.figure(figsize=(6/Y*X,6.3))
    plt.tricontour(S[:,0],S[:,1], isodepth, 7, linewidths=0.5, colors='k')
    plt.tricontourf(S[:,0],S[:,1], isodepth, 7,cmap='coolwarm')
    plt.title(title,fontsize=15,fontweight='demi')
    plt.savefig(save_dir)
    plt.close()

# Plot the isodepth result from trials, for K>=2 isodepths
def plot_trials_multi(output_dir, save_dir, S, trial_id='unknown'):
    X,Y = int(max(S[:,0])-min(S[:,0])+1),int(max(S[:,1])-min(S[:,1])+1)
    loss = np.load(output_dir+'loss_list.npy')[-1]
    lasso_loss = np.load(output_dir+'lasso_loss.npy')[-1]
    tru_loss = loss - lasso_loss
    isodepth = np.loadtxt(output_dir+'isodepth.txt')
    title=f'Trial {trial_id}: Isodepth'
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
        plt.imshow(vein_mat, cmap='coolwarm', interpolation='nearest')
        plt.title(title+f' {k}',fontsize=15,fontweight='demi')
        plt.savefig(save_dir+f'/loss{tru_loss:.4g}_trial{trial_id}_isodepth{k}.png')
        plt.close()

# Scale the learned isodepth into n bins
def scale(isodepth,S,n=100,show_plot=False,contour_plot=True):
    # get the width and height of the data slice
    X = int(max(S[:,0])-min(S[:,0]))+1
    Y = int(max(S[:,1])-min(S[:,1]))+1 
    def dist(a1,a2):
        (x1,y1) = a1
        (x2,y2) = a2
        return math.sqrt((x1-x2)**2+(y1-y2)**2)
    # 1. bin the isodepth into n bins
    # 2. set all isodepth in the 1st bin as 0, 
    # 3. for the i-th bin/contour area, compute its width by finding avg min distance from spots 
    # in (i+1)th bin and spots in (i-1)th bin. Set isodepths in i-th bin as sum(width[:i]).
    cut = pd.qcut(isodepth, n,retbins=True, duplicates='drop')
    N = len(cut[1])
    cut = pd.qcut(isodepth, n, labels=range(1,N),retbins=True,duplicates='drop')
    contours = np.array(cut[0])
    contours1 = contours.copy()
    width = np.zeros(N)
    for i in range(1,N-1):
        # all the spots in the (i+1)th bin or contour area
        contour_high = [(S[k,0],S[k,1]) for k in np.where(contours == i+1)][0]
        contour_high = list(zip(contour_high[0],contour_high[1]))
        # all the spots in the (i-1)th bin or contour area
        contour_low = [(S[k,0],S[k,1]) for k in np.where(contours == i-1)][0]
        contour_low = list(zip(contour_low[0],contour_low[1]))
        if len(contour_low) == 0 or len(contour_high) == 0: 
            width[i] = 0
            contours1[contours == i] = sum(width[:i])
            continue
        if len(contour_low) > 20: 
            contour_low_selected = random.sample(contour_low, 20)
            width[i] = min([min([dist(a1,a2) for a2 in contour_high]) for a1 in contour_low_selected])
        else: width[i] = min([min([dist(a1,a2) for a2 in contour_high]) for a1 in contour_low])
        contours1[contours == i] = sum(width[:i])
    contours1[contours == N-1] = sum(width[:N-1])
    contours1 = contours1 / max(contours1) * 1
    # Lastly, plot the isodepth
    if show_plot:
        radius_mat = np.ones((X,Y))*min(isodepth)
        radius_mat1 = np.ones((X,Y))*min(contours1)
        minX = min(S[:,0])
        minY = min(S[:,1])
        for i in range(0,S.shape[0]):
            row = S[i,:]
            x = int(row[0]-minX)
            y = int(row[1]-minY)
            radius_mat[x,y] = isodepth[i]
            radius_mat1[x,y] = contours1[i]
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(9, 5))
        ax1.imshow(radius_mat, cmap = 'coolwarm',interpolation='nearest')
        if contour_plot: ax1.contour(radius_mat)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title("Unscaled isodepth")
        ax2.imshow(radius_mat1, cmap = 'coolwarm',interpolation='nearest')
        if contour_plot: ax2.contour(radius_mat1)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title("Scaled isodepth")
        plt.show()
        plt.close()
    return contours1

# Scale the learned isodepth into n bins, for K = 1 isodepth
def adjust_isodepth(data, isodepth, names, marker_id, marker_id1,show_plot=False, scale_n = 200, axis_name='(PN -> CV)', contour_plot=True):
    # First adjust isodepth to be in range [0,1]
    isodepth -= np.min(isodepth)
    isodepth = isodepth*1/np.max(isodepth)
    # Invert the isodepth according to marker metabolite signal when needed
    # e.g. in murine liver, we use marker metabolite Taurocholic acid so that 
    # it has a negative slope w.r.t. the isodepth, so minimum isodepth corresponds
    # to PN regions.
    x = isodepth[data[:,2+marker_id]!=0]
    y = data[:,2+marker_id][data[:,2+marker_id]!=0]
    m, b = np.polyfit(x, y, 1)
    if m > 0: isodepth = -isodepth +1
    # Then scale the isodepth
    if scale_n > 1:
        isodepth = scale(isodepth,data[:,:2],scale_n,show_plot,contour_plot)
    # Visualize before-and-after isodepth and marker metabolite plots
    if show_plot:
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(16, 4))
        ax1.scatter(x,y,alpha = 2/5)
        ax1.plot(x, m*x+b,color='black',alpha = 3/5)
        ax1.set_xlabel(f"Isodepth {axis_name}")
        ax1.set_ylabel("Abundance")
        ax1.set_title(f"{names[marker_id][0]} before scaling and adjustment")
        x = isodepth[data[:,2+marker_id]!=0]
        y = data[:,2+marker_id][data[:,2+marker_id]!=0]
        m, b = np.polyfit(x, y, 1)
        ax2.scatter(x,y,alpha = 2/5)
        ax2.plot(x, m*x+b,color='black',alpha = 3/5)
        ax2.set_xlabel(f"Isodepth {axis_name}")
        ax2.set_ylabel("Abundance")
        ax2.set_title(f"{names[marker_id][0]} after scaling")
        x = isodepth[data[:,2+marker_id1]!=0]
        y = data[:,2+marker_id1][data[:,2+marker_id1]!=0]
        m, b = np.polyfit(x, y, 1)
        ax3.scatter(x,y,alpha = 2/5)
        ax3.plot(x, m*x+b,color='black',alpha = 3/5)
        ax3.set_xlabel(f"Isodepth {axis_name}")
        ax3.set_ylabel("Abundance")
        ax3.set_title(f"Marker 2 ({names[marker_id1][0]}) after scaling")
        plt.show()
        plt.close()


# Plot the isodepth results from all crops on top of the whole tissue region
# whole_slice: N x (1+M) whole tissue matrix including coordinates, TIC, and abundance
# crops: list of crops, where each element is a N x (1+1+M) crop matrix, with coordinates,
#       TIC, scaled isodepth, and abundance
# bg_met_id: index of metabolite to plot as the background abundance over the whole tissue
def plot_all_crops(whole_slice, crops, bg_met_id, save_dir):
    X = int(max(whole_slice[:,0])-min(whole_slice[:,0])+1)
    Y = int(max(whole_slice[:,1])-min(whole_slice[:,1])+1)
    # Background metabolite abundance over whole slice
    vein_mat = np.zeros((X,Y))
    for i in range(0,whole_slice.shape[0]):
        row = whole_slice[i,:]
        x = int(row[0])
        y = int(row[1])
        vein_mat[x,y] = whole_slice[i,2+bg_met_id]
    plt.imshow(vein_mat, interpolation='nearest',alpha=0.6)
    # Plot individual crop isodepths
    for crop in crops:
        plt.tricontour(crop[:,1],crop[:,0], crop[:,2], 7, linewidths=0.5, colors='k',alpha=0.9)
        plt.tricontourf(crop[:,1],crop[:,0], crop[:,2], 7, cmap='coolwarm',alpha=0.9)
    plt.xlabel("X",fontsize=13)
    plt.ylabel("Y",fontsize=13)
    plt.title(f'Isodepth from all crops',fontsize=15,fontweight='demi')
    plt.savefig(save_dir+f'/isodepth_all_crop.png')
    plt.show()
    plt.close()

# Inputs: 
#   save_dir: directory to save the image to
#   loss_list: training loss list over all epochs
#   lasso_loss: training lasso loss
def plot_loss(save_dir, loss_list, lasso_loss = None):
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

