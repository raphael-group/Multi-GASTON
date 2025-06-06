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








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as mpatches
import os
from itertools import product
from colour import Color


version = 'intest_dynamic'
# -----------------------------------------UNIVERSAL--------------------------------------
names=pd.read_excel(f'/Users/cloverzsq/Downloads/{version}/livm_meta.xlsx',header=None).to_numpy()
names = names[1:,:]
# name, chemical formula, neutral mass
pv_id = np.where(names[:,0]=='Taurocholic acid')[0][0]
heme_id = np.where(names[:,0]=='heme')[0][0]
cv_id = np.where(names[:,0]=='arachidonic acid')[0][0]


# -----------------------------------------Starting a slice--------------------------------------
# -----------------------------------------1. process data--------------------------------------
filenum =  'Full ROI/mouse M153/intm153_slide2_lowmass_ROI_11'



def process(filenum, plot = True):
    # Read in data
    folder = f'/Users/cloverzsq/Downloads/{version}/'
    if version == 'intestine': A=pd.read_csv(folder + f'intest_{filenum}.csv',header=None).to_numpy()
    elif version == 'Lactate_intest_Oct24': 
        for root, dirs, files in os.walk(folder):
            if root != folder: continue
            best = [i for i in files if i.startswith(filenum) and i.endswith('_tgt.csv')]
        print(best[0])
        A=pd.read_csv(folder + best[0],header=None).to_numpy()
    elif version.startswith('intest_dynamic') or version=='intest_glutamine' or version == 'epithelial_Nov24': 
        for root, dirs, files in os.walk(folder):
            if root != folder: continue
            print(root, dirs, files)
            best = [i for i in files if i.startswith(filenum)]
        print(best[0])
        A=pd.read_csv(folder + best[0],header=None).to_numpy()
    else: A=pd.read_csv(folder + f'{filenum}.csv',header=None).to_numpy()
    # Make sure both x and y coordinate start with 0
    A[:,0] = A[:,0] - min(A[:,0])
    A[:,1] = A[:,1] - min(A[:,1]) 
    S=A[:,:2]
    # Normalize by TIC and then log-transform
    A1 = (A[:,3:] / A[:,2][:,np.newaxis]) * A[:,2].mean()
    A1 = np.log(A1+1)
    A1 = np.concatenate((A[:,:3],A1),axis=1)
    print(S.shape)
    X = int(max(S[:,0])-min(S[:,0]))+1
    Y = int(max(S[:,1])-min(S[:,1]))+1
    print(X)
    print(Y)
    if plot == False: return X,Y,S,A1,folder
    path = folder+f'{filenum}/'
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    cv_mat,pv_mat,heme_mat,tic_mat = np.zeros((X,Y)),np.zeros((X,Y)),np.zeros((X,Y)),np.zeros((X,Y))
    if version == 'intestine_10um': 
        epi_mat,lac_mat = np.zeros((X,Y)),np.zeros((X,Y))
    for i in range(0,S.shape[0]):
        row = S[i,:]
        x = int(row[0])
        y = int(row[1])
        tic_mat[x,y],pv_mat[x,y],heme_mat[x,y],cv_mat[x,y] = A1[i,2],A1[i,3+pv_id],A1[i,3+heme_id],A1[i,3+cv_id]
        if version == 'intestine_10um': 
            lac_mat[x,y],epi_mat[x,y]= A1[i,3+lac_id],A1[i,3+epi_id]
    plt.imshow(cv_mat, interpolation='nearest')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Peri-Central Marker: arachidonic acid")
    plt.savefig(path+'cv.png')
    # plt.show()
    plt.close()
    # -----------plot
    plt.imshow(pv_mat, interpolation='nearest')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Peri-Portal Marker: Taurocholic acid")
    plt.savefig(path+'pv.png')
    # plt.show()
    plt.close()
    # -----------plot
    plt.imshow(heme_mat, interpolation='nearest')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Heme Marker")
    plt.savefig(path+'heme.png')
    # plt.show()
    plt.close()
    # -----------plot
    plt.imshow(tic_mat, interpolation='nearest')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Total Ion Count")
    plt.savefig(path+'tic.png')
    # plt.show()
    plt.close()
    if version == 'intestine_10um': 
        plt.imshow(lac_mat, interpolation='nearest')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Lac Marker ({names[lac_id,0]})")
        plt.savefig(path+'lac.png')
        # plt.show()
        plt.close()
        # -----------plot
        plt.imshow(epi_mat, interpolation='nearest')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Epi Marker ({names[epi_id,0]})")
        plt.savefig(path+'epi.png')
        # plt.show()
        plt.close()
    return X,Y,S,A1,folder

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

# =========================================3. finding crop=========================================
slice = 140
file = 'intm140_2_lowmass_ROI_02_villus_2'
filenum =  f'Lining pixels/mouse M{slice}/{file}_adjpixels'
X,Y,S,A1,folder = process(filenum)
np.savetxt(folder + f'LIN_{file}_adjpixels.csv',A1, delimiter=",")

filenum =  f'Isolated villi/mouse M{slice}/{file}'
X,Y,S,A1,folder = process(filenum)
np.savetxt(folder + f'ISO_{file}.csv',A1, delimiter=",")

for file in ["FULL_intm140_2_lowmass_ROI_02", "FULL_intm144_lowmass2_ROI_01", "FULL_intm144_lowmass2_ROI_02", "FULL_intm152_slide2_lowmass_ROI_02", "FULL_intm152_slide2_lowmass_ROI_03", "FULL_intm153_slide2_lowmass_ROI_10", "FULL_intm153_slide2_lowmass_ROI_11"]:
    file = file[file.find('i'):]
    slice = file[file.find('1'):file.find('1')+3]
    # file='intm144_lowmass2_ROI_01'
    filenum =  f'Full ROI/mouse M{slice}/{file}'
    X,Y,S,A1,folder = process(filenum,plot=False)
    # np.savetxt(folder + f'FULL_{file}.csv',A1, delimiter=",")
    dropout_rate = []
    for i in range(len(names)):
        dropout_rate+=[len(A1[A1[:,3+i]==0]) / A1.shape[0]]
    plt.plot(range(len(names)),sorted(dropout_rate))
    plt.xlabel("Metabolite index",fontsize=13)
    plt.ylabel("Dropout rate = #zeors / #spots",fontsize=13)
    plt.title(f'Slice {file}: Raw metabolite dropout rate',fontsize=15,fontweight='demi')
    plt.savefig(f'/Users/cloverzsq/Desktop/FULL_{file}_dropoutrate.png')
    # plt.show()
    plt.close()
    A2 = np.delete(A1,[i+3 for i in range(len(dropout_rate)) if dropout_rate[i] >= 0.8],1)
    np.savetxt(folder + f'filtered/FULL_{file}.csv',A2, delimiter=",")



crop = np.zeros(S.shape[0])
crop_id = 9
vein_mat = np.zeros((X,Y))
for i in range(0,S.shape[0]):
    row = S[i,:]
    x = int(row[0])
    y = int(row[1])
    vein_mat[x,y] = A1[i,2+111]
    (x1,x2),(y1,y2) = (300,400),(430,520)
    if x > x1 and x < x2 and y > y1 and y < y2:
        crop[i] = crop_id
    if ((y == y1 or y == y2) and x > x1 and x < x2) or ((x == x1 or x == x2) and y > y1 and y < y2):
        vein_mat[x,y] = 0

plt.imshow(vein_mat, interpolation='nearest')
plt.xlabel("Y")
plt.ylabel("X")
plt.title(f"Crop {crop_id}, (bkg Metabolite{111}: {names[111,0]})")
plt.savefig(folder+f'{filenum}/crop_{crop_id}.pdf')
plt.show()
plt.close()

np.savetxt(folder + f'intest_{filenum}_crop_{crop_id}.csv',A1[crop == crop_id], delimiter=",")


for i in range(names.shape[0]):
    plot_mat(X,Y,S,A1,i,names,save_dir = '/Users/cloverzsq/Desktop/data/',show_plot=False)
    break



# =================================intestine_10um: collage isolated villus together=========================
small_villi_list = ['ISO_intm140_2_lowmass_ROI_02_villus_5','ISO_intm152_lowmass_slide2_ROI_03_villus_3',
              'ISO_intm153_lowmass_slide2_ROI_10_villus_1','ISO_intm153_lowmass_slide2_ROI_10_villus_3',
              'ISO_intm153_lowmass_slide2_ROI_13_villus_9']
for root, dirs, files in os.walk(folder):
    if root != folder: continue
    med_villi_list = [i.split('.')[0] for i in files if i.startswith('ISO_')]
    print(med_villi_list)

y_counter,x_counter = 0,0
A = None
for file in med_villi_list:
    file = '_'.join(file.split('_')[1:])
    slice = file[file.find('1'):file.find('1')+3]
    filenum =  f'Isolated villi/mouse M{slice}/{file}'
    X,Y,S,A1,folder = process(filenum,plot=False)
    print(X,Y,A1.shape)
    A1[:,0] += x_counter
    A1[:,1] += y_counter
    if A is None: A = A1
    else:
        A = np.concatenate((A,A1),axis=0)
    y_counter+=Y
    print(f'y_counter = {y_counter}')
    if y_counter>=400:
        y_counter=0
        x_counter=max(A[:,0])

# plot_mat(A,pv_id,names,save_dir = folder+'collaged villi/5_known_villi', show_plot=True)
plot_mat(A,pv_id,names,save_dir = '', show_plot=True)
np.savetxt(folder + f'collaged villi/15_villi.csv',A, delimiter=",")

all_villi = {}
allvilli = []
dir = f'/Users/cloverzsq/Downloads/{version}/Isolated villi/'
for root, dirs, files in os.walk(dir):
    if root == dir: continue
    villi = [i[:-4] for i in files if i.endswith('.csv')]
    if len(villi) == 0 or root.find('mouse')< 0: continue
    slice = root[root.find('mouse')+7:root.find('mouse')+10]
    print(slice)
    print(villi)
    all_villi[slice] = villi
    if int(slice) >= 140:
        allvilli += villi

y_counter,x_counter = 0,0
A = None
slice = '144'
for file in all_villi[slice]:
    filenum =  f'Isolated villi/mouse M{slice}/{file}'
    X,Y,S,A1,folder = process(filenum,plot=False)
    print(X,Y,A1.shape)
    A1[:,0] += x_counter
    A1[:,1] += y_counter
    if A is None: A = A1
    else:
        A = np.concatenate((A,A1),axis=0)
    y_counter+=Y
    print(f'y_counter = {y_counter}')
    if y_counter>=300:
        y_counter=0
        x_counter=max(A[:,0])

plot_mat(A,pv_id,names,save_dir = '', show_plot=True)
plot_mat(A,pv_id,names,save_dir = f'/Users/cloverzsq/Downloads/{version}/collaged villi/all{slice}_', show_plot=False)
np.savetxt(folder + f'collaged villi/all{slice}_rm.csv',np.delete(A, (3+14,3+55), axis=1),delimiter=",")

y_counter,x_counter = 0,0
A = None
for slice in ['140','152','153']:
    for file in all_villi[slice]:
        filenum =  f'Isolated villi/mouse M{slice}/{file}'
        X,Y,S,A1,folder = process(filenum,plot=False)
        print(X,Y,A1.shape)
        A1[:,0] += x_counter
        A1[:,1] += y_counter
        if A is None: A = A1
        else:
            A = np.concatenate((A,A1),axis=0)
        y_counter+=Y
        print(f'y_counter = {y_counter}')
        if y_counter>=550:
            y_counter=0
            x_counter=max(A[:,0])

plot_mat(A,pv_id,names,save_dir = '', show_plot=True)
plot_mat(A,pv_id,names,save_dir = f'/Users/cloverzsq/Downloads/{version}/collaged villi/allvilli1_', show_plot=False)
np.savetxt(folder + f'collaged villi/allvilli1.csv',np.delete(A, (3+14,3+55), axis=1),delimiter=",")

slice='140'
villi_1401 = [a for a in all_villi[slice] if a.find('ROI_02')>= 0]
villi_1402 = [a for a in all_villi[slice] if a.find('ROI_06')>= 0]
slice='152'
villi_1521 = [a for a in all_villi[slice] if a.find('slide1')>= 0]
villi_1522 = [a for a in all_villi[slice] if a.find('slide2')>= 0]
slice='153'
# villi_1531 = all_villi[slice][1:-2] + [all_villi[slice][-1]]
villi_1531 = [a for a in all_villi[slice] if a.find('ROI_13')>= 0]
villi_1532 = [a for a in all_villi[slice] if a not in villi_1531]
slice='144'
# villi_1441 = all_villi[slice][2:-5]+all_villi[slice][-4:]
villi_1441 = [a for a in all_villi[slice] if a.find('lowmass_R')>= 0]
villi_1442 = [a for a in all_villi[slice] if a.find('lowmass_2')>= 0]

slice='140'
all_villi[slice+'_1'] = [a for a in all_villi[slice] if a.find('ROI_02')>= 0]
all_villi[slice+'_2'] = [a for a in all_villi[slice] if a.find('ROI_06')>= 0]
slice='152'
all_villi[slice+'_1'] = [a for a in all_villi[slice] if a.find('slide1')>= 0]
all_villi[slice+'_2'] = [a for a in all_villi[slice] if a.find('slide2')>= 0]
slice='153'
all_villi[slice+'_1'] = [a for a in all_villi[slice] if a.find('ROI_13')>= 0]
all_villi[slice+'_2'] = [a for a in all_villi[slice] if a not in villi_1531]
slice='144'
all_villi[slice+'_1'] = [a for a in all_villi[slice] if a.find('lowmass_R')>= 0]
all_villi[slice+'_2'] = [a for a in all_villi[slice] if a.find('lowmass_2')>= 0]

y_counter,x_counter = 0,0
A = None
slice='144'
for file in villi_1442:
    filenum =  f'Isolated villi/mouse M{slice}/{file}'
    X,Y,S,A1,folder = process(filenum,plot=False)
    print(X,Y,A1.shape)
    A1[:,0] += x_counter
    A1[:,1] += y_counter
    if A is None: A = A1
    else:
        A = np.concatenate((A,A1),axis=0)
    y_counter+=Y
    print(f'y_counter = {y_counter}')
    if y_counter>=200 and x_counter==0:
        y_counter=0
        x_counter=max(A[:,0])

plot_mat(A,pv_id,names,save_dir = '', show_plot=True)
plot_mat(A,pv_id,names,save_dir = f'/Users/cloverzsq/Downloads/{version}/collaged villi/all{slice}_2_', show_plot=False)
np.savetxt(folder + f'collaged villi/all{slice}_2.csv',A,delimiter=",")
# np.savetxt(folder + f'collaged villi/all{slice}_2_rm.csv',np.delete(A, (3+14,3+55), axis=1),delimiter=",")

# ======================================6.26 high mass new villi crop======================================
version = 'intestine_10um/high mass'
names=pd.read_excel(f'/Users/cloverzsq/Downloads/{version}/livm_meta.xlsx',header=None).to_numpy()
names = names[1:,:]
filenum =  'intm144_01_highmass_crop'
pv_id,heme_id,cv_id = 0,1,2
X,Y,S,A1,folder = process(filenum)
plot_mat(A1,pv_id,names,save_dir = '', show_plot=True)
np.savetxt(f'/Users/cloverzsq/Downloads/intestine_10um/collaged villi/highmass_villi.csv',A1, delimiter=",")



# ========================================lining pixels=====================================================
epi_id = np.where(names[:,0]=='Linoleic acid')[0][0]
plt.hist(A[:,3+epi_id])
plt.show()
plt.close()
plot_mat(A,epi_id,names,save_dir = '', show_plot=True,threshold=np.percentile(A[:,3+epi_id], 80))

bigdata = pd.read_csv(f'/Users/cloverzsq/Downloads/Intestine_10um/collaged villi/all140_2.csv',header=None,dtype=float).to_numpy()
radius =np.loadtxt('/Users/cloverzsq/Desktop/NN_8slices/intestine_10um/collaged villi/all140_2/lasso0.0005/loss0.78269_exp0.77720_nhs100,100_nhalin_trial2_lasso0.0005_radius.txt')
S=bigdata[:,:2]
X,Y = int(max(S[:,0])-min(S[:,0])+1),int(max(S[:,1])-min(S[:,1])+1)
thres = np.percentile(bigdata[:,3+epi_id], 75)
for k in range(radius.shape[1]):
    radius1 = radius[:,k]
    vein_mat, lining_mat = np.ones((X,Y))*0.1, np.zeros((X,Y))
    S[:,0] = S[:,0] - min(S[:,0])
    S[:,1] = S[:,1] - min(S[:,1])
    for i in range(0,S.shape[0]):
        row = S[i,:]
        x = int(row[0])
        y = int(row[1])
        vein_mat[x,y] = radius1[i]
        if bigdata[i,3+epi_id] >= thres: lining_mat[x,y] = 1
    lining = np.argwhere(lining_mat==1)
    plt.figure(figsize=(15,8))
    plt.imshow(vein_mat, cmap='hot', interpolation='nearest')
    plt.contour(vein_mat)
    plt.scatter(lining[:,1],lining[:,0], marker="s",edgecolors='none',s=20,color =[1,1,1,0.5],label="Epithelial lining")#, "r.",alpha=0.3)
    plt.xlabel("X",fontsize=13)
    plt.ylabel("Y",fontsize=13)
    plt.title(f'Mouse {slice}: Isodepth {k}',fontsize=15,fontweight='demi')
    plt.legend(fontsize=13)
    # plt.savefig(f'/Users/cloverzsq/Desktop/Mouse{slice}_iso{k}.pdf')
    plt.show()
    plt.close()


# ==========================================removed sus layer========================================
# all_villi = {}
allvilli = []
dir = f'/Users/cloverzsq/Downloads/{version}/removed_layer/'
for root, dirs, files in os.walk(dir):
    # if root == dir: continue
    villi = [i[:-4] for i in files if i.endswith('.csv')]
    if len(villi) == 0: continue
    # slice = root[root.find('mouse')+7:root.find('mouse')+10]
    # print(slice)
    print(villi)
    allvilli += villi

names=pd.read_excel(f'/Users/cloverzsq/Downloads/{version}/removed_layer/livm_meta.xlsx',header=None).to_numpy()
names = names[1:,:]
epi_id = np.where(names[:,0]=='Linoleic acid')[0][0]
lac_id = np.where(names[:,0]=='lacteal marker')[0][0]
slice='140'
all_villi[slice+'_1'] = [a for a in allvilli if a.find('ROI_02')>= 0]
all_villi[slice+'_2'] = [a for a in allvilli if a.find('ROI_06')>= 0]
y_counter,x_counter = 0,0
A = None
for file in all_villi[slice+'_2']:
    filenum =  f'removed_layer/{file}'
    X,Y,S,A1,folder = process(filenum,plot=False)
    print(X,Y,A1.shape)
    A1[:,0] += x_counter
    A1[:,1] += y_counter
    if A is None: A = A1
    else:
        A = np.concatenate((A,A1),axis=0)
    y_counter+=Y
    print(f'y_counter = {y_counter}')
    if y_counter>=170:
        y_counter=0
        x_counter=max(A[:,0])

plot_mat(A,epi_id,names,save_dir = '', show_plot=True)
plot_mat(A,pv_id,names,save_dir = '', show_plot=True)
plot_mat(A,heme_id,names,save_dir = '', show_plot=True)
plot_mat(A,cv_id,names,save_dir = '', show_plot=True)
plot_mat(A,pv_id,names,save_dir = f'/Users/cloverzsq/Downloads/{version}/removed_layer/all{140}_2_', show_plot=False)
np.savetxt(folder + f'removed_layer/all{140}_2.csv',A,delimiter=",")



# =========================================clutering data============================================
dir = f'/Users/cloverzsq/Downloads/{version}/clustering/'
names=pd.read_excel(f'/Users/cloverzsq/Downloads/{version}/clustering/livm_meta.xlsx',header=None).to_numpy()
names = names[1:,:]
epi_id = np.where(names[:,0]=='Linoleic acid')[0][0]
lac_id = np.where(names[:,0]=='lacteal marker')[0][0]
pv_id = np.where(names[:,0]=='Taurocholic acid')[0][0]
heme_id = np.where(names[:,0]=='heme')[0][0]
cv_id = np.where(names[:,0]=='arachidonic acid')[0][0]
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith('tgt.csv') == False: continue
        file = file.split('.')[0]
        filenum = 'clustering/'+file
        save_name = f'clustering/{file[:-4]}_processed.csv'
        X,Y,S,A1,folder = process(filenum)
        np.savetxt(folder + save_name,A1, delimiter=",")

filenum='clustering/intm140_2_02_cluster_3_ep'
X,Y,S,A1,folder = process(filenum)
plot_mat(A1,epi_id,names,save_dir = '', show_plot=True)
plot_mat(A1,pv_id,names,save_dir = '', show_plot=True)
plot_mat(A1,heme_id,names,save_dir = '', show_plot=True)
plot_mat(A1,cv_id,names,save_dir = '', show_plot=True)
np.savetxt(folder + f'removed_layer/cluster_epi.csv',A1, delimiter=",")


version = 'epithelial_Nov24'
names=pd.read_excel(f'/Users/cloverzsq/Downloads/{version}/livm_meta.xlsx',header=None).to_numpy()
names = names[1:,:]
dir = f'/Users/cloverzsq/Downloads/{version}/'
epi_id = np.where(names[:,0]=='Linoleic acid')[0][0]
lac_id = np.where(names[:,0]=='lacteal marker')[0][0]
pv_id = np.where(names[:,0]=='Taurocholic acid')[0][0]
heme_id = np.where(names[:,0]=='heme')[0][0]
cv_id = np.where(names[:,0]=='arachidonic acid')[0][0]
list=[]
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith('.csv') == False or file.endswith('_processed.csv') or file.endswith('_epi.csv') or file.endswith('intm140_2_02_cluster_3_ep.csv'): continue
        file = file.split('.')[0]
        save_name = f'{file}_processed.csv'
        list += [save_name.split('.')[0]]
        X,Y,S,A1,folder = process(file)
        np.savetxt(folder + save_name,A1, delimiter=",")
        break
    break

print(" ".join(str(x) for x in list))
print("\n".join(str(x)[:-10] for x in sorted(list)))
# Young fasted:
# m144_2_03_cluster_3_ep_n4_processed m153_slide2_13_cluster_3_ep_n3_processed m153_slide2_08-09-10_cluster_3_ep_n4_processed m152_slide1_10_cluster_1_combo_ep_processed m152_slide1_01_cluster_3_ep_n3_processed m144_2_06_cluster_3_ep_n4_processed m140_2_02_cluster_3_ep_n3_processed m140_2_06_cluster_3_ep_from3clus_processed m140_05_cluster_3_ep_n3_processed m152_slide2_02-03_cluster_3_ep_n4_processed m153_slide2_12_cluster_31_ep_n3_processed m144_2_01_cluster_3_ep_n4_processed m153_slide1_10_cluster_3_ep_n3_processed m153_slide1_09_cluster_3_ep_n4_processed m152_slide1_09_cluster_3_ep_processed m152_slide2_05_cluster_3_ep_n3_processed m153_slide2_11_cluster_3_ep_n3_processed m152_slide1_06_cluster_3_combo_ep_n4_processed m153_slide1_08_cluster_3_ep_n3_processed
# Fructose:
# m163-165_neg_11_cluster_ep_n4_processed m163-165_neg_07_cluster_ep_n4_processed m163-165_neg_15_cluster_ep_n4_processed m163-165_neg_03_cluster_ep_n4_processed m163-165_neg_09_cluster_ep_n4_processed m163-165_neg_01_cluster_ep_n4_processed m163-165_neg_12-13_cluster_ep_n4_processed m163-165_neg_05-06_cluster_ep_n4_processed m163-165_neg_10_cluster_ep_n4_processed m163-165_neg_04_cluster_ep_n4_processed m163-165_neg_02_cluster_ep_n4_processed m163-165_neg_08_cluster_ep_n4_processed
# 240615_intm163-165_neg_cluster_ep_n4_all_crops_processed 240428_intm135_neg_5um_cluster_3_ep_n4_processed 240428_intm136_neg_5um_3_cluster_ep_n4_processed 240428_intm136_neg_10um_cluster_ep_n4_processed 240428_intm135_neg_5um_2_cluster_3_ep_n5_processed 240428_intm136_neg_5um_cluster_ep_n4_processed 240428_intm136_neg_5um_2_cluster_ep_n4_processed 
# Aged
# m142_2_09_cluster_ep_n4_tgt_processed 240512_intm147_neg_slide2_06_cluster_ep_n5_processed 240512_intm147_neg_slide1_07_cluster_ep11_n2-n3_processed m142_2_07_cluster_ep_n4_tgt_processed m142_2_11_cluster_ep_n4_tgt_processed 240509_intm146_neg_2_04_cluster_ep_n5_processed m142_2_10_cluster_ep_n4_tgt_processed


version = 'intest_glutamine'
for i in [170,172,184]:
    if i in range(13,17): continue
    filenum = str(i).zfill(2)
    X,Y,S,A1,folder = process(filenum)
    np.savetxt(folder + f'{filenum}_processed.csv',A1, delimiter=",")


S = intest[:,:2]
S[:,0]=S[:,0]-min(S[:,0])
S[:,1]=S[:,1]-min(S[:,1])
X = int(max(S[:,0])-min(S[:,0]))+1
Y = int(max(S[:,1])-min(S[:,1]))+1
for id in range(intest.shape[1]-3):
    mat = np.zeros((X,Y))
    for i in range(0,S.shape[0]):
        row = S[i,:]
        x = int(row[0])
        y = int(row[1])
        mat[x,y] = intest[i,3+id]
    plt.imshow(mat, interpolation='nearest')
    plt.xlabel("X",fontsize=13)
    plt.ylabel("Y",fontsize=13)
    # plt.title(f"Metabolite {id}: {names[id,0]}",fontsize=15,fontweight='demi')
    # if len(save_dir)>0:
    plt.savefig(f'/Users/cloverzsq/Downloads/Intestine_10um/clustering/m153_slide1_10_cluster_3_ep_n3_tgt/M{id}.pdf')
    # if show_plot:
    # plt.show()
    plt.close()

names=pd.read_excel(f'',header=None).to_numpy()