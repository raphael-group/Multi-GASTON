import os
from tqdm import trange
from sklearn import linear_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2,mode

# Below code are modified from GASTON: https://github.com/raphael-group/GASTON/
######################

# INPUTS:
# counts_mat: N x G matrix of counts
# labels, learned_coord: N x 1 array with labels/learned coordinate for each spot (labels=0, ..., L-1)

# num_bins: number of bins to bin tissue-intrinsic coordinate into (for visualization)
# pseudocount: pseudocount to add to counts
# umi_threshold: restrict to genes with total UMI count across all spots > umi_threshold
# after filtering, there are G' genes
# t: p-value threshold for LLR test (slope = 0 vs slope != 0)
# coord_mult_factor: if the range of tissue-intrinsic coordinate values is too large (causes overflows)
#    then scale coordinate values by coord_mult_factor (ie tissue-intrinsic coordinate -> 
#    tissue-intrinsic coordinate * mult_factor) to compute Poisson regressoin fits that are more numerically 
#    stable
# OUTPUTS:
# pw_fit_dict: (slope_mat, intercept_mat, pv_mat)

# slope_mat, intercept_mat: G' x L, entries are slopes/intercepts
# pv_mat: G' x L, entries are p-values from LLR test (slope=0 vs slope != 0)
def pw_linear_fit(counts_mat, labels, learned_coord, 
                  umi_threshold=500, idx_kept=None, t=0.1,
                  coord_mult_factor=1, reg=0, zero_fit_threshold=0):

    counts_mat=counts_mat.T 
    if idx_kept is None:
        idx_kept=np.where(np.sum(counts_mat,1) > umi_threshold)[0]

    exposures=np.sum(counts_mat,0)
    cmat=counts_mat[idx_kept,:]
    G,N=cmat.shape
    learned_coord=learned_coord * coord_mult_factor
    L=len(np.unique(labels))
    s0_mat,i0_mat,s1_mat,i1_mat,pv_mat=segmented_poisson_regression(cmat,
                                                   exposures, 
                                                   labels, 
                                                   learned_coord,
                                                   L, reg=reg)
    slope_mat=np.zeros((len(idx_kept), L))
    intercept_mat=np.zeros((len(idx_kept), L))
    
    # Use s0 fit for genes with lots of zeros
    nonzero_per_domain = np.zeros((G,L))
    for l in range(L):
        cmat_l=cmat[:,labels==l]
        nonzero_per_domain[:,l]=np.count_nonzero(cmat_l,1)
    inds1= ((pv_mat < t) & (nonzero_per_domain >= zero_fit_threshold))
    inds0= ((pv_mat >= t) | (nonzero_per_domain < zero_fit_threshold))
    
    slope_mat[inds1] = s1_mat[inds1]
    intercept_mat[inds1] = i1_mat[inds1]
    slope_mat[inds0] = s0_mat[inds0]
    intercept_mat[inds0] = i0_mat[inds0]
    
    slope_mat = slope_mat * coord_mult_factor
    pw_fit_dict=(slope_mat,intercept_mat, pv_mat)
    return pw_fit_dict
    

######################

def llr_poisson(y, xcoords=None, exposure=None, alpha=0):
    s0, i0 = poisson_regression(y, xcoords=0*xcoords, exposure=exposure, alpha=alpha)
    s1, i1 = poisson_regression(y, xcoords=xcoords, exposure=exposure, alpha=alpha)
    
    ll0=poisson_likelihood(s0,i0,y,xcoords=xcoords,exposure=exposure)
    ll1=poisson_likelihood(s1,i1,y,xcoords=xcoords,exposure=exposure)
    
    return s0, i0, s1, i1, chi2.sf(2*(ll1-ll0),1)

def poisson_likelihood(slope, intercept, y, xcoords=None, exposure=None):
    lam=exposure * np.exp(slope * xcoords + intercept)
    return np.sum(y * np.log(lam) - lam)

def poisson_regression(y, xcoords=None, exposure=None, alpha=0):
    # run poisson fit on pooled data and return slope, intercept
    clf = linear_model.PoissonRegressor(fit_intercept=True,alpha=alpha,max_iter=500,tol=1e-10)
    clf.fit(np.reshape(xcoords,(-1,1)),y/exposure, sample_weight=exposure)

    return [clf.coef_[0], clf.intercept_ ]

def segmented_poisson_regression(count, totalumi, dp_labels, isodepth, num_domains,
                                 opt_function=poisson_regression, reg=0):
    """ Fit Poisson regression per gene per domain.
    :param count: UMI count matrix of SRT gene expression, G genes by n spots
    :type count: np.array
    :param totalumi: Total UMI count per spot, a vector of n spots.
    :type totalumi: np.array
    :param dp_labels: domain labels obtained by DP, a vector of n spots.
    :type dp_labels: np.array
    :param isodepth: Inferred domain isodepth, vector of n spots
    :type isodepth: np.array
    :return: A dataframe for the offset and slope of piecewise linear expression function, size of G genes by 2*L domains.
    :rtype: pd.DataFrame
    """

    G, N = count.shape
    unique_domains = np.sort(np.unique(dp_labels))
    # L = len(unique_domains)
    L=num_domains

    slope1_matrix=np.zeros((G,L))
    intercept1_matrix=np.zeros((G,L))
    
    # null setting
    slope0_matrix=np.zeros((G,L))
    intercept0_matrix=np.zeros((G,L))
    
    pval_matrix=np.zeros((G,L))

    for g in trange(G):
        for t in unique_domains:
            pts_t=np.where(dp_labels==t)[0]
            t=int(t)
            
            # need to be enough points in domain
            if len(pts_t) > 10:
                s0, i0, s1, i1, pval = llr_poisson(count[g,pts_t], xcoords=isodepth[pts_t], exposure=totalumi[pts_t], alpha=reg)
            else:
                s0=np.Inf
                i0=np.Inf
                s1=np.Inf
                i1=np.Inf
                pval=np.Inf
        
            slope0_matrix[g,t]=s0
            intercept0_matrix[g,t]=i0
            
            slope1_matrix[g,t]=s1
            intercept1_matrix[g,t]=i1
            
            pval_matrix[g,t]=pval
            
    return slope0_matrix,intercept0_matrix,slope1_matrix,intercept1_matrix, pval_matrix

######################

def bin_data(counts_mat, labels, learned_coord, gene_labels, num_bins=70, 
              num_bins_per_domain=None,idx_kept=None, umi_threshold=500, pc=0, extra_data=[]):
             
    if idx_kept is None:
        idx_kept=np.where(np.sum(counts_mat,0) > umi_threshold)[0]
    gene_labels_idx=gene_labels[idx_kept]
    
    exposure=np.sum(counts_mat,axis=1)
    cmat=counts_mat[:,idx_kept]
    
    N=len(exposure)
    cell_type_mat=np.ones((N,1))
    cell_type_names=['All']
    N,G=cmat.shape

    # BINNING
    if num_bins_per_domain is not None:
        bins=np.array([])
        L=len(np.unique(labels))
        
        for l in range(L):
            isodepth_l=learned_coord[np.where(labels==l)[0]]
            
            if l>0:
                isodepth_lm1=learned_coord[np.where(labels==l-1)[0]]
                isodepth_left=0.5*(np.min(isodepth_l) + np.max(isodepth_lm1))
            else:
                isodepth_left=np.min(isodepth_l)-0.01
                
            if l<L-1:
                isodepth_lp1=learned_coord[np.where(labels==l+1)[0]]
                isodepth_right=0.5*(np.max(isodepth_l) + np.min(isodepth_lp1))
            else:
                isodepth_right=np.max(isodepth_l)+0.01
            
            bins_l=np.linspace(isodepth_left, isodepth_right, num=num_bins_per_domain[l]+1)
            if l!=0:
                bins_l=bins_l[1:]
            bins=np.concatenate((bins, bins_l))
    else:
        isodepth_min, isodepth_max=np.floor(np.min(learned_coord))-0.5, np.ceil(np.max(learned_coord))+0.5
        bins=np.linspace(isodepth_min, isodepth_max, num=num_bins+1)

    unique_binned_isodepths=np.array( [0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)] )
    binned_isodepth_inds=np.digitize(learned_coord, bins)-1 #ie [1,0,3,15,...]
    binned_isodepths=unique_binned_isodepths[binned_isodepth_inds]
    
    # remove bins not used
    unique_binned_isodepths=np.delete(unique_binned_isodepths,
                                   [np.where(unique_binned_isodepths==t)[0][0] for t in unique_binned_isodepths if t not in binned_isodepths])

    N_1d=len(unique_binned_isodepths)
    binned_count=np.zeros( (N_1d,G) )
    binned_exposure=np.zeros( N_1d )
    to_subtract=np.zeros( N_1d )
    binned_labels=np.zeros(N_1d)
    binned_number_spots=np.zeros(N_1d)

    binned_count_per_ct={ct: np.zeros( (N_1d,G) ) for ct in cell_type_names}
    binned_exposure_per_ct={ct: np.zeros( N_1d ) for ct in cell_type_names}
    to_subtract_per_ct={ct:np.zeros( N_1d ) for ct in cell_type_names}
    binned_extra_data=[np.zeros(N_1d) for i in range(len(extra_data))]
    map_1d_bins_to_2d={} # map b -> [list of cells in bin b]
    for ind, b in enumerate(unique_binned_isodepths):
        bin_pts=np.where(binned_isodepths==b)[0]
        
        binned_count[ind,:]=np.sum(cmat[bin_pts,:],axis=0)
        binned_exposure[ind]=np.sum(exposure[bin_pts])
        if pc>0:
            to_subtract[ind]=np.log(10**6 * (len(bin_pts)/np.sum(exposure[bin_pts])))
        binned_labels[ind]= int(mode( labels[bin_pts],keepdims=False).mode)
        binned_number_spots[ind]=len(bin_pts)
        map_1d_bins_to_2d[b]=bin_pts

        for i, eb in enumerate(extra_data):
            binned_extra_data[i][ind]=np.mean(extra_data[i][bin_pts])
        
        for ct_ind, ct in enumerate(cell_type_names):
            
            ct_spots=np.where(cell_type_mat[:,ct_ind] > 0)[0]
            ct_spots_bin = [t for t in ct_spots if t in bin_pts]
            ct_spots_bin_proportions=cell_type_mat[ct_spots_bin,ct_ind]
            
            if len(ct_spots_bin)>0:
                binned_count_per_ct[ct][ind,:]=np.sum(cmat[ct_spots_bin,:] * np.tile(ct_spots_bin_proportions,(G,1)).T, axis=0)
                binned_exposure_per_ct[ct][ind]=np.sum(exposure[ct_spots_bin] * ct_spots_bin_proportions)
                if pc>0:
                    to_subtract_per_ct[ct]=np.log(10**6 * len(ct_spots_bin) / np.sum(exposure[ct_spots_bin]))
            
        
    L=len(np.unique(labels))
    segs=[np.where(binned_labels==i)[0] for i in range(L)]

    to_return={}
    
    to_return['L']=len(np.unique(labels))
    to_return['umi_threshold']=umi_threshold
    to_return['labels']=labels
    to_return['counts_mat_idx']=cmat
    to_return['idx_kept']=idx_kept
    to_return['gene_labels_idx']=gene_labels_idx
    
    to_return['binned_isodepths']=binned_isodepths
    to_return['unique_binned_isodepths']=unique_binned_isodepths
    to_return['binned_count']=binned_count
    to_return['binned_exposure']=binned_exposure
    to_return['to_subtract']=to_subtract
    to_return['binned_labels']=binned_labels
    to_return['binned_number_spots']=binned_number_spots
    
    to_return['binned_count_per_ct']=binned_count_per_ct
    to_return['binned_exposure_per_ct']=binned_exposure_per_ct
    to_return['to_subtract_per_ct']=to_subtract_per_ct
    to_return['binned_extra_data']=binned_extra_data
    to_return['binned_extra_data']=binned_extra_data
    
    to_return['map_1d_bins_to_2d']=map_1d_bins_to_2d
    to_return['segs']=segs

    return to_return

    
def plot_gene_pwlinear(gene_name, pw_fit_dict, binning_output, pt_size=10, 
                       colors=None, linear_fit=True, lw=2, domain_list=None, ticksize=20, figsize=(7,3),
                      offset=10**6, xticks=[0,0.2,0.4,0.6,0.8,1], yticks=None, alpha=1, domain_boundary_plotting=False, 
                      save=False, save_dir="./", variable_spot_size=False, show_lgd=False,
                      lgd_bbox=(1.05,1), extract_values = False):
    
    gene_labels_idx=binning_output['gene_labels_idx']
    if gene_name in gene_labels_idx:
        gene=np.where(gene_labels_idx==gene_name)[0]
    else:
        umi_threshold=binning_output['umi_threshold']
        raise ValueError(f'gene does not have UMI count above threshold {umi_threshold}')
    
    unique_binned_isodepths=binning_output['unique_binned_isodepths']
    unique_binned_isodepths-=unique_binned_isodepths.min()
    unique_binned_isodepths/=unique_binned_isodepths.max()
    binned_labels=binning_output['binned_labels']
    
    binned_count_list=[]
    binned_exposure_list=[]
    to_subtract_list=[]
    binned_count_list.append(binning_output['binned_count'])
    binned_exposure_list.append(binning_output['binned_exposure'])
    to_subtract_list.append(binning_output['to_subtract'])

    segs=binning_output['segs']
    L=len(segs)

    fig,ax=plt.subplots(figsize=figsize)

    if domain_list is None:
        domain_list=range(L)

    values_list = []
    for seg in domain_list:
        for i in range(len(binned_count_list)):
            pts_seg=np.where(binned_labels==seg)[0]
            binned_count=binned_count_list[i]
            binned_exposure=binned_exposure_list[i]
            to_subtract=np.log( offset*1 / np.mean(binned_exposure) )
            # set colors for domains
            if colors is None:
                c=None
            else:
                c=colors[seg]
                
            xax=unique_binned_isodepths[pts_seg]
            # print(binned_count.shape)
            yax=np.log((binned_count[pts_seg,gene] / binned_exposure[pts_seg]) * offset + 1)

            if extract_values:
                values_list.append(np.column_stack((xax, yax)))
            
            s=pt_size
            if variable_spot_size:
                s=s*binning_output['binned_number_spots'][pts_seg]
            plt.scatter(xax, yax, color=c, s=s, alpha=alpha)

            if linear_fit:
                slope_mat, intercept_mat, _ = pw_fit_dict
                slope=slope_mat[gene,seg]
                intercept=intercept_mat[gene,seg]
                plt.plot(unique_binned_isodepths[pts_seg], np.log(offset) + intercept + slope*unique_binned_isodepths[pts_seg], color='grey', alpha=1, lw=lw )

    if xticks is None:
        plt.xticks(fontsize=ticksize)
    else:
        plt.xticks(xticks,fontsize=ticksize)
        
    if yticks is None:
        plt.yticks(fontsize=ticksize)
    else:
        plt.yticks(yticks,fontsize=ticksize)
        
    if domain_boundary_plotting and len(domain_list)>1:
        binned_labels=binning_output['binned_labels']
        
        left_bps=[]
        right_bps=[]

        for i in range(len(binned_labels)-1):
            if binned_labels[i] != binned_labels[i+1]:
                left_bps.append(unique_binned_isodepths[i])
                right_bps.append(unique_binned_isodepths[i+1])
        
        for i in domain_list[:-1]:
            plt.axvline((left_bps[i]+right_bps[i])*0.5, color='black', ls='--', linewidth=1.5, alpha=0.2)

    sns.despine()
    if show_lgd:
        plt.legend(bbox_to_anchor=lgd_bbox)
    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{gene_name}_pwlinear.pdf", bbox_inches="tight")
        plt.close()
