import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from pos_encoding import positional_encoding
device = 'cuda' if torch.cuda.is_available() else 'cpu'

##################################################################################
# Neural network class
# Inputs: 
#   G: number of genes/features
#   S_hidden_list: list of hidden layer sizes for f_S 
#                  (e.g. [50] means f_S has one hidden layer of size 50)
#   A_hidden_list: list of hidden layer sizes for f_A 
#                  (e.g. [10,10] means f_A has two hidden layers, both of size 10)
#   activation_fn: activation function
##################################################################################

class Multi_GASTON(nn.Module):
    """
    Neural network class. Has two attributes: 
    (1) spatial embedding f_S : R^2 -> R^K, and 
    (2) expression function f_A : R^K -> R^G. 
    Each of these is parametrized by a neural network.
    
    Parameters
    ----------
    G
        number of genes/features
    S_hidden_list
        list of hidden layer sizes for f_S 
        (e.g. [50] means f_S has one hidden layer of size 50)
    A_hidden_list
        list of hidden layer sizes for f_A 
        (e.g. [10,10] means f_A has two hidden layers, both of size 10)
    activation_fn
        activation function for neural network
    A_linear
        boolean indicating whether f_A is a linear function
    K
        number of latent variables/isodepths to learn
    pos_encoding
        positional encoding option
    embed_size
        positional encoding embedding size
    sigma
        positional encoding sigma hyperparameter
    """
    
    def __init__(
        self, 
        G, 
        S_hidden_list, 
        A_hidden_list,
        activation_fn=nn.ReLU(),
        A_linear = False,
        K = 1,
        pos_encoding=False,
        embed_size=4,
        sigma=0.1
    ):
        super(Multi_GASTON, self).__init__()

        self.pos_encoding = pos_encoding
        self.embed_size = embed_size
        self.sigma = sigma
        
        input_size = 2*embed_size if self.pos_encoding else 2
        
        # create spatial embedding f_S
        S_layer_list=[input_size] + S_hidden_list + [K]
        S_layers=[]
        for l in range(len(S_layer_list)-1):
            # add linear layer
            S_layers.append(nn.Linear(S_layer_list[l], S_layer_list[l+1]))
            # add activation function except for last layer
            if l != len(S_layer_list)-2:
                S_layers.append(activation_fn)
                
        self.spatial_embedding=nn.Sequential(*S_layers)
        
        # create expression function f_A
        A_layer_list=[K] + A_hidden_list + [G]
        A_layers=[]
        for l in range(len(A_layer_list)-1):
            # add linear layer
            A_layers.append(nn.Linear(A_layer_list[l], A_layer_list[l+1]))
            # the expression mapping is be made linear when activation functions
            # are not added
            if A_linear == False:
                if l != len(A_layer_list)-2:
                    A_layers.append(activation_fn)
            
        self.expression_function=nn.Sequential(*A_layers)

    # Compute the lasso term (1-norm) on the expression function weights
    # with given lasso coefficient L
    def Lasso_reg(self,L):
        weights = self.expression_function[0].weight
        lasso  = L * torch.norm(weights, 1)
        return lasso
    
    # Convergence check
    def convergence_check(self,loss_diff,epoch,loss_threshold):
        if max(loss_diff[epoch-10:epoch]) < loss_threshold:
            return True
        return False

    def forward(self, x):
        z = self.spatial_embedding(x) # relative depth
        return self.expression_function(z)


##################################################################################
# Train NN
# Inputs: 
#   model: multi GASTON object
#   S: torch Tensor (N x 2) containing spot locations
#   A: torch Tensor (N x G) containing features
#   epochs: number of epochs to train
#   batch_size: batch size


#   A_hidden_list: list of hidden layer sizes for f_A 
#                  (e.g. [10,10] means f_A has two hidden layers, both of size 10)
#   activation_fn: activation function
##################################################################################
    
def train(S, A, 
          multi_gaston=None, S_hidden_list=None, A_hidden_list=None, activation_fn=nn.ReLU(),
          epochs=1000, batch_size=None, 
          checkpoint=100, SAVE_PATH=None, loss_reduction='mean', 
          optim='sgd', lr=1e-3, weight_decay=0, momentum=0, seed=0, 
          A_linear = False, lasso_lambda = 0, K = 2,
          loss_threshold = 1e-9, 
          schedule='linear',schedule_para=[0.05,12000],
          pos_encoding=False, embed_size=4, sigma=0.1):
    """
    Train a Multi_GASTON from scratch
    
    Parameters
    ----------
    multi_gaston
        Multi_GASTON object
    S
        torch Tensor (N x 2) containing spot locations
    A
        torch Tensor (N x G) containing features
    epochs
        number of epochs to train
    batch_size
        batch size of neural network
    checkpoint
        save the current NN when the epoch is a multiple of checkpoint
    SAVE_PATH
        folder to save NN at checkpoints
    loss_reduction
        either 'mean' or 'sum' for MSELoss
    optim
        optimizer to use (currently supports either 'sgd' or 'adam')
    lr
        learning rate for the optimizer
    weight_decay
        weight decay parameter for optimizer
    momentum
        momentum parameter, if using SGD optimizer
    A_linear
        if the expression mapping is linear
    lasso_lambda
        if positive, the lasso regularization coefficient
    K
        number of isodepths to learn
    loss_threshold 
        loss convergence threshold
    schedule
        type of learning rate decay, currently supports 'linear' and 'expo'
    schedule_para
        learning rate decay parameters
    """
    torch.manual_seed(seed)
    N,G=A.shape

    if multi_gaston == None:
        multi_gaston=Multi_GASTON(G, S_hidden_list, A_hidden_list, activation_fn=activation_fn, A_linear=A_linear, K=K, pos_encoding=pos_encoding, embed_size=embed_size, sigma=sigma)
    
    if optim=='sgd':
        opt = torch.optim.SGD(multi_gaston.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim=='adam':
        opt = torch.optim.Adam(multi_gaston.parameters(), lr=lr, weight_decay=weight_decay)

    if schedule=='linear' and len(schedule_para) >= 2:
        scheduler = lr_scheduler.LinearLR(opt, start_factor=1, end_factor=float(schedule_para[0]), total_iters=int(schedule_para[1]))
    elif schedule=='expo' and len(schedule_para) >= 1:
        scheduler = lr_scheduler.ExponentialLR(opt, gamma=float(schedule_para[0]))
    
    # record the loss for both result logging and convergence check
    loss_list=np.zeros(epochs)
    lasso_loss=np.zeros(epochs)
    loss_diff=np.zeros(epochs)

    S_init = torch.clone(S)
    if multi_gaston.pos_encoding:
        S = positional_encoding(S, multi_gaston.embed_size, multi_gaston.sigma)

    loss_function=torch.nn.MSELoss(reduction='none')

    for epoch in range(epochs):
        if epoch%checkpoint==0:
            # print(f"Epoch %d: {optim} lr = %.3E" % (epoch, opt.param_groups[0]["lr"]))
            if SAVE_PATH is not None:
                torch.save(multi_gaston, SAVE_PATH + f'model_epoch_{epoch}.pt')
                np.save(SAVE_PATH+'loss_list.npy', loss_list)
                np.save(SAVE_PATH+'lasso_loss.npy', lasso_loss)
                radius = multi_gaston.spatial_embedding(S).detach().numpy()
                np.savetxt(SAVE_PATH+f'{epoch}_radius.txt', radius)
        
        if batch_size is not None:
            # take non-overlapping random samples of size batch_size
            permutation = torch.randperm(N)
            for i in range(0, N, batch_size):
                opt.zero_grad()
                indices = permutation[i:i+batch_size]

                S_ind=S[indices,:]
                S_ind.requires_grad_()

                A_ind=A[indices,:]

                loss = loss_function(multi_gaston(S_ind), A_ind)
                loss_list[epoch] += loss.item()
                if lasso_lambda > 0:  
                    lasso = multi_gaston.Lasso_reg(lasso_lambda)
                    lasso_loss[epoch] = lasso.item()
                    # if epoch%checkpoint==0:
                    #     print(f"original loss is {loss}, and lasso term is {lasso/lasso_lambda}")
                    loss+=lasso
            
                loss_list[epoch] += loss.item()
                if epoch>0:
                    loss_diff[epoch] = loss_list[epoch]-loss_list[epoch-1]
                loss.backward()
                opt.step()
        else:
            opt.zero_grad()
            S.requires_grad_()

            loss = loss_function(multi_gaston(S), A)
            if lasso_lambda > 0:  
                lasso = multi_gaston.Lasso_reg(lasso_lambda)
                lasso_loss[epoch] = lasso.item()
                # if epoch%checkpoint==0:
                #     print(f"original loss is {loss}, and lasso term is {lasso/lasso_lambda}")
                loss+=lasso
                
            loss_list[epoch] += loss.item()
            if epoch>0:
                loss_diff[epoch] = abs(loss_list[epoch]-loss_list[epoch-1])
            loss.backward()
            opt.step()
            
        # Convergence check and update loss 
        if epoch > 1000 and multi_gaston.convergence_check(loss_diff,epoch,loss_threshold):
            loss_list = loss_list[:epoch]
            lasso_loss = lasso_loss[:epoch]
            corr_loss = corr_loss[:epoch]
            # print(f'convergence loss threshold reached at epoch {epoch}')
            break

        # Adjust learning rate
        if schedule == 'expo' and opt.param_groups[0]["lr"] < float(schedule_para[1]):
            continue
        before_lr = opt.param_groups[0]["lr"]
        scheduler.step()
        after_lr = opt.param_groups[0]["lr"]
        # if epoch<5:
        #     print(f"Epoch %d: {optim} lr %.3E -> %.3E" % (epoch, before_lr, after_lr))

    if SAVE_PATH is not None:
        torch.save(multi_gaston, f'{SAVE_PATH}/final_model.pt')
        np.save(f'{SAVE_PATH}/loss_list.npy', loss_list)
        np.save(f'{SAVE_PATH}/lasso_loss.npy', lasso_loss)
        radius = multi_gaston.spatial_embedding(S).detach().numpy()
        np.savetxt(f'{SAVE_PATH}/radius.txt', radius)
        torch.save(A, f'{SAVE_PATH}/Atorch.pt')
        if multi_gaston.pos_encoding:
            torch.save(S_init, f'{SAVE_PATH}/Storch.pt')
        else:
            torch.save(S, f'{SAVE_PATH}/Storch.pt')

    return multi_gaston, loss_list, lasso_loss