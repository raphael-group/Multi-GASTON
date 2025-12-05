import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from multi_gaston.pos_encoding import positional_encoding
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

# Structured linear layer class: implement a structured layer using a given mask on weights
class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, mask: torch.Tensor = None):
        super().__init__(in_features, out_features, bias)
        if mask is not None:
            if mask.shape != (out_features, in_features):
                raise ValueError("Mask shape must match (out_features, in_features)")
            self.register_buffer('mask', mask) # Register mask as a buffer, not a parameter
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'mask'):
            # Apply the mask to the weights before performing the matrix multiplication
            masked_weight = self.weight * self.mask
            return nn.functional.linear(input, masked_weight, self.bias)
        else:
            return nn.functional.linear(input, self.weight, self.bias)

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
        boolean indicating if we want f_A to be linear
    K
        number of tissue-intrinsic coordinates to learn
    slices
        list of number of spots in each input sample
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
        slices = [],
        pos_encoding=False, 
        embed_size=4,
        sigma=0.1,
    ):
        super(Multi_GASTON, self).__init__()
        self.slices = slices
        self.pos_encoding = pos_encoding
        self.embed_size = embed_size
        self.sigma = sigma
        self.K = K
        self.A_linear = A_linear
        self.mask = None

        input_size = 2*embed_size if self.pos_encoding else 2
        M = len(slices)

        self.coordinate_function = nn.ModuleList()
        # create a spatial embedding f_S for each sample
        for s in range(M):
            S_layer_list = [input_size] + S_hidden_list + [K]
            S_layers=[]
            for l in range(len(S_layer_list)-1):
                # add linear layer
                S_layers.append(nn.Linear(S_layer_list[l], S_layer_list[l+1]))
                # add activation function except for last layer
                if l != len(S_layer_list)-2:
                    S_layers.append(activation_fn)
            self.coordinate_function.append(nn.Sequential(*S_layers))
        # print(f'f_S has structure {S_layer_list}')
        
        # create expression function f_A
        A_layer_list=[K] + A_hidden_list + [G]
        A_layers=[]
        if K == 1 or A_linear:
            # when learning a single coordinate or linear f_A, we have just one f_A
            for l in range(len(A_layer_list)-1):
                # add linear layer
                A_layers.append(nn.Linear(A_layer_list[l], A_layer_list[l+1]))
                # add activation function except the last layer
                if A_linear == False:
                    # when A_linear, we don't add activation layer
                    if l != len(A_layer_list)-2:
                        A_layers.append(activation_fn)
                
            self.expression_function = nn.Sequential(*A_layers)
        else: 
            # In the case where K>1 and A_linear == False
            self.expression_function = nn.ModuleList()
            A_layer_list = [1] + A_hidden_list + [G] + [G]
            for i in range(K):
                # For each coordinate, we construct all layers of f_A same fashion as in the 
                # previous case except the last layer
                A_layers=[]
                for l in range(len(A_layer_list)-2):
                    A_layers.append(nn.Linear(A_layer_list[l], A_layer_list[l+1]))
                    A_layers.append(activation_fn)                   
                self.expression_function.append(nn.Sequential(*A_layers))
            # We want a structured last linear layer, where coordinate K's prediction of gene g
            # is only connected to gene g.
            self.mask = torch.hstack([torch.eye(G)] * K)
            self.expression_function.append(MaskedLinear(G*K, A_layer_list[-1],mask=self.mask))
            
        # print(f'each f_A has structure {A_layer_list}')

    # Compute the lasso term (1-norm) on the expression function last layer weights
    # with given lasso coefficient L
    def Lasso_reg(self,L):
        weights = self.expression_function[-1].weight
        if self.mask is not None: 
            # when we have a structure last linear layer
            weights * self.mask
        lasso  = L * torch.norm(weights, 1)
        return lasso

    # Convergence check
    def convergence_check(self,loss_diff,epoch,loss_threshold):
        if max(loss_diff[epoch-10:epoch]) < loss_threshold:
            return True
        return False

    def forward(self, x):
        # First compute tissue-intrinsic coordinates d
        if len(self.slices) <= 1:
            d = self.coordinate_function[0](x) # relative depth
        else:
            d = []
            counter = 0
            for l in range(len(self.slices)):
                d.append(self.coordinate_function[l](x[counter:counter+self.slices[l],:]))
                counter += self.slices[l]
            d = torch.cat(d,dim=0)
        # Then compute gene expression prediction
        if self.K>1 and self.A_linear == False:
            g = []
            for i in range(self.K):
                g.append(self.expression_function[i](d[:,i].unsqueeze(1)))
            g = torch.cat(g,dim=1)
            return self.expression_function[-1](g)
        else:
            return self.expression_function(d)


def train(S, A,
          multi_gaston=None, S_hidden_list=None, A_hidden_list=None, activation_fn=nn.ReLU(),
          epochs=1000, checkpoint=100, SAVE_PATH=None, 
          loss_func = 'mse', loss_reduction='mean', optim='adam', lr=1e-3, weight_decay=0, momentum=0, seed=0, 
          A_linear = False, lasso_lambda = 0, loss_threshold = 1e-8, K = 1,
          schedule='linear',schedule_para=[0.05,12000],
          pos_encoding=False, embed_size=4, sigma=0.1, slices=[], all_avg=True):
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
    checkpoint
        save the current NN when the epoch is a multiple of checkpoint
    SAVE_PATH
        folder to save NN at checkpoints
    loss_reduction
        either 'mean' or 'sum' for MSELoss
    optim
        optimizer to use (currently supports either 'adam' or 'sgd')
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
        number of tissue-intrinsic coordinates to learn
    slices
        list of number of spots in each sample 
    all_avg
        given loss_reduction is 'mean', when True, the loss is average of all spots across 
        all samples; otherwise, the loss is the average of losses from each sample
    loss_threshold
        loss convergence threshold
    schedule
        type of learning rate decay, currently supports 'linear' and 'expo'
    schedule_para
        learning rate decay parameters
    """

    # Check if slices parameter is valid
    if len(slices) >= 1:
        assert sum(slices) == S.shape[0]
        assert sum(slices) == A.shape[0]
    else:
        # when there is just M = 1 sample
        slices = [S.shape[0]]
    M = len(slices)

    # Initialize Multi_GASTON object and training parameters
    torch.manual_seed(seed)
    if multi_gaston == None:
        multi_gaston=Multi_GASTON(A.shape[1], S_hidden_list, A_hidden_list, activation_fn=activation_fn, A_linear=A_linear, K=K, pos_encoding=pos_encoding, embed_size=embed_size, sigma=sigma, slices=slices)
    
    if optim=='sgd':
        opt = torch.optim.SGD(multi_gaston.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim=='adam':
        opt = torch.optim.Adam(multi_gaston.parameters(), lr=lr, weight_decay=weight_decay)
    if schedule=='linear' and len(schedule_para) >= 2:
        scheduler = lr_scheduler.LinearLR(opt, start_factor=1, end_factor=float(schedule_para[0]), total_iters=int(schedule_para[1]))
    if schedule=='linearWARM' and len(schedule_para) >= 3:
        scheduler_warm = lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1, total_iters=int(schedule_para[-1]))
        scheduler_1 = lr_scheduler.LinearLR(opt, start_factor=1, end_factor=float(schedule_para[0]), total_iters=int(schedule_para[1]))
        scheduler = lr_scheduler.SequentialLR(opt, schedulers=[scheduler_warm, scheduler_1], milestones=[int(schedule_para[-1])])
    elif schedule=='expo' and len(schedule_para) >= 1:
        scheduler = lr_scheduler.ExponentialLR(opt, gamma=float(schedule_para[0]))
    elif schedule=='constant':
        scheduler = lr_scheduler.ConstantLR(opt, factor=1, total_iters=0)
    else: 
        scheduler = lr_scheduler.LinearLR(opt, start_factor=1, end_factor=0.1,total_iters=10000)

    # Initialize variables
    loss_list=np.zeros(epochs)
    lasso_loss=np.zeros(epochs)
    loss_diff=np.zeros(epochs)
    loss_slices = []
    for l in multi_gaston.slices:
        loss_slices.append(np.zeros(epochs))
    S_init = torch.clone(S)
    if multi_gaston.pos_encoding:
        S = positional_encoding(S, multi_gaston.embed_size, multi_gaston.sigma)
    if loss_func == 'mse':
        loss_function=torch.nn.MSELoss(reduction='none')
    else:
        print(f'loss function = {loss_func} currently not supported!')
        return
        
    # Start training
    for epoch in range(epochs):
        if epoch%checkpoint==0:
            if SAVE_PATH is not None:
                torch.save(multi_gaston, SAVE_PATH + f'model_epoch_{epoch}.pt')
                np.save(SAVE_PATH+'loss_list.npy', loss_list)
                np.save(SAVE_PATH+'lasso_loss.npy', lasso_loss)
                if M > 1: np.save(SAVE_PATH+'loss_slices.npy', loss_slices)
                coordinate = torch.zeros((S.shape[0],K))
                counter = 0
                for l in range(M):
                    coordinate[counter:counter+slices[l],:] = multi_gaston.coordinate_function[l](S[counter:counter+slices[l],:])
                    counter += slices[l]
                coordinate = coordinate.detach().numpy()
                if M == 1: coordinate = coordinate.flatten()
                np.savetxt(SAVE_PATH+f'{epoch}_coordinate.txt', coordinate)
                        
        opt.zero_grad()
        S.requires_grad_()
        
        # Update loss
        loss_l = loss_function(multi_gaston(S), A)
        loss_l.requires_grad_()
        if loss_reduction=='mean': 
            if M > 1:
            # When there are multiple samples, we record the loss in each sample
                losses_all_samples = []
                counter = 0
                for l in range(len(slices)):
                    loss_slice = torch.mean(loss_l[counter:counter+slices[l],:])
                    loss_slice.requires_grad_()
                    losses_all_samples.append(loss_slice)
                    loss_slices[l][epoch] = loss_slice.item()
                    counter+=slices[l]
                if all_avg:
                    loss=torch.mean(loss_l)
                else:
                    loss=torch.mean(torch.stack(losses_all_samples))
            else:
                loss=torch.mean(loss_l)
        if loss_reduction=='sum':  loss=torch.sum(loss_l)
        loss.requires_grad_()

        # Apply lasso regularization
        if lasso_lambda > 0:  
            lasso = multi_gaston.Lasso_reg(lasso_lambda)
            lasso_loss[epoch] = lasso.item()
            loss += lasso
            
        loss_list[epoch] = loss.item()
        if epoch>0:
            loss_diff[epoch] = abs(loss_list[epoch]-loss_list[epoch-1])
        loss.backward()
        opt.step()

        # Convergence check
        if epoch > 1000 and multi_gaston.convergence_check(loss_diff,epoch,loss_threshold):
            loss_list = loss_list[:epoch]
            lasso_loss = lasso_loss[:epoch]
            if len(slices)>0:
                loss_slices = [L[:epoch] for L in loss_slices]
            break

        # Adjust learning rate
        if (schedule == 'expo' and opt.param_groups[0]["lr"] < float(schedule_para[1])) == False:
            # We enforce a lower bound on lr schedule = "expo" as schedule_para[1]
            scheduler.step()
    
    # Save final results
    if SAVE_PATH is not None:
        torch.save(multi_gaston, f'{SAVE_PATH}/final_model.pt')
        np.save(f'{SAVE_PATH}/loss_list.npy', loss_list)
        np.save(f'{SAVE_PATH}/lasso_loss.npy', lasso_loss)
        if M == 1:
            coordinate=multi_gaston.coordinate_function[0](S).detach().numpy()
        else:
            np.save(f'{SAVE_PATH}/loss_slices.npy', loss_slices)
            coordinate = torch.zeros((S.shape[0],K))
            counter = 0
            for l in range(len(multi_gaston.slices)):
                coordinate[counter:counter+slices[l],:] = multi_gaston.coordinate_function[l](S[counter:counter+slices[l],:])
                counter += slices[l]
            coordinate = coordinate.detach().numpy()
        np.savetxt(SAVE_PATH+f'coordinate.txt', coordinate)
        torch.save(A, f'{SAVE_PATH}/Atorch.pt')
        if multi_gaston.pos_encoding:
            torch.save(S_init, f'{SAVE_PATH}/Storch.pt')
        else:
            torch.save(S, f'{SAVE_PATH}/Storch.pt')

    return multi_gaston, loss_list, lasso_loss, loss_slices