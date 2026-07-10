'''
This script contains all definition of neural networks used in the algorithm.
'''
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Literal, Union, Optional

# ---------------------------------------------------------------------------------------------------------
# Supporting blocks 
# ---------------------------------------------------------------------------------------------------------

    # Custom activation function - Swish
class Swish(nn.Module):
    def forward(self, x, coeff = 1.0):
        return x * torch.sigmoid(coeff * x)

    # Time encoder 
class Sinusoidal_Time_Encoder(nn.Module):
    def __init__(self, num_frequencies = 4):
        super(Sinusoidal_Time_Encoder, self).__init__()
        self.num_frequencies = num_frequencies

    def forward(self, t) :  # dim(t) = (n_samples, 1)
        if t.dim() == 1 : 
            t = t.unsqueeze(-1).to(t.device)    # in case dim(t) = (n_samples, ), then enlarge it to (n_samples, 1)

        freqs =  torch.exp(
                    torch.arange(self.num_frequencies, device=t.device) * - (math.log(10000.0)/(self.num_frequencies - 1))
                )
        sin = torch.sin(freqs * t).to(t.device) # dim = (batch_size, num_frequencies)
        cos = torch.cos(freqs * t).to(t.device)
        encoding = torch.cat([sin, cos], 
                        dim=-1).to(t.device)    # dim = (batch_size, 2 * num_frequencies)
        return encoding

    # Residual block
class Residual_MLP(nn.Module):
    def __init__(self, 
            dim_input : int,                                                    # input dimension (for the residual block)
            dim_output : int,                                                   # output dimension (for the residual block)
            dim_hidden : int,                                                   # number of neurons per hidden layer 
            n_hidden_layers : int,                                              # number of hidden layers in the residual block 
            hidden_activation_type : Literal['GELU', 'ReLU', 'Swish'] = 'GELU', # type of activation function for the hidden layers 
            norm_type : Literal['batch', 'identity', 'layer'] = 'batch'         # type of normalization to be used
            ):   
        super(Residual_MLP, self).__init__()
        self.input_proj = nn.Linear(dim_input, dim_hidden)  
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(n_hidden_layers):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))   

            if hidden_activation_type == "GELU":
                self.layers.append(nn.GELU())
            elif hidden_activation_type == "ReLU":
                self.layers.append(nn.ReLU())
            elif hidden_activation_type == "Swish":
                self.layers.append(Swish())
            else:
                raise ValueError(f"Unknown activation function for hidden layers: {hidden_activation_type}.")

            self.layers.append(nn.Linear(dim_hidden, dim_hidden))

            if norm_type == 'layer':
                self.norms.append(nn.LayerNorm(dim_hidden))
            elif norm_type == 'identity':
                self.norms.append(nn.Identity(dim_hidden))
            elif norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(dim_hidden))
            else:
                raise ValueError(f"Unknown normalization type: {norm_type}.")

        self.output_layer = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        x = self.input_proj(x)     
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x))  # apply layer norm on residual
        return self.output_layer(x) # dim = (batch_size, dim_output)
    
# ---------------------------------------------------------------------------------------------------------
# Neural Network for estimating the optimal control on the boundary
# ---------------------------------------------------------------------------------------------------------

class Boundary_Optimal_Control_Net(nn.Module):
    def __init__(self, 
            dim_state : int,                                                            # dimension of the state variable
                dim_output : int,                                                       # dimension of the output variable ( = dimension of the control variable)
                dim_hidden : int,                                                       # number of neurons in each hidden layer
                n_hidden_layers : int,                                                  # number of hidden layers in the network
                lower_bounds : Union[np.ndarray, torch.Tensor],                         # lower bound for the control variable
                upper_bounds : Union[np.ndarray, torch.Tensor],                         # upper bound for the control variable
                num_time_freqs : int = 5,                                               # number of frequencies (for time encoding)
                hidden_activation_type : Literal["Swish", "ReLU", "GELU"] = "Swish",    # type of activation functin for hidden layers
                output_activation_type : Literal["Tanh", "Sigmoid"] = "Tanh",           # type of activation function for the last layer before output (which makes sure that all controls stay within the given intervals)
                norm_type : Literal['batch', 'identity', 'layer'] = 'batch'             # type of normalization to be used
                ):       
        
            # initialization
        super(Boundary_Optimal_Control_Net, self).__init__()
        self.dim_state = dim_state

            # enlarge input dimension to fit the time encoder
        self.time_encoder = Sinusoidal_Time_Encoder(num_frequencies = num_time_freqs)
        dim_input = 2 * num_time_freqs + dim_state  

            # stack layers        
        layers = []
        dim_in = dim_input    # dim_in of the first layer is dim_input, otherwise dim_in = dim_hidden for all the other hidden layers

        for _ in range(n_hidden_layers):                # first + hidden layers
                # normalization 
            if norm_type == 'batch':
                layers.append(nn.BatchNorm1d(dim_in))
            elif norm_type == 'identity':
                layers.append(nn.Identity(dim_in))
            elif norm_type == 'layer':
                layers.append(nn.LayerNorm(dim_in))
            else : 
                raise ValueError(f'Unknown normalization type : {norm_type}')
        
                # linear pass with chosen activation function
            layers.append(nn.Linear(dim_in, dim_hidden))

            if hidden_activation_type == "Swish":
                layers.append(Swish())
            elif hidden_activation_type == "GELU":
                layers.append(nn.GELU())
            elif hidden_activation_type == "ReLU":
                layers.append(nn.ReLU())
            else :
                raise ValueError(f'Unknown activation function for hidden layers : {hidden_activation_type}')

            dim_in = dim_hidden             # update dim_in (for the next hidden layer)
        
        layers.append(nn.Linear(dim_in, dim_output))   # output layer
        if output_activation_type == "Tanh":
            layers.append(nn.Tanh())        # should be either nn.Tanh() or nn.Sigmoid() for bounding the output  
        elif output_activation_type == "Sigmoid":
            layers.append(nn.Sigmoid())


        self.mlp = nn.Sequential(*layers)   # stack all layers - output of this has dim = (batch_size, dim_output)

            # output layer
        self.lower_bounds = lower_bounds    # lower bounds for the control variable
        self.upper_bounds = upper_bounds    # upper bounds for the control variable
        self.output_activation_type = output_activation_type
        self.dim_output = dim_output

    def forward(self, 
                t: torch.Tensor,            # dim = (sample_size, 1) or (sample_size, ) 
                X: torch.Tensor             # dim = (sample_size, dim_state)
            ) -> torch.Tensor:     # u(t,X) - dim = (sample_size, dim_control)
        device = X.device
        
        assert t.shape[0] == X.shape[0], f'Dimension Mismatch : batch size for t = {t.shape[0]} does NOT match the batch size for X = {X.shape[0]}.'
        assert X.shape[1] == self.dim_state, f'Dimension Mismatch : 2nd dimension for X = {X.shape[1]} does NOT match the expected self.dim_state = {self.dim_state}.'

        t_encoded = self.time_encoder(t)    # encoding time (sinudoial) - dim = (sample_size, 2*num_time_freqs)
        t_encoded = t_encoded.to(device)  

        t_X = torch.cat([t_encoded, X], dim=1).to(device)   # dim = (sample_size, dim_state + 2*num_time_freqs)
        out = self.mlp(t_X).to(device)      # all values between (0,1) if sigmoid activation or (-1, 1) if tanh activation 
        assert out.shape == torch.Size([t_X.shape[0], self.dim_output]), f'Dimension Mismatch : out has shape = {out.size()} while expecting (batch_size, dim_control) = ({t_X.shape[0], self.dim_output}).'
        
        if self.output_activation_type == "Tanh" :
            out = self.lower_bounds.to(device) + (self.upper_bounds.to(device) - self.lower_bounds.to(device)) * (out + 1)/2
        elif self.output_activation_type == "Sigmoid" :
            out = self.lower_bounds.to(device) + (self.upper_bounds.to(device) - self.lower_bounds.to(device)) * out
        else :
            raise ValueError(f'Unknown output activation function {self.output_activation}')
        
        out = out.to(device)

        return out                          # dim = (sample_size, dim_control)
    
# ---------------------------------------------------------------------------------------------------------
# Neural Network for estimating the boundary of the domain
# ---------------------------------------------------------------------------------------------------------

class Domain_Boundary_Net(nn.Module):
    def __init__(self, 
            dim_state : int,                                                        # dimension of the state variable
            dim_hidden : int,                                                       # number of neurons in each hidden layer
            n_hidden_layers : int,                                                  # number of hidden layers in the network
            num_time_freqs : int = 5,                                               # number of time frequencies for the time encoder
            hidden_activation_type : Literal["Swish", "ReLU", "GELU"] = "Swish",    # type of activation function for hidden layers
            norm_type : Literal['batch', 'identity', 'layer'] = 'batch'             # type of normalization to be used 
            ):
        # initialization
        super(Domain_Boundary_Net, self).__init__()
        self.dim_state = dim_state

        # enlarge input dimension to fit the time encoder
        self.time_encoder = Sinusoidal_Time_Encoder(num_frequencies = num_time_freqs)
        dim_input = 2 * num_time_freqs + dim_state  

        # layers
        self.mlp = Residual_MLP(dim_input = dim_input,
                                dim_output = 1,             
                                dim_hidden = dim_hidden,
                                hidden_activation_type = hidden_activation_type,
                                n_hidden_layers = n_hidden_layers,
                                norm_type = norm_type)

    def forward(self, 
            t,                          # dim = (sample_size, 1) or (sample_size, )
            X                           # dim = (sample_size, dim_state)
            ) -> torch.Tensor: # w(t,X) - dim = (sample_size, 1)
        
        # device    
        device = X.device
        t = t.to(device)

        # check dimension
        assert t.shape[0] == X.shape[0], f'Dimension Mismatch : batch size for t = {t.shape[0]} does NOT match the batch size for X = {X.shape[0]}.'
        assert X.shape[1] == self.dim_state, f'Dimension Mismatch : 2nd dimension for X = {X.shape[1]} does NOT match the expected self.dim_state = {self.dim_state}.'

        # encode time 
        t_encoded = self.time_encoder(t)    # encoding time (sinudoial) - dim = (sample_size, 2 * num_time_freqs)
        t_encoded = t_encoded.to(device)  

        t_X = torch.cat([t_encoded, X], 
                        dim=1).to(device)   # dim = (sample_size, dim_state + 2 * num_time_freqs)
        out = self.mlp(t_X).to(device)          

        assert out.shape == torch.Size([t_X.shape[0], 1]), f'Dimension Mismatch : out has shape = {out.size()} while expecting (batch_size, 1) = ({t_X.shape[0], 1}).'
        out = out.to(device)

        return out                          # dim = (sample_size, 1)

# ---------------------------------------------------------------------------------------------------------
# Neural Network for estimating the value function at the boundary
# ---------------------------------------------------------------------------------------------------------

class Boundary_Value_Function_Net(nn.Module):
    '''
    Note that Vb(t,X) = V(t,X,w(t,X)). 
    Structually, Vb_NN and w_NN are similar (dim_output = 1, dim_input = 2 * n_time_freqs + 1 + dim_state),
        but Vb_NN aims to estimate the expected value of utility function F (to be maximized) 
        while w_NN aims to estimate the expected value of cost function G (to be minimized)
    '''
    def __init__(self, 
            dim_state : int,                                                        # dimension of the state variable
            dim_hidden : int,                                                       # number of neurons in each hidden layer
            n_hidden_layers : int,                                                  # number of hidden layers in the network
            num_time_freqs : int = 5,                                               # number of time frequencies for the time encoder
            hidden_activation_type : Literal["Swish", "ReLU", "GELU"] = "Swish",    # type of activation function for hidden layers
            norm_type : Literal['batch', 'identity', 'layer'] = 'batch'             # type of normalization to be used 
            ):

        # initiation
        super(Boundary_Value_Function_Net, self).__init__()
        self.dim_state = dim_state
        
        # enlarge input dimension to fit the time encoder
        self.time_encoder = Sinusoidal_Time_Encoder(num_frequencies = num_time_freqs)
        dim_input = 2 * num_time_freqs + dim_state  

        # layers
        self.mlp = Residual_MLP(dim_input = dim_input,
                                dim_output = 1,             
                                dim_hidden = dim_hidden,
                                hidden_activation_type = hidden_activation_type,
                                n_hidden_layers = n_hidden_layers,
                                norm_type = norm_type)
        
    def forward(self, 
            t,                      # dim = (sample_size, 1) or (sample_size, )
            X                       # dim = (sample_size, dim_state)
        ) -> torch.Tensor:# Vb(t,X) - dim = (sample_size, 1)

        device = X.device

        assert t.shape[0] == X.shape[0], f'Dimension Mismatch : batch size for t = {t.shape[0]} does NOT match the batch size for X = {X.shape[0]}.'
        assert X.shape[1] == self.dim_state, f'Dimension Mismatch : 2nd dimension for X = {X.shape[1]} does NOT match the expected self.dim_state = {self.dim_state}.'

        t_encoded = self.time_encoder(t)    # encoding time (sinudoial) - dim = (sample_size, 2 * num_time_freqs)
        t_encoded = t_encoded.to(device)  

        t_X = torch.cat([t_encoded, X], 
                        dim=1).to(device)   # dim = (sample_size, dim_state + 2 * num_time_freqs)
        out = self.mlp(t_X).to(device)          

        assert out.shape == torch.Size([t_X.shape[0], 1]), f'Dimension Mismatch : out has shape = {out.size()} while expecting (batch_size, 1) = ({t_X.shape[0], 1}).'
        out = out.to(device)
        
        return out                          # dim = (sample_size, 1)
    
# ---------------------------------------------------------------------------------------------------------
# Neural Network for estimating the (maximizing + augmented) optimal control on the boundary
# ---------------------------------------------------------------------------------------------------------

class Optimal_Control_Net(nn.Module):
    def __init__(self,
            dim_state : int,                                                        # dimension of the state variable
            dim_control : int,                                                      # dimension of the control variable (NOT including the control for martingale)
            dim_sto : int,                                                          # dimension of the stochastic factor = dimension fo the control for martingale
            dim_hidden : int,                                                       # number of neurons in each hidden layer
            n_hidden_layers : int,                                                  # number of hidden layers in the network
            lower_bounds : Union[np.ndarray, torch.Tensor],                         # lower bound for the control variable 
            upper_bounds : Union[np.ndarray, torch.Tensor],                         # upper bound for the control variable 
            martingale_control_limit : float,                                       # limit for the control for martingale 
            device : torch.device,
            hidden_activation_type : Literal["Swish", "ReLU", "GELU"] = "Swish",    # type of activation function for hidden layers
            output_activation_type : Literal["Tanh", "Sigmoid"] = "Tanh",           # type of activation function for the last layer before output (which makes sure that all controls stay within the given intervals)
            num_time_freqs : int = 5,                                               # number of time frequencies for the time encoder
            norm_type : Literal['batch', 'identity', 'layer'] = 'batch'             # type of normalization to be used 
        ):  

        # initiate
        super(Optimal_Control_Net, self).__init__()
        self.dim_state = dim_state
        self.dim_control = dim_control
        self.dim_sto = dim_sto
        self.dim_output = dim_control + dim_sto

        # enlarge the input dimension to include time encoder
        self.time_encoder = Sinusoidal_Time_Encoder(num_frequencies = num_time_freqs)
        dim_input = 2 * num_time_freqs + dim_state + 1          # this network takes as input the augmented state variable, which includes the martingale (hence the + 1)

        # stack layers
        layers = []
        dim_in = dim_input  

            # hidden layers
        for _ in range(n_hidden_layers) :   # dim_in = dim_input for the first layer, then dim_in = dim_hidden for all other hidden layers
                # normalize
            if norm_type == 'batch':
                layers.append(nn.BatchNorm1d(dim_in))
            elif norm_type == 'identity':
                layers.append(nn.Identity(dim_in))
            elif norm_type == 'layer':
                layers.append(nn.LayerNorm(dim_in))
            else : 
                raise ValueError(f'Unknown normalization type : {norm_type}')
        
                # linear pass with chosen activation function
            layers.append(nn.Linear(dim_in, dim_hidden))

            if hidden_activation_type == "Swish":
                layers.append(Swish())
            elif hidden_activation_type == "GELU":
                layers.append(nn.GELU())
            elif hidden_activation_type == "ReLU":
                layers.append(nn.ReLU())
            else :
                raise ValueError(f'Unknown activation function for hidden layers : {hidden_activation_type}')

            dim_in = dim_hidden             # update dim_in

            # output layer
        layers.append(nn.Linear(dim_in, self.dim_output))  
        if output_activation_type == "Tanh":
            layers.append(nn.Tanh())        # should be either nn.Tanh() or nn.Sigmoid() for bounding the output  
        elif output_activation_type == "Sigmoid":
            layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)   # stack all layers - output of this has dim = (batch_size, dim_output)

        # info for the output layer
        self.lower_bounds = torch.Tensor(lower_bounds)      # lower bounds for the control variable (not including the control for martingale)
        self.upper_bounds = torch.Tensor(upper_bounds)      # upper bounds for the control variable (not including the control for martingale)
        self.martingale_control_limit = abs(martingale_control_limit)  # limit for the control for martingale
        self.aug_lower_bounds = torch.concat([self.lower_bounds, 
                                              torch.full((dim_sto, ), fill_value = -1 * self.martingale_control_limit, device = device)],
                                            dim = 0)        # lower bounds for the augmented control variable   - dim = (dim_control + dim_sto, )
        self.aug_upper_bounds = torch.concat([self.upper_bounds,
                                              torch.full((dim_sto, ), fill_value = self.martingale_control_limit, device = device)],
                                            dim = 0)        # upper bounds for the augmented control variable   - dim = (dim_control + dim_sto, )
        self.output_activation_type = output_activation_type

    def forward(self,
            t: torch.Tensor,            # time variable     - dim = (sample_size, 1) or (sample_size, ) 
            X: torch.Tensor,            # state variable    - dim = (sample_size, dim_state)
            P: torch.Tensor             # martingale        - dim = (sample_size, 1) of (sample_size, )
        ) -> torch.Tensor:     # augmented control a(t,X,P) - dim = (sample_size, dim_control + dim_sto)
        
        # device
        device = X.device   
        t = t.to(device)
        P = P.to(device)

        # dimension check
        sample_size = X.shape[0]
        assert t.shape[0] == sample_size, f'Dimension Mismatch : batch size for t = {t.shape[0]} does NOT match the batch size for X = {sample_size}.'
        assert P.shape[0] == sample_size, f'Dimension Mismatch : batch size for P = {P.shape[0]} does NOT match the batch size for X = {sample_size}'
        assert X.shape[1] == self.dim_state, f'Dimension Mismatch : 2nd dimension for X = {X.shape[1]} does NOT match the expected self.dim_state = {self.dim_state}.'
        P = P.reshape(([sample_size, 1])).to(device) 

        # encode time
        t_encoded = self.time_encoder(t)    # encoding time (sinudoial) - dim = (sample_size, 2*num_time_freqs)
        t_encoded = t_encoded.to(device)  

        # pass through the hidden layers
        t_X_P = torch.cat([t_encoded, X, P], dim=1).to(device)          # dim = (sample_size, 2 * num_time_freqs +  dim_state + 1)
        out = self.mlp(t_X_P).to(device)        # all values between (0,1) if sigmoid activation or (-1, 1) if tanh activation 
        assert out.shape == torch.Size([sample_size, self.dim_output]), f'Dimension Mismatch : out has shape = {out.size()} while expecting (batch_size, dim_control) = ({sample_size, self.dim_output}).'
        
        # scale to the desired range
        
        if self.output_activation_type == "Tanh" :
            out = self.aug_lower_bounds.to(device) + (self.aug_upper_bounds.to(device) - self.aug_lower_bounds.to(device)) * (out + 1)/2
        elif self.output_activation_type == "Sigmoid" :
            out = self.aug_lower_bounds.to(device) + (self.aug_upper_bounds.to(device) - self.aug_lower_bounds.to(device)) * out
        else :
            raise ValueError(f'Unknown output activation function {self.output_activation}')
        out = out.to(device)

        u_out = out[:, :self.dim_control].to(device)        # control variable      - dim = (sample_size, dim_control)
        a_out = out[:, self.dim_control:].to(device)        # martingale control    - dim = (sample_size, dim_sto) 
    
        return u_out, a_out
    
# ---------------------------------------------------------------------------------------------------------
# Neural Network for estimating the value function on the entire domain
# ---------------------------------------------------------------------------------------------------------

class Value_Function_Net(nn.Module):
    '''
    Note that the value function takes 3 arguments : V(t, X, M)
        We use the same architecture as for w_NN and Vb_NN, but the dimension is different.
        Recall : V_NN aims to estimate the expected value of the utility function F (to be maximized)
            over the ENTIRE viable domain.
    '''
    def __init__(self,
            dim_state : int,                                                        # dimension of the state variable
            dim_hidden : int,                                                       # number of neurons in each hidden layer
            n_hidden_layers : int,                                                  # number of hidden layers in the network
            num_time_freqs : int = 5,                                               # number of time frequencies for the time encoder
            hidden_activation_type : Literal["Swish", "ReLU", "GELU"] = "Swish",    # type of activation function for hidden layers
            norm_type : Literal['batch', 'identity', 'layer'] = 'batch'             # type of normalization to be used 
                ):
        # initiation
        super(Value_Function_Net, self).__init__()
        self.dim_state = dim_state
        
        # enlarge the input dimension to fit the time encoder
        self.time_encoder = Sinusoidal_Time_Encoder(num_frequencies = num_time_freqs)
        dim_input = 2 * num_time_freqs + dim_state + 1      # plus 1 for the martingale

        # layers 
        self.mlp = Residual_MLP(dim_input = dim_input,
                                dim_output = 1, 
                                dim_hidden = dim_hidden,
                                hidden_activation_type = hidden_activation_type,
                                n_hidden_layers = n_hidden_layers,
                                norm_type = norm_type)
        
    def forward(self, 
            t : torch.Tensor,       # dim = (sample_size, 1) or (sample_size, )
            X : torch.Tensor,       # dim = (sample_size, dim_state)
            P : torch.Tensor        # dim = (sample_size, 1) or (sample_size, )
    ) -> torch.Tensor : # V(t,X,P)  - dim = (sample_size, 1)

        device = X.device
        t = t.to(device)
        P = P.to(device)

        assert t.shape[0] == X.shape[0], f'Dimension Mismatch : batch size for t = {t.shape[0]} does NOT match the batch size for X = {X.shape[0]}.'
        assert P.shape[0] == X.shape[0], f'Dimension Mismatch : batch size for P = {P.shape[0]} does NOT match the batch size for X = {X.shape[0]}.'
        assert X.shape[1] == self.dim_state, f'Dimension Mismatch : 2nd dimension for X = {X.shape[1]} does NOT match the expected self.dim_state = {self.dim_state}.'

        t_encoded = self.time_encoder(t)    # encoding time (sinudoial) - dim = (sample_size, 2 * num_time_freqs)
        t_encoded = t_encoded.to(device)  

        if P.dim() == 1 : 
            P = P.unsqueeze(-1).to(device)  # in case dim(t) = (n_samples, ), then enlarge it to (n_samples, 1)

        t_X_P = torch.cat([t_encoded, X, P], 
                    dim = 1).to(device)     # dim = (sample_size, 2 * num_time_freqs + dim_state + 1)
        out = self.mlp(t_X_P).to(device)

        assert out.shape == torch.Size([X.shape[0], 1]), f'Dimension Mismatch : out has shape = {out.size()} while expecting (batch_size, 1) = ({X.shape[0], 1}).'

        return out                          # dim = (sample_size, 1)

