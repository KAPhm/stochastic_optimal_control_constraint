'''
This script contains all functions concerning the portfolio modeling.
In this model, there is 1 risky asset (without any coupon), and the interest rate is constant. 
'''

import json, copy
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from torch import vmap
from math import sqrt
from typing import Union, Literal, Optional

# ------------------------
# Supporting functions
# ------------------------

    # Function to sent tensors to device
def send_to_device(
        tensor_list : list,             # list of tensors to be sent to device
        device : torch.device           # device
        ):
    for tensor in tensor_list :
        tensor = tensor.to(device)

    # Function to check dimension
def dimension_check(
        list_label : list,              # list of name of dimension variables to check
        list_expected_value : list,     # list of expected values of these dimension variables
        list_actual_value : list        # list of actual values of these dimension variables
        ):
    assert len(list_actual_value) == len(list_expected_value) and len(list_label)==len(list_actual_value), f'Check the input of the function dimension_check!'
    
    for i in range(len(list_expected_value)):
        assert list_expected_value[i] == list_actual_value[i], f'Dimension Mismatch : expectd value for {list_label[i]} to be {list_expected_value[i]} but currently it is {list_actual_value[i]}.'

    # Function to approximate the negative part of a value using a smooth function
def approx_neg_part(value : float, epsilon = 1e-40):
    return (- value + torch.sqrt(value**2 + epsilon))/2

    # Function to read config file with model parameters into a dictionary
def read_model_param(
        config_file : str,              # name of the .json file to be read
        device : torch.device           # device 
        ):
    '''
    Outputs : 
        portfolio_param : dictionary that contains model parameters
        sampling_param : dictionary that contains the sampling space information
        scaling_param : dictionary that contains the scaling factors
    '''
    # Read all into initial dictionaries
    config_dict = json.load(open(config_file, 'r'))         # load the entire config file into a dictionnary
    portfolio_param = config_dict["model_param"]            # all basic parameters for the portfolio modeling
    sampling_param = config_dict["sampling_param"]          # sampling interval for each variable in the portfolio state
    scaling_param = config_dict["scaling_param"]            # scaling factor for each variable in the portfolio state
    
    # Scaling parameters
        # complete the parameter set
    scaling_param["cash"] = scaling_param["price"] * scaling_param["quantity"]  # this ensures that cash is of the same magnitude as asset value = price * quantity
    scaling_param["liability"] = scaling_param["cash"]      # liability is on the same scale as cash

    # Portfolio parameters 
        # set fixed data type for certain parameters
    for k in ["n_asset", "n_period"]: 
        portfolio_param[k] = int(portfolio_param[k])        # dimension-like variables must be integer
    d = portfolio_param["n_asset"]
    
    for k in portfolio_param.keys():
        if k.startswith('id'):
            portfolio_param[k] = bool(portfolio_param[k])   # indicators take on boolean values

        # trim price-related parameters (based on the number of assets) + send their tensors to device (ready for computing drift)
    for k in ["mu_price", "sigma_price", "book_value", "trading_intensity_max", "trading_intensity_min"]:
        portfolio_param[k] = torch.tensor(portfolio_param[k], device=device)[0:d]           # dim = (n_asset, )
    portfolio_param["corr_price"] = torch.tensor(portfolio_param["corr_price"])[0:d, 0:d]   # dim = (n_asset, n_asset)

        # determine dimension based on model specifications
    if portfolio_param["id_update_book_value"]:     # if book value is updated in each time period, dim_state = dim(price) + dim()
        portfolio_param["dim_state"] = 3 * d + 2
    else :          # book value is not updated at each time period 
        portfolio_param["dim_state"] = 2 * d + 2
    
    if portfolio_param["id_noise"] :                # if additional brownians (noises) are added for non-stochastic variables (like quantity, cash, and liability)
        portfolio_param["dim_sto"] = portfolio_param["dim_state"]
    else :          # only stochastic variable (price) have brownians
        portfolio_param["dim_sto"] = portfolio_param["n_asset"]
    
    portfolio_param["dim_control"] = portfolio_param["n_asset"] + 1 

        # scale if needed
    if portfolio_param["id_scaling"]:
        for k in ["trading_intensity_max", "trading_intensity_min"]:
            portfolio_param[k] = torch.mul(portfolio_param[k], scaling_param["quantity"])
        
        portfolio_param["book_value"] = torch.mul(portfolio_param["book_value"], scaling_param["price"])

        # complete the parameter set
    portfolio_param["dt"] = portfolio_param["T"] / portfolio_param["n_period"]
    portfolio_param["control_lower_bound"] = torch.cat([portfolio_param["trading_intensity_min"], torch.tensor([portfolio_param["profit_sharing_min"]]).to(device)],
                                                       dim = 0).to(device)                  # dim = (n_asset + 1, ) = (dim_control, )
    portfolio_param["control_upper_bound"] = torch.cat([portfolio_param["trading_intensity_max"], torch.tensor([portfolio_param["profit_sharing_max"]]).to(device)],
                                                       dim = 0).to(device)                  # dim = (n_asset + 1, ) = (dim_control, )

    # Sampling intervals
        # trim price-related parameters (based on the number of assets) + send their tensors to device
    for k in sampling_param.keys():
        if k in ["price", "quantity"]:
            sampling_param[k] = torch.tensor(copy.deepcopy(sampling_param[k]), device=device, dtype=torch.float32, 
                                             requires_grad=False)[0:d, :]                   # dim = (n_asset, 2)
        else :  # k = "cash", "cash_ratio", "liability", or "liability_ratio"
            sampling_param[k] = torch.tensor(copy.deepcopy(sampling_param[k]), device=device, dtype=torch.float32, 
                                             requires_grad=False).unsqueeze(0)              # dim = (1, 2)

        # if activated, scale values of state-related variables
    if portfolio_param["id_scaling"]:
        for k in sampling_param.keys():
            sampling_param[k] = torch.mul(sampling_param[k], scaling_param[k])              

        return portfolio_param, sampling_param, scaling_param
    
    
# ------------------------
# Portfolio dynamic
# ------------------------
'''
In this section, we use the following notations (unless otherwise specified)
  > state variables : 
    X : (portfolio) state variable          - dim = (n_samples, dim_state) 
    _ the 2nd dimension includes the following variables in the exact order
                    price                   - dim = n_asset
                    cash                    - dim = 1
                    quantity                - dim = n_asset
                    liability               - dim = 1
        _ optional: book_value              - dim = n_asset
    _ dim_state = 2 * n_asset + 2 if the option to update book_value at each time step is not actvated
            or  = 3 * n_asset + 2 if this option is activated.

    P : Martingale representation           - dim = (n_samples, 1)

  > control variables :    
    u : controls to apply to portfolio      - dim = (n_samples, dim_control)
    _ the 2nd dimension includes the following variables in the exact order
                    trade intensity         - dim = n_asset
                    profit-sharing rate     - dim = 1
    _ dim_control = n_asset + 1

    a : stochastic increment for Martingale - dim = (n_samples, dim_sto)
    
  > portfolio parameters : 
    param_dict : dictionary that contains all parameters concerning the portfolio modelization, including
        dim_sto : dimension of the stochastic factors (represented by brownians)
        dim_state : dimension of state variable X
        dim_control : dimension of control u
        dt = T/n_period
        n_asset : number of assets in the portfolio 
        mu_price : drift param for prices   - dim = (n_asset, )
        sigma_price : vol param for prices  - dim = (n_asset, )
            _ price follows Black-Scholes model 
        dynamic_lapse_coeff : coefficient for dynamic lapse which is linear 
            in the difference between amount of profit shared and amount of interest that would be generated by interest rate
        id_update_book_value ; indicator about whether to include book value as a state variable
        book_value : if book_value is provided in param_dict, it is constant for the entire trajectory 
            _ dim = (n_asset, )

  > dynamic parameters :
    dW : brownian increment for 1 period, dim = (n_samples, dim_sto)
'''

    # Function to compute the drift 
def compute_drift_complete(
        X : torch.tensor,                           # a sample of state variables X_t   - dim = (n_samples, dim_state)
        u : torch.tensor,                           # a sample of control variables u_t - dim = (n_samples, dim_control)
        param_dict : dict,                          # dictionary of all parameters pertaining to the portfolio model
        device : torch.device,                      # device
        book_value : Optional[torch.tensor] = None  # tensor of book value (not implemeted for now)
        ):  

    send_to_device([X, u], device)

    # check 
    assert X.shape[1] == param_dict["dim_state"], f'Dimension Mismatch : expect dim_state = {param_dict["dim_state"]} but state X.shape[1] = {X.shape[1]}.'
    assert u.shape[1] == param_dict["dim_control"], f'Dimension Mismatch : expect dim_control = {param_dict["dim_control"]} but control u.shape[1] = {u.shape[1]}.'
    assert X.shape[0] == u.shape[0], f'Dimension Mismatch : number of samples in X = {X.shape[0]} does NOT match with that in u = {u.shape[0]}.'
    if not param_dict["id_update_book_value"] :
        assert book_value is not None, f'Missing Data : if book_value is not included in the state variable, then it must be provided as an argument !'
        book_value = book_value.to(device)  
    
    # dimension
    d = param_dict["n_asset"]

    # drift for exogeneous variables (price)
    drift_price = torch.mul(param_dict["mu_price"], X[:, 0:d])                  # dim = (n_samples, n_asset)

    # financial production intensity
    if param_dict["id_update_book_value"]:  # book value is included in the state variable
        latent_capital_gain = X[:, :d] - X[:, -d:]                              # dim = (n_samples, n_asset)
    else :
        latent_capital_gain = X[:, :d] - book_value
    
    if param_dict["id_approx_neg_part"]:    # if negative part is approximated using a smooth function
        sale_quantity_intensity = approx_neg_part(u[:, :d],
                            epsilon = param_dict["epsilon"]).to(device)         # dim = (n_samples, n_asset)
    else :
        sale_quantity_intensity = torch.clamp(torch.neg(u[:, :d]), min = 0).to(device)
    realized_capital_gain_intensity = torch.sum(sale_quantity_intensity * latent_capital_gain,
                                      dim = 1).unsqueeze(1).to(device)          # dim = (n_samples, 1)   

    interest_from_cash = param_dict["interest_rate"] * X[:, 2*d:2*d+1]     # dim = (n_samples, 1)
    fin_prod_intensity = realized_capital_gain_intensity + interest_from_cash   # dim = (n_samples, 1)
    fin_prod_intensity = fin_prod_intensity.to(device)

    # lapse intensity
    structural_lapse_intensity = param_dict["structural_lapse_rate"] * X[:, 2*d+1:2*d+2]   
    dynamic_lapse_intensity = param_dict["dynamic_lapse_coeff"] * (param_dict["interest_rate"]*X[:, 2*d+1:2*d+2] - u[:, -1:]*fin_prod_intensity)
    lapse_intensity = structural_lapse_intensity + dynamic_lapse_intensity      # dim = (n_samples, 1)
    lapse_intensity = lapse_intensity.to(device)

    # drift for endogeneous variables (cash, quantity, and liab)
    drift_quantity = u[:, 0:d]                                                  # dim = (n_samples, n_asset)
    drift_liab = u[:, -1:] * fin_prod_intensity - lapse_intensity               # dim = (n_samples, 1)
        
    drift_cash = -1 * torch.sum(u[:, 0:d] * X[:, 0:d],                          # cost of asset reallocation (since cash is residual for trading)
                                        dim = 1).unsqueeze(1).to(device)        # dim = (n_samples, 1)
    drift_cash = drift_cash + param_dict["interest_rate"]*X[:,2*d+1:2*d+2]      # interest from cash is integrated into cash
    drift_cash = drift_cash - lapse_intensity                                   # claim payout is withdrawn from cash

    if param_dict["id_update_book_value"] : # if book value is a part of the state variable
        print('NOT IMPLEMENTED YET !')
    else :
        drift = torch.cat([drift_price, drift_quantity, drift_cash, drift_liab], 
                            dim = 1).to(device)    # dim = (n_samples, 2*n_asset + 2) = (n_samples, dim_state)
        
    return drift, fin_prod_intensity, lapse_intensity, structural_lapse_intensity, dynamic_lapse_intensity

def compute_drift(
        X : torch.tensor, 
        u : torch.tensor, 
        param_dict : dict, 
        device : torch.device, 
        book_value : Optional[torch.tensor] = None
        ) -> torch.tensor :
    
    if book_value is None:
        book_value = param_dict["book_value"]

    drift, _, _, _, _ = compute_drift_complete(X, u, param_dict, device, book_value)
    return drift    # dim = (n_samples, dim_state)

    # Function to compute volatility
def compute_vol(
        X : torch.tensor, 
        param_dict : dict, 
        device : torch.device
        ) -> torch.tensor:

    send_to_device([X], device)
    n_samples = X.shape[0] 
    d = param_dict["n_asset"]

    # vol for exogeneous variables (price)
    vol_price = torch.mul(param_dict["mu_price"], X[:, 0:d]).to(device)     # dim = (n_samples, n_asset)

    # vol for endogeneous variables (quantity, cash, liability)
    if param_dict["id_noise"]:
        vol_endo = torch.full(size = (n_samples, param_dict["dim_state"] - param_dict["n_asset"]), 
                        fill_value = param_dict["epsilon"]).to(device)      # dim = (n_samples, dim_state - n_asset)
        vol_vect = torch.cat([vol_price, vol_endo], dim = 1).to(device)     # dim = (n_samples, dim_state)

    else :  # not adding any noise into endogeneous variables' dynamics
        vol_vect = vol_price

    vol = torch.zeros((n_samples, param_dict["dim_state"], param_dict["dim_sto"])).to(device)
    vol[:, :param_dict["dim_sto"], :param_dict["dim_sto"]] = torch.diag_embed(vol_vect).to(device)

    return vol  # dim = (n_samples, dim_state, dim_sto)

    # Function to update the spatial state
def update_spatial_state(
        X : torch.tensor,   # sample of state variable      - dim = (n_samples, dim_state)
        u : torch.tensor,   # sample of control variable    - dim = (n_samples, dim_control)
        dW : torch.tensor,  # sample of brownian increment  - dim = (n_samples, dim_sto)
        param_dict : dict,  
        device : torch.device, 
        book_value : Optional[torch.tensor] = None
        ) -> torch.tensor:  # sample of updated state       - dim = (n_samples, dim_state)
    
    shape = X.size()
    send_to_device([X, u, dW], device)

    if book_value is not None :
        book_value = book_value.to(device)

    drift = compute_drift(X, u, param_dict, device, book_value)     # dim = (n_samples, dim_state)
    vol = compute_vol(X, param_dict, device)                        # dim = (n_samples, dim_state, dim_sto)

    X = X + torch.mul(drift, param_dict["dt"]).to(device)           # add drift part = drift * dt
    X = X + torch.bmm(vol, dW.unsqueeze(2)).squeeze(2)              # add defusion part = vol * dW
               
    assert X.size() == shape, f'Dimension Mismatch : After updating, the dimension of X = {X.size()} but the original shape is {shape}.'
    
    return X    # dim = (n_samples, dim_state)

    # Function to update the temporal-spatial state
def update_state(
        t : torch.tensor, 
        X : torch.tensor, 
        u : torch.tensor, 
        dW : torch.tensor, 
        param_dict : dict, 
        device : torch.device, 
        book_value : Optional[torch.tensor] = None
        ):
    
        # check dimension
    assert X.shape[1] == param_dict["dim_state"], f'Dimension Mismatch : expect 2nd dimension of X to has dim_state = {param_dict['dim_state']}, but as of now, it has X.shape[1]={X.shape[1]}.'
    assert t.shape[1] == 1, f'Dimension Mismatch : expect 2nd dimension of t to be 1, but as of now it is {t.shape[1]}.'
    assert dW.shape[1] == param_dict["dim_sto"], f'Dimension Mismatch : expect 2nd dimension of dW to has dim_sto = {param_dict['dim_sto']}, but as of now, it has dW.shape[1]={dW.shape[1]}.'
    assert u.shape[1] == param_dict["dim_control"], f'Dimension Mismatch : expect 2nd dimension of u to has dim_control = {param_dict['dim_control']}, but as of now, it has u.shape[1] = {u.shape[1]}.'

    assert t.shape[0] == X.shape[0], f'Dimension Mismatch : expect batch dimension of t = {t.shape[0]} to match that of X = {X.shape[0]}.'
    assert dW.shape[0] == X.shape[0], f'Dimension Mismatch : expect batch dimension of dW = {dW.shape[0]} to match that of X = {X.shape[0]}.'
    assert u.shape[0] == u.shape[0], f'Dimension Mismatch : expect batch dimension of u = {u.shape[0]} to match that of X = {X.shape[0]}.'
    
        # device
    send_to_device([t, X, u, dW], device)            
    if book_value is not None :
        book_value = book_value.to(device)

        # update 
    t = t + torch.full(size=(t.shape[0], 1), fill_value = param_dict["dt"]).to(device) 
    X = update_spatial_state(X, u, dW, param_dict, device, book_value)

    return t, X      # dim(t) = (n_samples, 1) and dim(X) = (n_samples, dim_state)

    # Function to update the martingale
def update_martingale(
        P : torch.tensor,       # sample of martingale process      - dim = (n_samples, 1)
        a : torch.tensor,       # sample of control for martingale  - dim = (n_samples, dim_sto)
        dW : torch.tensor,      # sample of brownian increments     - dim = (n_samples, dim_sto)
        device : torch.device   # device
        ) -> torch.tensor:

    assert P.shape[1] == 1, f'Dimension Mismatch : expect 2nd dimension of M to be 1, but as of now it is {P.shape[1]}.'
    assert a.shape[0] == P.shape[0], f'Dimension Mismatch : expect batch dimension of M = {P.shape[0]} to match that of a = {a.shape[0]}'
    assert a.shape == dW.shape, f'Dimension Mismatch : the shape of a = {a.shape} does NOT match with that of dW = {dW.shape}'

    send_to_device([P, a, dW], device)
    P = P + torch.sum(a * dW, dim=1).unsqueeze(1).to(device)        # dim = (n_samples, 1)
    
    return P        # dim(P) = (n_samples, 1)

    # Function to update the augmented temporal-spatial state (i.e with martingale)
def update_aug_state(
        t : torch.tensor,       # sample of time variable           - dim = (n_samples, 1)
        X : torch.tensor,       # sample of state variable          - dim = (n_samples, dim_state)
        P : torch.tensor,       # sample of maringale process       - dim = (n_samples, 1)
        u : torch.tensor,       # sample of control variable        - dim = (n_samples, dim_control)
        a : torch.tensor,       # sample of control for martingale  - dim = (n_samples, dim_sto)
        dW : torch.tensor,      # sample of brownian increments     - dim = (n_samples, dim_sto)
        param_dict: dict,       # dictionary of all parameters pertaining to the portfolio model
        device : torch.device,  # device
        book_value : Optional[torch.tensor] = None  # book value - not implemented 
        ):
        # check 
    assert X.shape[0] == P.shape[0], f'Dimension Mismatch : expect batch dimension of X = {X.shape[0]} to match that of M = {P.shape[0]}.'
        
        # device
    send_to_device([t, X, P, u, a, dW], device)       
    if book_value is not None :
        book_value = book_value.to(device)

        # update
    t, X = update_state(t, X, u, dW, param_dict, device, book_value)    
    P = update_martingale(P, a, dW, device)

    return t, X, P

# ----
# Cost functions (to be accumulated throughout the time horizon)

    # Function to compute wealth (to be used multiple times)
def compute_wealth(
        X : torch.Tensor,       # a sample of state variable    - dim = (n_samples, dim_state)   
        param_dict : dict       # dictionary of parameters concerning the portfolio model
        ) -> torch.Tensor:      # a sample of wealth            - dim = (n_samples, 1)
    
    device = X.device
    d = param_dict["n_asset"]

    wealth = X[:,2*d]                                   # + cash
    wealth = wealth + torch.sum(X[:, 0:d]*X[:, d:2*d],  
                        dim = 1).to(device)             # + asset
    wealth = wealth - X[:, 2*d+1]                       # - liability
    wealth = wealth.unsqueeze(1).to(device)             
    assert wealth.shape[1] == 1, f'Dimension Mismatch : expect the 2nd dimension of wealth to be 1, but it is {wealth.shape[1]}.'
    
    return wealth   

    # Function to compute penalties at the boundary - used in training the (hedging) boundary optimal control 
def boundary_penalty(
        X : torch.Tensor,       # a sample of state variable    - dim = (n_samples, dim_state)
        param_dict : dict,      # dictionary of parameters concerning the portfolio model
        device : torch.device,  # device for all computation 
        id_keep_tensor = False  # indicator of whether the full tensors of penalties are to be kept (if not activated, only the sum of penalties over all sample points will be returned)
        ):
    
    d = param_dict["n_asset"]

    wealth = compute_wealth(X, param_dict).to(device)  # dim = (n_samples, 1)

    if param_dict["id_approx_neg_part"] :   # if negative part function is approximated with a smooth function (to avoid disappearance of gradient)
        penalty_neg_liab_tensor = approx_neg_part(X[:, 2*d+1:2*d+2],
                                    epsilon = param_dict["epsilon"]).to(device)     # dim = (n_samples, 1) - negative part of liability
        penalty_neg_cash_tensor = approx_neg_part(X[:, 2*d:2*d+1],
                                    epsilon = param_dict["epsilon"]).to(device)     # dim = (n_samples, 1) - negative part of cash
        penalty_bankrupt_tensor = approx_neg_part(wealth,
                                    epsilon = param_dict["epsilon"]).to(device)     # dim = (n_samples, 1) - negative part of wealth
    else :
        penalty_neg_liab_tensor = torch.clamp(torch.neg(X[:, 2*d+1:2*d+2]), min = 0).to(device)     
        penalty_neg_cash_tensor = torch.clamp(torch.neg(X[:, 2*d:2*d+1]), min = 0).to(device)      
        penalty_bankrupt_tensor = torch.clamp(torch.neg(wealth), min = 0).to(device)               

    penalty_neg_liab = penalty_neg_liab_tensor.sum().to(device)             # dim = ()
    penalty_neg_cash = penalty_neg_cash_tensor.sum().to(device)             # dim = () 
    penalty_bankrupt = penalty_bankrupt_tensor.sum().to(device)             # dim = () 

    if id_keep_tensor : 
        penalty_tensor = torch.cat([penalty_neg_liab_tensor,
                                 penalty_neg_cash_tensor, 
                                 penalty_bankrupt_tensor], 
                                 dim = 1).to(device)            # dim = (n_samples, 3)
        return penalty_neg_liab, penalty_neg_cash, penalty_bankrupt, penalty_tensor
    
    else :      # only return the penalty sums (no tensor returned)
        return penalty_neg_liab, penalty_neg_cash, penalty_bankrupt, None

    # Function to compute penalties in case of touching the boundary - used in training the (maximizing) inner optimal control
def breach_penalty( 
        w_NN,                       # a trained neural network to estimate the boundary of the viable domain
        t : torch.Tensor,           # a sample of time variable   - dim = (n_samples, 1)
        X : torch.Tensor,           # a sample of state variables - dim = (n_samples, dim_state)
        P : torch.Tensor,           # a sample of martingales     - dim = (n_samples, 1)   
        param_dict : dict,          # dictionary of parameters concerning the portfolio model
        device : torch.device,      # device for all computation
        id_keep_tensor = False      # indicator for whether the full tensor of penalty will be kept
        ): 
    
        # device
    w_NN.to(device)
    send_to_device([t, X, P], device)

        # compute the boundary at each point (t, X)
    w = w_NN(t, X).to(device)       # dim = (n_samples, 1)

        # compute the penalty = max{ w(t,X) - P, 0} = - min{ P - w(t,X), 0}, effectively the negative part of P - w(t,X)
    if param_dict["id_approx_neg_part"]:    # if negative part function is approximated with a smooth function (to avoid disappearance of gradient)
        penalty_breach_tensor = approx_neg_part(P - w, 
                                    epsilon=param_dict["epsilon"]).to(device)   # dim = (n_samples, 1)
    else :
        penalty_breach_tensor = torch.clamp(w - P, min=0).to(device)            # dim = (n_samples, 1)
    
    penalty_breach = penalty_breach_tensor.sum().to(device)                     # dim = ()
    
    if id_keep_tensor : 
        return penalty_breach, penalty_breach_tensor
    else :  
        return penalty_breach, None
    
# ----
# Terminal functions
    # Function to compute capital loss (for finding the hedging strategy at the boundary)
def terminal_capital_loss(
        X : torch.Tensor,               # dim = (n_samples, dim_state)
        param_dict : dict,              # should contain terminal_min and terminal_max (to keep the function bounded)
        id_keep_tensor = False          # indicator on whether to keep the entire tensor (if not, only total loss is returned)
        ):
    
    device = X.device
    final_loss = -1 * compute_wealth(X, param_dict)                     # dim = (n_samples, 1)

    terminal_loss_tensor = torch.clamp(final_loss.to(device),           # dim = (n_samples, 1)
                                    min=param_dict["terminal_min"], 
                                    max=param_dict["terminal_max"]).to(device)
    
    terminal_loss = terminal_loss_tensor.sum().to(device)               # dim = ()

    if id_keep_tensor :
        return terminal_loss, terminal_loss_tensor
    else :
        return terminal_loss, None
    
    # Function to compute utility at end of horizon (for finding the maximizing strategy in the viable domain)
def terminal_utility(
        X : torch.Tensor,               # sample of state variable      - dim = (n_samples, dim_state)          
        param_dict : dict,              # should contain utility_min and utility_max (to bound the utility function) and utility_coeff (risk aversion coefficient for exponential utility function)
        id_keep_tensor = False          # indicatior on whether to keep the entire tensor (if not, only the total utility is returned)
    ):     
    ''' power utility function F(X)= -1/a * exp(- a * X)'''

    device = X.device
    final_wealth = compute_wealth(X, param_dict)                        # dim = (n_samples, 1)
    a = param_dict["utility_coeff"]                                     # scalar - coefficient for the power utility function 

    terminal_utility_tensor = -1/a * torch.exp(-a * final_wealth)       # dim = (n_samples, 1)
    terminal_utility_tensor = torch.clamp(terminal_utility_tensor.to(device),         
                                min = param_dict["utility_min"],
                                max = param_dict["utility_max"]).to(device)
    
    terminal_utility = terminal_utility_tensor.sum().to(device)         # dim = ()

    if id_keep_tensor : 
        return terminal_utility, terminal_utility_tensor
    else :  # only the sum is returned
        return terminal_utility, None
    

# ------------------------
# Sampling functions
# ------------------------

    # Function to sample initial state (X)
def sample_initial_state(
        sample_size : int,          # number of trajectories to generate
        param_dict : dict,          # dictionary for all the parameters pertaining to the portfolio model
        sampling_dict : dict,       # dictionary of ranges to sample state variables from
        device : torch.device,      # device
        seed : Optional[int] = None # seed for sampling
        ):
    
    d = param_dict["n_asset"]

        # create upper and lower limits for the interval 
            # - note that if the values in sampling_dict need to be scaled, it's already done at the parameter reading step !
    lower = torch.cat([sampling_dict["price"][:, 0], sampling_dict["quantity"][:, 0],       # each has dim = (n_asset, )
                sampling_dict["cash_ratio"][:, 0], sampling_dict["liability_ratio"][:, 0]], # each has dim = (1, )
                                                                    dim = 0).to(device)     # dim = (2*n_asset + 1, )
    upper = torch.cat([sampling_dict["price"][:, 1], sampling_dict["quantity"][:, 1],       # each has dim = (n_asset, )
                sampling_dict["cash_ratio"][:, 1], sampling_dict["liability_ratio"][:, 1]], # each has dim = (1, )
                                                                    dim = 0).to(device)     # dim = (2*n_asset + 1, )
    assert lower.shape[0] == param_dict["n_asset"] * 2 + 2, f'Dimension Mismatch : 1st dimension of lower limit (for the sampling interval) is {lower.shape[0]} while it is expected to be param_dict["n_asset"] * 2 + 2 = {param_dict["n_asset"] * 2 + 2}.'

        # randomize
    generator = torch.Generator(device=device)
    if seed is not None :
        generator.manual_seed(seed)

    random_draw = torch.rand((sample_size, param_dict["n_asset"]*2 + 2),
                            generator = generator, device = device) # dim = (sample_size, 2*n_asset + 2)
    random_in_range = random_draw * (upper - lower) + lower         # dim = (sample_size, 2*n_asset + 2)

        # compute cash and liability based on asset
    random_asset = torch.sum(random_in_range[:, 0:d]*random_in_range[:, d:2*d], dim=1)      # (non-cash) asset = price * quantity
    random_asset = random_asset.to(device)                                                  # dim = (sample_size,)
    random_cash = random_asset * random_in_range[:, 2*d]            # cash = (non-cash) asset * cash        > dim = (sample_size,)
    random_asset = random_asset + random_cash                       # total asset = (non-cash) asset + cash > dim = (sample_size,)
    random_liability = random_in_range[:, 2*d+1] * random_asset     # liab = liab_ratio * total asset       > dim = (sample_size, )

    sample = torch.cat([random_in_range[:, 0:2*d],                  # price and quantity                    > dim = (sample_size, 2*n_asset)
                        random_cash.unsqueeze(1).to(device),        # cash                                  > dim = (sample_size, 1)
                        random_liability.unsqueeze(1).to(device)    # liability                             > dim = (sample_size, 1)
                        ], dim = 1).to(device)
    
    if param_dict["id_update_book_value"]:
        print('NOT IMPLEMENTED YET !')
    
    assert sample.shape == torch.Size([sample_size, param_dict["dim_state"]]),f'Dimension Mismatch : expect initial state sample to be of the size {[sample_size, param_dict["dim_state"]]} but it has size = {sample.size} instead.'

    return sample       # dim = (sample_size, dim_state)

    # Function to sample full set of brownians
def sample_brownian_trajectories(
        sample_size : int,                  # number of trajectories to generate
        param_dict : dict,                  # dictionary of all parameters pertaining to the portfolio model
        device : torch.device,              # device
        seed : Optional[int] = None,        # seed for sampling
        n_timestep : Optional[int] = None   # number of time steps 
        ):
    
        # set seed (otherwise automatically chosen by the machine)
    if seed is not None :
        torch.manual_seed(seed)

        # number of steps to generate 
    if n_timestep is None :
        n_timestep = param_dict["n_period"] # without precision, generate full trajectories for the entire time horizon

        # create generator
    generator = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc = torch.zeros((param_dict["dim_sto"], ), dtype = torch.float32).to(device),                        # mean for brownian
                    covariance_matrix = (param_dict["corr_price"].to(torch.float32) * sqrt(param_dict["dt"])).to(device)   # covariance matrix for brownian
                    )          
    
        # create antithetic brownian trajectories (for better convergence)
    assert sample_size > 0, f'Value Error : sample_size for sample_brownian_trajectories must be at least 1; right now, it is {sample_size}.'

    if sample_size > 1 :
        half_size = int(sample_size / 2)
        first_half = generator.sample((half_size, n_timestep)).permute(0, 2, 1).to(device)  # dim = (half_size, dim_sto, n_step)
        second_half = -1 * first_half
        brownians = torch.cat([first_half, second_half], dim = 0).to(device)                # dim = (sample_size, dim_sto, n_step)
    else :
        brownians = generator.sample((1, n_timestep)).permute(0,2,1).to(device)             

    assert brownians.shape == torch.Size([sample_size, param_dict["dim_sto"], n_timestep]), f'Dimension Mismatch : expect brownian trajectories sample to be of size {[sample_size, param_dict["dim_sto"], n_timestep]} but it has size = {brownians.size()} instead. '
    
    return brownians    # dim = (sample_size, dim_sto, n_step)

    # Function to sample state trajectories using a trained control neural network
def simulate_trajectories(
        control_NN,                     # neural network used to estimate control; when pass forward, control_NN(t,X) = u 
        param_dict : dict,              # dictionary of all parameters pertaining to the portfolio model
        sampling_dict : dict,           # dictionary of ranges to sample state variables from
        device : torch.device,          # device
        seed_initial_state : int,       # seed for sampling initial states
        seed_brownian :  int,           # seed for sampling brownians
        sample_size : Optional[int] = 1 # number of trajectories to simulate
        ):
    '''
    Outputs :
        a sample of temporal state t from t=0 to t=T                        - dim = (sample_size, n_period + 1, 1)
        a sample of spatial state X following control_NN from t=0 to t=T    - dim = (sample_size, n_period + 1, dim_state)
        a sample of controls estimated by control_NN from t=0 to t=T-1      - dim = (sample_size, n_period, dim_control)
    '''
    control_NN.to(device)   # device
    control_NN.eval()       # switch to evaluation mode
   
    with torch.no_grad():

            # sampling initial data at t = 0
        current_t = torch.full(size=(sample_size, 1), 
                               fill_value = 0.0).to(device)                 # dim = (sample_size, 1)
        current_X = sample_initial_state(sample_size = sample_size, param_dict = param_dict, 
                                sampling_dict = sampling_dict, device = device, 
                                seed = seed_initial_state)                  # dim = (sample_size, dim_state)
        brownians = sample_brownian_trajectories(sample_size = sample_size, param_dict = param_dict, 
                        device = device, seed = seed_brownian)              # dim = (sample_size, dim_sto, n_period)
        
            # initiation
        t_hist = current_t.detach().clone().unsqueeze(1).to(device)         # dim = (sample_size, 1, 1)
        X_hist = current_X.detach().clone().unsqueeze(1).to(device)         # dim = (sample_size, 1, dim_state)

        for i in range(param_dict["n_period"]):
                
            u_t = control_NN(current_t, current_X)                          # dim = (sample_size, dim_control)
            dW_t = brownians[:, :, i]                                       # dim = (sample_size, dim_sto)

            current_t, current_X = update_state(t=current_t, X=current_X, u=u_t, dW=dW_t, param_dict=param_dict, device=device)
        
            t_hist = torch.cat([t_hist, current_t.detach().clone().unsqueeze(1).to(device)], 
                                    dim=1).to(device)                       # dim = (sample_size, 1 + i, 1)
            X_hist = torch.cat([X_hist, current_X.detach().clone().unsqueeze(1).to(device)], 
                                    dim=1).to(device)                       # dim = (sample_size, 1 + i, dim_state)

            if i == 0 :
                u_hist = u_t.detach().clone().unsqueeze(1).to(device)       # dim = (sample_size, 1, dim_control)
            else :
                u_hist = torch.cat([u_hist, u_t.detach().clone().unsqueeze(1).to(device)],
                                    dim=1).to(device)                       # dim = (sample_size, i, dim_control)

        return t_hist, X_hist, u_hist

def generate_initial_constraints(
        domain_boundary_NN,                 # neural network used to estimate domain boundary; when pass forward, domain_boundary_NN(t,X) = w(t,X)
        initial_states : torch.tensor,      # sample of initial states X_0  - dim = (sample_size, dim_state)
        device : torch.device,              # device
        seed : Optional[int] = None,        # seed for sampling
        constraint_sample_range: float=1.0, # buffer zone (above the minimal threshold) to sample constraints from 
        float_type = torch.float32          # float type 
        ) -> torch.tensor:  # sample of initial constraints p >= w(0, X_0)  - dim = (sample_size, 1)

    domain_boundary_NN.to(device)
    send_to_device([initial_states], device)
    sample_size = initial_states.shape[0]
    
    if seed is not None : 
        torch.manual_seed(seed)

    initial_time = torch.full((sample_size, 1), fill_value = 0.0, device = device, dtype = float_type)
    with torch.no_grad():
        constraints_min = domain_boundary_NN(initial_time, initial_states)  # for (0,X_0,p) to be feasible, it must be that p >= w(0,X_0)
        constraints = constraints_min + torch.rand([sample_size, 1]).to(device) * constraint_sample_range

    return constraints.to(device)   # dim = (sample_size, 1)

def simulate_aug_trajectories(
        aug_control_NN,                     # neural network used to estimate augmented control; when pass forward, control_NN(t, X, P) = (u,a) 
        domain_boundary_NN,                 # neural network used to estimate domain boundary; when pass forward, domain_boundary_NN(t,X) = w(t,X)
        
        param_dict : dict,                  # dictionary of all parameters pertaining to the portfolio model
        sampling_dict : dict,               # dictionary of ranges to sample state variables from
        constraint_sample_range : float,    # buffer zone to draw constraints above the minimal threshold
        
        device : torch.device,              # device
        seed_initial_state : int,           # seed for sampling initial states (and initial constraints)
        seed_brownian : int,                # seed for sampling brownians

        sample_size : Optional[int] = 1,    # number of trajectories to simulate
        float_type = torch.float32,         # float type 
        book_value : Optional[torch.tensor] = None  # book value - not implemented
        ):
    '''
    Outputs :
        a sample of temporal state t from t=0 to t=T                                        - dim = (sample_size, n_periods + 1, 1)           
        a sample of spacial state X following aug_control_NN from t=0 to t=T                - dim = (sample_size, n_periods + 1, dim_state)   
        a sample of martingale P following aug_control_NN from t=0 to t=T                   - dim = (sample_size, n_periods + 1, 1)   

        a sample of control for state estimated by aug_control_NN from t=0 to t=T-1         - dim = (sample_size, n_periods, dim_control)   
        a sample of control for martingale estimated by aug_control_NN from t=0 to t=T-1    - dim = (sample_size, n_periods, dim_sto)   
    '''

    aug_control_NN.to(device)
    aug_control_NN.eval()

    domain_boundary_NN.to(device)
    domain_boundary_NN.eval()

    with torch.no_grad():

        # sampling initial data at t = 0
        current_t = torch.full(size=(sample_size, 1), 
                            fill_value = 0.0).to(device)                        # dim = (sample_size, 1)
        current_X = sample_initial_state(sample_size = sample_size, 
                            sampling_dict = sampling_dict, param_dict = param_dict,
                            seed = seed_initial_state, device = device)         # dim = (sample_size, dim_state)
        current_P = generate_initial_constraints(domain_boundary_NN = domain_boundary_NN,
                            initial_states = current_X.detach().clone(),
                            seed = seed_initial_state, device = device, 
                            constraint_sample_range = constraint_sample_range,
                            float_type = float_type)                            # dim = (sample_size, 1)
    
        # sampling entire brownian increment trajectories
        brownians = sample_brownian_trajectories(sample_size = sample_size,
                            param_dict = param_dict, seed = seed_brownian,
                            device = device)                                    # dim = (sample_size, dim_sto, n_period)
        
        # initation
        t_hist = current_t.detach().clone().unsqueeze(1).to(device)             # dim = (sample_size, 1, 1)
        X_hist = current_X.detach().clone().unsqueeze(1).to(device)             # dim = (sample_size, 1, dim_state)
        P_hist = current_P.detach().clone().unsqueeze(1).to(device)             # dim = (sample_size, 1, 1)

        # iteration
        for i in range(param_dict["n_period"]):

            u_t, a_t = aug_control_NN(current_t, current_X, current_P)          # dim(u_t) = (sample_size, dim_control) and dim(a_t) = (sample_size, dim_sto)
            dW_t = brownians[:, :, i].to(device)                                # dim = (sample_size, dim_sto)

            current_t, current_X, current_P = update_aug_state(t = current_t, X = current_X, M = current_P,
                                                    u = u_t, a = a_t, dW = dW_t, param_dict = param_dict, 
                                                    device = device, book_value = book_value)
            
            t_hist = torch.cat([t_hist, current_t.detach().clone().unsqueeze(1).to(device)],
                                    dim = 1).to(device)                         # dim = (sample_size, 1 + i, 1)
            X_hist = torch.cat([X_hist, current_X.detach().clone().unsqueeze(1).to(device)], 
                                    dim=1).to(device)                           # dim = (sample_size, 1 + i, dim_state)
            P_hist = torch.cat([P_hist, current_P.detach().clone().unsqueeze(1).to(device)], 
                                    dim=1).to(device)                           # dim = (sample_size, 1 + i, 1)
            
            if i == 0 :     # initiation of control history
                u_hist = u_t.detach().clone().unsqueeze(1).to(device)           # dim = (sample_size, 1, dim_control)
                a_hist = a_t.detach().clone().unsqueeze(1).to(device)           # dim = (sample_size, 1, dim_sto)
            else :
                u_hist = torch.cat([u_hist, u_t.detach().clone().unsqueeze(1).to(device)],
                                    dim=1).to(device)                           # dim = (sample_size, i, dim_control)
                a_hist = torch.cat([a_hist, a_t.detach().clone().unsqueeze(1).to(device)],
                                    dim=1).to(device)                           # dim = (sample_size, i, dim_sto)
        
        return t_hist, X_hist, P_hist, u_hist, a_hist
            
            