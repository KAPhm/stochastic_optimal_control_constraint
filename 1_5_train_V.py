'''
This script is to train the neural network which estimates the value function on the entire viable domain.
'''

# ------------------------
# Packages
# ------------------------

import os, json, shutil, time, copy, math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Union, Literal, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import vmap

import importlib
pm = importlib.import_module('0_portfolio_model', package=None)
custom_nn = importlib.import_module('0_neural_networks', package=None)
utils = importlib.import_module('0_utils', package=None)

# ------------------------
# Support funcions and Loss functions
# ------------------------
'''
For the following functions, 
    model : the current neural network (V_NN) to be trained
        _ this model takes as arguments t of dim = (sample_size, 1), X of dim = (sample_size, dim_state), and P of dim = (sample_size, 1)
                and returns as output V(t, X, P) of dim = (sample_size, 1)
'''

def LossFunction_PDE_residual(model,    # the neural network to evaluate
                              
        t_hist_inner : torch.Tensor,    # trajectories of t where t < T             - dim = (sample_size, 1)     
        X_hist_inner : torch.Tensor,    # trajectories of X_t where t < T           - dim = (sample_size, dim_state)
        P_hist_inner : torch.Tensor,    # trajectories of P_t where t < T           - dim = (sample_size, 1)

        u_hist : torch.Tensor,          # trajectories of control u_t for state X_t - dim = (sample_size, dim_control) 
        a_hist : torch.Tensor,          # trajectories of control a_t for martingale -dim = (sample_size, dim_sto) 

        device : torch.device,          # device 
        param_dict : dict,              # dictionary for all parameters pertaining to the portfolio model
    
        constraint_penalty_weight : float = 1.0,                # weight for constraint penalty (from training u_NN and w_NN)

        id_keep_tensor : bool = False   # indicator for whether tensor are to be retained 
    ):
    '''
    Output : 
        PDE loss = avg of squared Dynkin operator HV
        if id_extra_info, also returns the tensor of HV for each sample point
    '''

    pm.send_to_device([t_hist_inner, X_hist_inner, P_hist_inner, u_hist, a_hist], device)
    model.to(device)

    if id_keep_tensor : 
        tensor_dict = {}

    # elements of computation
        # gradients and hessians
    X_P_hist_inner = torch.cat([X_hist_inner, P_hist_inner],
                        dim = 1).to(device)                     # dim = (sample_size, dim_state + 1)
    dV_t, dV_XP, dV_2_XP = utils.compute_deriv_aug(NN = model, device = device,
                        t = t_hist_inner, X_P = X_P_hist_inner)        
    dV_t = dV_t.squeeze().to(device)                            # dim = (sample_size, )
    dV_X = dV_XP[:, :-1].to(device)                             # dim = (sample_size, dim_state)
    dV_P = dV_XP[:, -1].to(device)                              # dim = (sample_size, )
    
        # drift term
    drift = pm.compute_drift(X = X_hist_inner, u = u_hist, param_dict = param_dict, device = device)
    drift_T = drift.unsqueeze(1).to(device)                     # dim = (sample_size, 1, dim_state)
    drift_term = torch.bmm(drift_T, dV_X.unsqueeze(2)).squeeze().to(device) # dim = (sample_size, )

        # diffusion term
    vols = pm.compute_vol(X = X_hist_inner, param_dict = param_dict, 
                        device = device)                        # dim = (sample_size, dim_state, dim_sto)
    a_hist_extended = a_hist.unsqueeze(1).to(device)            # dim = (sample_size, 1, dim_sto)
    aug_vols = torch.cat([vols, a_hist_extended],
                        dim = 1).to(device)                     # dim = (sample_size, dim_state + 1, dim_sto)
    aug_variances = torch.bmm(aug_vols, torch.transpose(aug_vols, 1, 2)).to(device)
    diffusion = torch.bmm(aug_variances, dV_2_XP).to(device)    # dim = (sample_size, dim_state + 1, dim_state + 1)
    diffsion_term = 0.5 * torch.vmap(torch.trace)(diffusion).to(device)                     # dim = (sample_size, )

        # penalty term
            # constraint penalty for boundary (from training u_NN and w_NN)
    _, _, _, penalties = pm.boundary_penalty(X = X_hist_inner, param_dict = param_dict, device = device, id_keep_tensor = True)
    constraint_penalties = penalties.sum(dim = 1).to(device)                                # dim = (sample_size, )

    if id_keep_tensor :
        tensor_dict.update({"penalties" : copy.deepcopy(constraint_penalties).to(device)})

    constraint_penalties = constraint_penalty_weight * constraint_penalties                 
    penalty_term = - torch.mul(dV_P, constraint_penalties).to(device)                       # dim = (sample_size, )

    # HV operator 
    dynkins_without_penalty = dV_t + drift_term + diffsion_term 
    dynkins_without_penalty = dynkins_without_penalty.to(device)

    error = dynkins_without_penalty + penalty_term              # dim = (sample_size, )
    error = error.to(device)

    loss = error ** 2
    loss = loss.to(device)

    if id_keep_tensor :
        tensor_dict.update({"dynkins_without_penalty" : copy.deepcopy(dynkins_without_penalty).to(device),
                            "error": copy.deepcopy(error)})
        return loss.mean().to(device), tensor_dict
    else :
        return loss.mean().to(device), None

def LossFunction_at_terminal(model,     # the neural network to evaluate
        X_terminal : torch.Tensor,      # state variable X_T at terminal time T         - dim = (sample_size, dim_state)
        P_terminal : torch.Tensor,      # martingale variable P_T at terminal time T    - dim = (sample_size, 1)
        device : torch.device,
        param_dict : dict,
        T_tensor : Optional[torch.Tensor] = None,   # time variable at T (as a tensor)  - dim = (sample_size, 1)
        T_value : Union[None, int, float] = None,   # value of T, only needed if T_tensor is not provided
        ):
    '''
    Output : 
        MSE loss between the predictions V(T, X_T, P_T) and the targets F(X_T) which is terminal utility
    '''
    
    sample_size = X_terminal.shape[0]

    if T_tensor is None :
        T_tensor = torch.full((sample_size, 1), fill_value = T_value).to(device)        # dim = (sample_size, 1)
    
    pm.send_to_device([T_tensor, X_terminal, P_terminal], device)
    model.to(device)

    predicted = model(T_tensor, X_terminal, P_terminal)     # dim = (sample_size, 1)
    _, target = pm.terminal_utility(X = X_terminal, param_dict = param_dict, 
                        id_keep_tensor = True)              # dim = (sample_size, 1)
    
    loss = nn.MSELoss(reduction = "mean")(predicted, target)
    return loss.to(device)              # tensor of size([])

def LossFunction_on_boundary(model,     # the neural network to evaluate     
        boundary_value_function_model,  # trained neural network (Vb_NN) to estimate the value function on the boundary
        t_hist : torch.Tensor,          # trajectories of t where t < T         - dim = (sample_size, 1)
        X_hist : torch.Tensor,          # trajectories of X_t where t < T       - dim = (sample_size, dim_state)
        w_hist : torch.Tensor,          # domain boundary w(t,X_t) where t < T  - dim = (sample_size, 1)
        device : torch.device
        ):
    '''
    Output : 
        MSE loss between the predictions V(t, X_t, w(t, X_t)) and the target Vb(t, X_t)
            which is the value function at the domain boundary
    '''

    model.to(device)
    boundary_value_function_model.to(device)
    pm.send_to_device([t_hist, X_hist, w_hist], device)

    predicted = model(t_hist, X_hist, w_hist)                   # dim = (sample_size, 1)
    target = boundary_value_function_model(t_hist, X_hist)      # dim = (sample_size, 1)

    loss = nn.MSELoss(reduction = "mean")(predicted, target)
    return loss.to(device)

def generate_training_data(
        aug_control_NN,                 # trained neural network used to estimate the augmented optimal control
        domain_boundary_NN,             # trained neural network used to estimate the domain boundary

        param_dict : dict,
        sampling_dict : dict, 
        constraint_sample_range : float,

        device : torch.device, 
        seed_initial_state : int, 
        seed_brownian : int, 

        sample_size : Optional[int] = 1, 
        float_type = torch.float32
    ):      
    
    aug_control_NN.to(device)
    domain_boundary_NN.to(device)
    n_period = param_dict["n_period"]
    dim_state = param_dict["dim_state"]

    t_hist, X_hist, P_hist, u_hist, a_hist = pm.simulate_aug_trajectories(
                                                    aug_control_NN = aug_control_NN, domain_boundary_NN = domain_boundary_NN,
                                                    param_dict = param_dict, sampling_dict = sampling_dict,
                                                    constraint_sample_range = constraint_sample_range, device = device,
                                                    seed_initial_state = seed_initial_state, seed_brownian = seed_brownian,
                                                    sample_size = sample_size, float_type = float_type
                                                    )
    t_before_terminal = t_hist[:, :-1, :].reshape((sample_size * n_period, 1)).to(device)
    X_before_terminal = X_hist[:, :-1, :].reshape((sample_size * n_period, dim_state)).to(device)
    P_before_terminal = P_hist[:, :-1, :].reshape((sample_size * n_period, 1)).to(device)
    u_before_terminal = u_hist.reshape((sample_size * n_period, param_dict["dim_control"])).to(device)
    a_before_terminal = a_hist.reshape((sample_size * n_period, param_dict["dim_sto"])).to(device)

        # data for the interior of the domain boundary
    w_before_terminal = domain_boundary_NN(t_before_terminal, X_before_terminal)    # dim = (sample_size * n_period, 1)
    distance = P_before_terminal - w_before_terminal  
    rebounce_distance = torch.where(distance < 0, -2 * distance, 0.0).to(device)
    P_inner = P_before_terminal + rebounce_distance     # if martingale goes outside of the viable domain, reflect it back into the interior
    interior_data_dict = {"t": t_before_terminal, "X": X_before_terminal, 
                          "P": P_inner, "w": w_before_terminal,
                          "u": u_before_terminal, "a": a_before_terminal}

        # data for the terminal 
    t_terminal = t_hist[:, -1, :].squeeze().to(device)      # dim = (sample_size, 1)
    X_terminal = X_hist[:, -1, :].squeeze().to(device)      # dim = (sample_size, dim_state)
    P_terminal = P_hist[:, -1, :].squeeze().to(device)      # dim = (sample_size, 1)
    terminal_data_dict = {"t": t_terminal, "X": X_terminal, "P": P_terminal}

    return interior_data_dict, terminal_data_dict
    
if __name__ == "__main__":
# ------------------------
# Parameters + Configuration
# ------------------------

    J = os.path.join    # alias for frequently used function

    # create parser
    parser = ArgumentParser(description='Train : Value Function', formatter_class = ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--portfolio', type=str, required=True, help='path to the config (.json) file for all portfolio model parameters, including the sampling space.')
    parser.add_argument('-w', '--config_w', type=str, required=True, help='path to the result folder of the trained domain boundary (w_NN).')
    parser.add_argument('-Vb', '--config_Vb', type=str, required=True, help='path to the result folder of the trained value function at the domain boundary (Vb_NN).')
    parser.add_argument('-a', '--config_a', type=str, required=True, help='path to the result folder of the trained optimal control on viable domain (a_NN).')
    parser.add_argument('-V', '--config_V', type=str, required=True, help='path to the config (.json) file to train the value function on the boundary (V_NN).')

    parser.add_argument('-o', '--result_dir', type=str, default='./results/5_Vnn', help='directory to store results.')
    parser.add_argument('-s', '--seed', type=int, default=1851794, help='seed for the random number generator.')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='indicator for what type of information to include in the log.', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
   
    args = parser.parse_args()

        # create the result directory and a logger
    result_dir = utils.create_dir(basedir = args.result_dir, dirname = 'train_V', suffix = None)
    checkpoint_dir = utils.create_dir(basedir = result_dir, dirname = 'checkpoints', suffix = '')
    logger = utils.setup_logging(result_dir, log_level = args.log_level, fname = 'train_V.log')

        # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # read parameters 
    param_dict, sampling_dict, scaling_dict = pm.read_model_param(args.portfolio, device = device)

        # seed
    torch.manual_seed(args.seed)

        # load trained w_NN for the domain boundary
    w_nn_dir = args.config_w                                    
    config_w = json.load(open(J(w_nn_dir, 'config_w.json'), 'r'))   
    w_dir_list = [J(w_nn_dir, 'checkpoints', f) for f in os.listdir(J(w_nn_dir, 'checkpoints'))]
    w_nn_checkpoint_path = max(w_dir_list, key=os.path.getmtime)

    w_NN = custom_nn.Domain_Boundary_Net(
                            dim_state = param_dict["dim_state"],
                            dim_hidden = config_w["n_neuron"],
                            n_hidden_layers = config_w["n_hidden_layers"],
                            hidden_activation_type = config_w["hidden_activation_type"],
                            norm_type = config_w["norm_type"],
                            num_time_freqs = config_w["n_time_freqs"]
                            )
    w_NN.to(device)
    w_NN.to(torch.float32)
    w_NN.load_state_dict(torch.load(w_nn_checkpoint_path, map_location=device)["model_state_dict"])
    w_NN.eval()

        # load trained Vb_NN for the value function on the domain boundary
    Vb_nn_dir = args.config_Vb
    config_Vb = json.load(open(J(Vb_nn_dir, 'config_Vb.json'), 'r'))
    Vb_dir_list = [J(Vb_nn_dir, 'checkpoints', f) for f in os.listdir(J(Vb_nn_dir, 'checkpoints'))]
    Vb_nn_checkpoint_path = max(Vb_dir_list, key=os.path.getmtime)

    Vb_NN = custom_nn.Boundary_Value_Function_Net(
                            dim_state = param_dict["dim_state"],
                            dim_hidden = config_Vb["n_neuron"],
                            n_hidden_layers = config_Vb["n_hidden_layers"],
                            hidden_activation_type = config_Vb["hidden_activation_type"],
                            norm_type = config_Vb["norm_type"],
                            num_time_freqs = config_Vb["n_time_freqs"]
                        )
    Vb_NN.to(device)
    Vb_NN.to(torch.float32)
    Vb_NN.load_state_dict(torch.load(Vb_nn_checkpoint_path, map_location=device)["model_state_dict"])
    Vb_NN.eval()

        # load trained a_NN for the augmented optimal control
    a_nn_dir = args.config_a
    config_a = json.load(open(J(a_nn_dir, 'config_a.json'), 'r'))
    a_dir_list = [J(a_nn_dir, 'checkpoints', f) for f in os.listdir(J(a_nn_dir, 'checkpoints'))]
    a_nn_checkpoint_path = max(a_dir_list, key=os.path.getmtime)

    a_NN = custom_nn.Optimal_Control_Net(dim_state = param_dict["dim_state"],
                            dim_control = param_dict["dim_control"], dim_sto = param_dict["dim_sto"],
                            dim_hidden = config_a["n_neuron"], n_hidden_layers = config_a["n_hidden_layers"],
                            num_time_freqs = config_a["n_time_freqs"], norm_type = config_a["norm_type"],
                            hidden_activation_type = config_a["hidden_activation_type"], 
                            output_activation_type = config_a["output_activation_type"],
                            lower_bounds = param_dict["control_lower_bound"], upper_bounds = param_dict["control_upper_bound"],
                            martingale_control_limit = config_a["martingale_control_limit"],
                            device = device
                            )
    a_NN.to(device)
    a_NN.to(torch.float32)
    a_NN.load_state_dict(torch.load(a_nn_checkpoint_path, map_location=device)["model_state_dict"])
    a_NN.eval()

        # load config file for training
    config_V = json.load(open(args.config_V, 'r'))
    n_batches = config_V["n_samples"] // config_V["batch_size"]

        # save all config files for replication, if needed
    json.dump(json.load(open(args.portfolio, "r")), open(f'{result_dir}/config_portfolio.json', 'w'), indent=4)
    json.dump(config_w, open(f'{result_dir}/config_u.json', 'w'), indent=4)
    json.dump(config_Vb, open(f'{result_dir}/config_Vb.json', 'w'), indent=4)
    json.dump(config_a, open(f'{result_dir}/config_a.json', 'w'), indent=4)
    json.dump(config_V, open(f'{result_dir}/config_V.json', 'w'), indent=4)

    args_dict = vars(args)
    args_dict["config_w"] = os.path.abspath(args.config_w)
    args_dict["config_Vb"] = os.path.abspath(args.config_Vb)
    args_dict["config_a"] = os.path.abspath(args.config_a)
    args_dict["result_dir"] = os.path.abspath(args.result_dir)
    json.dump(args_dict, open(f'{result_dir}/args_V.json', 'w'), indent=4)

    script_name = os.path.basename(__file__)    # save the training script (keeping track of version)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(result_dir, script_name))
    
# ------------------------
# Training
# ------------------------
    
    # Pre-training
    logger.info(f'****** Pre-training ******')

        # construct the model by loading hyperparameters
    model = custom_nn.Value_Function_Net(dim_state = param_dict["dim_state"],
                            dim_hidden = config_V["n_neuron"],
                            n_hidden_layers = config_V["n_hidden_layers"],
                            num_time_freqs = config_V["n_time_freqs"],
                            hidden_activation_type = config_V["hidden_activation_type"],
                            norm_type = config_V["norm_type"])
    model.to(device)
    model.to(torch.float32)

        # optimizer
    optimizer = optim.Adam(model.parameters(), lr = config_V["lr"], weight_decay = config_V["weight_decay"])
    scheduler_step = optim.lr_scheduler.MultiStepLR(optimizer, milestones = config_V["milestones"], gamma = config_V["gamma_step"])
    scheduler_expo = optim.lr_scheduler.ExponentialLR(optimizer, gamma = config_V["gamma_expo"])

        # initiation
            # variables to keep track of losses
    best_epoch = 0
    best_epoch_loss = None

    losses = {
        "PDE_residuals": [],        # mean squared Dynkin operator HV for (t, X_t, P_T) when t < T with (u_t, a_t) estimated by a_NN
        "boundary_losses": [],      # MSE loss between the target Vb(t, X_t) and the predicted V(t, X_t, w(t, X_t)) for t < T
        "terminal_losses": [],      # MSE loss between the target F(X_T) and the predictedd V(T, X_T, P_T) 
        "total_losses": [],         # total_loss = PDE_residual + boundary_loss + terminal_loss (without any weight for PDE_residual)
        "val_total_losses": []      # validation total loss
        }
    
            # load previously saved checkpoints if it's continued training
    if config_V["checkpoint_path"] is None :
        logger.info(f'New training      - no checkpoint loaded, starting from scratch.')
        start_epoch = 0
    else :
        logger.info(f'Continued training - loading checkpoints from {config_V["checkpoint_path"]}.')
        start_epoch, losses = utils.load_checkpoint(config_V["checkpoint_path"], model, optimizer, scheduler_step, scheduler_expo)
        best_epoch = start_epoch 

            # create a validation sample if the option is activated
    if config_V["id_validate_sample"]:
        val_seed = args.seed + 1    # fix a seed for validation sample
        val_before_terminal_dict, val_terminal_dict = generate_training_data(device = device, 
                                                            aug_control_NN = a_NN, domain_boundary_NN = w_NN,
                                                            param_dict = param_dict, sampling_dict = sampling_dict,
                                                            constraint_sample_range = config_a["constraint_sample_range"],
                                                            seed_initial_state = val_seed, seed_brownian = val_seed,
                                                            sample_size = config_V["batch_size"]
                                                            ) 
        
        logger.debug(f'Validation sample : batch_size = {config_V["batch_size"]}')
        logger.debug(f'     - before terminal : t.shape = {val_before_terminal_dict["t"].shape}, X.shape = {val_before_terminal_dict["X"].shape}, P.shape = {val_before_terminal_dict["P"].shape}, w.shape = {val_before_terminal_dict["w"].shape}')
        logger.debug(f'                         u.shape = {val_before_terminal_dict["u"].shape}, a.shape = {val_before_terminal_dict["a"].shape}')
        logger.debug(f'     - at terminal     : t.shape = {val_terminal_dict["t"].shape}, X.shape = {val_terminal_dict["X"].shape}, P.shape = {val_terminal_dict["P"].shape}')
        
            # initate weights for PDE residual losses
    if start_epoch == 0 :                               # if new training
        weight_PDE_residual = config_V["lambda_start"]
    elif start_epoch > config_V["lambda_end_epoch"] :   # continued training + lambda is no longer being updated
        weight_PDE_residual = config_V["lambda_end"]
    else : 
        last_epoch_update = math.floor(start_epoch/config_V["lambda_update_interval"]) * config_V["lambda_update_interval"]
        
        weight_PDE_residual = config_V["lambda_start"] + (config_V["lambda_end"] - config_V["lambda_start"])*(last_epoch_update / config_V["lambda_end_epoch"])

        # log before training
    logger.info(f'Device : {device} and Seed : {args.seed}')
    logger.info('Hyper-parameters :')
    logger.info(f'  > n_asset = {param_dict["n_asset"]}, n_period = {param_dict["n_period"]}')
    logger.info(f'  > utility function coeff (risk aversion coeff for exponential utility function) = {param_dict['utility_coeff']}')
    logger.info(f'  > buffer range for initial constraint limit = {config_a["constraint_sample_range"]}, limit for martingale control = {config_a["martingale_control_limit"]}')
    
    
    logger.info(f'  > n_samples = {config_V["n_samples"]}, batch_size = {config_V["batch_size"]}, n_batches = {n_batches}, hidden_activation_type = {config_V["hidden_activation_type"]}')
    logger.info(f'    n_hidden_layers = {config_V["n_hidden_layers"]}, n_neuron = {config_V["n_neuron"]}, n_time_freqs = {config_V["n_time_freqs"]}, norm_type = {config_V["norm_type"]},')
    
    logger.info(f'  > (initial) lr = {config_V["lr"]}, weight_decay = {config_V["weight_decay"]}')
    logger.info(f'  > lambda_start (initial weight for PDE residual loss) = {config_V["lambda_start"]}, lambda_end (final weight for PDE residual loss) = {config_V["lambda_end"]}, ')
    logger.info(f'  > lambda_end_epoch (last epoch at which the PDE residual weight will get updated) = {config_V["lambda_end_epoch"]}, lambda_update_interval (number of epoch between two updates for the PDE residual weight)= {config_V["lambda_update_interval"]}')
    logger.info(f'  > current lambda = {weight_PDE_residual}')
    logger.info(f'  > gamma_expo (for exponential scheduler) = {config_V["gamma_expo"]}, gamma_step (for step scheduler) = {config_V["gamma_step"]} at milestones = {config_V["milestones"]}.')
    
    logger.info(f'  > resampling initial states at each epoch ? {config_V["id_resample_initial_state"]}')
    logger.info(f'  > resampling brownians at each epoch ? {config_V["id_resample_brownian"]}')
    logger.info(f'  > using a separate validation sample ? {config_V["id_validate_sample"]}')
    logger.info('****** Begin training ******')

        # start timer
    start = time.time() # keep track of time

        # generate batch seeds to maintain coherence between epochs
    batch_seeds = torch.multinomial(torch.ones(100000), n_batches, replacement = False).to(int)

        # training loop
    for epoch in range(start_epoch, config_V["n_epoch"] + start_epoch):
        
        model.train()   # set model to training mode
        logger.info(f'Epoch {epoch + 1}/{config_V["n_epoch"] + start_epoch} : learning rate = {scheduler_expo.get_last_lr()[0]} and PDE residual weight (lambda) = {weight_PDE_residual}')

        # create a loss dictionary to keep track
        epoch_total_loss = 0.0
        epoch_PDE_residual = 0.0
        epoch_terminal_loss = 0.0
        epoch_boundary_loss = 0.0

        # shuffle the seed
        idx = torch.randperm(batch_seeds.size(0))
        batch_seeds = batch_seeds[idx]  

        # loop through the batches
        for j in range(n_batches):

            # data for the batch
            logger.debug(f' + Batch {j+1}/{n_batches} : seed = {batch_seeds[j].item()}')

                # seed(s)
            temp_seed = batch_seeds[j].item()   

            if config_V["id_resample_initial_state"] : 
                batch_seed_initial_state = torch.seed()
                logger.debug(f'     - seed for initial states : {batch_seed_initial_state}')
            else :
                batch_seed_initial_state = temp_seed

            if config_w["id_resample_brownian"]:
                batch_seed_brownian = torch.seed()
                logger.debug(f'     - seed for brownians : {batch_seed_brownian}')
            else :
                batch_seed_brownian = temp_seed

                # generate data
            batch_before_terminal_dict, batch_terminal_dict = generate_training_data(device = device,
                                            aug_control_NN = a_NN, domain_boundary_NN = w_NN,
                                            param_dict = param_dict, sampling_dict = sampling_dict,
                                            constraint_sample_range = config_a["constraint_sample_range"],
                                            seed_initial_state = batch_seed_initial_state, 
                                            seed_brownian = batch_seed_brownian,
                                            sample_size = config_V["batch_size"]
                                                                )
            # reset the optimizer 
            optimizer.zero_grad()

            # loss at terminal
            batch_terminal_loss = LossFunction_at_terminal(model = model,  
                                            T_tensor = batch_terminal_dict["t"],
                                            X_terminal = batch_terminal_dict["X"],
                                            P_terminal = batch_terminal_dict["P"],
                                            device = device, param_dict = param_dict)
            
            # loss on the boundary
            batch_boundary_loss = LossFunction_on_boundary(model = model,
                                            boundary_value_function_model = Vb_NN,
                                            t_hist = batch_before_terminal_dict["t"],
                                            X_hist = batch_before_terminal_dict["X"],
                                            w_hist = batch_before_terminal_dict["w"],
                                            device = device)
            
            # loss in the interior
            if args.log_level == "DEBUG":   
                batch_PDE_residual, batch_tensor_dict = LossFunction_PDE_residual(model = model, 
                                            t_hist_inner = batch_before_terminal_dict["t"],
                                            X_hist_inner = batch_before_terminal_dict["X"],
                                            P_hist_inner = batch_before_terminal_dict["P"],
                                            u_hist = batch_before_terminal_dict["u"],
                                            a_hist = batch_before_terminal_dict["a"],
                                            device = device, param_dict = param_dict,
                                            constraint_penalty_weight = 1.0, 
                                            id_keep_tensor = True)     
                if j == 0 :
                    logger.debug(f'     _ batch penalty (avg) = {batch_tensor_dict["penalty"].mean().item(): .15f}')
                    logger.debug(f'     _ batch dynkin without penalty (avg) = {batch_tensor_dict["dynkins_without_penalty"].mean().item(): .15f}')
                    logger.debug(f'     _ batch error (avg) = {batch_tensor_dict["error"].mean().item(): .15f}')

            else :
                batch_PDE_residual, _ = LossFunction_PDE_residual(model = model, 
                                            t_hist_inner = batch_before_terminal_dict["t"],
                                            X_hist_inner = batch_before_terminal_dict["X"],
                                            P_hist_inner = batch_before_terminal_dict["P"],
                                            u_hist = batch_before_terminal_dict["u"],
                                            a_hist = batch_before_terminal_dict["a"],
                                            device = device, param_dict = param_dict,
                                            constraint_penalty_weight = 1.0, 
                                            id_keep_tensor = False)     
            
            # loss to back propagate
            batch_loss = batch_PDE_residual * weight_PDE_residual + batch_terminal_loss + batch_boundary_loss

            # back propagation
            batch_loss.backward()
            optimizer.step()

            # accumulate batch losses for the record (no need for batch propagation here, just recording)
            epoch_PDE_residual  += batch_PDE_residual.item() / n_batches
            epoch_terminal_loss += batch_terminal_loss.item() / n_batches
            epoch_boundary_loss += batch_boundary_loss.item() / n_batches
            epoch_total_loss    += (batch_PDE_residual.item() + batch_terminal_loss.item() + batch_boundary_loss.item()) / n_batches

        # keep track of epoch losses
        losses["PDE_residuals"].append(epoch_PDE_residual)
        losses["boundary_losses"].append(epoch_boundary_loss)
        losses["terminal_losses"].append(epoch_terminal_loss)
        losses["total_losses"].append(epoch_total_loss)

        logger.info(f'  + Training    _ Total loss : {epoch_total_loss: .15f} = PDE residual : {epoch_PDE_residual: .15f} + Boundary loss : {epoch_boundary_loss: .15f} + Terminal loss : {epoch_terminal_loss: .15f}')
            
        # update the PDE weight
        if (epoch + 1 + start_epoch <= config_V["lambda_end_epoch"]) and ((epoch + 1 + start_epoch) % config_V["lambda_update_interval"] == 0):
            weight_PDE_residual = weight_PDE_residual + (config_V["lambda_end"] - config_V["lambda_start"])*(config_V["lambda_update_interval"]/config_V["lambda_end_epoch"])

        # update learning rate
        scheduler_step.step()
        scheduler_expo.step()

        # validation 
        if config_V["id_validate_sample"] :     # if validation sample is in use
        
            model.eval()    # switch to evaluation mode 
            with torch.no_grad():
                val_terminal_loss = LossFunction_at_terminal(model = model, T_tensor = val_terminal_dict["t"], 
                                        X_terminal = val_terminal_dict["X"], P_terminal = val_terminal_dict["P"], 
                                        device = device, param_dict = param_dict)
                val_boundary_loss = LossFunction_on_boundary(model = model, boundary_value_function_model = Vb_NN,
                                        t_hist = val_before_terminal_dict["t"], X_hist = val_before_terminal_dict["X"],
                                        w_hist = val_before_terminal_dict["w"], device = device)
                val_PDE_residual, _ = LossFunction_PDE_residual(model = model, device = device, 
                                        t_hist_inner = val_before_terminal_dict["t"], X_hist_inner = val_before_terminal_dict["X"],
                                        P_hist_inner = val_before_terminal_dict["P"], constraint_penalty_weight = 1.0,
                                        u_hist = val_before_terminal_dict["u"], a_hist = val_before_terminal_dict["a"],
                                        param_dict = param_dict, id_keep_tensor = False)
                
            val_epoch_total_loss = val_PDE_residual.item() + val_terminal_loss.item() + val_boundary_loss.item()
            logger.info(f'  + Validation   _ Total loss : {val_epoch_total_loss: .15f} = PDE residual : {val_PDE_residual.item(): .15f} + Boundary loss : {val_boundary_loss.item(): .15f} + Terminal loss : {val_terminal_loss.item(): .15f}')
        else :
            val_epoch_total_loss = epoch_total_loss

        losses["val_total_losses"].append(val_epoch_total_loss)

        if best_epoch_loss is None or val_epoch_total_loss < best_epoch_loss :
            best_epoch = epoch
            best_epoch_loss = val_epoch_total_loss

            utils.save_checkpoint(model = model, optimizer = optimizer, scheduler_step = scheduler_step, scheduler_expo = scheduler_expo,
                                epoch = epoch, basedir = checkpoint_dir, suffix = "best", additional_info = losses)
            
            logger.info(f'  >>>>> Found best model in epoch = {epoch + 1}')

    # Closing 
    end = time.time()
    logger.info(f'****** End training ______ Execution time = {(end - start)/60 : .2f} minutes + Best model found at epoch [{best_epoch + 1}/{start_epoch + config_V["n_epoch"]}].')
            
        
    

        
