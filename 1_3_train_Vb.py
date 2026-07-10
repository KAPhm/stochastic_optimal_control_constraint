'''
This script is to train the neural network which estimates the value function at the boundary.
'''
# ------------------------
# Packages
# ------------------------
import os, json, shutil, time, math
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
# Supporting / Loss functions
# ------------------------
'''
For the following functions, model is the current neural network (Vb_NN) to be evaludated
    _ this model takes as arguments t of dim = (sample_size, 1) and X of dim = (sample_size, dim_state)
            and returns Vb(t, X) of dim = (sample_size, 1)
'''

def LossFunction_at_terminal(
        model,
        X_T : torch.Tensor,                         # state variable X at terminal time T           - dim = (sample_size, dim_state)
        device : torch.device,
        param_dict : dict,
        T_tensor : Optional[torch.Tensor] = None,   # time variable at T (tensor of only value T)   - dim = (sample_size, 1)
        T_value : Union[None, int, float] = None    # value of T, only needed if T_tensor is not provided
    ):
    
    sample_size = X_T.shape[0]
    
    if T_tensor is None : 
        T_tensor = torch.full((sample_size, 1), fill_value = T_value).to(device)    # dim = (sample_size, 1)

    pm.send_to_device([T_tensor, X_T], device)
    model.to(device)

    predicted = model(T_tensor, X_T)        # dim = (sample_size, 1)
    _, target = pm.terminal_utility(X = X_T, param_dict = param_dict, 
                    id_keep_tensor = True)  # dim = (sample_size, 1)
    
    loss = nn.MSELoss(reduction="mean")(predicted, target)      # average across batch dimension

    return loss.to(device)                  # tensor of size([])

def LossFunction_PDE_residual(  
        model,                              # model (for Vb) in training
        t_hist_inner : torch.Tensor,        # trajectories of t where t < T             - dim = (sample_size, 1)
        X_hist_inner : torch.Tensor,        # trajectories of X_t where t < T           - dim = (sample_size, dim_state)
        u_hist : torch.Tensor,              # trajectories of control u_t for (t, X_t)  - dim = (sample_size, dim_control)
        device : torch.device, 
        param_dict : dict, 
        id_extra_info = False               # indicator of whether extra information will be returned
    ):

    pm.send_to_device([t_hist_inner, X_hist_inner, u_hist], device)
    model.to(device)

    # compute the gradients and hessians
    dVb_t, dVb_X, dVb_XX = utils.compute_deriv(NN = model, t = t_hist_inner, Y = X_hist_inner, device = device)

    # components : drift and vol
    drift = pm.compute_drift(X = X_hist_inner, u = u_hist,                      # dim = (sample_size, dim_state)
                param_dict = param_dict, device = device)
    drift_T = torch.transpose(drift.unsqueeze(2), 1, 2).to(device)              # dim = (sample_size, 1, dim_state)

    vols = pm.compute_vol(X = X_hist_inner, param_dict = param_dict, device = device)     
    variances = torch.bmm(vols, torch.transpose(vols, 1, 2)).to(device)         # dim = (sample_size, dim_state, dim_state)
    diffusion = torch.bmm(variances, dVb_XX).to(device)                         # dim = (sample_size, dim_state, dim_state)

    # dynkin operator
    dynkins = torch.bmm(drift_T, dVb_X.unsqueeze(2)).squeeze().to(device)       # dim = (sample_size, )
    dynkins = dynkins + dVb_t + 0.5 * vmap(torch.trace)(diffusion).to(device)   # dim = (sample_size, )

    loss = (dynkins.squeeze()) ** 2                                             # dim = (sample_size, )
    loss = loss.to(device)

    if id_extra_info : 
        print(f'    __ stats for dynkins : min={torch.min(dynkins)}, max={torch.max(dynkins)}, mean={torch.mean(dynkins)}, std={torch.std(dynkins)}')
        return loss.mean().to(device), dynkins.mean().to(device)
    else : 
        return loss.mean().to(device)   # tensor of size([])
    
if __name__ == "__main__":

# ------------------------
# Parameters + Configuration
# ------------------------
    
    J = os.path.join

    # create parser
    parser = ArgumentParser(description="Train : Value Function at the Boundary - method : PINN", formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--portfolio', type=str, required=True, help='path to the config (.json) file for all portfolio model parameters, including the sampling space.')
    parser.add_argument('-u', '--config_u', type=str, required=True, help='path to the result folder of the trained optimal control on the boundary (u_NN).')
    parser.add_argument('-Vb', '--config_Vb', type=str, required=True, help='path to the config (.json) file to train the value function on the boundary (Vb_NN).')

    parser.add_argument('-o', '--result_dir', type=str, default='./results/3_Vbnn', help='directory to store results.')
    parser.add_argument('-s', '--seed', type=int, default=904, help='seed for the random number generator.')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='indicator for what type of information to include in the log.', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
   
    args = parser.parse_args()

        # create the result directory and a logger
    result_dir = utils.create_dir(basedir = args.result_dir, dirname = 'train_Vb', suffix = None)
    checkpoint_dir = utils.create_dir(basedir = result_dir, dirname = 'checkpoints', suffix = '')
    logger = utils.setup_logging(result_dir, log_level = args.log_level, fname = 'train_Vb.log')

        # device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # read parameters 
    param_dict, sampling_dict, scaling_dict = pm.read_model_param(args.portfolio, device = device)

        # seed
    torch.manual_seed(args.seed)

        # load trained u_NN for the optimal control on the boundary
    u_nn_dir = args.config_u                                        # locate
    config_u = json.load(open(J(u_nn_dir, 'config_u.json'), 'r'))   # retrieve configuration
    args_u = json.load(open(J(u_nn_dir, 'args_u.json'), 'r'))       # retrieve arguments when launching
    u_dir_list = [J(u_nn_dir, 'checkpoints', f) for f in os.listdir(J(u_nn_dir, 'checkpoints'))]
    u_nn_checkpoint_path = max(u_dir_list, key=os.path.getmtime)    # get the last saved checkpoint

    u_NN = custom_nn.Boundary_Optimal_Control_Net(
                                                    dim_state = param_dict["dim_state"],
                                                    dim_output = param_dict["dim_control"],
                                                    dim_hidden = config_u["n_neuron"],
                                                    n_hidden_layers = config_u["n_hidden_layers"],
                                                    hidden_activation_type = args_u["hidden_activation_type"],
                                                    output_activation_type = args_u["output_activation_type"],
                                                    lower_bounds = param_dict["control_lower_bound"], 
                                                    upper_bounds = param_dict["control_upper_bound"], 
                                                    num_time_freqs = config_u["n_time_freqs"], 
                                                    norm_type = config_u["norm_type"]
                                                )
    u_NN.to(device)
    u_NN.to(torch.float32)
    u_NN.load_state_dict(torch.load(u_nn_checkpoint_path, map_location=device)["model_state_dict"])
    u_NN.eval()

        # load config file for training
    config_Vb = json.load(open(args.config_Vb, 'r'))
    n_batches = config_Vb["n_samples"] // config_Vb["batch_size"]

        # save all config files in case replication is needed
    json.dump(json.load(open(args.portfolio, "r")), open(f'{result_dir}/config_portfolio.json', 'w'), indent=4)
    json.dump(config_u, open(f'{result_dir}/config_u.json', 'w'), indent=4)
    json.dump(args_u, open(f'{result_dir}/args_u.json', 'w'), indent=4)
    json.dump(config_Vb, open(f'{result_dir}/config_Vb.json', 'w'), indent=4)

    args_dict = vars(args)                      # save parameters parsed from terminal
    args_dict["config_u"] = os.path.abspath(args.config_u)
    args_dict["config_Vb"] = os.path.abspath(args.config_Vb)
    args_dict["result_dir"] = os.path.abspath(args.result_dir)
    json.dump(args_dict, open(f'{result_dir}/args_Vb.json', 'w'), indent=4)

    script_name = os.path.basename(__file__)    # save the current script (keeping track of version)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(result_dir, script_name))

# ------------------------
# Training
# ------------------------

    # Pre-training 
    logger.info(f'****** Pre-training ******')

        # construct the model by loading hyperparameters
    model = custom_nn.Boundary_Value_Function_Net(
                            dim_state = param_dict["dim_state"],
                            dim_hidden = config_Vb["n_neuron"],
                            n_hidden_layers = config_Vb["n_hidden_layers"],
                            hidden_activation_type = config_Vb["hidden_activation_type"],
                            norm_type = config_Vb["norm_type"],
                            num_time_freqs = config_Vb["n_time_freqs"]
                            )
    model.to(device)
    model.to(torch.float32)

        # optimizer
    optimizer = optim.Adam(model.parameters(), lr = config_Vb["lr"], weight_decay = config_Vb["weight_decay"])
    scheduler_step = optim.lr_scheduler.MultiStepLR(optimizer, milestones = config_Vb["milestones"], gamma = config_Vb["gamma_step"])
    scheduler_expo = optim.lr_scheduler.ExponentialLR(optimizer, gamma = config_Vb["gamma_expo"])

        # initiation
            # variables to keep track of losses
    best_epoch = 0
    best_epoch_loss = None

    losses = {
        "PDE_residuals": [],    # mean squared Dynkin operator HVb for (t,X_t) when t < T
        "terminal_losses" : [], # MSE loss between predicted Vb(T, X_T) and target function F(X_T)
        "total_losses" : [],    # total_loss = PDE_loss + terminal_loss (without any weight for PDE_loss)
        "val_total_losses" : [] # validation total loss can either be training total_loss (if id_validate_sample = False) or the total loss of an independent fixed validation sample
        }
            # load previouly saved checkpoints in case it's continued training 
    if config_Vb["checkpoint_path"] is None :
        logger.info(f'New training       - no checkpoint loaded, starting from scratch.')
        start_epoch = 0
    else :
        logger.info(f'Continued training - loading checkpoints from {config_Vb["checkpoint_path"]}.')
        start_epoch, losses = utils.load_checkpoint(config_Vb["checkpoint_path"], model, optimizer, scheduler_step, scheduler_expo)
        best_epoch = start_epoch 

            # create a validation sample if the option is activated
    if config_Vb["id_validate_sample"]:
        val_seed = args.seed + 1    # fix a seed for validation sample
        val_t_hist, val_X_hist, val_u_hist = pm.simulate_trajectories(
                                                                control_NN = u_NN, 
                                                                sample_size = config_Vb["batch_size"],
                                                                param_dict = param_dict, 
                                                                sampling_dict = sampling_dict,
                                                                device = device, 
                                                                seed_initial_state = val_seed,
                                                                seed_brownian = val_seed
                                                                )
        val_t_before_terminal = val_t_hist[:, :-1, :].reshape((config_Vb["batch_size"] * param_dict["n_period"], 1)).to(device)
        val_X_before_terminal = val_X_hist[:, :-1, :].reshape((config_Vb["batch_size"] * param_dict["n_period"], param_dict["dim_state"])).to(device)
        val_u_before_terminal = val_u_hist.reshape((config_Vb["batch_size"] * param_dict["n_period"], param_dict["dim_control"])).to(device)

        logger.debug(f'Validation sample : val_t_hist.shape = {val_t_hist.shape}, val_X_hist.shape = {val_X_hist.shape}, val_u_hist.shape = {val_u_hist.shape}')

            # initiate weights for PDE residual losses
    if start_epoch == 0 :                               # new training 
        weight_PDE_residual = config_Vb["lambda_start"]
    elif start_epoch > config_Vb["lambda_epoch_end"]:   # continued training + lambda is no longer being updated
        weight_PDE_residual = config_Vb["lambda_end"]
    else :                                              # continued training + lambda is still being updated
        last_epoch_update = math.floor(start_epoch/config_Vb["lambda_update_interval"]) * config_Vb["lambda_update_interval"]
        
        weight_PDE_residual = config_Vb["lambda_start"] + (config_Vb["lambda_end"] - config_Vb["lambda_start"])*(last_epoch_update / config_Vb["lambda_end_epoch"])

        # log before training
    logger.info(f'Device : {device} and Seed : {args.seed}')
    logger.info('Hyper-parameters :')
    logger.info(f'  > n_asset = {param_dict["n_asset"]}, n_period = {param_dict["n_period"]}')
    logger.info(f'  > utility function coeff (risk aversion coeff for exponential utility function) = {param_dict['utility_coeff']}')
    
    logger.info(f'  > n_samples = {config_Vb["n_samples"]}, batch_size = {config_Vb["batch_size"]}, n_batches = {n_batches}, hidden_activation_type = {config_Vb["hidden_activation_type"]}')
    logger.info(f'    n_hidden_layers = {config_Vb["n_hidden_layers"]}, n_neuron = {config_Vb["n_neuron"]}, n_time_freqs = {config_Vb["n_time_freqs"]}, norm_type = {config_Vb["norm_type"]},')
    
    logger.info(f'  > (initial) lr = {config_Vb["lr"]}, weight_decay = {config_Vb["weight_decay"]}')
    logger.info(f'  > lambda_start (initial weight for PDE residual loss) = {config_Vb["lambda_start"]}, lambda_end (final weight for PDE residual loss) = {config_Vb["lambda_end"]}, ')
    logger.info(f'  > lambda_end_epoch (last epoch at which the PDE residual weight will get updated) = {config_Vb["lambda_end_epoch"]}, lambda_update_interval (number of epoch between two updates for the PDE residual weight)= {config_Vb["lambda_update_interval"]}')
    logger.info(f'  > current lambda = {weight_PDE_residual}')
    logger.info(f'  > gamma_expo (for exponential scheduler) = {config_Vb["gamma_expo"]}, gamma_step (for step scheduler) = {config_Vb["gamma_step"]} at milestones = {config_Vb["milestones"]}.')
    
    logger.info(f'  > resampling initial states at each epoch ? {config_Vb["id_resample_initial_state"]}')
    logger.info(f'  > resampling brownians at each epoch ? {config_Vb["id_resample_brownian"]}')
    logger.info(f'  > using a separate validation sample ? {config_Vb["id_validate_sample"]}')
    logger.info('****** Begin training ******')

    # Training
        # start timer
    start = time.time() 
           
        # generate batch seeds to maintain coherence between epochs
    batch_seeds = torch.multinomial(torch.ones(100000), n_batches, replacement = False).to(int)

        # training loop
    for epoch in range(start_epoch, config_Vb["n_epoch"] + start_epoch):
        
        model.train()   # set model to training mode
        logger.info(f'Epoch {epoch+1}/{config_Vb["n_epoch"] + start_epoch} : learning rate = {scheduler_expo.get_last_lr()[0]} and PDE residual weight (lambda) = {weight_PDE_residual}')

        # create a loss dictionary to keep track
        epoch_total_loss = 0.0
        epoch_PDE_residual = 0.0
        epoch_terminal_loss = 0.0

        # shuffle the seed
        idx = torch.randperm(batch_seeds.size(0))
        batch_seeds = batch_seeds[idx]  

        # loop through batches
        for j in range(n_batches):

            # data for the batch
            logger.debug(f' + Batch {j+1}/{n_batches} : seed = {batch_seeds[j].item()}')

                # seed(s)
            temp_seed = batch_seeds[j].item()       

            if config_u["id_resample_initial_state"] :
                batch_seed_initial_state = torch.seed()
                logger.debug(f'     - seed for initial states : {batch_seed_initial_state}')
            else :
                batch_seed_initial_state = temp_seed

            if config_u["id_resample_brownian"]:
                batch_seed_brownian = torch.seed()
                logger.debug(f'     - seed for brownians : {batch_seed_brownian}')
            else :
                batch_seed_brownian = temp_seed

                # generate data
            batch_t_hist, batch_X_hist, batch_u_hist = pm.simulate_trajectories(
                                                            control_NN = u_NN,
                                                            sample_size = config_Vb["batch_size"],
                                                            param_dict = param_dict, 
                                                            sampling_dict = sampling_dict, 
                                                            device = device, 
                                                            seed_initial_state = batch_seed_initial_state,
                                                            seed_brownian = batch_seed_brownian)
            # reset the optimizer
            optimizer.zero_grad()

            # loss at terminal 
            batch_terminal_loss = LossFunction_at_terminal(
                                        model = model, param_dict = param_dict, device = device, 
                                        X_T = batch_X_hist[:, -1, :].squeeze(),     # dim = (batch_size, dim_state)
                                        T_tensor = batch_t_hist[:, -1, :].squeeze() # dim = (batch_size, )  
                                        )
            if j < 10 : # debugging the first few batches   
                logger.debug(f'     _ batch_terminal_loss (avg) = {batch_terminal_loss.item(): .15f}')

            # loss before terminal (PDE residual)
                # reshape (flatten tensors so that batch dimension = batch_size * n_period)
            batch_t_before_terminal = batch_t_hist[:, :-1, :].reshape((config_Vb["batch_size"] * param_dict["n_period"], 1)).to(device)
            batch_X_before_terminal = batch_X_hist[:, :-1, :].reshape((config_Vb["batch_size"] * param_dict["n_period"], param_dict["dim_state"])).to(device)
            batch_u_before_terminal = batch_u_hist.reshape((config_Vb["batch_size"] * param_dict["n_period"], param_dict["dim_control"])).to(device)

            if args.log_level == "DEBUG":
                batch_PDE_residual, batch_dynkins = LossFunction_PDE_residual(
                                                            model = model, param_dict = param_dict, device = device,
                                                            t_hist_inner = batch_t_before_terminal,     # dim = (batch_size * n_period, 1)
                                                            X_hist_inner = batch_X_before_terminal,     # dim = (batch_size * n_period, dim_state)
                                                            u_hist = batch_u_before_terminal,           # dim = (batch_size * n_period, dim_control)
                                                            id_extra_info = True)
                if j < 10 : # for debugging
                    logger.debug(f'     _ batch_dynkins (avg) = {batch_dynkins.item(): .15f}')    
            else :
                batch_PDE_residual = LossFunction_PDE_residual(
                                                            model = model, param_dict = param_dict, device = device,
                                                            t_hist_inner = batch_t_before_terminal,     # dim = (batch_size * n_period, 1)
                                                            X_hist_inner = batch_X_before_terminal,     # dim = (batch_size * n_period, dim_state)
                                                            u_hist = batch_u_before_terminal,           # dim = (batch_size * n_period, dim_control)
                                                            id_extra_info = False)
            if j < 10 : # for debugging
                logger.debug(f'     _ batch_PDE_residual (avg) = {batch_PDE_residual.item(): .15f}')    
            
            # loss to back propagate
            batch_loss = batch_terminal_loss + weight_PDE_residual * batch_PDE_residual

            # back propagate
            batch_loss.backward()
            optimizer.step()

            # accumulate batch losses for the record (no back-propagation here, just recording)
            epoch_PDE_residual += batch_PDE_residual.item() / n_batches
            epoch_terminal_loss += batch_terminal_loss.item() / n_batches
            epoch_total_loss += batch_PDE_residual.item() / n_batches + batch_terminal_loss.item() / n_batches

            if j < 10 : # for debugging
                logger.debug(f'     _ loss to back propagate (which includes weight for PDE loss) = {batch_loss.item(): .15f}')

        # keep track of epoch losses
        losses["total_losses"].append(epoch_total_loss)
        losses["PDE_residuals"].append(epoch_PDE_residual)
        losses["terminal_losses"].append(epoch_terminal_loss)

        logger.info(f'  + Training    _ Total loss : {epoch_total_loss: .15f} = PDE residual : {epoch_PDE_residual: .15f} + Terminal loss : {epoch_terminal_loss: .15f}')
                        
        # update the PDE weight
        if (epoch + 1 + start_epoch <= config_Vb["lambda_end_epoch"]) and ((epoch + 1 + start_epoch) % config_Vb["lambda_update_interval"] == 0):
            weight_PDE_residual = weight_PDE_residual + (config_Vb["lambda_end"] - config_Vb["lambda_start"])*(config_Vb["lambda_update_interval"]/config_Vb["lambda_end_epoch"])

        # update learning rate
        scheduler_step.step()
        scheduler_expo.step()

        # validate through validation set
        if config_Vb["id_validate_sample"]: # if validation sample is in use

            model.eval()    # switch to evaluation mode
            with torch.no_grad() : 
                val_terminal_loss = LossFunction_at_terminal(model = model, param_dict = param_dict, device = device, 
                                        X_T = val_X_hist[:, -1, :].squeeze(), T_tensor = val_t_hist[:, -1, :].squeeze())
                val_PDE_residual = LossFunction_PDE_residual(model = model, param_dict = param_dict, device = device, 
                                        t_hist_inner = val_t_before_terminal, X_hist_inner = val_X_before_terminal,
                                        u_hist = val_u_before_terminal, id_extra_info = False)
                val_epoch_total_loss = val_PDE_residual.item() + val_terminal_loss.item()

            logger.info(f'  + Validation  _ Total loss : {val_epoch_total_loss: .15f} = PDE residual : {val_PDE_residual: .15f} + Terminal loss : {val_terminal_loss: .15f}')
        else : 
            val_epoch_total_loss = epoch_total_loss

        losses["val_total_losses"].append(val_epoch_total_loss)

        if best_epoch_loss is None or val_epoch_total_loss < best_epoch_loss : 
            best_epoch = epoch
            best_epoch_loss = val_epoch_total_loss 

            utils.save_checkpoint(model = model, optimizer = optimizer, scheduler_step = scheduler_step, scheduler_expo = scheduler_expo,
                                epoch = epoch, basedir = checkpoint_dir, suffix = "best", additional_info = losses)
            
            logger.info(f'  >>>>> Found best model in epoch = {epoch + 1}, saving in {checkpoint_dir}')

    # Closing 
    end = time.time()
    logger.info(f'****** End training ______ Execution time = {(end - start)/60 : .2f} minutes + Best model found at epoch [{best_epoch + 1}/{start_epoch + config_Vb["n_epoch"]}].')