'''
This script is to train the neural network which estimates the optimal control on the entire viable domain, i.e. maximizing strategy.
'''
# ------------------------
# Packages
# ------------------------

import os, json, shutil, time, copy
from typing import Literal, Union, Optional
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

import importlib
pm = importlib.import_module('0_portfolio_model', package=None)
custom_nn = importlib.import_module('0_neural_networks', package=None)
utils = importlib.import_module('0_utils', package=None)

float_type = torch.float32

# ------------------------
# Supporting / Loss functions
# ------------------------

def LossAccumulative(
        model,                                      # the neural network in training
        w_NN,                                       # a trained neural network to estimate the boundary of the viable domain
        initial_states : torch.Tensor,              # a sample of initial states X_0        - dim = (sample_size, dim_state)
        initial_constraints : torch.Tensor,         # a sample of initial constraint p      - dim = (sample_size, 1)
        brownians : torch.Tensor,                   # a sample of brownian increment paths  - dim = (sample_size, dim_sto, n_period)
        device : torch.device,                      # device for all computation
        param_dict : dict,                          # dictionary of all parameters concerning the portfolio model
        book_value = None,                          # book value if not using the one given in param_dict - NOT implemented as of now
        id_save_states = False                      # indicator whether to keep a record of state trajectories (if not activated, keep only the average loss)
        ): 
    
    if book_value is None : # if book value is not provided, use the one from param_dict (supposed to be constant over time)
        book_value = param_dict["book_value"]       # dim = (n_asset, )
        
    # device
    utils.send_to_device([initial_states, brownians, initial_constraints, book_value], device)
    model.to(device)
    
    # dimension check
    sample_size = initial_states.shape[0]
    n_period = param_dict["n_period"]
    utils.dimension_check(list_label = ['sample size of brownians', 'number of time steps in brownian trajectories', 'number of stochastic factors', 'number of state variables in initial_states', 'sample size of initial_constraints'],
                          list_expected_value = [sample_size, n_period, param_dict["dim_sto"], param_dict["dim_state"], sample_size],
                          list_actual_value = [brownians.shape[0], brownians.shape[2], brownians.shape[1], initial_states.shape[1], initial_constraints.shape[0]])

    # initiation
    current_t = torch.full(size=(sample_size, 1), dtype = float_type,           
                        fill_value=0.0).to(device)                              # dim = (sample_size, 1)
    current_X = initial_states.detach().clone().to(float_type)                  # dim = (sample_size, dim_state)
    current_M = initial_constraints.detach().clone().to(float_type)             # dim = (sample_size, 1)
    
    if id_save_states:
        aug_state_history = torch.empty(size=(sample_size, param_dict["dim_state"] + 2, n_period + 1),
                                device = device, dtype = float_type)            # dim = (sample_size, 1 + dim_state + 1, 1 + n_period)
        aug_state_history[:, :, 0] = torch.cat((current_t, current_X, current_M), dim = 1).to(device)     

        aug_control_history = torch.empty(size=(sample_size, param_dict["dim_control"] + param_dict["dim_sto"], n_period),
                                device = device, dtype = float_type)            # dim = (sample_size, dim_control + dim_sto, n_period)
        
        breach_penalty_history = torch.empty(size=(sample_size, 1, n_period),
                                    device = device, dtype = float_type)        # dim = (sample_size, 1, n_period)
    
    avg_breach_penalty = 0

    # push forward and accumulate penalties
    for i in range(n_period):
        # estimate the control
        u_t, a_t = model(current_t, current_X, current_M)
        
        # update the current augmented states : (t,X_t, M_t) -> (t_new, X_new, M_new)
        dW_t = brownians[:, :, i].detach().clone()      # dim = (sample_size, dim_sto)
        current_t, current_X, current_M = pm.update_aug_state(device = device, param_dict = param_dict,
                                                t = current_t, X = current_X, M = current_M, 
                                                u = u_t, a = a_t, dW = dW_t, book_value = book_value)
        
        # compute penalty
        breach_penalty_t, breach_penalty_tensor_t = pm.breach_penalty(w_NN = w_NN, 
                                                            t = current_t, X = current_X, M = current_M,
                                                            param_dict = param_dict, device = device, 
                                                            id_keep_tensor = id_save_states)    
        avg_breach_penalty = avg_breach_penalty + breach_penalty_t / n_period

        # track if needed
        if id_save_states :
            aug_state_history[:, :, i+1] = torch.cat([current_t.detach().clone(), 
                                                      current_X.detach().clone(),
                                                      current_M.detach().clone()],
                                                    dim = 1).to(device) 
            aug_control_history[:, :, i] = torch.cat([u_t.detach().clone(), a_t.detach().clone()],
                                                    dim = 1).to(device)
            breach_penalty_history[:, :, i] = breach_penalty_tensor_t.detach().clone().to(device)

    # terminal loss = - terminal utility (maximizing utility = minimizing loss, hence the negative)
    terminal_utility, terminal_utility_tensor = pm.terminal_utility(X = current_X, param_dict = param_dict, 
                                                            id_keep_tensor = id_save_states)
    
    # average across the sameple size
    terminal_utility = terminal_utility / sample_size
    avg_breach_penalty = avg_breach_penalty / sample_size
    
    if id_save_states :
        history = {'aug_state' : aug_state_history, 'aug_control': aug_control_history,
                   'utility': terminal_utility_tensor.detach(),
                   'breach_penalty': breach_penalty_history.detach()}
        return terminal_utility, avg_breach_penalty, history
    else :
        return terminal_utility, avg_breach_penalty, None
    

if __name__ == "__main__":
# ------------------------
# Parameters + Configuration
# ------------------------

    J = os.path.join    # alias for frequently used function

        # create parser
    parser = ArgumentParser(description='Train : Optimal Control', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--portfolio', type=str, required=True, help='path to the config (.json) file for all portfolio model parameters, including the sampling space.')
    parser.add_argument('-w', '--config_w', type=str, required=True, help='path to the result folder of the trained domain boundary (w_NN).')
    parser.add_argument('-a', '--config_a', type=str, required=True, help='path to the config (.json) file for training the optimal control on viable domain (a_NN).')

    parser.add_argument('-o', '--result_dir', type=str, default='./results/2_wnn', help='directory to store results.')
    parser.add_argument('-s', '--seed', type=int, default=1851794, help='seed for the random number generator.')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='indicator for what type of information to include in the log.', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
   
    args = parser.parse_args()

        # create the result directory and a logger
    result_dir = utils.create_dir(basedir = args.result_dir, dirname = 'train_a', suffix = None)
    checkpoint_dir = utils.create_dir(basedir = result_dir, dirname = 'checkpoints', suffix = '')
    logger = utils.setup_logging(result_dir, log_level = args.log_level, fname = 'train_a.log')

        # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # read parameters 
    param_dict, sampling_dict, scaling_dict = pm.read_model_param(args.portfolio, device = device)
    logger.debug(f'Parameters :\n + param_dict = {param_dict} \n + sampling_dict = {sampling_dict} \n + scaling_dict = {scaling_dict}')
        
        # seed 
    torch.manual_seed(args.seed)        

        # load the trained w_NN for the domain boundary
    w_nn_dir = args.config_w                                    
    config_w = json.load(open(J(w_nn_dir, 'config_w.json'), 'r'))   
    args_w = json.load(open(J(w_nn_dir, 'args_w.json'), 'r'))       
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

        # read config file for the training
    config_a = json.load(open(args.config_a, 'r'))
    n_batches = config_a["n_samples"] // config_a["batch_size"]

        # save all config files in case replication is needed
    json.dump(json.load(open(args.portfolio, "r")), open(f'{result_dir}/config_portfolio.json', 'w'), indent=4)
    json.dump(config_w, open(f'{result_dir}/config_w.json', 'w'), indent=4)
    json.dump(config_a, open(f'{result_dir}/config_a.json', 'w'), indent=4)

    args_dict = vars(args)
    args_dict['config_w'] = os.path.abspath(args.config_w)      # save all parameters parsed from terminal
    args_dict['result_dir'] = os.path.abspath(args.result_dir)  # save result directory
    json.dump(args_dict, open(f'{result_dir}/args_a.json', 'w'), indent=4)

    script_name = os.path.basename(__file__)    # save the current script (keeping track of version)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(result_dir, script_name))
    
# ------------------------
# Training
# ------------------------

        # construct the model by loading hyperparameters
    model = custom_nn.Optimal_Control_Net(dim_state = param_dict["dim_state"], 
                        dim_control = param_dict["dim_control"], dim_sto = param_dict["dim_sto"],
                        dim_hidden = config_a["n_neuron"], n_hidden_layers = config_a["n_hidden_layers"],
                        num_time_freqs = config_a["n_time_freqs"], norm_type = config_a["norm_type"],
                        hidden_activation_type = config_a["hidden_activation_type"], output_activation_type = config_a["output_activation_type"],
                        lower_bounds = param_dict["control_lower_bound"], upper_bounds = param_dict["control_upper_bound"],
                        martingale_control_limit = config_a["martingale_control_limit"],
                        device = device
                        )
    model.to(device)

        # optimizer
    optimizer = optim.Adam(model.parameters(), lr = config_a['lr'], weight_decay = config_a["weight_decay"])
    scheduler_step = optim.lr_scheduler.MultiStepLR(optimizer, milestones = config_a["milestones"], gamma = config_a["gamma_step"])
    scheduler_expo = optim.lr_scheduler.ExponentialLR(optimizer, gamma = config_a["gamma_expo"])

        # initialization
    start_epoch = 0

    losses = {"total_losses" : [], 
            "terminal_utility" : [],
            "penalty_breach" : [],
            "val_total_losses": []
            }

    best_epoch = 0
    best_epoch_loss = None

    logger.info(f'****** Pre-training ******')

        # create a validation sample if activated
    if config_a["id_validate_sample"]:
        val_seed = args.seed + 1    # fix a seed for validation sample
        val_initial_states = pm.sample_initial_state(sample_size = config_a["batch_size"], 
                                                    param_dict = param_dict,
                                                    sampling_dict = sampling_dict, 
                                                    device = device, seed = val_seed)
        val_brownians = pm.sample_brownian_trajectories(sample_size = config_a["batch_size"], 
                                                    param_dict = param_dict,
                                                    device = device, seed = val_seed)
        val_initial_constraints = pm.generate_initial_constraints(
                                                    domain_boundary_NN = w_NN, 
                                                    initial_states = val_initial_states,
                                                    device = device, seed = val_seed, 
                                                    constraint_sample_range = config_a["constraint_sample_range"],
                                                    float_type = float_type)
        
        # load checkpoint if needed 
    if config_a["checkpoint_path"] is not None :
        logger.info(f"Continued training - loading checkpoints from {config_a["checkpoint_path"]}.")
        if args.new_lr:     # replace schedulers and optimizer from the checkpoint by those created in-place
            logger.info("                   - Ignoring optimizer and learning rate from previous training(s).")
            start_epoch, additional_info = utils.load_checkpoint(config_a["checkpoint_path"], model)
        else :              # retain schedulers and optimizer from the checkpoint
            start_epoch, additional_info = utils.load_checkpoint(config_a["checkpoint_path"], model,
                                                                optimizer, scheduler_step, scheduler_expo)
        
        losses = additional_info    # keep the record from last training
        
        logger.info(f"Continued training - last checkpoint is at epoch = {start_epoch}.")
        epoch = start_epoch
    else : 
        logger.info(f"New training       - no checkpoint loaded, starting from scratch.")

        # pre-training log
    logger.info(f'Device : {device} and Seed : {args.seed}')
    logger.info('Hyper-parameters :')
    logger.info(f'  > n_asset = {param_dict["n_asset"]}, n_period = {param_dict["n_period"]}')
    logger.info(f'  > utility function coeff = {param_dict["utility_coeff"]}')

    logger.info(f'  > n_samples = {config_a["n_samples"]}, batch_size = {config_a["batch_size"]}, n_batches = {n_batches}, penalty_weight = {config_a["penalty_weight"]}, ')
    logger.info(f'    n_hidden_layers = {config_a["n_hidden_layers"]}, n_neuron = {config_a["n_neuron"]}, n_time_freqs = {config_a["n_time_freqs"]}, norm_type = {config_a["norm_type"]}')
    logger.info(f'  > activation for hidden layers = {config_a["hidden_activation_type"]} and for output layer = {config_a["output_activation_type"]}')
    logger.info(f'  > buffer range for initial constraint limit = {config_a["constraint_sample_range"]}, limit for martingale control = {config_a["martingale_control_limit"]}')
    
    logger.info(f'  > (initial) lr = {config_a["lr"]}, weight_decay = {config_a["weight_decay"]}, gamma_expo (for exponential scheduler) = {config_a["gamma_expo"]},')
    logger.info(f'  > gamma_step (for step scheduler) = {config_a["gamma_step"]} at milestones = {config_a["milestones"]}.')

    logger.info(f'  > resampling initial states at each epoch ? {config_a["id_resample_initial_state"]}')
    logger.info(f'  > resampling brownians at each epoch ? {config_a["id_resample_brownian"]}')
    logger.info(f'  > using a separate validation sample ? {config_a["id_validate_sample"]}')

    logger.info(f'  > compute stats on the breaching of domain boundary ? {config_a["id_breach_stats"]}')

    logger.info('****** Begin training ******')

    start = time.time()     # start time counter

        # generate batch seeds to maintain coherence between epochs
    batch_seeds = torch.multinomial(torch.ones(100000), n_batches, replacement = False).to(int)

        # training loop
    for epoch in range(start_epoch, config_a["n_epoch"] + start_epoch):

        model.train()       # set to training mode
        logger.info(f'Epoch [{epoch + 1}/{config_a["n_epoch"] + start_epoch}]: learning rate = {optimizer.param_groups[-1]['lr']}')

            # create variables to keep track of the epoch losses
        epoch_total_loss = 0.0
        epoch_terminal_utility = 0.0
        epoch_penalty_breach = 0.0

        if config_a["id_breach_stats"]:
            epoch_n_path_breached = 0.0

            # shuffle the seed
        idx = torch.randperm(batch_seeds.size(0))
        batch_seeds = batch_seeds[idx]

            # loop through the batches
        for j in range(n_batches):

                # data for the batch
            logger.debug(f'Batch{j+1}/{n_batches}: batch_seed = {batch_seeds[j].item()}')
            temp_seed = batch_seeds[j].item()               # preset seed for the batch

                    # initial states and intial constraints
            if config_a["id_resample_initial_state"]:       # resample initial states 
                temp_seed = torch.seed()
                logger.debug(f'         - changing seed to {temp_seed} for resampling initial states.')

            batch_initial_states = pm.sample_initial_state(sample_size = config_a["batch_size"], 
                                                        param_dict = param_dict, 
                                                        sampling_dict = sampling_dict, 
                                                        device = device, 
                                                        seed = temp_seed)
            batch_initial_constraints = pm.generate_initial_constraints(
                                                        domain_boundary_NN = w_NN, 
                                                        initial_states = batch_initial_states,
                                                        seed = temp_seed, 
                                                        device = device,
                                                        constraint_sample_range = config_a["constraint_sample_range"],
                                                        float_type = float_type)
                # brownians
            if config_a["id_resample_brownian"] :       # resampling brownians
                temp_seed = torch.seed()
                logger.debug(f'         - changing seed to {temp_seed} for resampling brownians.')

            batch_brownians = pm.sample_brownian_trajectories(sample_size = config_a["batch_size"], 
                                                        param_dict = param_dict, 
                                                        device = device, 
                                                        seed = temp_seed)
                # reset the optimizer
            optimizer.zero_grad()

                # forward pass
            batch_terminal_utility, batch_penaly_breach, batch_history = LossAccumulative(model = model, w_NN = w_NN, 
                                                                        param_dict = param_dict,
                                                                        initial_states = batch_initial_states,
                                                                        initial_constraints = batch_initial_constraints,
                                                                        brownians = batch_brownians,
                                                                        device = device, 
                                                                        id_save_states = config_a["id_breach_stats"])   # only keep track if in debug mode
            batch_loss = - batch_terminal_utility + config_a["penalty_weight"] * batch_penaly_breach

                # back propagation
            batch_loss.backward()
            optimizer.step()

                # accumulate batch losses for the record (no need for back propagation here)
            epoch_total_loss += batch_loss.item()/n_batches
            epoch_terminal_utility += batch_terminal_utility.item()/n_batches
            epoch_penalty_breach += batch_penaly_breach.item()/n_batches

                # if in debug mode, print out state history to check coherence
            if args.log_level == "DEBUG" and j == 1 and epoch == start_epoch :
                logger.debug(f'         - terminal_utility_tensor (first 10 values) = {batch_history["utility"][:10, :]}')
                logger.debug(f'         - breach_penalty_tensor (first 10 values + first 10 periods) = {batch_history["breach_penalty"][:10, :, :10]}')

                # if statistics about domain breaches are retained
            if config_a["id_breach_stats"]:
                with torch.no_grad():
                    batch_penalty_breach_tensor = batch_history["breach_penalty"].squeeze().to(device)         # dim = (sample_size, n_period)
                    batch_breaching_moment_tensor = torch.where(batch_penalty_breach_tensor > 0, 1, 0).to(device)
                    batch_number_breach_tensor = batch_breaching_moment_tensor.sum(axis = 1).to(device) # dim = (sample_size, )
                
                    batch_path_inside = torch.where(batch_number_breach_tensor == 0, 1, 0).sum().item()
                    epoch_n_path_breached += config_a["batch_size"] - batch_path_inside
    
        # keep track of epoch losses
        losses["total_losses"].append(epoch_total_loss)
        losses["terminal_utility"].append(epoch_terminal_utility)
        losses["penalty_breach"].append(epoch_penalty_breach)

        logger.info(f'  + Training :    _ Total loss (with penalty weight) : {epoch_total_loss : .15f}')
        logger.info(f'                      __ terminal utility = {epoch_terminal_utility: .15f}')
        logger.info(f'                      __ penalty for breaching domain boundary = {epoch_penalty_breach: .15f}.')
        if config_a["id_breach_stats"]:
            logger.info(f'                      __ percentage of path breached = {round(100 * epoch_n_path_breached/config_a["n_samples"]): .3f} %')

        # update learning rate
        scheduler_step.step()
        scheduler_expo.step()

        # validation
        if config_a["id_validate_sample"] :     # if using the validation sample
            model.eval()

            with torch.no_grad():
                val_terminal_utility, val_penalty_breach, _ = LossAccumulative(model = model, w_NN = w_NN, 
                                                                initial_states = val_initial_states, 
                                                                initial_constraints = val_initial_constraints,
                                                                brownians = val_brownians, device = device, 
                                                                param_dict = param_dict, id_save_states = False)
                val_total_loss = - val_terminal_utility + config_a["penalty_weight"] * val_penalty_breach
                val_total_loss = val_total_loss.item()

            logger.info(f'  + Validation :  _ Total loss (with penalty weight) : {val_total_loss : .15f}')
            logger.info(f'                      __ terminal utility = {val_terminal_utility: .15f}')
            logger.info(f'                      __ penalty for breaching domain boundary = {val_penalty_breach: .15f}.')
        else :  # use the training loss as the validation loss
            val_total_loss = epoch_total_loss

        losses["val_total_losses"].append(val_total_loss)

        if best_epoch_loss == None or best_epoch_loss > val_total_loss :
            best_epoch = epoch
            best_epoch_loss = val_total_loss

            utils.save_checkpoint(model = model, optimizer = optimizer, scheduler_step = scheduler_step, scheduler_expo = scheduler_expo,
                    epoch = epoch, basedir = checkpoint_dir, suffix = 'best', additional_info = losses)
            logger.info(f'  >>>>> Found best model in epoch = {epoch + 1}')

    # closing
    end = time.time()
    logger.info(f'****** End training ______ Execution time = {(end - start)/60 : .2f} minutes + Best model found at epoch [{best_epoch + 1}/{start_epoch + config_a["n_epoch"]}].')
        



            



