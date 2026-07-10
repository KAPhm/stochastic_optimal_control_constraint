'''
This script is to train the neural network which estimates the optimal control on the boundary, i.e hedging strategy.
'''
# ------------------------
# Packages
# ------------------------

import os, json, shutil, time, copy
import numpy as np
from typing import Literal, Union, Optional
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

import importlib
pm = importlib.import_module('0_portfolio_model', package=None)
custom_nn = importlib.import_module('0_neural_networks', package=None)
utils = importlib.import_module('0_utils', package=None)

# ----
# Logistic variable 
float_type = torch.float32

# ------------------------
# Supporting / Loss functions
# ------------------------

def LossAccumulative(
        model,                                          # the neural network in training
        initial_states : torch.Tensor,                  # a sample of initial states X_0        - dim = (n_samples, dim_state)
        brownians : torch.Tensor,                       # a sample of brownian increment paths  - dim = (n_samples, dim_sto, n_period)
        device : torch.device,                          # device for all computation
        param_dict : dict,                              # dictionary of all parameters for the portfolio model
        book_value : Optional[np.ndarray] = None,       # book value if not using the one given in param_dict - NOT implemented as of now
        id_save_states = False                          # indicator whether to keep a record of state trajectories (if not activated, keep only the average loss)
        ):  # Output : terminal loss and accumulated penalties across the time horizon, averaged over sample size.

    if book_value is None : # if book value is not provided, use the one from param_dict (supposed to be constant over time)
        book_value = param_dict["book_value"]       # dim = (n_asset, )
        
        # device
    utils.send_to_device([initial_states, brownians, book_value], device)
    model.to(device)

        # dimension
    sample_size = initial_states.shape[0]
    n_period = param_dict["n_period"]
    utils.dimension_check(list_label = ['sample size of brownians', 'number of time steps in brownian trajectories', 'number of stochastic factors', 'number of state variables in initial_states'],
                          list_expected_value = [sample_size, n_period, param_dict["dim_sto"], param_dict["dim_state"]],
                          list_actual_value = [brownians.shape[0], brownians.shape[2], brownians.shape[1], initial_states.shape[1]])

        # initiation
    current_t = torch.full(size=(sample_size, 1), dtype = float_type,           
                        fill_value=0.0).to(device)                              # dim = (sample_size, 1)
    current_X = initial_states.detach().clone().to(float_type)                  # dim = (sample_size, dim_state)
    
    if id_save_states:
        state_history = torch.empty(size=(sample_size, param_dict["dim_state"]+1, n_period+1),
                                    device = device, dtype = float_type)        # dim = (sample_size, 1 + dim_state, 1 + n_period)
        state_history[:, :, 0] = torch.cat((current_t, current_X), dim = 1).to(device)               
        control_history = torch.empty(size=(sample_size, param_dict["dim_control"], n_period),
                                    device = device, dtype = float_type)        # dim = (sample_size, dim_control, n_period)
        penalties_history = torch.empty(size=(sample_size, 3, n_period),
                                    device = device, dtype = float_type)        # dim = (sample_size, 3, n_period)
        
    avg_penalty_neg_cash = 0
    avg_penalty_neg_liab = 0
    avg_penalty_bankrupt = 0

        # push forward and accumulate penalties
    for i in range(n_period):
            # control estimated 
        u_t = model(current_t, current_X) # dim = (sample_size, dim_control)

            # update the current states
        dW_t = brownians[:, :, i].detach().clone()                          # dim = (sample_size, dim_sto)
        current_t, current_X = pm.update_state(t = current_t, X = current_X, u = u_t, dW = dW_t, 
                                    param_dict = param_dict, book_value = book_value, device = device)
        
            # compute penalties
        pen_neg_liab_t, pen_neg_cash_t, pen_bankrupt_t, penalty_tensor = pm.boundary_penalty(X = current_X, param_dict = param_dict, device = device, id_keep_tensor = id_save_states)
        avg_penalty_neg_liab = avg_penalty_neg_liab + pen_neg_liab_t/n_period        # note that this is solely averaged across the time horizon, not yet across the sample size
        avg_penalty_neg_cash = avg_penalty_neg_cash + pen_neg_cash_t/n_period
        avg_penalty_bankrupt = avg_penalty_bankrupt + pen_bankrupt_t/n_period
        
            # if track if desired
        if id_save_states :
            state_history[:, :, i+1] = torch.cat((current_t.detach().clone(), current_X.detach().clone()),
                                                dim = 1).to(device)
            control_history[:, :, i] = u_t.detach().clone().to(device)
            penalties_history[:, :, i] = penalty_tensor.detach().clone().to(device)
                    
        # terminal loss 
    terminal_loss, terminal_loss_tensor = pm.terminal_capital_loss(X = current_X, param_dict = param_dict, id_keep_tensor = id_save_states)

        # average across the sample size
    terminal_loss = terminal_loss/sample_size
    avg_penalty_neg_liab = avg_penalty_neg_liab/sample_size
    avg_penalty_neg_cash = avg_penalty_neg_cash/sample_size
    avg_penalty_bankrupt = avg_penalty_bankrupt/sample_size

    if id_save_states :
        history = {'state': state_history, 'control': control_history, 
                   'loss' : terminal_loss_tensor.detach(),                  # dim = (sample_size, 1)
                   'penalties' : penalties_history.detach()}                # dim = (sample_size, 3, n_period)
        return terminal_loss, avg_penalty_neg_liab, avg_penalty_neg_cash, avg_penalty_bankrupt, history
    else :
        return terminal_loss, avg_penalty_neg_liab, avg_penalty_neg_cash, avg_penalty_bankrupt, None

if __name__ == '__main__':
# ------------------------
# Parser
# ------------------------

        # create parser 
    parser = ArgumentParser(description='Train : Optimal Control on the Boundary', 
                    formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--portfolio', type=str, required=True, help='path to the config (.json) file for all portfolio model parameters, including the sampling space.')
    parser.add_argument('-u', '--config_u', type=str, required=True, help='path to the config (.json) file for training the optimal control on the boundary (u_NN).')
    parser.add_argument('--hidden_activation_type', type=str, default='Swish', help='type of activation function to be used in the hidden layers of the neural network (u_NN).',
                        choices=["GELU", "Swish", "ReLU"])
    parser.add_argument('--output_activation_type', type=str, default="Tanh", help='type of activation function to be used in the output layer of the neural network (u_NN).',
                        choices=["Tanh", "Sigmoid"])

    parser.add_argument('-o', '--result_dir', type=str, default='./results/1_u_nn', help='directory to store results.')
    parser.add_argument('-s', '--seed', type=int, default=29, help='seed for the random number generator.')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='indicator for what type of information to include in the log.', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
    parser.add_argument('-c', '--save_every', type=int, default=100, help='number of epochs after which the result is automatically saved.')

    parser.add_argument('--new_lr', action='store_true', help='if passed and there is a checkpoint, the learning rate scheduler will be set to the new scheduler built from the config file, thus ignoring the learning rate scheduler in the checkpoint')

    args = parser.parse_args()

        # create the result directory and a logger
    result_dir = utils.create_dir(basedir = args.result_dir, dirname = 'train_u', suffix = None)
    checkpoint_dir = utils.create_dir(basedir = result_dir, dirname = 'checkpoints', suffix = '')
    logger = utils.setup_logging(result_dir, log_level = args.log_level, fname = 'train_u.log')

        # device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # read parameters 
    param_dict, sampling_dict, scaling_dict = pm.read_model_param(args.portfolio, device = device)
    logger.debug(f'Parameters :\n + param_dict = {param_dict} \n + sampling_dict = {sampling_dict} \n + scaling_dict = {scaling_dict}')
        # seed 
    torch.manual_seed(args.seed)        

        # read config file for the training
    config_u = json.load(open(args.config_u, 'r'))
    n_batches = config_u["n_samples"] // config_u["batch_size"]

        # save all config files in case replication is needed
    json.dump(json.load(open(args.portfolio, "r")), open(f'{result_dir}/config_portfolio.json', 'w'), indent=4)
    json.dump(config_u, open(f'{result_dir}/config_u.json', 'w'), indent=4)

    args_dict = vars(args)
    args_dict['config_u'] = os.path.abspath(args.config_u)     # save all parameters parsed from terminal
    args_dict['result_dir'] = os.path.abspath(args.result_dir)  # save result directory
    json.dump(args_dict, open(f'{result_dir}/args_u.json', 'w'), indent=4)

    script_name = os.path.basename(__file__)    # save the current script (keeping track of version)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(result_dir, script_name))
    
    # ----
    # Training
        # construct the model by loading hyperparameters
    model = custom_nn.Boundary_Optimal_Control_Net(
                        dim_state = param_dict["dim_state"], dim_output = param_dict["dim_control"], 
                        dim_hidden = config_u["n_neuron"], n_hidden_layers = config_u["n_hidden_layers"], 
                        hidden_activation_type = args.hidden_activation_type, output_activation_type = args.output_activation_type,
                        lower_bounds = param_dict["control_lower_bound"], upper_bounds = param_dict["control_upper_bound"], 
                        num_time_freqs = config_u["n_time_freqs"], norm_type = config_u["norm_type"]
                        )
    model.to(device)    # move model to device   

        # optimizer
    optimizer = optim.Adam(model.parameters(), lr = config_u['lr'], weight_decay = config_u["weight_decay"])
    scheduler_step = optim.lr_scheduler.MultiStepLR(optimizer, milestones = config_u["milestones"], gamma = config_u["gamma_step"])
    scheduler_expo = optim.lr_scheduler.ExponentialLR(optimizer, gamma = config_u["gamma_expo"])

        # initialization
    start_epoch = 0

    losses = {"total_losses" : [], 
            "terminal_losses" : [],
            "penalties_neg_liab" : [],
            "penalties_neg_cash" : [],
            "penalties_bankrupt" : [],
            "val_total_losses": []
            }

    best_epoch = 0
    best_epoch_loss = None

    logger.info(f'****** Pre-training ******')
        
        # create a validation sample if activated
    if config_u["id_validate_sample"]:
        val_seed = args.seed + 1    # fix a seed for validation sample
        val_initial_states = pm.sample_initial_state(sample_size = config_u["batch_size"], param_dict = param_dict,
                                    sampling_dict = sampling_dict, device = device, seed = val_seed)
        val_brownians = pm.sample_brownian_trajectories(sample_size = config_u["batch_size"], param_dict = param_dict,
                                                device = device, seed = val_seed)
        
        # load checkpoint if needed 
    if config_u["checkpoint_path"] is not None :
        logger.info(f"Continued training - loading checkpoints from {config_u["checkpoint_path"]}.")
        if args.new_lr:     # replace schedulers and optimizer from the checkpoint by those created in-place
            logger.info("                   - Ignoring optimizer and learning rate from previous training(s).")
            start_epoch, additional_info = utils.load_checkpoint(config_u["checkpoint_path"], model)
        else :              # retain schedulers and optimizer from the checkpoint
            start_epoch, additional_info = utils.load_checkpoint(config_u["checkpoint_path"], model,
                                                                optimizer, scheduler_step, scheduler_expo)
        
        losses = additional_info    # keep the record from last training
        
        logger.info(f"Continued training - last checkpoint is at epoch = {start_epoch}.")
        epoch = start_epoch
    else : 
        logger.info(f"New training       - no checkpoint loaded, starting from scratch.")
        
        # pre-training log
    logger.info(f'Device : {device} and Seed : {args.seed}')
    logger.info('Hyper-parameters :')
    logger.info(f'  > n_asset = {param_dict["n_asset"]}, n_period = {param_dict["n_period"]}, ')
    logger.info(f'  > n_samples = {config_u["n_samples"]}, batch_size = {config_u["batch_size"]}, n_batches = {n_batches}, penalty_weight = {config_u["penalty_weight"]}, ')
    logger.info(f'    n_hidden_layers = {config_u["n_hidden_layers"]}, n_neuron = {config_u["n_neuron"]}, n_time_freqs = {config_u["n_time_freqs"]}, norm_type = {config_u["norm_type"]}')
    logger.info(f'  > (initial) lr = {config_u["lr"]}, weight_decay = {config_u["weight_decay"]}, gamma_expo (for exponential scheduler) = {config_u["gamma_expo"]},')
    logger.info(f'  > gamma_step (for step scheduler) = {config_u["gamma_step"]} at milestones = {config_u["milestones"]}.')
    logger.info(f'  > activation for hidden layers = {args.hidden_activation_type} and for output layer = {args.output_activation_type}')
    logger.info(f'  > resampling initial states at each epoch ? {config_u["id_resample_initial_state"]}')
    logger.info(f'  > resampling brownians at each epoch ? {config_u["id_resample_brownian"]}')
    logger.info(f'  > using a separate validation sample ? {config_u["id_validate_sample"]}')
    logger.info('****** Begin training ******')

        # starter
    start = time.time() # keep track of time

        # generate batch seeds to maintain coherence between epochs
    batch_seeds = torch.multinomial(torch.ones(100000), n_batches, replacement = False).to(int)

        # training loop
    for epoch in range(start_epoch, config_u["n_epoch"] + start_epoch):
            
        model.train()   # set model to training mode
        
            # create variables to keep track of the epoch losses
        epoch_total_loss = 0.0
        epoch_terminal_loss = 0.0
        epoch_penalty_neg_liab = 0.0
        epoch_penalty_neg_cash = 0.0
        epoch_penalty_bankrupt = 0.0

            # shuffle the seed
        idx = torch.randperm(batch_seeds.size(0))
        batch_seeds = batch_seeds[idx]

            # loop through batches
        for j in range(n_batches):

                # data for the batch
            logger.debug(f'Batch {j+1}/{n_batches}: batch_seed = {batch_seeds[j].item()}')
            temp_seed = batch_seeds[j].item()           # preset seed for the batch
                    # initial states
            if config_u["id_resample_initial_state"] :  # resampling initial states 
                temp_seed = torch.seed()    
                logger.debug(f'         - changing seed to {temp_seed} for resampling initial states.')
            batch_initial_states = pm.sample_initial_state(sample_size = config_u["batch_size"], 
                                                    param_dict = param_dict, 
                                                    sampling_dict = sampling_dict, 
                                                    device = device, 
                                                    seed = temp_seed)
                    # brownians
            if config_u["id_resample_brownian"] :       # resampling brownians
                temp_seed = torch.seed()
                logger.debug(f'         - changing seed to {temp_seed} for resampling brownians.')
            batch_brownians = pm.sample_brownian_trajectories(sample_size = config_u["batch_size"], 
                                                        param_dict = param_dict, 
                                                        device = device, 
                                                        seed = temp_seed)

                # reset the optimizer
            optimizer.zero_grad()

                # forward pass 
            batch_terminal_loss, batch_penalty_neg_liab, batch_penalty_neg_cash, batch_penalty_bankrupt, batch_history = LossAccumulative(model = model, 
                                                                                                                            param_dict = param_dict,
                                                                                                                            initial_states = batch_initial_states,
                                                                                                                            brownians = batch_brownians,
                                                                                                                            device = device,
                                                                                                                            id_save_states = (args.log_level == 'DEBUG'))   # only keep track if in debug mode
            batch_loss = batch_terminal_loss + config_u["penalty_weight"] * (batch_penalty_neg_liab + batch_penalty_neg_cash + batch_penalty_bankrupt)
            
                # back propagation
            batch_loss.backward()
            optimizer.step()

                # accumulate batch losses for the record (no need for back propagation here, just recording)
            epoch_total_loss        +=             batch_loss.item() / n_batches
            epoch_terminal_loss     +=    batch_terminal_loss.item() / n_batches
            epoch_penalty_neg_liab  += batch_penalty_neg_liab.item() / n_batches
            epoch_penalty_neg_cash  += batch_penalty_neg_cash.item() / n_batches
            epoch_penalty_bankrupt  += batch_penalty_bankrupt.item() / n_batches 

                # if in debug mode, print out state history to check coherence
            if args.log_level == 'DEBUG' and j == 1 and epoch == config_u["n_epoch"] + start_epoch - 1 : 
                logger.debug(f'         - terminal_loss_tensor (first 10 values) = {batch_history["loss"][:10, :]} ')
                logger.debug(f'         - penalties (first 10 values of all 3 types + first 10 periods) = {batch_history["penalties"][:10, :, :10]}')
            
        # keep track of epoch losses
        losses["total_losses"].append(epoch_total_loss)
        losses["terminal_losses"].append(epoch_terminal_loss)
        losses["penalties_neg_liab"].append(epoch_penalty_neg_liab)
        losses["penalties_neg_cash"].append(epoch_penalty_neg_cash)
        losses["penalties_bankrupt"].append(epoch_penalty_bankrupt)

        logger.info(f'Epoch [{epoch + 1}/{config_u["n_epoch"] + start_epoch}]: learning rate = {optimizer.param_groups[-1]['lr']}')
        logger.info(f'  + Training :    _ Total loss (with penalty weight) = {epoch_total_loss : .15f}, Terminal = {epoch_terminal_loss: .15f},')
        logger.info(f'                  _ Penalties : Negative Liability = {epoch_penalty_neg_liab: .15f}, Negative Cash = {epoch_penalty_neg_cash: .15f}, Bankruptcy = {epoch_penalty_bankrupt: .15f}.')

            # update learning rate
        scheduler_expo.step()
        scheduler_step.step()

            # validation 
        if config_u["id_validate_sample"] : # if using the validation sample
            model.eval()

            with torch.no_grad():
                val_terminal_loss, val_penalty_neg_liab, val_penalty_neg_cash, val_penalty_bankrupt, _ = LossAccumulative(model = model, 
                                                                                                                        initial_states = val_initial_states, 
                                                                                                                        brownians = val_brownians, 
                                                                                                                        device = device,
                                                                                                                        param_dict = param_dict, 
                                                                                                                        id_save_states=False)
                val_total_loss = val_terminal_loss + config_u["penalty_weight"] * (val_penalty_neg_liab + val_penalty_neg_cash + val_penalty_bankrupt)
                val_total_loss = val_total_loss.item()

            logger.info(f'  + Validation :  _ Total loss (with penalty weight) = {val_total_loss: .15f}, Terminal = {val_terminal_loss: .15f},')
            logger.info(f'                  _ Penalties : Negative Liability = {val_penalty_neg_liab: .15f}, Negative Cash = {val_penalty_neg_cash: .15f}, Bankruptcy = {val_penalty_bankrupt: .15f}.')

        else :                  # validate using training loss
            val_total_loss = epoch_total_loss
        
        losses["val_total_losses"].append(val_total_loss)

        if best_epoch_loss == None or best_epoch_loss > val_total_loss : 
            best_epoch = epoch 
            best_epoch_loss = val_total_loss 
            
            utils.save_checkpoint(model = model, optimizer = optimizer, scheduler_step = scheduler_step, scheduler_expo = scheduler_expo, 
                                epoch = epoch, basedir = checkpoint_dir, suffix='best',
                                additional_info = losses)
            logger.info(f'  >>>>> Found best model in epoch = {epoch + 1}, saving in {checkpoint_dir}')

    # closing
    end = time.time()
    logger.info(f'****** End training ______ Execution time = {(end - start)/60 : .2f} minutes + Best model found at epoch [{best_epoch + 1}/{start_epoch + config_u["n_epoch"]}].')

        
            
                


