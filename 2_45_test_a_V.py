'''
This script is to test the training of the neural networks 
    a_NN, which estimates the (augmented) optimal control, and V_NN, which estimates the value function.

Note : this is specified for the current model with 1 stochastic factor = 1 risky asset.
'''

# ------------------------
# Packages
# ------------------------
import os, json, shutil, time, copy, math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Union, Literal, Optional
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import vmap

import importlib
pm = importlib.import_module('0_portfolio_model', package=None)
custom_nn = importlib.import_module('0_neural_networks', package=None)
utils = importlib.import_module('0_utils', package=None)

# ------------------------
# Support / Loss functions
# ------------------------

def generate_testing_data(
        aug_control_NN,                 # trained neural network (a_NN) to estimate the augmented optimal control
        domain_boundary_NN,             # trained neural network (w_NN) to estimate the domain boundary

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
                                                    aug_control_NN = aug_control_NN, 
                                                    domain_boundary_NN = domain_boundary_NN,
                                                    param_dict = param_dict, 
                                                    sampling_dict = sampling_dict,
                                                    constraint_sample_range = constraint_sample_range, 
                                                    device = device,
                                                    seed_initial_state = seed_initial_state, 
                                                    seed_brownian = seed_brownian,
                                                    sample_size = sample_size, 
                                                    float_type = float_type
                                                    )
    t_initial = t_hist[:, 0, :].reshape((sample_size, 1)).to(device)
    X_initial = X_hist[:, 0, :].reshape((sample_size, dim_state)).to(device)
    P_initial = P_hist[:, 0, :].reshape((sample_size, 1)).to(device)
    initial_data_dict = {"t": t_initial, "X": X_initial, "P": P_initial}

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
    t_terminal = t_hist[:, -1, :].reshape((sample_size, 1)).to(device)              # dim = (sample_size, 1)
    X_terminal = X_hist[:, -1, :].reshape((sample_size, dim_state)).to(device)      # dim = (sample_size, dim_state)
    P_terminal = P_hist[:, -1, :].reshape((sample_size, 1)).to(device)              # dim = (sample_size, 1)
    terminal_data_dict = {"t": t_terminal, "X": X_terminal, "P": P_terminal}

    return initial_data_dict, interior_data_dict, terminal_data_dict
    
def compute_error_interior(     # Note that in function is specific to the current version of the model (only 1 asset = 1 stochastic factor, min_control_martingal = - max_control_martingale)
        value_func_NN,                  # trained neural network (V_NN) to estimate the value function
       
        t_hist_inner : torch.Tensor,    # trajectories of t where t < T                 - dim = (sample_size, 1)
        X_hist_inner : torch.Tensor,    # trajectories of X_t where t < T               - dim = (sample_size, dim_state)
        P_hist_inner : torch.Tensor,    # trajectories of P_t where t < T               - dim = (sample_size, 1)

        u_hist : torch.Tensor,          # trajectories of control u_t for state X_t     - dim = (sample_size, dim_control)
        a_hist : torch.Tensor,          # trajectories of control a_t for martingale P_t- dim = (sample_size, dim_sto)

        device : torch.device,          # device
        param_dict : dict,              # dictionary for parameters pertaining to the portfolio model
        
        martingale_control_limit : float,               # limit for the martingale control
        constraint_penalty_weight : float = 1.0,        # weight for constraint penalty (from training u_NN and w_NN)

        id_keep_tensor : bool = False,  # indicateur of whether tensors are to be retained
        id_extra_info : bool = True     # indicateur of whether extra information (statistics on the computation) will be given
    ):

    '''
    Output :
        error V = avg( abs( H_{u_nn, a_nn} V) ) where H_{u_nn,a_nn} V = PDE residual evaluated at control (u_nn, a_nn) estimated by the aug_control_NN
        error a = avg( abs( H_{u_nn, a_nn} V - H^* V )) where H^* V = max H_{u,a} V with (u,a) from the authorized control range 
    '''

    pm.send_to_device([t_hist_inner, X_hist_inner, P_hist_inner, u_hist, a_hist], device)
    value_func_NN.to(device)
    sample_size = X_hist_inner.shape[0] 

    if id_keep_tensor : 
        tensor_dict = {}

    # (shared) computation component 
        # gradients and hessians
    X_P_hist_inner = torch.cat([X_hist_inner, P_hist_inner], 
                        dim = 1).to(device)             # dim = (sample_size, dim_state + 1)
    dV_t, dV_XP, dV_2_XP = utils.compute_deriv_aug(NN = value_func_NN,
                        device = device, t = t_hist_inner, X_P = X_P_hist_inner)
    dV_t = dV_t.squeeze().to(device)                    # dim = (sample_size, )
    dV_X = dV_XP[:, :-1].to(device)                     # dim = (sample_size, dim_state)
    dV_P = dV_XP[:, -1].to(device)                      # dim = (sample_size, )

        # penalty
    _, _, _, penalties = pm.boundary_penalty(X = X_hist_inner, param_dict = param_dict,
                            device = device, id_keep_tensor = True)
    constraint_penalties = penalties.sum(dim = 1).to(device)    # dim = (sample_size, )
    if id_keep_tensor :
        tensor_dict["penalties"] = constraint_penalties
    constraint_penalties = constraint_penalty_weight * constraint_penalties
    penalty_term = - torch.mul(dV_P, constraint_penalties).to(device)

        # volatility
    vols = pm.compute_vol(X = X_hist_inner,                     
                    param_dict = param_dict, device = device)   # dim = (sample_size, dim_state, dim_sto)

    # support function : computation of the HV operator with any given controls (u,a)
    def HV_operator(
            u : torch.Tensor,                           # dim = (sample_size, dim_control)
            a : torch.Tensor,                           # dim = (sample_size, dim_sto)
        ) -> torch.Tensor:

        # drift term 
        drift = pm.compute_drift(X = X_hist_inner, u = u,
                            param_dict = param_dict, device = device)
        drift_T = drift.unsqueeze(1).to(device)         # dim = (sample_size, 1, dim_state)
        drift_term = torch.bmm(drift_T, dV_X.unsqueeze(2)).squeeze().to(device) 

        # diffusion term
        aug_vols = torch.cat([vols, a.unsqueeze(1).to(device)],
                            dim = 1).to(device)         # dim = (sample_size, dim_state + 1, dim_sto)
        aug_variances = torch.bmm(aug_vols, torch.transpose(aug_vols, 1, 2)).to(device)
        diffusion = torch.bmm(aug_variances, dV_2_XP).to(device)    # dim = (sample_size, dim_state + 1, dim_state + 1)
        diffusion_term = 0.5 * torch.vmap(torch.trace)(diffusion).to(device)

        # operator 
        dynkins = dV_t + drift_term + diffusion_term    # without penalty term
        if id_keep_tensor:
            tensor_dict["dynkins_without_penalty"] = copy.deepcopy(dynkins)
 
        dynkins = dynkins + penalty_term                # add penalty term

        return dynkins.to(device)                       # dim = (sample_size, )
    
    # HV with the estimated controls by aug_control_NN
    dynkins_NN = HV_operator(u = u_hist, a = a_hist)    # dim = (sample_size, )

    # HV optimized (i.e. maximized)   
        # find maximizer a* of A * a^2 + 2 * B * a where A = dV_2_PP (scalar) and B = dV_dP_dX * sigma (scalar if dim_sto = 1) 
    dV_2_PP = dV_2_XP[:, -1, -1].to(device)             # dim = (sample_size, )
    A = dV_2_PP

    N = torch.full((len(A), ),                          # dim = (sample_size, )
            fill_value = abs(martingale_control_limit)).to(device)

    dV_dP_dX = dV_2_XP[:, -1, :-1].to(device)           # dim = (sample_size, dim_state)
    B = torch.bmm(dV_dP_dX.unsqueeze(1).to(device), 
                    vols).to(device)                    # dim = (sample_size, 1, dim_sto)
    if param_dict["dim_sto"] != 1 :
        raise ValueError(f' -- Code is currently not implemented for dim_sto = {param_dict["dim_sto"]}')
    else :  # dim_sto = 1
        B = B.squeeze().to(device)                      # dim = (sample_size)
        
            # case 1 : if A < 0, then concave -> maximizer at critical point = - B/A
        id_case_1 = torch.where(A < 0, 1, 0)

            # case 2 : if A >= 0 and B < 0 (either convex "leaning" right or decreasing linear) -> maximizer at lower end of range = - N
        id_case_2 = torch.where(A < 0, 0, 1) * torch.where(B < 0, 1, 0)

            # case 3 : if A >= 0 and B >= 0 (either convex "leaning" left or increasing linear) -> maximizer at upper end of range = N
        id_case_3 = torch.where(A < 0, 0, 1) * torch.where(B < 0, 0, 1)

        pm.send_to_device([id_case_1, id_case_2, id_case_3], device)
        assert id_case_1 + id_case_2 + id_case_3 == torch.ones((sample_size, )).to(device),f'Dimension Mismatch : please check the factorization of cases for finding a maximizer'               
        
        a_maximizer = torch.nan_to_num(-B/A, nan=0.0) * id_case_1 + (-N) * id_case_2 + N * id_case_3
        a_maximizer = a_maximizer.reshape((sample_size, param_dict["dim_sto"])).to(device)

        # find maximizer u* : check difference permutation of points at the corners of the feasible range
    trade_intensity_candidates = (param_dict["trading_intensity_in"], 0, param_dict["trading_intensity_ax"])
    profit_sharing_candidates = (param_dict["profit_sharing_min"], param_dict["profit_sharing_max"])
    u_list = [trade_intensity_candidates for i in range(param_dict["dim_control"] - 1)]
    u_list.append(profit_sharing_candidates)            # dim = 3 * (dim_control - 1) + 2 * 1

    u_candidates = list(product(*u_list))               # dim = [3^(dim_control - 1)] * 2  
    u_candidates_tensor = torch.Tensor(u_candidates).to(device) # dim = (3^(dim_control - 1) * 2, dim_control)
    n_candidates = u_candidates_tensor.shape[0]

    dynkins_candidates = torch.empty((sample_size, n_candidates), device = device)
    for i in range(n_candidates):
        u_candidate_i = u_candidates_tensor[i,:].unsqueeze(0).repeat(sample_size, 1).to(device)

        dynkins_candidate_i = HV_operator(u = u_candidate_i,    # dim = (sample_size, dim_control)
                                    a = a_maximizer)            # dim = (sample_size, dim_sto)
        dynkins_candidates[:, i] = dynkins_candidate_i          # dim = (sample_size, )

        # optimal value of the operator
    dynkins_optimal = dynkins_candidates.max(dim = 1).values    # dim = (sample_size, )
    dynkins_optimal = dynkins_optimal.to(device)

    if id_keep_tensor:
        tensor_dict.update({'dynkins_NN': dynkins_NN, 'dynkins_optimal': dynkins_optimal})

    error_control = torch.abs(dynkins_optimal - dynkins_NN).mean().item()
    error_value_func = torch.abs(dynkins_NN).mean().item()

    if id_extra_info:
        error_optim = torch.abs(dynkins_optimal).mean().item()
    else :
        error_optim = None

    if not id_keep_tensor :
        tensor_dict = None

    return error_control, error_value_func, error_optim, tensor_dict

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
    parser.add_argument('-V', '--config_V', type=str, required=True, help='path to the result folder of the trained value function on the boundary (V_NN).')

    parser.add_argument('-o', '--result_dir', type=str, default='./results/5_Vnn', help='directory to store results.')
    parser.add_argument('-s', '--seed', type=int, default=1851794, help='seed for the random number generator.')
    parser.add_argument('-n', '--n_samples', type=int, default = 1000, help='total sample size for the test.')
    parser.add_argument('-b', '--batch_size', type=int, default = 100, help='batch size to break the computation into manageable chunks, if necessary.')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='indicator for what type of information to include in the log.', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
   
    args = parser.parse_args()

    # create the result directory and a logger
    result_dir = utils.create_dir(basedir = args.result_dir, dirname = 'test_a_V', suffix = None)
    checkpoint_dir = utils.create_dir(basedir = result_dir, dirname = 'checkpoints', suffix = '')
    logger = utils.setup_logging(result_dir, log_level = args.log_level, fname = 'test_a_V.log')

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

        # load trained V_NN for the value function on the entire domain
    V_nn_dir = args.config_V
    config_V = json.load(open(J(V_nn_dir, 'config_V.json'), 'r'))
    V_dir_list = [J(V_nn_dir, 'checkpoints', f) for f in os.listdir(J(V_nn_dir, 'checkpoints'))]
    V_nn_checkpoint_path = max(V_dir_list, key=os.path.getmtime)

    V_NN = custom_nn.Value_Function_Net(
                            dim_state = param_dict["dim_state"],
                            dim_hidden = config_V["n_neuron"],
                            n_hidden_layers = config_V["n_hidden_layers"],
                            num_time_freqs = config_V["n_time_freqs"],
                            hidden_activation_type = config_V["hidden_activation_type"],
                            norm_type = config_V["norm_type"]
                        )
    V_NN.to(device)
    V_NN.to(torch.float32)
    V_NN.load_state_dict(torch.load(V_nn_checkpoint_path, map_location=device)["model_state_dict"])
    V_NN.eval()

        # save information for replication, if needed
    args_dict = vars(args)
    args_dict["config_w"] = os.path.abspath(args.config_w)
    args_dict["config_Vb"] = os.path.abspath(args.config_Vb)
    args_dict["config_a"] = os.path.abspath(args.config_a)
    args_dict["config_V"] = os.path.abspath(args.config_V)
    args_dict["result_dir"] = os.path.abspath(args.result_dir)
    json.dump(args_dict, open(f'{result_dir}/args_V.json', 'w'), indent=4)

    script_name = os.path.basename(__file__)    # save the training script (keeping track of version)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(result_dir, script_name))

# ------------------------
# Computation
# ------------------------

    # Pre-computation
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_batches = n_samples // batch_size

        # log before test
    logger.info(f'****** Pre-testing ******')
    logger.info(f'w_NN from {w_nn_dir} \nVb_NN from {Vb_nn_dir} \na_NN from {a_nn_dir} \nV_NN from {V_nn_dir}')
    logger.info(f'Device : {device} and Seed : {args.seed}')
    logger.info(f' > n_samples = {n_samples} = {n_batches} batches x {batch_size} samples per batches')

    logger.info(f'****** Begin testing ******')

        # initiation
    start = time.time()
    error_control = 0.0
    error_value_func = 0.0
    error_boundary = 0.0
    error_terminal = 0.0
    avg_value_func = 0.0

        # seed 
    batch_seeds = torch.multinomial(torch.ones(100000), n_batches, replacement = False).to(int)
    idx = torch.randperm(batch_seeds.size(0))
    batch_seeds = batch_seeds[idx]              # shuffle the seeds

    # Loop through the batches
    for j in range(n_batches):
        logger.info(f'Batch {j+1}/{n_batches}:')

        # generate data
        logger.debug(f' _ seed = {batch_seeds[j].item()}')

        batch_initial_dict, batch_before_terminal_dict, batch_terminal_dict = generate_testing_data(
                                            aug_control_NN = a_NN,
                                            domain_boundary_NN = w_NN,
                                            param_dict = param_dict, 
                                            sampling_dict = sampling_dict,
                                            constraint_sample_range = config_a["constraint_sample_range"],
                                            device = device,
                                            seed_initial_state = batch_seeds[j].item(),
                                            seed_brownian = batch_seeds[j].item(),
                                            sample_size = batch_size
                                                )
        logger.debug(f' _ shape : before_terminal X = {batch_before_terminal_dict["X"].shape}')
        logger.debug(f'           initial X = {batch_initial_dict["X"].shape}')
        logger.debug(f'           terminal X = {batch_terminal_dict["X"].shape}')
        
        # compute the error components
            # error in the interior
        batch_error_control, batch_error_value_func, batch_error_optim, batch_tensor_dict = compute_error_interior(
                                            value_func_NN = V_NN,
                                            t_hist_inner = batch_before_terminal_dict["t"],
                                            X_hist_inner = batch_before_terminal_dict["X"],
                                            P_hist_inner = batch_before_terminal_dict["P"],
                                            u_hist = batch_before_terminal_dict["u"],
                                            a_hist = batch_before_terminal_dict["a"],
                                            device = device, param_dict = param_dict,
                                            martingale_control_limit = config_a["martingale_control_limit"],
                                            constraint_penalty_weight = 1.0
                                                )
        error_control    += batch_error_control     / n_batches
        error_value_func += batch_error_value_func  / n_batches

        logger.info(f' _ batch error : \n   + control error = {batch_error_control: .10f} \n   + value func error = {batch_error_value_func: .10f}')
        logger.debug(f'   + optim error = {batch_error_optim: .10f}')

            # error on the boundary
        targets_b = Vb_NN(batch_before_terminal_dict['t'], batch_before_terminal_dict['X'])
        predictions_b = V_NN(batch_before_terminal_dict['t'], batch_before_terminal_dict['X'], batch_before_terminal_dict['w'])
        batch_error_boundary = torch.abs(predictions_b - targets_b).mean().item()
        error_boundary += batch_error_boundary      / n_batches

        logger.info(f'   + boundary error = {batch_error_boundary: .10f}')

            # error at the terminal
        _, targets_t = pm.terminal_utility(X = batch_terminal_dict["X"], param_dict = param_dict,
                                            id_keep_tensor = True)
        predictions_t = V_NN(batch_terminal_dict["t"], batch_terminal_dict["X"], batch_terminal_dict["P"])
        batch_error_terminal = torch.abs(predictions_t - targets_t).mean().item()

        error_terminal += batch_error_terminal      / n_batches

        logger.info(f'   + terminal error = {batch_error_terminal: .10f}')

            # denominator
        batch_value_func = V_NN(batch_initial_dict["t"], batch_initial_dict["X"], batch_initial_dict["P"])
        avg_batch_V = batch_value_func.mean().item()

        avg_value_func += avg_batch_V               / n_batches

        logger.info(f' _ batch average value function = {avg_batch_V: .10f}')

    # Final computation (for the error measure for the entire sample)
    logger.info(f'\n >>>>> Global level computation :')
        # numerator = total error
    total_error = error_control + error_value_func + error_boundary + error_terminal
    logger.info(f' _ Total error = {total_error: .10f}')
    logger.info(f'   + error_control = {error_control:.10f}\n   + error_value_func = {error_value_func: .10f}')
    logger.info(f'   + error_boundary = {error_boundary: .10f}\n   + error_terminal = {error_terminal: .10f}')
        
        # denominator = (avg) value function at t = 0
    logger.info(f' _ Avg value function at t=0, i.e. V(t_0, X_0, P_0) = {avg_value_func}')

        # error measure
    error_measure = total_error / avg_value_func
    logger.info(f' _ Error measure = {error_measure: .10f}, equivalent of {100 * error_measure: .8f}%')

    # Closing
    logger.info(f'****** End testing ______ Execution time = {(time.time() - start)/60 : .2f} minutes.')
    
    
    

    