'''
This script generates graphs to represent the optimal control on the viable domain (i.e. maximizing strategy).
    There is no training required.
'''
# ------------------------
# Packages
# ------------------------

import json, os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib import rc       # for text rendering in graphs
rc('text', usetex = True)

import importlib
pm = importlib.import_module('0_portfolio_model', package = None)
custom_nn = importlib.import_module('0_neural_networks', package=None)
utils = importlib.import_module('0_utils', package=None)

float_type = torch.float32

if __name__ == "__main__":
# ------------------------
# Parameters + Configuration
# ------------------------
    
    J = os.path.join    # alias for frequently used function 

    # create parser
    parser = ArgumentParser(description='Visualize : Trajectories of state process by optimal control network a_nn', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--portfolio', type=str, required=True, help='path to the config (.json) file for all portfolio model parameters, including the sampling space.')
    parser.add_argument('-w', '--config_w', type=str, required=True, help='path to the result folder of the trained domain boundary (w_NN).')
    parser.add_argument('-a', '--config_a', type=str, required=True, help='path to the result folder of the trained optimal control on the viable domain (a_NN).')

    parser.add_argument('-o', '--result_dir', type=str, default='./debug', help='directory to store results.')
    parser.add_argument('-s', '--seed', type=int, default=1851794, help='seed for the random number generator.')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='indicator for what type of information to include in the log.', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
   
    parser.add_argument('-n', '--n_paths', type=int, default=2, help='number of trajectories to plot (= 2 by default)')
    parser.add_argument('-d', '--descale_ind', type=int, default=1, help='indicator on whether to descale values for all variables (state + control) - 0 = no descale, 1 = descale; 0 by default')

    args = parser.parse_args()
    
    # create result directory + logger 
    result_dir = utils.create_dir(basedir = args.result_dir, dirname = 'visual_a', suffix = None)
    logger = utils.setup_logging(result_dir, log_level = args.log_level, fname = 'visual_a.log')

        # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # seed
    torch.manual_seed(args.seed)
    logger.info(f'Device : {device} and Seed : {args.seed}')

        # read parameters 
    param_dict, sampling_dict, scaling_dict = pm.read_model_param(args.portfolio, device = device)

        # load the trained w_NN for the domain boundary
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

        # load the trained a_NN for the optimal control
    a_nn_dir = args.config_a
    config_a = json.load(open(J(a_nn_dir, 'config_a.json'), 'r'))
    a_dir_list = [J(a_nn_dir, 'checkpoints', f) for f in os.listdir(J(a_nn_dir, 'checkpoints'))]
    a_nn_checkpoint_path = max(a_dir_list, key=os.path.getmtime)

    a_NN = custom_nn.Optimal_Control_Net(dim_state = param_dict["dim_state"], 
                        dim_control = param_dict["dim_control"], dim_sto = param_dict["dim_sto"],
                        dim_hidden = config_a["n_neuron"], n_hidden_layers = config_a["n_hidden_layers"],
                        num_time_freqs = config_a["n_time_freqs"], norm_type = config_a["norm_type"],
                        hidden_activation_type = config_a["hidden_activation_type"], output_activation_type = config_a["output_activation_type"],
                        lower_bounds = param_dict["control_lower_bound"], upper_bounds = param_dict["control_upper_bound"],
                        martingale_control_limit = config_a["martingale_control_limit"],
                        device = device
                        )
    a_NN.to(device)
    a_NN.to(torch.float32)
    a_NN.load_state_dict(torch.load(a_nn_checkpoint_path, map_location=device)["model_state_dict"])
    a_NN.eval()
    
# ------------------------
# Data simulation
# ------------------------

    n_simu = args.n_paths               # number of trajectories to simulate
    n_period = param_dict["n_period"]   # number of periods in the time horizon
    d = param_dict["n_asset"]           # number of asset in the portfolio
    constraint_buffer = config_a['constraint_sample_range'] * 0.2

    logger.info(f'n_period = {n_period}, n_simu = {n_simu}, constraint_buffer = {constraint_buffer}')

    with torch.no_grad():
        # generate trajectories 
        t_hist, X_hist, P_hist, u_hist, _ = pm.simulate_aug_trajectories(
                                                    sample_size = n_simu, device = device,
                                                    aug_control_NN = a_NN, domain_boundary_NN = w_NN,
                                                    param_dict = param_dict, sampling_dict = sampling_dict,
                                                    constraint_sample_range = constraint_buffer,
                                                    seed_initial_state = args.seed, seed_brownian = args.seed,
                                                    float_type = float_type
                                                    )

    # common setting for plots
    colors = {
        'price' : 'blue',
        'quantity' : 'red',
        'liability' : 'purple',
        'cash' : 'green',
        'profit_sharing' : 'orange',
        'domain_boundary' : 'maroon',
        'martingale' : 'forestgreen'
        }
    if args.descale_ind == 1 : 
        units = {
                'price': r'\texteuro', 
                'trade_intensity': 'units', 
                'quantity': 'units', 
                'profit_sharing': r'\%',
                'liability':r'\texteuro', 
                'cash': r'\texteuro', 
                'wealth': r'\texteuro',
                'martingale': r'\texteuro',
                'domain_boundary': r'\texteuro'
                }
    else : 
        units = {
                'price': r'1e3 \texteuro', 
                'trade_intensity': '1e3 units', 
                'quantity': '1e3 units', 
                'profit_sharing': r'\%',
                'liability':r'1e6 \texteuro', 
                'cash': r'1e6 \texteuro', 
                'wealth': r'1e6 \texteuro',
                'martingale': r'1e6 \texteuro',
                'domain_boundary': r'1e6 \texteuro'
                }
    plot_titles = [
                r'Martingale $P$ and domain boundary $w$',
                r'Price of Asset $S$ and Trade Intensity $\dot\phi$',
                r'Liability $L$ and Profit Sharing Proportion $\pi$',
                'Composition of Wealth'
                ]
    
    fontsizes = {'suptitle' : 12, 'subplot_title': 12, 'legend' : 8, 'axis_label' : 8}

    x_axis_state = np.arange(0, n_period + 1)
    x_axis_control = np.arange(0, n_period)

    if args.descale_ind :
        book_value_scalar = param_dict["book_value"].item() / scaling_dict["price"] 

        trade_intensity_max_scalar = param_dict["trading_intensity_min"] / scaling_dict["quantity"]
        trade_intensity_min_scalar = param_dict["trading_intensity_max"] / scaling_dict["quantity"]

        profit_sharing_max_scalar = param_dict["profit_sharing_max"] * 100
        profit_sharing_min_scalar = param_dict["profit_sharing_min"] * 100
        profit_sharing_mid_point = ( profit_sharing_max_scalar + profit_sharing_min_scalar )/2
        profit_sharing_dev = ( profit_sharing_max_scalar - profit_sharing_min_scalar ) * 1.2/2

    # plot sample by sample 
    for j in range(n_simu):
        
        fig, ax = plt.subplots(nrows = 4, ncols = 1, figsize = (6, 10), sharex = True, sharey = False)

        # data
        with torch.no_grad():
            t = t_hist[j, :, :]             # dim = (n_period + 1, 1)
            X = X_hist[j, :, :]             # dim = (n_period + 1, dim_state)
            w = w_NN(t, X)                  # dim = (n_period + 1, 1)
            
            P = P_hist[j, :, :]             # dim = (n_period + 1, 1)
            u = u_hist[j, :, :]             # dim = (n_period, dim_control)

        if args.descale_ind :
            X[:, 0:d] = torch.mul(X[:, 0:d], 1/scaling_dict["price"])
            X[:, d:2*d] = torch.mul(X[:, d:2*d], 1/scaling_dict["quantity"])
            X[:, 2*d:2*d+1] = torch.mul(X[:, 2*d:2*d+1], 1/scaling_dict["cash"])
            X[:, 2*d+1:2*d+2] = torch.mul(X[:, 2*d+1:2*d+2], 1/scaling_dict["liability"])

            u[:, 0:d] = torch.mul(u[:, 0:d], 1/scaling_dict["quantity"])    # trade intensity has the same scaling as quantity
            u[:, -1] = torch.mul(u[:, -1], 100)                             # this is a percentage

            w = w / scaling_dict["cash"]    # w is an estimate of loss = liability - cash - price * quantity, so w has the same scale as cash and liability
            P = P / scaling_dict["cash"]    # P is on the same scale as w

        non_cash_asset = torch.mul(X[:, 0:d], X[:, d:2*d]).sum(dim = 1) # non-cash asset
        total_asset = non_cash_asset + X[:, 2*d]                        # total asset = cash + non-cash asset 

        X = X.detach().numpy()
        w = w.detach().squeeze().numpy()    # dim = (n_period + 1,)
        P = P.detach().squeeze().numpy()    # dim = (n_period + 1,)

        # first plot : domain boundary vs martingale
            # domain boundary
        domain_boundary_plot = ax[0].plot(x_axis_state, w, label = r'Domain Boundary $w(t, X_t)$',
                                        color = colors['domain_boundary'], ls = 'solid', lw = 0.75)
            # martingale
        martingale_plot = ax[0].plot(x_axis_state, P, label = r'Martingale $P_t$',
                                        color = colors['martingale'], ls = 'solid', lw = 0.75)
            # combined
        ax[0].set_title(plot_titles[0], pad=25, fontsize = fontsizes['subplot_title'])
        ax[0].grid(True, alpha = 0.5)
        ax[0].legend(loc='lower center', ncol = 2, 
                    bbox_to_anchor = (0.5, 1.01), fontsize = fontsizes['legend'])

        # second plot : price and trade intensity
            # price 
        price = X[:, 0]                     # dim = (n_period +1, )
        price_plot = ax[1].plot(x_axis_state, price, label = r'Asset price $S_t$',
                                        color = colors['price'], ls = 'solid', lw = 0.75)
        ax[1].set_ylabel(f'Price ({units['price']})', 
                                        color = colors['price'], fontsize = fontsizes['axis_label'])
        book_value_plot = ax[1].axhline(y = book_value_scalar, label = r'Bookvalue $\title S$', 
                                        color = colors['price'], ls = 'dashed', lw = 0.75)
        ax[1].yaxis.set_label_position("left")
        ax[1].yaxis.tick_left()
        max_dev = max(abs(price - book_value_scalar)) * 1.2
        ax[1].set_ylim(ymin = book_value_scalar - max_dev, ymax = book_value_scalar + max_dev)

            # trade intensity
        trade_intensity = u[:, 0]           # dim = (n_period, )    - there is only 1 asset in this model
        ax_second = ax[1].twinx()
        trade_intensity_plot = ax_second.scatter(x_axis_control, trade_intensity, 
                                        label = r'Trade intensity $\dot \phi_t$',
                                        color = colors['quantity'], marker = 'o', 
                                        edgecolors = None, alpha = 1, s = 2)
        ax_second.axhline(y = trade_intensity_max_scalar, color = colors['quantity'], ls='dotted', lw=0.75)
        ax_second.axhline(y = trade_intensity_min_scalar, color = colors['quantity'], ls='dotted', lw=0.75)
        
        ax_second.set_ylabel(f'Trade Intensity ({units["trade_intensity"]})', 
                                        color = colors['quantity'], fontsize = fontsizes['axis_label'])
        ax_second.yaxis.set_label_position('right')
        ax_second.yaxis.tick_right()

            # combined 
        ax[1].set_title(plot_titles[1], pad = 25, fontsize = fontsizes['subplot_title'])
        ax[1].grid(True, alpha = 0.5)
        
        handles_left, labels_left = ax[1].get_legend_handles_labels()
        handles_right, labels_right = ax_second.get_legend_handles_labels()
        ax[1].legend(handles_left + handles_right,
                    labels_left + labels_right,
                    loc = "lower center", bbox_to_anchor = (0.5, 1.01),
                    ncol = 3, fontsize = fontsizes['legend'])
        
        # third plog : liability and profit sharing
            # liability
        liability = X[:, -1]                # dim = (n_period + 1,)
        liability_plot = ax[2].plot(x_axis_state, liability, label=r'Liability $L_t$',
                                        color = colors["liability"], ls='solid', lw=0.75)
        ax[2].set_ylabel(f'Liability ({units['liability']})', 
                                        color = colors['liability'], fontsize = fontsizes['axis_label'])
        ax[2].yaxis.set_label_position('left')
        ax[2].yaxis.tick_left()

            # profit sharing
        profit_sharing = u[:, -1]
        ax_second = ax[2].twinx()
        profit_sharing_plot = ax_second.scatter(x_axis_control, profit_sharing, 
                                        label = r'Profit sharing rate $\pi_t$',
                                        color = colors['profit_sharing'], marker = 'o',
                                        edgecolors = None, alpha = 1, s = 2)
        ax_second.axhline(y = profit_sharing_max_scalar, color = colors['profit_sharing'], ls = 'dotted', lw = 0.75)
        ax_second.axhline(y = profit_sharing_min_scalar, color = colors['profit_sharing'], ls = 'dotted', lw = 0.75)
        ax_second.set_ylabel(f'Profit sharing rate ({units['profit_sharing']})', 
                                        color = colors['profit_sharing'], fontsize = fontsizes['axis_label'])
        ax_second.yaxis.set_label_position("right")
        ax_second.yaxis.tick_right()
        ax_second.set_ylim(ymin = profit_sharing_mid_point - profit_sharing_dev, ymax = profit_sharing_mid_point + profit_sharing_dev)

            # combined 
        ax[2].set_title(plot_titles[2], pad = 25, fontsize = fontsizes['subplot_title'])
        ax[2].grid(True, alpha = 0.5)

        handles_left, labels_left = ax[2].get_legend_handles_labels()
        handles_right, labels_right = ax_second.get_legend_handles_labels()
        ax[2].legend(handles_left + handles_right,
                    labels_left + labels_right,
                    loc = "lower center", bbox_to_anchor = (0.5, 1.01),
                    ncol = 3, fontsize = fontsizes['legend'])
        
        # fourth plot : wealth decomposition
        cash = X[:, 2*d]
        non_cash_asset = non_cash_asset.detach().numpy()
        total_asset = total_asset.detach().numpy()

            # asset decomposition : stack plot 
        ax[3].stackplot(x_axis_state, cash, non_cash_asset,
                    labels = [r'Cash $\beta_t$', r'Non-cash asset $\phi_t S_t$'],
                    colors = [colors['cash'], 'yellow'], 
                    alpha = 0.2)
        
            # total asset vs liability
        ax[3].plot(x_axis_state, total_asset, label = 'Total Asset', 
                    color = 'red', ls = 'solid', lw = 0.8)
        ax[3].plot(x_axis_state, liability, label = 'Liability', 
                    color = colors['liability'], ls = 'solid', lw = 0.8)
        
            # combined
        ax[3].set_title(plot_titles[3], pad = 25, fontsize = fontsizes['subplot_title'])
        ax[3].grid(True, alpha = 0.5)
        ax[3].xaxis.set_tick_params(labelsize = fontsizes['axis_label'])
        ax[3].set_xlabel('Timestep', fontsize = fontsizes['axis_label'])
        ax[3].legend(loc = 'lower center', bbox_to_anchor = (0.5, 1.01),
                    ncol = 4, fontsize = fontsizes['legend'])
        
        # finalization + save plot
        initial_quantity = X[0, d].item()
        suptitle = 'Initial condition'
        suptitle += r': $S_0$' +'='                 + str(round(price[0], 2))           # initial price
        suptitle += r', $\phi_0$'+'='               + str(round(initial_quantity,2))    # initial quantity
        suptitle += r', $\beta_0$'+'='              + str(round(cash[0],2))             # initial cash
        suptitle += r', $L_0$'+'='                  + str(round(liability[0],2))        # initial liability
        suptitle += '\n Initial constraint: '+r'$p$' + '=' + str(round(M[0],2))         # initial constraint

        fig.suptitle(suptitle, fontsize = fontsizes['suptitle'])

        fig.tight_layout()

        file_name = f'visual_a_seed_{args.seed}_scenario_{j+1}.pdf'
        fig.savefig(J(result_dir, file_name))

        plt.close(fig)




