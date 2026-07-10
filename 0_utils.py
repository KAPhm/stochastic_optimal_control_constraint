# ------------------------
# Packages
# ------------------------

import time, os, logging, sys
from scipy.optimize import minimize_scalar

import torch
import torch.nn as nn
from torch.func import jacrev, jacfwd, vmap, functional_call
from torch.func import hessian, grad


# ------------------------
# Support functions for computation
# ------------------------

    # Function to compute derivatives using vmap (used for big batch) for w_NN and Vb_NN
def compute_deriv(
        NN,                         # neural network to be evaluated
        t : torch.Tensor,           # a sample of time variable     - dim = (n_samples, 1) or dim = (n_samples, )
        Y : torch.Tensor,           # a sample of spatial variable  - dim = (n_samples, dim_Y) 
        device : torch.device
        ):
    '''
    Notes : _ this is the method similar to the one used in old training scripts, 
                but instead of using autograd.grad, this one uses grad and hessian (from torch.func)
    Outputs :
        dNN_t : gradient of NN wrt time dimension t     - dim = (n_samples, 1)
        dNN_Y : gradient of NN wrt space dimension Y    - dim = (n_samples, dim_Y)
        dNN_YY : hessian of NN wrt space dimension Y    - dim = (n_samples, dim_Y, dim_Y)
    '''
        # put NN in eval mode + send to device
    is_training = NN.training   # original mode
    NN.eval()                   # switch to eval mode
    NN.to(device)           

        # prepare the inputs
    t = t.squeeze()                 # dim = (n_samples, )
    send_to_device([t,Y], device)   # send inputs to the same device
    
        # create a function equivalent for 1 data point
    def make_NN_single(NN, NN_params : dict, NN_buffers : dict):
        '''
        model_params : learnable parameters of the model
        model_buffers : non-learnable parameters of the model
         > these are needed for functional_call
        '''
        def NN_single(t_single : torch.Tensor, Y_single: torch.Tensor) -> torch.Tensor:
            t_single = t_single.squeeze().to(device)   # dim = ()

            t = t_single.unsqueeze(0).to(device)       # dim = (1, )
            Y = Y_single.unsqueeze(0).to(device)       # dim = (1, dim_Y)
            estimate = functional_call(NN, (NN_params, NN_buffers), (t, Y))
            return estimate.squeeze().to(device)       # dim = ()
        return NN_single

        # create a function that compute derivatives for 1 data point
    def compute_deriv_func(NN, NN_params : dict, NN_buffers : dict):
        NN_func = make_NN_single(NN, NN_params, NN_buffers)
        
        dNN_dt_func = grad(NN_func, argnums=0)
        dNN_dY_func = grad(NN_func, argnums=1)
        dNN_dYY_func = hessian(NN_func, argnums=1)

        def deriv_single(t_single, Y_single):
            ''' reminder :  t_single has dim = (1, ) and Y_single has dim = (dim_Y, ) '''
            t_single = t_single.squeeze().to(device)               # dim = ()

            dNN_dt = dNN_dt_func(t_single, Y_single).to(device)    # dim = ()
            dNN_dY = dNN_dY_func(t_single, Y_single).to(device)    # dim = (dim_Y, )
            dNN_dYY = dNN_dYY_func(t_single, Y_single).to(device)  # dim = (dim_Y, dim_Y)

            return dNN_dt, dNN_dY, dNN_dYY
        
        return deriv_single
    
        # compute derivatives in batch using vmap
    NN_params = dict(NN.named_parameters())
    NN_buffers = dict(NN.named_buffers())
    deriv_func = compute_deriv_func(NN, NN_params, NN_buffers)
    batched_derv_func = vmap(deriv_func, in_dims=(0,0))

    dNN_dt, dNN_dY, dNN_dYY = batched_derv_func(t, Y)
    dNN_dt = dNN_dt.to(device)          # dim = (n_samples, )
    dNN_dY = dNN_dY.to(device)          # dim = (n_samples, dim_Y)
    dNN_dYY = dNN_dYY.to(device)        # dim = (n_samples, dim_Y, dim_Y)

        # set NN back to original mode
    NN.train(is_training)

    return dNN_dt, dNN_dY, dNN_dYY

# Function to compute derivatives using vmap (used for big batch) for V_NN
def compute_deriv_aug(
        NN,                     # neural network to be evaluated
        t : torch.Tensor,       # a sample of time variable             - dim = (n_samples, 1)
        X_P : torch.Tensor,     # a sample of augmented state variable  - dim = (n_samples, dim_state + 1) 
        device : torch.device
    ):
    '''
    Note : Y = X_P in this function
    Outputs :
        dNN_t : gradient of NN wrt time dimension t                 - dim = (n_samples, 1)
        dNN_XP : gradient of NN wrt augmented space dimension X_P   - dim = (n_samples, dim_state + 1)
        dNN_2_XP : hessian of NN wrt augmented space dimension X_P  - dim = (n_samples, dim_state + 1, dim_state + 1)
    '''
        # put NN in eval mode + send to device
    is_training = NN.training   # original mode
    NN.eval()                   # switch to eval mode
    NN.to(device)           

        # prepare the inputs
    t = t.squeeze()                 # dim = (n_samples, )
    send_to_device([t, X_P], device)# send inputs to the same device
    
        # create a function equivalent for 1 data point
    def make_NN_single(NN, NN_params : dict, NN_buffers : dict):
        '''
        model_params : learnable parameters of the model
        model_buffers : non-learnable parameters of the model
         > these are needed for functional_call
        '''
        def NN_single(t_single : torch.Tensor, X_P_single : torch.Tensor) -> torch.Tensor:
            t_single = t_single.squeeze().to(device)    # dim = ()

            t = t_single.unsqueeze(0).to(device)        # dim = (1, )
            X = X_P_single[:-1].unsqueeze(0).to(device) # dim = (1, dim_state)
            P = X_P_single[-1].unsqueeze(0).to(device)  # dim = (1, )

            estimate = functional_call(NN, (NN_params, NN_buffers), (t, X, M))
            return estimate.squeeze().to(device)       # dim = ()
        
        return NN_single

        # create a function that compute derivatives for 1 data point
    def compute_deriv_func(NN, NN_params : dict, NN_buffers : dict):
        NN_func = make_NN_single(NN, NN_params, NN_buffers)
        
        dNN_dt_func = grad(NN_func, argnums=0)
        dNN_dY_func = grad(NN_func, argnums=1)
        dNN_dYY_func = hessian(NN_func, argnums=1)

        def deriv_single(t_single, Y_single):
            ''' reminder :  t_single has dim = (1, ) and Y_single has dim = (dim_Y, ) '''
            t_single = t_single.squeeze().to(device)               # dim = ()

            dNN_dt = dNN_dt_func(t_single, Y_single).to(device)    # dim = ()
            dNN_dY = dNN_dY_func(t_single, Y_single).to(device)    # dim = (dim_Y, )
            dNN_dYY = dNN_dYY_func(t_single, Y_single).to(device)  # dim = (dim_Y, dim_Y)

            return dNN_dt, dNN_dY, dNN_dYY
        
        return deriv_single
    
        # compute derivatives in batch using vmap
    NN_params = dict(NN.named_parameters())
    NN_buffers = dict(NN.named_buffers())
    deriv_func = compute_deriv_func(NN, NN_params, NN_buffers)
    batched_derv_func = vmap(deriv_func, in_dims=(0,0))

    dNN_dt, dNN_dY, dNN_dYY = batched_derv_func(t, X_P)
    dNN_dt = dNN_dt.to(device)          # dim = (n_samples, )
    dNN_dY = dNN_dY.to(device)          # dim = (n_samples, dim_Y)
    dNN_dYY = dNN_dYY.to(device)        # dim = (n_samples, dim_Y, dim_Y)

        # set NN back to original mode
    NN.train(is_training)

    return dNN_dt, dNN_dY, dNN_dYY
    
# ------------------------
# Support functions for logistics (device, logger, etc)
# ------------------------

    # Function to sent tensors to device
def send_to_device(tensor_list, device):
    for tensor in tensor_list :
        tensor = tensor.to(device)

    # Function to create timestamp
def timestamp():
    return time.strftime('%Y%m%d_%H%M%S')

    # Function to check dimension
def dimension_check(list_label, list_expected_value, list_actual_value):
    assert len(list_actual_value) == len(list_expected_value) and len(list_label)==len(list_actual_value), f'Check the input of the function dimension_check!'
    
    for i in range(len(list_expected_value)):
        assert list_expected_value[i] == list_actual_value[i], f'Dimension Mismatch : expectd value for {list_label[i]} to be {list_expected_value[i]} but currently it is {list_actual_value[i]}.'

    # Function to create directory with timestamp
def create_dir(basedir='./', dirname = "results", suffix = None):
    if suffix is None:
        suffix = "_" + timestamp()

    dir_path = os.path.join(basedir, f"{dirname}{suffix}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

    # Function to save model (including its optimizer, scheduler, and current epoch)
def save_checkpoint(model, optimizer = None, scheduler_step = None, scheduler_expo = None, 
                    epoch = None, basedir='models', suffix = None, additional_info = {}):
    if suffix is None:
        suffix = timestamp()
    checkpoint_path = os.path.join(basedir, f'checkpoint_{suffix}.pth')
    to_save = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if not optimizer is None else None,
        'scheduler_step_state_dict': scheduler_step.state_dict()if not scheduler_step is None else None,
        'scheduler_expo_state_dict': scheduler_expo.state_dict()if not scheduler_expo is None else None,
        'additional_info': additional_info
    }
    torch.save(to_save, checkpoint_path)
    
    return checkpoint_path

    # Function to load a torch model, optimizer, sceduler and epoch
def load_checkpoint(checkpoint_path, model, optimizer = None, 
                    scheduler_step = None, scheduler_expo = None):
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if not optimizer is None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if not scheduler_step is None:
        scheduler_step.load_state_dict(checkpoint['scheduler_step_state_dict'])

    if not scheduler_expo is None:
        scheduler_expo.load_state_dict(checkpoint['scheduler_expo_state_dict'])

    return checkpoint['epoch'], checkpoint['additional_info']

    # Function to set the logger
def setup_logging(root_path, log_level = 'INFO', fname = 'log.log'):
    level_dict = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARN': logging.WARNING,
        'ERROR': logging.ERROR,
        'FATAL': logging.CRITICAL
    }
    level = level_dict.get(log_level, logging.INFO)

    format_ = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    filename = '{}/{}'.format(root_path, fname)

    logger = logging.getLogger('kim')
    logger.setLevel(level)
    fh = logging.FileHandler(filename)
    fh.setFormatter(logging.Formatter(format_))
    fh.setLevel(level)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(format_))
    ch.setLevel(level)

    logger.addHandler(ch)

    return logger
