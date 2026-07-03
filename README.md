## stochastic_optimal_control_constraint

# Introduction

This repository is an illustration for the implementation of the alternative training algorithm proposed in Sections 4.1-4.2-4.3 of the article *Optimal Control with Expectation Constraint in a Smooth Boundary Case* by Bruno BOUCHARD, Kim-Anh PHAM, and Lucas GNECCO HEREDIA. 

# Organization of the repository
In this repository, one can find 
- The documentation of this implementation can be found in the folder `docs`, and we refer to Sections 2 and 4 of the associated paper for a detailed description of the theoretical mathematical framework and of the algorithm. 
- The set of `python` scripts used for training. These scripts begins with a number, either `0` for pre-training scripts, `1` for training scripts, `2` for testing scripts, and `3` for visualization scripts. 
- An example of configuration for training the algorithm and our training results can be found in the folders `config` and `results`, respectively. These are the parameters and results shown in the article. 

# Environment requirements
Install `conda`, then create an environments using the following command

    conda env create -f env.yml

This will create the environment `optimal_control`. Now activate the environment

    conda activate optimal_control

If there are problems installing `torch`, then activate the environment and run

    pip3 install torch --index-url https://download.pytorch.org/whl/cu118


# Guide to executing the training scripts
We highly encourage the readers to take a look at the training scripts before execution, particularly the part named `Parser` which details the necessary parameters/arguments for each training step. Since our algorithm requires consequential training steps of different elements, the set of parameters needed for each step is different from another. 

Suppose that we want to run a script called `train_X.py`, we run

    python train_X.py -arg1 value_for_arg1 -arg2 value_for_arg2

where `-arg1` and `-arg2` are parameters needed for the training (as described in the `Parser` section of `train.py`) while `value_for_arg1` and `value_for_arg2` are the values to be assigned to these parameters. Note that there can be multiple parameters needed for one training script, and the values can be pathways to the configuration files, numerical values, or string values. 

After the training, a copy of the configuration files used during the training as well as a log file and the checkpoints from the training process can be found in a directory with a name similar to `./results/train_X_20250321_173151` which indicates the starting time of the training. The checkpoints are stocked in a subfolder named `checkpoints` inwhichfile `checkpoint_final.pth` is the final model.

In case of difficulty in executing the script, run  `python train_X.py --help` for more details.
