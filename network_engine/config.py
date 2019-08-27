#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# August 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# config.py                                  oN88888UU[[[/;::-.        dP^
# set and get experiment parameters         dNMMNN888UU[[[/;:--.   .o@P^
#                                          ,MMMMMMN888UU[[/;::-. o@^
#                                          NNMMMNN888UU[[[/~.o@P^
# Markus Ernst                             888888888UU[[[/o@^-..
#                                         oI8888UU[[[/o@P^:--..
#                                      .@^  YUU[[[/o@^;::---..
#                                    oMP     ^/o@P^;:::---..
#                                 .dMMM    .o@^ ^;::---...
#                                dMMMMMMM@^`       `^^^^
#                               YMMMUP^
#                                ^^
# _____________________________________________________________________________
#
#
# Copyright 2019 Markus Ernst
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# _____________________________________________________________________________


# ----------------
# import libraries
# ----------------


# standard libraries
# -----

import os
import numpy as np

# custom functions
# -----


# --------------------------
# main experiment parameters
# --------------------------


def get_par():
    """
    Get main parameters.
    For each experiment, change these parameters manually for different
    experiments.
    """

    par = {}

    par['exp_name'] = ["noname_experiment"]
    # TODO: par['name'] must be defined in the run_engine
    # TODO: add documentation i.e. parameter possibilities
    par['dataset'] = ["mnist"]

    par['n_occluders'] = [2]
    par['occlusion_percentage'] = [20]
    par['label_type'] = ["onehot"]
    par['connectivity'] = ['B', 'BK', 'BF', 'BL', 'BLT']
    par['network_depth'] = [2]
    par['time_depth'] = [3]
    par['timedepth_beyond'] = [0]
    par['feature_mult'] = [1]
    par['keep_prob'] = [1.0]

    par['stereo'] = [False, True]
    par['downsampling'] = ['ds4']
    par['color'] = ['grayscale']
    par['cropped'] = [False]
    par['augmented'] = [False]

    par['write_every'] = [1]
    par['test_every'] = [1]
    par['buffer_size'] = [20000]
    par['verbose'] = [True]
    par['visualization'] = [True]

    par['batchsize'] = [100]
    par['epochs'] = [100]
    par['learning_rate'] = [0.003]

    return par

# ----------------------------
# auxiliary network parameters
# ----------------------------


def get_aux():
    """
    Get auxiliary parameters.
    These auxiliary parameters do not have to be changed manually for the most
    part. Configure once in the beginning of setup.
    """

    aux = {}
    aux['wdir'] = ["/Users/markus/Research/Code/saturn/"]
    aux['input_dir'] = ["/Users/markus/Research/Code/saturn/datasets/"]
    # aux['input_dir'] = ["/home/aecgroup/aecdata/Textures/"]
    aux['output_dir'] = ["/Users/markus/Research/Code/saturn/experiments/"]
    # aux['output_dir'] = ["/home/aecgroup/aecdata/Results_python/markus/"]
    aux['network_module'] = ["utilities.networks.simplercnn"]
    aux['training_dir'] = [""]
    aux['validation_dir'] = [""]
    aux['test_dir'] = [""]
    aux['evaluation_dir'] = [""]

    aux['decaying_lrate'] = [False]
    aux['lr_eta'] = [0.1]
    aux['lr_delta'] = [0.1]
    aux['lr_d'] = [40.]
    aux['l2_lambda'] = [0.]
    aux['batchnorm'] = [True]

    aux['iterations'] = [1]
    return aux


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
