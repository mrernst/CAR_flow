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
from platform import system
IS_MACOSX = True if system() == 'Darwin' else False
PWD_STEM = "/Users/markus/Research/Code/" if IS_MACOSX else "/home/mernst/git/"


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

    par['exp_name'] = ["fmnist_testrun"]
    # par['name'] must be defined as a FLAG to engine, b/c it resembles the
    # iteration number that gets passed by the sbatch script
    # TODO: add documentation i.e. parameter possibilities
    par['dataset'] = ["osfashionmnist"]

    par['n_occluders'] = [0]
    par['occlusion_percentage'] = [0]
    par['label_type'] = ["onehot"]
    par['connectivity'] = ['B', 'BK', 'BF', 'BT', 'BL', 'BLT']
    par['network_depth'] = [2]
    par['time_depth'] = [3]
    par['time_depth_beyond'] = [0]
    par['feature_multiplier'] = [1]
    par['keep_prob'] = [1.0]

    par['stereo'] = [False]
    par['downsampling'] = ['fine']
    par['color'] = ['grayscale']
    par['cropped'] = [False]
    par['augmented'] = [False]

    par['write_every'] = [1]
    par['test_every'] = [1]
    par['buffer_size'] = [600000]
    par['verbose'] = [False]
    par['visualization'] = [True]
    par['projector'] = [True]

    par['batchsize'] = [100]
    par['epochs'] = [1]
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
    aux['wdir'] = ["{}saturn/".format(PWD_STEM)]
    aux['input_dir'] = ["{}saturn/datasets/".format(PWD_STEM)]
    # aux['input_dir'] = ["/home/aecgroup/aecdata/Textures/"]
    aux['output_dir'] = ["{}saturn/experiments/".format(PWD_STEM)]
    # aux['output_dir'] = ["/home/aecgroup/aecdata/Results_python/markus/"]
    aux['network_module'] = ["utilities.networks.simplercnn"]
    aux['norm_by_stat'] = [False]
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
    aux['global_weight_init_mean'] = ['None']
    aux['global_weight_init_std'] = ['None']
    # Info: None-Values have to be strings b/c of csv text conversion

    aux['iterations'] = [1]
    return aux


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
