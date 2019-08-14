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

    par['newexperiment'] = True                  # create a new exp. folder
    par['N_e'] = 1600                            # excitatory neurons
    par['N_u'] = int(par.N_e/60)                 # neurons in each input pool

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
    aux['datadirectory'] = "/home/mernst/Code/saturn/datasets/"
    aux['experimentdirectory'] = "/home/mernst/Code/saturn/experiments/"

    aux['N_i'] = int(0.2 * par.N_e)              # inhibitory neurons
    aux['N'] = par.N_e + aux.N_i               # total number of neurons

    return aux


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
