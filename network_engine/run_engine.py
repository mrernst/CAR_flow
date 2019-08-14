#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# August 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# run_engine.py                              oN88888UU[[[/;::-.        dP^
# setup and initialization                  dNMMNN888UU[[[/;:--.   .o@P^
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
import tensorflow as tf
import numpy as np

# custom functions
# -----
from utilities.io import iohelper, pathfinder
from config import get_par, get_aux

# custom functions 2
# -----
# TODO: Add these to the utilities folder


def gen_sbatch(parameters, auxiliary_parameters):
    """
    gen_sbatch takes a parameter dict and a auxiliary_parameters dict and
    generates a sbatch file for the corresponding experiment
    """
    pass


def est_folders(arg):
    """
    est_folders takes X and establishes a folder structure given the parameters
    of the function
    """
    pass


class SbatchDocument(object):
    """docstring for SbatchDocument."""

    def __init__(self, arg):
        super(SbatchDocument, self).__init__()
        self.arg = arg


# _____________________________________________________________________________
# Description:
#
# This program is supposed to create an folder structure for the experiment,
# copy the necessary files to that environment, create an sbatch file and run
# the file given the environment-variables predefined in an external document.
# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
