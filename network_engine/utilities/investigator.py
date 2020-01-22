#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# January 2020                                   _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# investigator.py                            oN88888UU[[[/;::-.        dP^
# Description Description                   dNMMNN888UU[[[/;:--.   .o@P^
# Description Description                  ,MMMMMMN888UU[[/;::-. o@^
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
#
# The investigator
#
#           .\"\"\"-.
#          /      \\
#          |  _..--'-.
#          >.`__.-\"\";\"`
#         / /(     ^\\
#         '-`)     =|-.
#          /`--.'--'   \\ .-.
#        .'`-._ `.\\    | J /
#       /      `--.|   \\__/


# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import tensorflow as tf
import numpy as np
import argparse

# custom functions
# -----


def mkdir_p(path):
    """
    mkdir_p takes a string path and creates a directory at this path if it
    does not already exist.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# commandline arguments
# -----

parser = argparse.ArgumentParser()
parser.add_argument(
     "-cfgf",
     "--config_file",
     type=str,
     default=None,
     help='path to config file')
parser.add_argument(
     "-cfgdir",
     "--config_dir",
     type=str,
     default=None,
     help='path to config directory')
parser.add_argument(
     "-mem",
     "--memory",
     type=int,
     default=20,
     help='memory to be reserved (GB)')
args = parser.parse_args()

# -----------------
# import config txt data
# -----------------


# Store weight matrices in a dict of arrays
# -----


# -----------------
# statistical analysis
# -----------------

# PCA
# -----

# Distribution and Histograms
# -----

# Correlation of Data
# -----

# Fourier
# -----

# Overall Statistics
# -----

# Comparison
# -----

if __name__ == __main__:
    pass

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
