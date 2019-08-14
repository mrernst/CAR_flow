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
import os

# custom functions
# -----
from utilities.io import iohelper, pathfinder
from config import get_par, get_aux


class sbatch_document(object):
    """docstring for sbatch_document."""

    def __init__(self, parameters):
        super(sbatch_document, self).__init__()
        self.arg = arg

    def gen_sbatch(parameters):
        """
        gen_sbatch takes a parameter dict and a auxiliary_parameters dict and
        generates a sbatch file for the corresponding experiment
        """

        header = \
            "#!/bin/sh\n" + \
            "#SBATCH -c 8     # cores requested\n" + \
            "#SBATCH --mem=24000 #memory in Mb\n" + \
            "#SBATCH -o experiments/{}/logs/{}_outfile  # send stdout\n" \
            "#SBATCH -e experiments/{}/logs/{}_errfile  # send stderr\n" + \
            "export PYTHONPATH=~software/tensorflow-py3-amd64-gpu\n" + \
            "srun python3 engine.py --run {} --config_to_run {} --tb_loc {}"

        middle = \
            "#!/bin/sh\n" + \
            "#SBATCH -c 8     # cores requested\n" + \
            "#SBATCH --mem=24000 #memory in Mb\n" + \
            "#SBATCH -o experiments/{}/logs/{}_outfile  # send stdout\n" \
            "#SBATCH -e experiments/{}/logs/{}_errfile  # send stderr\n" + \
            "export PYTHONPATH=~software/tensorflow-py3-amd64-gpu\n" + \
            "srun python3 engine.py --run {} --config_to_run {} --tb_loc {}"

        footer = \
            "#!/bin/sh\n" + \
            "#SBATCH -c 8     # cores requested\n" + \
            "#SBATCH --mem=24000 #memory in Mb\n" + \
            "#SBATCH -o experiments/{}/logs/{}_outfile  # send stdout\n" \
            "#SBATCH -e experiments/{}/logs/{}_errfile  # send stderr\n" + \
            "export PYTHONPATH=~software/tensorflow-py3-amd64-gpu\n" + \
            "srun python3 engine.py --run {} --config_to_run {} --tb_loc {}"

        return (header + middle + footer)

    def write_to_file(parameters):
        file = open("run_experiment.sbatch", "w")
        file.write(self.gen_sbatch(parameters))
        file.close()
        pass

    def run_sbatch():
        self.write_to_file(parameters)
        os.system("sbatch run_experiment.sbatch")
        pass


class Experiment(object):
    """docstring for Experiment."""

    def __init__(self, arg):
        super(Experiment, self).__init__()
        self.arg = arg


    def est_folders(parameters):
        """
        est_folders takes X and establishes a folder structure given the parameters
        of the function
        """
        pass


if __name__ == '__main__':
    pass

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
