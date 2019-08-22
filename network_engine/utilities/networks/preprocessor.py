#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# August 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# simplercnn.py                              oN88888UU[[[/;::-.        dP^
# network definition of                     dNMMNN888UU[[[/;:--.   .o@P^
# Spoerer 2017 network                     ,MMMMMMN888UU[[/;::-. o@^
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
import utilities.networks.buildingblocks as bb


class PreprocessorNetwork(bb.ComposedModule):
    def define_inner_modules(self, name):
        # create all modules of the network
        # -----

        self.layers = {}



        # crop input if desired
        if FLAGS.cropped:
            IMAGE_HEIGHT = IMAGE_WIDTH // 10 * 4
            IMAGE_WIDTH = IMAGE_WIDTH // 10 * 4
            inp_prep = bb.CropModule("input_cropped", IMAGE_HEIGHT, IMAGE_WIDTH)
            inp_prep.add_input(inp)
        # augment data if desired
        elif FLAGS.augmented:
            inp_prep = bb.AugmentModule(
                "input_prep", is_training.placeholder, IMAGE_WIDTH)
            IMAGE_HEIGHT = IMAGE_WIDTH // 10 * 4
            IMAGE_WIDTH = IMAGE_WIDTH // 10 * 4
            inp_prep.add_input(inp)
        else:
            inp_prep = inp

        with tf.name_scope('input_normalization'):
            self.layers["inp_norm"] = bb.NormalizationModule("inp_norm")
        # TODO: remove input normalization from the network definition

        # connect all modules of the network in a meaningful way
        # -----

        with tf.name_scope('wiring_of_modules'):
            self.layers["conv0"].add_input(self.layers["inp_norm"], 0)
            self.layers["pool0"].add_input(self.layers["conv0"])
        with tf.name_scope('input_output'):
            self.input_module = self.layers["inp_norm"]
            self.output_module = self.layers["fc0"]




# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
