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
    """
    PreprocessorNetwork inherits from ComposedModule. It is a input network
    that preprocesses the network input. It can crop parts of the image and
    applies normalization before the image is handed to the neural network.
    """
    def define_inner_modules(self, name, inp_min, inp_max, cropped_bool,
                             augmented_bool, norm_by_stat_bool,
                             image_height, image_width, is_training):
        # create all modules of the network
        # -----

        self.layers = {}

        # added a module that does nothing b/c it is not clear whether
        # any of the following operations is indeed triggered.
        self.layers['pass_through'] = bb.ActivationModule(
            'pass_through', tf.identity)

        # crop input if desired
        with tf.name_scope('image_manipulation'):
            if cropped_bool:
                self.layers['manipulated'] = bb.CropModule(
                    "input_manipulated", image_height, image_width)

            # augment data if desired
            elif augmented_bool:
                self.layers['manipulated'] = bb.AugmentModule(
                    "input_manipulated", is_training, image_width * 10 // 4)

            else:
                pass

        with tf.name_scope('input_normalization'):
            self.layers["inp_norm"] = bb.NormalizationModule(
                "inp_norm", inp_max, inp_min)
        # TODO: remove input normalization from the network definition
        # TODO: integrate the normalization by input statistics
        # connect all modules of the network in a meaningful way
        # -----

        with tf.name_scope('wiring_of_modules'):
            if cropped_bool or augmented_bool:
                self.layers["manipulated"].add_input(
                    self.layers['pass_through'])
                self.layers['input_normalization'].add_input(
                    self.layers['manipulated'])

            else:
                self.layers['input_normalization'].add_input(
                    self.layers['pass_through'])

        with tf.name_scope('input_output'):
            self.input_module = self.layers["pass_through"]
            self.output_module = self.layers["input_normalization"]


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
