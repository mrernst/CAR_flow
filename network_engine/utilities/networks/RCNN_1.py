#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# August 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# Filename.py                                oN88888UU[[[/;::-.        dP^
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


# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import tensorflow as tf
import numpy as np

# custom functions
# -----
import buildingblocks as bb


class RCNN_1(bb.ComposedModule):
    def define_inner_modules(self, name, is_training, activations,
                             conv_filter_shapes, bias_shapes, ksizes,
                             pool_strides, topdown_filter_shapes,
                             topdown_output_shapes, keep_prob, FLAGS):
        # TODO: Fix this input mess, just input FLAGS
        # create all modules of the network
        # -----

        self.layers = {}
        with tf.name_scope('input_normalization'):
            self.layers["inp_norm"] = bb.NormalizationModule("inp_norm")
        with tf.name_scope('convolutional_layer_0'):
            if FLAGS.batchnorm:
                self.layers["conv0"] = \
                    bb.TimeConvolutionalLayerWithBatchNormalizationModule(
                        "conv0", bias_shapes[0][-1], is_training, 0.0, 1.0,
                        0.5, activations[0], conv_filter_shapes[0],
                        [1, 1, 1, 1], bias_shapes[0])
            else:
                self.layers["conv0"] = bb.TimeConvolutionalLayerModule(
                    "conv0", activations[0], conv_filter_shapes[0],
                    [1, 1, 1, 1], bias_shapes[0])
        with tf.name_scope('lateral_layer_0'):
            lateral_filter_shape = conv_filter_shapes[0]
            tmp = lateral_filter_shape[2]
            lateral_filter_shape[2] = lateral_filter_shape[3]
            lateral_filter_shape[3] = tmp
            self.layers["lateral0"] = bb.Conv2DModule(
                "lateral0", lateral_filter_shape, [1, 1, 1, 1])
            self.layers["lateral0_batchnorm"] = bb.BatchNormalizationModule(
                "lateral0_batchnorm", lateral_filter_shape[-1], is_training,
                beta_init=0.0, gamma_init=0.1, ema_decay_rate=0.5,
                moment_axes=[0, 1, 2], variance_epsilon=1e-3)
        with tf.name_scope('pooling_layer_0'):
            self.layers["pool0"] = bb.MaxPoolingModule(
                "pool0", ksizes[0], pool_strides[0])
        with tf.name_scope('dropout_layer_0'):
            self.layers['dropoutc0'] = bb.DropoutModule(
                'dropoutc0', keep_prob=keep_prob)
        with tf.name_scope('convolutional_layer_1'):
            if FLAGS.batchnorm:
                self.layers["conv1"] = \
                    bb.TimeConvolutionalLayerWithBatchNormalizationModule(
                        "conv1", bias_shapes[1][-1], is_training, 0.0, 1.0,
                        0.5, activations[1], conv_filter_shapes[1],
                        [1, 1, 1, 1], bias_shapes[1])
            else:
                self.layers["conv1"] = bb.TimeConvolutionalLayerModule(
                    "conv1", activations[1], conv_filter_shapes[1],
                    [1, 1, 1, 1], bias_shapes[1])
        with tf.name_scope('topdown_layer_0'):
            self.layers["topdown0"] = bb.Conv2DTransposeModule(
                "topdown0", topdown_filter_shapes[0], [1, 2, 2, 1],
                topdown_output_shapes[0])
            self.layers["topdown0_batchnorm"] = bb.BatchNormalizationModule(
                "topdown0_batchnorm", topdown_output_shapes[0][-1],
                is_training, beta_init=0.0, gamma_init=0.1, ema_decay_rate=0.5,
                moment_axes=[0, 1, 2], variance_epsilon=1e-3)
        with tf.name_scope('lateral_layer_1'):
            lateral_filter_shape = conv_filter_shapes[1]
            tmp = lateral_filter_shape[2]
            lateral_filter_shape[2] = lateral_filter_shape[3]
            lateral_filter_shape[3] = tmp
            self.layers["lateral1"] = bb.Conv2DModule(
                "lateral1", lateral_filter_shape, [1, 1, 1, 1])
            self.layers["lateral1_batchnorm"] = bb.BatchNormalizationModule(
                "lateral1_batchnorm", lateral_filter_shape[-1], is_training,
                beta_init=0.0, gamma_init=0.1, ema_decay_rate=0.5,
                moment_axes=[0, 1, 2], variance_epsilon=1e-3)
        with tf.name_scope('pooling_layer_1'):
            self.layers["pool1"] = bb.MaxPoolingModule(
                "pool1", ksizes[0], pool_strides[1])
            self.layers["flatpool1"] = bb.FlattenModule("flatpool1")
        with tf.name_scope('dropout_layer_1'):
            self.layers['dropoutc1'] = bb.DropoutModule(
                'dropoutc1', keep_prob=keep_prob)
        with tf.name_scope('fully_connected_layer_0'):
            if FLAGS.batchnorm:
                self.layers["fc0"] = \
                    bb.FullyConnectedLayerWithBatchNormalizationModule(
                        "fc0", bias_shapes[-1][-1], is_training, 0.0, 1.0, 0.5,
                        activations[2],
                        int(np.prod(np.array(bias_shapes[1]) /
                            np.array(pool_strides[1]))),
                        np.prod(bias_shapes[2]))
            else:
                self.layers["fc0"] = \
                    bb.FullyConnectedLayerModule(
                        "fc0", activations[2],
                        int(np.prod(np.array(bias_shapes[1]) /
                            np.array(pool_strides[1]))),
                        np.prod(bias_shapes[2]))

        # connect all modules of the network in a meaningful way
        # -----

        with tf.name_scope('wiring_of_modules'):
            self.layers["conv0"].add_input(self.layers["inp_norm"], 0)
            self.layers["pool0"].add_input(self.layers["conv0"])
            self.layers["dropoutc0"].add_input(self.layers["pool0"])
            self.layers["conv1"].add_input(self.layers["dropoutc0"], 0)
            self.layers["pool1"].add_input(self.layers["conv1"])
            self.layers["dropoutc1"].add_input(self.layers["pool1"])
            self.layers["flatpool1"].add_input(self.layers["dropoutc1"])
            self.layers["fc0"].add_input(self.layers["flatpool1"])
            if "L" in FLAGS.architecture:
                if FLAGS.batchnorm:
                    self.layers["lateral0"].add_input(
                        self.layers["conv0"].preactivation)
                    self.layers["lateral0_batchnorm"].add_input(
                        self.layers["lateral0"])
                    self.layers["conv0"].add_input(
                        self.layers["lateral0_batchnorm"], -1)
                    self.layers["lateral1"].add_input(
                        self.layers["conv1"].preactivation)
                    self.layers["lateral1_batchnorm"].add_input(
                        self.layers["lateral1"])
                    self.layers["conv1"].add_input(
                        self.layers["lateral1_batchnorm"], -1)
                else:
                    self.layers["lateral0"].add_input(
                        self.layers["conv0"].preactivation)
                    self.layers["conv0"].add_input(
                        self.layers["lateral0"], -1)
                    self.layers["lateral1"].add_input(
                        self.layers["conv1"].preactivation)
                    self.layers["conv1"].add_input(
                        self.layers["lateral1"], -1)
            if "T" in FLAGS.architecture:
                if FLAGS.batchnorm:
                    self.layers["topdown0_batchnorm"].add_input(
                        self.layers["topdown0"])
                    self.layers["conv0"].add_input(
                        self.layers["topdown0_batchnorm"], -1)
                    self.layers["topdown0"].add_input(
                        self.layers["conv1"].preactivation)
                else:
                    self.layers["conv0"].add_input(
                        self.layers["topdown0"], -1)
                    self.layers["topdown0"].add_input(
                        self.layers["conv1"].preactivation)
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
