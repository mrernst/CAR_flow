#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# November 2019                                  _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# advancedrcnn.py                            oN88888UU[[[/;::-.        dP^
# a dynamic network constructor with        dNMMNN888UU[[[/;:--.   .o@P^
# variable depth and complexity            ,MMMMMMN888UU[[/;::-. o@^
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

# Block 1:       F = 96, K = 7 (11 for BK)
# Block 2:       F = 125, K = 5 (7)
# Block 3:       F = 192, K = 3 (5)
# Block 4:       F = 256, K = 3 (5)
# Block 5:       F = 512, K = 3 (5)
# Block 6:       F = 1024, K = 3 (5)
# Block 7:       F = 2048, K = 1 (3)
# GAP
# class readout

# 8 timesteps

# Additional mode: BD (deeper network)


# TODO: Write description for this function
def constructor(name,
                configuration_dict,
                is_training,
                keep_prob,
                custom_net_parameters=None):
    """
    constructor takes a name, a configuration dict the booleans is_training,
    keep_prob and optionally a custom_net_parameters dict and returns
    an initialized NetworkClass instance.
    """

    def get_net_parameters(configuration_dict):
        net_param_dict = {}
        net_param_dict["receptive_pixels"] = np.array(
            [7, 5, 3, 3, 3, 3, 1])
        net_param_dict["n_features"] = np.array(
            [96, 125, 196, 256, 512, 1024, 2048])
        net_param_dict["depth"] = len(net_param_dict["receptive_pixels"])

        if "F" in configuration_dict['connectivity']:
            net_param_dict["n_features"] = \
                net_param_dict["n_features"]*2

        if "K" in configuration_dict['connectivity']:
            net_param_dict["receptive_pixels"] = np.array(
                [11, 7, 5, 5, 5, 5, 3])

        net_param_dict["activations"] = [
            bb.lrn_relu,
            bb.lrn_relu,
            bb.lrn_relu,
            bb.lrn_relu,
            bb.lrn_relu,
            bb.lrn_relu,
            tf.identity]

        net_param_dict["conv_filter_shapes"] = [
            [net_param_dict["receptive_pixels"][0],
             net_param_dict["receptive_pixels"][0],
             configuration_dict['image_channels'],
             net_param_dict["n_features"][0]],
            [net_param_dict["receptive_pixels"][1],
             net_param_dict["receptive_pixels"][1],
             net_param_dict["n_features"][0],
             net_param_dict["n_features"][1]],
            [net_param_dict["receptive_pixels"][2],
             net_param_dict["receptive_pixels"][2],
             net_param_dict["n_features"][1],
             net_param_dict["n_features"][2]],
            [net_param_dict["receptive_pixels"][3],
             net_param_dict["receptive_pixels"][3],
             net_param_dict["n_features"][2],
             net_param_dict["n_features"][3]],
            [net_param_dict["receptive_pixels"][4],
             net_param_dict["receptive_pixels"][4],
             net_param_dict["n_features"][3],
             net_param_dict["n_features"][4]],
            [net_param_dict["receptive_pixels"][5],
             net_param_dict["receptive_pixels"][5],
             net_param_dict["n_features"][4],
             net_param_dict["n_features"][5]],
            [net_param_dict["receptive_pixels"][6],
             net_param_dict["receptive_pixels"][6],
             net_param_dict["n_features"][5],
             net_param_dict["n_features"][6]]
        ]

        net_param_dict["bias_shapes"] = [
            [1, configuration_dict['image_height'],
                configuration_dict['image_width'],
                net_param_dict["n_features"][0]],
            [1, int(np.ceil(configuration_dict['image_height'] / 2)),
                int(np.ceil(configuration_dict['image_width'] / 2)),
                net_param_dict["n_features"][1]],
            [1, int(np.ceil(configuration_dict['image_height'] / (2**2))),
                int(np.ceil(configuration_dict['image_width'] / (2**2))),
                net_param_dict["n_features"][2]],
            [1, int(np.ceil(configuration_dict['image_height'] / (2**3))),
                int(np.ceil(configuration_dict['image_width'] / (2**3))),
                net_param_dict["n_features"][3]],
            [1, int(np.ceil(configuration_dict['image_height'] / (2**4))),
                int(np.ceil(configuration_dict['image_width'] / (2**4))),
                net_param_dict["n_features"][4]],
            [1, int(np.ceil(configuration_dict['image_height'] / (2**5))),
                int(np.ceil(configuration_dict['image_width'] / (2**5))),
                net_param_dict["n_features"][5]],
            [1, int(np.ceil(configuration_dict['image_height'] / (2**6))),
                int(np.ceil(configuration_dict['image_width'] / (2**6))),
                net_param_dict["n_features"][6]],
            [1, configuration_dict['classes']]
        ]

        net_param_dict["pool_sizes"] = [
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1]
        ]

        net_param_dict["pool_strides"] = [
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1]
        ]

        net_param_dict["topdown_filter_shapes"] = [
            [net_param_dict['receptive_pixels'][0],
             net_param_dict['receptive_pixels'][0],
             net_param_dict['n_features'][0],
             net_param_dict['n_features'][1]],
            [net_param_dict['receptive_pixels'][0],
             net_param_dict['receptive_pixels'][0],
             net_param_dict['n_features'][1],
             net_param_dict['n_features'][2]],
            [net_param_dict['receptive_pixels'][0],
             net_param_dict['receptive_pixels'][0],
             net_param_dict['n_features'][2],
             net_param_dict['n_features'][3]],
            [net_param_dict['receptive_pixels'][0],
             net_param_dict['receptive_pixels'][0],
             net_param_dict['n_features'][3],
             net_param_dict['n_features'][4]],
            [net_param_dict['receptive_pixels'][0],
             net_param_dict['receptive_pixels'][0],
             net_param_dict['n_features'][4],
             net_param_dict['n_features'][5]],
            [net_param_dict['receptive_pixels'][0],
             net_param_dict['receptive_pixels'][0],
             net_param_dict['n_features'][5],
             net_param_dict['n_features'][6]]
        ]

        net_param_dict["topdown_output_shapes"] = [
            [configuration_dict['batchsize'],
             configuration_dict['image_height'],
             configuration_dict['image_width'],
             net_param_dict['n_features'][0]],
            [configuration_dict['batchsize'],
             int(np.ceil(configuration_dict['image_height'] / (2**1))),
             int(np.ceil(configuration_dict['image_width'] / (2**1))),
             net_param_dict['n_features'][1]],
            [configuration_dict['batchsize'],
             int(np.ceil(configuration_dict['image_height'] / (2**2))),
             int(np.ceil(configuration_dict['image_width'] / (2**2))),
             net_param_dict['n_features'][2]],
            [configuration_dict['batchsize'],
             int(np.ceil(configuration_dict['image_height'] / (2**3))),
             int(np.ceil(configuration_dict['image_width'] / (2**3))),
             net_param_dict['n_features'][3]],
            [configuration_dict['batchsize'],
             int(np.ceil(configuration_dict['image_height'] / (2**4))),
             int(np.ceil(configuration_dict['image_width'] / (2**4))),
             net_param_dict['n_features'][4]],
            [configuration_dict['batchsize'],
             int(np.ceil(configuration_dict['image_height'] / (2**5))),
             int(np.ceil(configuration_dict['image_width'] / (2**5))),
             net_param_dict['n_features'][5]]
        ]

        return net_param_dict

    if custom_net_parameters:
        net_parameters = custom_net_parameters
    else:
        net_parameters = get_net_parameters(configuration_dict)

    # copy necessary items from configuration
    net_parameters['connectivity'] = configuration_dict['connectivity']
    net_parameters['batchnorm'] = configuration_dict['batchnorm']

    # longrange connection parameters for B, L, and T connections
    net_parameters['BLT_longrange'] = configuration_dict['BLT_longrange']

    return NetworkClass(name, net_parameters, is_training, keep_prob)

class NetworkClass(bb.ComposedModule):
    def define_inner_modules(self, name, net_param_dict,
                             is_training, keep_prob):

        # create default input parameters for network if none given
        # -----

        if not(net_param_dict):
            self.net_params = get_net_parameters(net_param_dict)
        else:
            self.net_params = net_param_dict

        # create input/output modules of the network
        # -----

        self.layers = {}

        # with tf.name_scope('flatpool_0'):
        #     self.layers['flatpool0'] = bb.FlattenModule('flatpool0')

        with tf.name_scope('global_average_pooling'):
            self.layers['gap'] = bb.GlobalAveragePoolingModule('gap')

        with tf.name_scope('fully_connected_layer_0'):
            if self.net_params['batchnorm']:
                self.layers['fc0'] = \
                    bb.FullyConnectedLayerWithBatchNormalizationModule(
                    'fc0',
                    self.net_params['bias_shapes'][-1][-1],
                    is_training,
                    0.0,
                    1.0,
                    0.5,
                    self.net_params['activations'][-1],
                    # input_shape
                    int(np.prod(np.ceil(np.array(
                        self.net_params['bias_shapes'][-2]) /
                        np.array(self.net_params['pool_strides'][-2])
                                        )
                                )
                        ),
                    # output_shape
                    np.prod(
                        self.net_params['bias_shapes'][-1])
                )
            else:
                self.layers['fc0'] = bb.FullyConnectedLayerModule(
                    'fc0',
                    self.net_params['activations'][-1],
                    int(np.prod(np.ceil(np.array(
                        self.net_params['bias_shapes'][-2]) /
                        np.array(self.net_params['pool_strides'][-2])))),
                    np.prod(
                        self.net_params['bias_shapes'][-1])
                )

        # create middle modules of the network
        # -----

        # convolutions, lateral, pooling
        for lnr in range(self.net_params['depth']):
            with tf.name_scope('convolutional_layer_{}'.format(lnr)):
                if self.net_params['batchnorm']:
                    self.layers["conv{}".format(lnr)] = \
                        bb.TimeConvolutionalLayerWithBatchNormalizationModule(
                        "conv{}".format(lnr),
                        self.net_params['bias_shapes'][lnr][-1],
                        is_training,
                        0.0,
                        1.0,
                        0.5,
                        self.net_params['activations'][lnr],
                        self.net_params['conv_filter_shapes'][lnr],
                        [1, 1, 1, 1],
                        self.net_params['bias_shapes'][lnr]
                    )
                else:
                    self.layers["conv{}".format(lnr)] = \
                        bb.TimeConvolutionalLayerModule(
                        "conv{}".format(lnr),
                        self.net_params['activations'][lnr],
                        self.net_params['conv_filter_shapes'][lnr],
                        [1, 1, 1, 1],
                        self.net_params['bias_shapes'][lnr]
                    )

            with tf.name_scope('dropout_conv{}'.format(lnr)):
                self.layers['dropoutc{}'.format(lnr)] = bb.DropoutModule(
                    'dropoutc{}'.format(lnr), keep_prob=keep_prob)

            with tf.name_scope('lateral_layer_{}'.format(lnr)):
                lateral_filter_shape = self.net_params['conv_filter_shapes'][lnr]
                #tmp = lateral_filter_shape[2]
                lateral_filter_shape[2] = lateral_filter_shape[3]
                #lateral_filter_shape[3] = tmp
                self.layers["lateral{}".format(lnr)] = bb.Conv2DModule(
                    "lateral{}".format(lnr),
                    lateral_filter_shape,
                    [1, 1, 1, 1]
                )
                self.layers["lateral{}_batchnorm".format(lnr)] = \
                    bb.BatchNormalizationModule(
                    "lateral{}_batchnorm".format(lnr),
                    lateral_filter_shape[-1],
                    is_training,
                    beta_init=0.0,
                    gamma_init=0.1,
                    ema_decay_rate=0.5,
                    moment_axes=[0, 1, 2],
                    variance_epsilon=1e-3
                )

            with tf.name_scope('pooling_layer_{}'.format(lnr)):
                self.layers["pool{}".format(lnr)] = bb.MaxPoolingModule(
                    "pool{}".format(lnr),
                    self.net_params['pool_sizes'][lnr],
                    self.net_params['pool_strides'][lnr]
                )

        # topdown connections

        for lnr in range(self.net_params['depth'] - 1):
            with tf.name_scope('topdown_layer_{}'.format(lnr)):
                self.layers["topdown{}".format(lnr)] = \
                    bb.Conv2DTransposeModule(
                    "topdown{}".format(lnr),
                    self.net_params['topdown_filter_shapes'][lnr],
                    [1, 2, 2, 1],
                    self.net_params['topdown_output_shapes'][lnr]
                )
                self.layers["topdown{}_batchnorm".format(lnr)] = \
                    bb.BatchNormalizationModule("topdown{}_batchnorm".format(lnr),
                    self.net_params[
                     'topdown_output_shapes'][lnr][-1],
                    is_training,
                    beta_init=0.0,
                    gamma_init=0.1,
                    ema_decay_rate=0.5,
                    moment_axes=[0, 1, 2],
                    variance_epsilon=1e-3
                    )

        # longrange topdown connections
        # TODO: implement long range connections
        # for i in range(1, self.net_params['depth']):
        #     for lnr in range(self.net_params['depth'] - i):
        #         with tf.name_scope('longrange_topdown_layer_{}'.format(lnr)):
        #             self.layers["longrange_topdown{}".format(lnr)] = \
        #                 bb.Conv2DTransposeModule(
        #                 "topdown{}".format(lnr),
        #                 self.net_params['topdown_filter_shapes'][lnr],
        #                 [1, 2, 2, 1],
        #                 self.net_params['topdown_output_shapes'][lnr]
        #             )
        #             self.layers["longrange_topdown{}_batchnorm".format(lnr)] = \
        #                 bb.BatchNormalizationModule("longrange_topdown{}_batchnorm".format(lnr),
        #                 self.net_params[
        #                  'topdown_output_shapes'][lnr][-1],
        #                 is_training,
        #                 beta_init=0.0,
        #                 gamma_init=0.1,
        #                 ema_decay_rate=0.5,
        #                 moment_axes=[0, 1, 2],
        #                 variance_epsilon=1e-3
        #                 )
        #         pass

        # connect all modules of the network in a meaningful way
        # -----
        with tf.name_scope('input_output'):
            self.input_module = self.layers["conv0"]
            self.output_module = self.layers["fc0"]

        with tf.name_scope('wiring_of_modules'):
            self.layers["gap"].add_input(self.layers["dropoutc{}".format(
                self.net_params['depth']-1)])
            #])  # TODO: is this correct?

            # self.layers["flatpool0"].add_input(
            #     self.layers["dropoutc{}".format(self.net_params['depth'] - 1)])
            self.layers["fc0"].add_input(self.layers["gap"])

            for lnr in range(self.net_params['depth'] - 1):
                # convolutional layers
                self.layers["conv{}".format(
                    (lnr + 1))].add_input(self.layers["dropoutc{}".format(lnr)], 0)

            for lnr in range(self.net_params['depth']):
                # pooling layers
                self.layers["pool{}".format(lnr)].add_input(
                    self.layers["conv{}".format(lnr)])
                self.layers["dropoutc{}".format((lnr))].add_input(
                    self.layers["pool{}".format(lnr)])
                if "L" in self.net_params["connectivity"]:
                    # lateral layers
                    if self.net_params["batchnorm"]:
                        self.layers["lateral{}".format(lnr)].add_input(
                            self.layers["conv{}".format(lnr)])
                        self.layers["lateral{}_batchnorm".format(lnr)].add_input(
                            self.layers["lateral{}".format(lnr)])
                        self.layers["conv{}".format(lnr)].preactivation.add_input(
                            self.layers["lateral{}_batchnorm".format(lnr)], -1)
                    else:
                        self.layers["lateral{}".format(lnr)].add_input(
                            self.layers["conv{}".format(lnr)])
                        self.layers["conv{}".format(lnr)].preactivation.add_input(
                            self.layers["lateral{}".format(lnr)], -1)

            for lnr in range(self.net_params['depth'] - 1):
                # topdown layers
                if "T" in self.net_params["connectivity"]:
                    if self.net_params["batchnorm"]:
                        self.layers["topdown{}_batchnorm".format(lnr)].add_input(
                            self.layers["topdown{}".format(lnr)])
                        self.layers["conv{}".format(lnr)].preactivation.add_input(
                            self.layers["topdown{}_batchnorm".format(lnr)], -1)
                        self.layers["topdown{}".format(lnr)].add_input(
                            self.layers["conv{}".format((lnr + 1))])
                    else:
                        self.layers["conv{}".format(lnr)].preactivation.add_input(
                            self.layers["topdown{}".format(lnr)], -1)
                        self.layers["topdown{}".format(lnr)].add_input(
                            self.layers["conv{}".format((lnr + 1))])

    def get_all_weights(self):
        weights = []
        for lyr in self.layers.keys():
            if 'batchnorm' not in lyr:
                if 'conv' in lyr:
                    weights.append(self.layers[lyr].conv.weights)
                elif 'fc' in lyr:
                    weights.append(self.layers[lyr].input_module.weights)
                elif ('lateral' in lyr):
                    weights.append(self.layers[lyr].weights)
                elif ('topdown' in lyr):
                    weights.append(self.layers[lyr].weights)
        return weights

    def get_all_biases(self):
        biases = []
        if self.net_params['batchnorm']:
            for lyr in self.layers.keys():
                if ('conv' in lyr):
                    biases.append(self.layers[lyr].batchnorm.beta)
                elif ('fc' in lyr):
                    biases.append(self.layers[lyr].batchnorm.beta)
                elif ('batchnorm' in lyr):
                    if ('lateral' in lyr):
                        biases.append(self.layers[lyr].beta)
                    elif ('topdown' in lyr):
                        biases.append(self.layers[lyr].beta)
        else:
            for lyr in self.layers.keys():
                if 'batchnorm' not in lyr:
                    if ('conv' in lyr):
                        biases.append(self.layers[lyr].bias.bias)
                    elif ('fc' in lyr):
                        biases.append(self.layers[lyr].bias.bias)
        return biases






# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
