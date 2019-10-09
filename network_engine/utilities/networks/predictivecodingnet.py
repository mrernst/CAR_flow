#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# October 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# predictivecodingnet.py                     oN88888UU[[[/;::-.        dP^
# implementation of a simple                dNMMNN888UU[[[/;:--.   .o@P^
# predictive coding network (Wen, 2018)    ,MMMMMMN888UU[[/;::-. o@^
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

# TODO:
# CAR composed module
# AR composed module
# CR composed module
# S module
# TakeLastInputModule


class InitialRepresentation(bb.ComposedModule):
    def define_inner_modules(self, name):
        self.input_module = bb.ConvolutionalLayerModule()
        self.output_module = DoNothingModule()  # bb.MaxPoolingModule()
        self.output_module.add_input(self.input_module)


class Prediction(bb.ComposedModule):
    def define_inner_modules(self, name):
        self.input_module = bb.ConvolutionalLayerModule()
        self.output_module = DoNothingModule()
        self.output_module.add_input(self.input_module)

class FBRepresentation(bb.ComposedModule):
    def define_inner_modules(self, name):
        self.input_module = bb.AddModule()
        self.output_module = bb.ActivationModule()
        self.output_module.add_input(self.input_module)


class PredictionError(bb.AddModule):
    def operation(self, *arg):
        arg1 = args[0]
        arg2 = args[-1]
        return tf.subtract(arg1, arg2, name=self.name)


class FFRepresentation(bb.ComposedModule):
    def define_inner_modules(self, name):
        self.input_module = bb.ConvolutionalLayerModule()
        self.addition = bb.AddModule()
        self.output_module = bb.ActivationModule()
        self.addition.add_input(self.input_module)
        self.output_module.add_input(self.addition)


class TakeLastInputModule(bb.TimeOperationModule):
    def operation(self, *arg):
        return args[-1]


class DoNothingModule(TakeLastInputModule):
    pass


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
        receptive_pixels = 3
        n_features = 32
        feature_multiplier = configuration_dict['feature_multiplier']

        if "F" in configuration_dict['connectivity']:
            n_features = 64
        if "K" in configuration_dict['connectivity']:
            receptive_pixels = 5

        net_param_dict["activations"] = [bb.lrn_relu, bb.lrn_relu, tf.identity]
        net_param_dict["conv_filter_shapes"] = [
            [receptive_pixels, receptive_pixels,
                configuration_dict['image_channels'], n_features],
            [receptive_pixels, receptive_pixels, n_features,
                configuration_dict['feature_multiplier'] * n_features]
                                               ]
        net_param_dict["bias_shapes"] = [
            [1, configuration_dict['image_height'],
                configuration_dict['image_width'], n_features],
            [1, int(np.ceil(configuration_dict['image_height']/2)),
                int(np.ceil(configuration_dict['image_width']/2)),
                configuration_dict['feature_multiplier']*n_features],
            [1, configuration_dict['classes']]]
        net_param_dict["ksizes"] = [
            [1, 2, 2,  1], [1, configuration_dict['image_height']//2,
                            configuration_dict['image_width']//2, 1]]
        net_param_dict["pool_strides"] = [[1, 2, 2, 1], [1, 2, 2, 1]]
        net_param_dict["topdown_filter_shapes"] = [
            [3, 3, configuration_dict['image_channels'],
                feature_multiplier * n_features]]
        net_param_dict["topdown_output_shapes"] = [
            [configuration_dict['batchsize'],
                configuration_dict['image_height'],
                configuration_dict['image_width'],
                configuration_dict['image_channels']]]

        return net_param_dict

    if custom_net_parameters:
        net_parameters = custom_net_parameters
    else:
        net_parameters = get_net_parameters(configuration_dict)

    # copy necessary items from configuration
    net_parameters['connectivity'] = configuration_dict['connectivity']
    net_parameters['batchnorm'] = configuration_dict['batchnorm']

    return NetworkClass(name, net_parameters, is_training, keep_prob)


class NetworkClass(bb.ComposedModule):
    def define_inner_modules(self, name, net_param_dict,
                             is_training, keep_prob):
        self.net_params = net_param_dict
        self.layers = {}
        # TODO: integrate into net_parameters
        L = 2
        T = 4

        with tf.name_scope('initialization'):
            self.layers['image'] = DoNothingModule()
            self.layers['r_ff_0'] = DoNothingModule()
            self.layers['r_fb_0'] = DoNothingModule()
            self.layers['r_init_0'] = DoNothingModule()

            for l in range(L):
                self.layers['r_init_{}'.format(l+1)] = bb.TimeConvolutionalLayerModule()
        with tf.name_scope('main_circuit'):
            for l in range(L, 0, -1):
                self.layers['takelast_for_p_{}'.format(l-1)] = TakeLastInputModule()
                self.layers['p_{}'.format(l-1)] = bb.Conv2DTransposeModule()
                if (l > 1):
                    self.layers['takelast_for_r_fb_{}'.format(l-1)] = TakeLastInputModule()
                    self.layers['r_fb_{}'.format(l-1)] = ARModule()
            for l in range(L):
                self.layers['e_{}'.format(l)] = SubtractModule()
                self.layers['r_ff_{}'.format(l+1)] = CARPModule()
                # TODO: weight sharing between init and recurrent modules
                self.layers['r_ff_{}'.format(l+1)].weights = self.layers['r_init_{}'.format(l+1)].weights

        with tf.name_scope('output_modules'):
            self.layers['gap'] = bb.GlobalAveragePoolingModule()
            self.layers['fc0'] = bb.FullyConnectedLayerModule()



        # connect all modules of the network in a meaningful way
        # -----

        with tf.name_scope('wiring_of_modules'):
            # initialization
            # -----

            self.layers['r_ff_0'].add_input(self.layers['image'])
            self.layers['r_fb_0'].add_input(self.layers['image'])
            self.layers['r_init_0'].add_input(self.layers['image'])

            self.layers['r_init_1'].add_input(self.layers['image'])
            for l in range(0, L):
                self.layers['r_init{}'.format(l+1)].add_input(self.layers['r_init_{}'.format(l)])

            # main circuit
            # -----

            # feedforward pass
            for l in range(L, 0, -1):
                # init connection
                self.layers['takelast_for_p_{}'.format(l-1)].add_input(self.layers['init_pool_{}'.format(l)])
                # recurrent connection
                self.layers['takelast_for_p_{}'.format(l-1)].add_input(self.layers['r_ff_{}'.format(l)], -1)
                # link
                self.layers['p_{}'.format(l-1)].add_input(self.layers['takelast_for_p_{}'.format(l-1)])
                if (l > 1):
                    # init connection
                    self.layers['takelast_for_r_fb_{}'.format(l-1)].add_input(self.layers['r_init_{}'.format(l-1)])
                    # recurrent connection
                    self.layers['takelast_for_r_fb_{}'.format(l-1)].add_input(self.layers['r_ff_{}'.format(l-1)], -1)
                    # link
                    self.layers['r_fb_{}'.format(l-1)].add_input(self.layers['takelast_for_r_fb_{}'.format(l-1)])
                    self.layers['r_fb_{}'.format(l-1)].add_input(self.layers['p_{}'.format(l-1)])

            # feedback pass
            for l in range(L):
                self.layers['e_{}'.format(l)].add_input(self.layers['r_ff_{}'.format(l)])
                self.layers['e_{}'.format(l)].add_input(self.layers['p_{}'.format(l)])
                self.layers['r_ff_{}'.format(l+1)].add_input(self.layers['r_fb_{}'.format(l+1)])
                self.layers['r_ff_{}'.format(l+1)].add_input(self.layers['e_{}'.format(l)])


            # output modules
            # -----

            self.layers['gap'].add_input(self.layers['r_ff_{}'.format(L)])
            self.layers['fc0'].add_input(self.layers['gap'])

        with tf.name_scope('input_output'):
            self.input_module = self.layers["image"]
            self.output_module = self.layers["fc0"]

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
