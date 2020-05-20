#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# November 2019                                  _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# simplemrcnn.py                             oN88888UU[[[/;::-.        dP^
# network definition inspired by            dNMMNN888UU[[[/;:--.   .o@P^
# Spoerer 2017, with mult. input           ,MMMMMMN888UU[[/;::-. o@^
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
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np

# custom functions
# -----
import utilities.networks.buildingblocks as bb


def return_network_layers(connectivity):
    """
    return_network_layers takes a string, that describes the connectivity
    of the network i.e. "BLT", and returns the number of layers the
    network has.
    """
    return 2

# custom variants of the recurrent modules

class MultBiasModule(bb.VariableModule):
    """
    MultBiasModule inherits from VariableModule. It holds on to a bias
    variable, that can be added to another tensor.
    The difference to bb.BiasModule is that biases are initialized as a
    tensor of ones. Eventual input modules to MultBiasModule are disregarded.
    """

    def __init__(self, name, bias_shape):
        """
        Creates a BiasModule Object

        Args:
          name:                 string, name of the module
          bias_shape:           array, shape of the bias, i.e. [B,H,W,C]
        """
        self.bias_shape = bias_shape
        super().__init__(name, bias_shape)

    def operation(self, *args):
        """
        operation takes a BiasModule and returns the tensorflow variable which
        holds the variable created by create_variables
        """
        return self.bias

    def create_variables(self, name):
        """
        create_variables takes a BiasModule and a name and instantiates a
        tensorflow variable with the shape specified in the constructor.
        It returns nothing.

        @param name str, name can be accessed from self
        """
        self.bias = tf.Variable(tf.ones(shape=self.bias_shape), name=name)


class TimeConvolutionalLayerModule(bb.TimeComposedModule):
    """
    TimeConvolutionalLayerModule inherits from TimeComposedModule. This
    composed module adds up inputs, performs a convolution and applies a bias
    and an activation function. It does allow recursions on two different hooks
    (input and preactivation).
    """
    def define_inner_modules(self, name, activation, filter_shape, strides,
                             bias_shape, w_init_m=None, w_init_std=None,
                             padding='SAME'):
        # multiply_inputs=True

        self.input_module = bb.TimeAddModule(name + "_input")
        self.conv = bb.Conv2DModule(name + "_conv", filter_shape, strides,
                                    w_init_m, w_init_std, padding=padding)
        self.adder = bb.AddModule(
            name + "_add_bias")
        self.bias = MultBiasModule(name + "_bias", bias_shape)
        self.preactivation = bb.TimeMultModule(name + "_preactivation")

        self.output_module = bb.ActivationModule(name + "_output", activation)

        # wiring of modules
        self.conv.add_input(self.input_module)
        self.adder.add_input(self.conv)
        self.adder.add_input(self.bias)

        self.preactivation.add_input(self.adder, 0)
        self.output_module.add_input(self.preactivation)


class TimeConvolutionalLayerWithBatchNormalizationModule(
        bb.TimeComposedModule):
    """
    TimeConvolutionalLayerWithBatchNormalizationModule inherits from
    TimeComposedModule. This composed module performs a convolution,
    applies a bias, batchnormalizes the preactivation and then applies an
    activation function. It does allow recursions on two different hooks
    (input and preactivation)
    """
    def define_inner_modules(self, name, n_out, is_training, beta_init,
                             gamma_init, ema_decay_rate, activation,
                             filter_shape, strides, bias_shape,
                             w_init_m=None, w_init_std=None, padding='SAME'):

        self.input_module = bb.TimeAddModule(name + "_input")
        self.conv = bb.Conv2DModule(name + "_conv", filter_shape, strides,
                                    w_init_m, w_init_std, padding=padding)
        self.preactivation = bb.TimeMultModule(name + "_preactivation")
        self.batchnorm = bb.BatchNormalizationModule(name + "_batchnorm",
                                                     n_out,
                                                     is_training, beta_init,
                                                     gamma_init,
                                                     ema_decay_rate,
                                                     moment_axes=[0, 1, 2],
                                                     variance_epsilon=1e-3)
        self.output_module = bb.ActivationModule(name + "_output", activation)

        # wiring of modules
        self.conv.add_input(self.input_module)
        self.batchnorm.add_input(self.conv)
        self.preactivation.add_input(self.batchnorm, 0)
        self.output_module.add_input(self.preactivation)


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
            [1, 2, 2, 1], [1, 2, 2, 1]]
        net_param_dict["pool_strides"] = [[1, 2, 2, 1], [1, 2, 2, 1]]
        net_param_dict["topdown_filter_shapes"] = [
            [receptive_pixels, receptive_pixels, n_features,
                feature_multiplier * n_features]]
        net_param_dict["topdown_output_shapes"] = [
            [configuration_dict['batchsize'],
                configuration_dict['image_height'],
                configuration_dict['image_width'],
                n_features]]

        net_param_dict["global_weight_init_mean"] = \
            configuration_dict['global_weight_init_mean']
        net_param_dict["global_weight_init_std"] = \
            configuration_dict['global_weight_init_std']

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

        with tf.name_scope('convolutional_layer_0'):
            if self.net_params['batchnorm']:
                self.layers["conv0"] = \
                    TimeConvolutionalLayerWithBatchNormalizationModule(
                        "conv0", self.net_params['bias_shapes'][0][-1],
                        is_training, 1.0, 1.0, 0.5,
                        self.net_params['activations'][0],
                        self.net_params['conv_filter_shapes'][0],
                        [1, 1, 1, 1], self.net_params['bias_shapes'][0],
                        self.net_params['global_weight_init_mean'],
                        self.net_params['global_weight_init_std'])
            else:
                self.layers["conv0"] = TimeConvolutionalLayerModule(
                    "conv0", self.net_params['activations'][0],
                    self.net_params['conv_filter_shapes'][0],
                    [1, 1, 1, 1], self.net_params['bias_shapes'][0],
                    self.net_params['global_weight_init_mean'],
                    self.net_params['global_weight_init_std'])
        if 'L' in self.net_params['connectivity']:
            with tf.name_scope('lateral_layer_0'):
                # mutability kicked my ass here
                lateral_filter_shape = \
                    self.net_params['conv_filter_shapes'][0].copy()
                tmp = lateral_filter_shape[2]
                lateral_filter_shape[2] = lateral_filter_shape[3]
                self.layers["lateral0"] = bb.Conv2DModule(
                    "lateral0", lateral_filter_shape, [1, 1, 1, 1],
                    self.net_params['global_weight_init_mean'],
                    self.net_params['global_weight_init_std'])
                self.layers["lateral0_batchnorm"] = \
                    bb.BatchNormalizationModule(
                    "lateral0_batchnorm", lateral_filter_shape[-1],
                    is_training, beta_init=1.0, gamma_init=0.1,
                    ema_decay_rate=0.5, moment_axes=[0, 1, 2],
                    variance_epsilon=1e-3)
                self.layers["l0_adder"] = bb.AddModule(
                    name + "lateral0_add_bias")
                self.layers["l0_multbias"] = \
                    MultBiasModule(name + "_bias",
                                   self.net_params['bias_shapes'][0])

        with tf.name_scope('pooling_layer_0'):
            self.layers["pool0"] = bb.MaxPoolingModule(
                "pool0", self.net_params['ksizes'][0],
                self.net_params['pool_strides'][0])

        with tf.name_scope('dropout_layer_0'):
            self.layers['dropoutc0'] = bb.DropoutModule(
                'dropoutc0', keep_prob=keep_prob)

        with tf.name_scope('convolutional_layer_1'):
            if self.net_params['batchnorm']:
                self.layers["conv1"] = \
                    TimeConvolutionalLayerWithBatchNormalizationModule(
                        "conv1", self.net_params['bias_shapes'][1][-1],
                        is_training, 1.0, 1.0, 0.5,
                        self.net_params['activations'][1],
                        self.net_params['conv_filter_shapes'][1],
                        [1, 1, 1, 1], self.net_params['bias_shapes'][1],
                        self.net_params['global_weight_init_mean'],
                        self.net_params['global_weight_init_std'])
            else:
                self.layers["conv1"] = TimeConvolutionalLayerModule(
                    "conv1", self.net_params['activations'][1],
                    self.net_params['conv_filter_shapes'][1],
                    [1, 1, 1, 1], self.net_params['bias_shapes'][1],
                    self.net_params['global_weight_init_mean'],
                    self.net_params['global_weight_init_std'])

        if 'T' in self.net_params['connectivity']:
            with tf.name_scope('topdown_layer_0'):
                self.layers["topdown0"] = bb.Conv2DTransposeModule(
                    "topdown0", self.net_params['topdown_filter_shapes'][0],
                    [1, 2, 2, 1], self.net_params['topdown_output_shapes'][0],
                    self.net_params['global_weight_init_mean'],
                    self.net_params['global_weight_init_std'])
                self.layers["topdown0_batchnorm"] = \
                    bb.BatchNormalizationModule(
                    "topdown0_batchnorm",
                    self.net_params['topdown_output_shapes'][0][-1],
                    is_training, beta_init=0.0, gamma_init=0.1,
                    ema_decay_rate=0.5, moment_axes=[0, 1, 2],
                    variance_epsilon=1e-3)
                self.layers["t0_adder"] = bb.AddModule(
                    name + "topdown0_add_bias")
                self.layers["t0_multbias"] = \
                    MultBiasModule(name + "_bias",
                                   self.net_params['bias_shapes'][0])

        if 'L' in self.net_params['connectivity']:
            with tf.name_scope('lateral_layer_1'):
                lateral_filter_shape = self.net_params['conv_filter_shapes'][1]
                tmp = lateral_filter_shape[2]
                lateral_filter_shape[2] = lateral_filter_shape[3]
                self.layers["lateral1"] = bb.Conv2DModule(
                    "lateral1", lateral_filter_shape, [1, 1, 1, 1],
                    self.net_params['global_weight_init_mean'],
                    self.net_params['global_weight_init_std'])
                self.layers["lateral1_batchnorm"] = \
                    bb.BatchNormalizationModule(
                    "lateral1_batchnorm", lateral_filter_shape[-1],
                    is_training, beta_init=1.0, gamma_init=0.1,
                    ema_decay_rate=0.5, moment_axes=[0, 1, 2],
                    variance_epsilon=1e-3)
                self.layers["l1_adder"] = bb.AddModule(
                    name + "lateral1_add_bias")
                self.layers["l1_multbias"] = \
                    MultBiasModule(name + "_bias",
                                   self.net_params['bias_shapes'][1])

        with tf.name_scope('pooling_layer_1'):
            self.layers["pool1"] = bb.MaxPoolingModule(
                "pool1", self.net_params['ksizes'][0],
                self.net_params['pool_strides'][1])
            self.layers["flatpool1"] = bb.FlattenModule("flatpool1")

        with tf.name_scope('dropout_layer_1'):
            self.layers['dropoutc1'] = bb.DropoutModule(
                'dropoutc1', keep_prob=keep_prob)

        with tf.name_scope('fully_connected_layer_0'):
            if self.net_params['batchnorm']:
                self.layers["fc0"] = \
                    bb.FullyConnectedLayerWithBatchNormalizationModule(
                        "fc0", self.net_params['bias_shapes'][-1][-1],
                        is_training, 0.0, 1.0, 0.5,
                        self.net_params['activations'][2],
                        int(np.prod(
                            np.array(self.net_params['bias_shapes'][1]) /
                            np.array(self.net_params['pool_strides'][1]))),
                        np.prod(self.net_params['bias_shapes'][2]),
                        self.net_params['global_weight_init_mean'],
                        self.net_params['global_weight_init_std'])
            else:
                self.layers["fc0"] = \
                    bb.FullyConnectedLayerModule(
                        "fc0", self.net_params['activations'][2],
                        int(np.prod(
                            np.array(self.net_params['bias_shapes'][1]) /
                            np.array(self.net_params['pool_strides'][1]))),
                        np.prod(self.net_params['bias_shapes'][2]),
                        self.net_params['global_weight_init_mean'],
                        self.net_params['global_weight_init_std'])

        # connect all modules of the network in a meaningful way
        # -----

        with tf.name_scope('wiring_of_modules'):
            self.layers["pool0"].add_input(self.layers["conv0"])
            self.layers["dropoutc0"].add_input(self.layers["pool0"])
            self.layers["conv1"].add_input(self.layers["dropoutc0"], 0)
            self.layers["pool1"].add_input(self.layers["conv1"])
            self.layers["dropoutc1"].add_input(self.layers["pool1"])
            self.layers["flatpool1"].add_input(self.layers["dropoutc1"])
            self.layers["fc0"].add_input(self.layers["flatpool1"])
            if "L" in self.net_params['connectivity']:
                if self.net_params['batchnorm']:
                    self.layers["lateral0"].add_input(
                        self.layers["conv0"])
                    self.layers["lateral0_batchnorm"].add_input(
                        self.layers["lateral0"])
                    self.layers["conv0"].preactivation.add_input(
                        self.layers["lateral0_batchnorm"], -1)
                    self.layers["lateral1"].add_input(
                        self.layers["conv1"])
                    self.layers["lateral1_batchnorm"].add_input(
                        self.layers["lateral1"])
                    self.layers["conv1"].preactivation.add_input(
                        self.layers["lateral1_batchnorm"], -1)
                else:
                    self.layers["lateral0"].add_input(
                        self.layers["conv0"])
                    self.layers["l0_adder"].add_input(
                        self.layers["lateral0"])
                    self.layers["l0_adder"].add_input(
                        self.layers["l0_multbias"])
                    self.layers["conv0"].preactivation.add_input(
                        self.layers["l0_adder"], -1)
                    self.layers["lateral1"].add_input(
                        self.layers["conv1"])
                    self.layers["l1_adder"].add_input(
                        self.layers["lateral1"])
                    self.layers["l1_adder"].add_input(
                        self.layers["l1_multbias"])
                    self.layers["conv1"].preactivation.add_input(
                        self.layers["l1_adder"], -1)

            if "T" in self.net_params['connectivity']:
                if self.net_params['batchnorm']:
                    self.layers["topdown0_batchnorm"].add_input(
                        self.layers["topdown0"])
                    self.layers["conv0"].preactivation.add_input(
                        self.layers["topdown0_batchnorm"], -1)
                    self.layers["topdown0"].add_input(
                        self.layers["conv1"])
                else:
                    self.layers["topdown0"].add_input(
                        self.layers["conv1"])
                    self.layers["t0_adder"].add_input(
                        self.layers["topdown0"])
                    self.layers["t0_adder"].add_input(
                        self.layers["t0_multbias"])
                    self.layers["conv0"].preactivation.add_input(
                        self.layers["t0_adder"], -1)

        with tf.name_scope('input_output'):
            self.input_module = self.layers["conv0"]
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
