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
# TakeLastInput


class InitialRepresentation(bb.ComposedModule):
    def define_inner_modules(
        self, name, n_out, is_training, beta_init,
        gamma_init, ema_decay_rate, activation,
            filter_shape, strides, bias_shape):
        self.input_module = bb.ConvolutionalLayerWithBatchNormalizationModule(
            name, n_out, is_training, beta_init,
            gamma_init, ema_decay_rate, activation,
            filter_shape, strides, bias_shape,
            padding='SAME')
        self.output_module = DoNothingModule(name + "_output")
        self.output_module.add_input(self.input_module)


class Prediction(bb.ComposedModule):
    def define_inner_modules(self, name, filter_shape, strides, output_shape):
        self.input_module = bb.Conv2DTransposeModule(
            name + "_deconv", filter_shape, strides, output_shape)
        # wiring of modules
        self.output_module = DoNothingModule(name + "_output")
        self.output_module.add_input(self.input_module)


class FBRepresentation(bb.OperationModule):
    def define_inner_modules(self, name, activation_function):
        # self.beta = tf.variable(0.5, trainable=True)
        # self.sigma2 = tf.variable(1.0, trainable=False)
        # self.b = 2*tf.abs(self.beta) / self.sigma2
        self.b = tf.variable(0.5, trainable=True)
        # wiring of modules
        self.input_module = WeightedSum(name + "_input", 1.-self.b, self.b)
        self.output_module = bb.ActivationModule(name + "_output", activation_function)
        self.output_module.add_input(self.input_module)


class PredictionError(bb.AddModule):
    def operation(self, *arg):
        arg1 = args[0]
        arg2 = args[-1]
        return tf.subtract(arg1, arg2, name=self.name)


class FFRepresentation(bb.ComposedModule):
    def define_inner_modules(self, name, name, n_out, is_training, beta_init,
                             gamma_init, ema_decay_rate, activation,
                             filter_shape, strides, bias_shape):
        # self.alpha = tf.variable(1.0, trainable=True)
        # self.sigma2 = tf.variable(1.0, trainable=False)
        # self.a = 2*tf.abs(self.alpha) / self.sigma2
        self.a = tf.variable(1.0, trainable=True)


        self.input_module = bb.ConvolutionalLayerWithBatchNormalizationModule(
            name, n_out, is_training, beta_init,
            gamma_init, ema_decay_rate, activation,
            filter_shape, strides, bias_shape,
            padding='SAME')
        self.old_representation = bb.AddModule(name + "_oldrep")  # hook to 2nd input
        self.addition = WeightedSum(name + "_addition", 1., self.a)
        self.output_module = bb.ActivationModule()
        # wiring of modules
        self.addition.add_input(self.old_representation)
        self.addition.add_input(self.input_module)
        self.output_module.add_input(self.addition)



class TakeLastInput(bb.TimeOperationModule):
    def operation(self, *arg):
        return args[-1]


class DoNothingModule(TakeLastInput):
    pass

class WeightedSum(bb.OperationModule):
    def __init__(self, name, w1, w2):
        self.w1 = w1
        self.w2 = w2
    def operation(self, *arg):
        ret = (args[0]*self.w1 + args[1]+self.w2)
        return ret


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
        net_param_dict['network_depth'] = configuration_dict['network_depth']

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
        L = net_param_dict['network_depth']

        with tf.name_scope('initialization'):
            self.layers['image'] = DoNothingModule('image')
            self.layers['r_ff_0'] = DoNothingModule('r_ff_0')
            self.layers['r_fb_0'] = DoNothingModule('r_fb_0')
            self.layers['r_init_0'] = DoNothingModule('r_init_0')

            for l in range(L):
                self.layers['r_init_{}'.format(l+1)] = InitialRepresentation(
                    name='r_init_{}'.format(l+1),
                    n_out=,
                    is_training=,
                    beta_init=,
                    gamma_init=,
                    ema_decay_rate=,
                    activation=,
                    filter_shape=,
                    strides=,
                    bias_shape=,
                )
        with tf.name_scope('main_circuit'):
            for l in range(L, 0, -1):
                self.layers['takelast_p_{}'.format(l-1)] = TakeLastInput('takelast_p_{}')
                self.layers['p_{}'.format(l-1)] = Prediction(
                    name='p_{}'.format(l-1),
                    filter_shape=,
                    strides=,
                    output_shape=
                )
                if (l > 1):
                    self.layers['takelast_r_fb_{}'.format(l-1)] = TakeLastInput('takelast_r_fb_{}')
                    self.layers['r_fb_{}'.format(l-1)] = FBRepresentation(
                        name='r_fb_{}'.format(l-1),
                        n_out=,
                        is_training=,
                        beta_init=,
                        gamma_init=,
                        ema_decay_rate=,
                        activation=,
                        filter_shape=,
                        strides=,
                        bias_shape=
                    )
            for l in range(L):
                self.layers['e_{}'.format(l)] = PredictionError('e_{}'.format(l))
                self.layers['r_ff_{}'.format(l+1)] = FFRepresentation(
                    name='r_ff_{}'.format(l+1),
                    n_out=,
                    is_training=,
                    beta_init=,
                    gamma_init=,
                    ema_decay_rate=,
                    activation=,
                    filter_shape=,
                    strides=,
                    bias_shape=
                )
                # weight sharing between init and recurrent modules
                self.layers['r_ff_{}'.format(l+1)].input_module.input_module.weights = self.layers['r_init_{}'.format(l+1)].input_module.input_module.weights

        with tf.name_scope('output_modules'):
            self.layers['gap'] = bb.GlobalAveragePoolingModule('gap')
            self.layers['fc0'] = bb.FullyConnectedLayerModule(
                name='fc0',
                activation=tf.identity,
                in_size=,
                out_size=)



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
                self.layers['takelast_p_{}'.format(l-1)].add_input(self.layers['init_pool_{}'.format(l)])
                # recurrent connection
                self.layers['takelast_p_{}'.format(l-1)].add_input(self.layers['r_ff_{}'.format(l)], -1)
                # link
                self.layers['p_{}'.format(l-1)].add_input(self.layers['takelast_p_{}'.format(l-1)])
                if (l > 1):
                    # init connection
                    self.layers['takelast_r_fb_{}'.format(l-1)].add_input(self.layers['r_init_{}'.format(l-1)])
                    # recurrent connection
                    self.layers['takelast_r_fb_{}'.format(l-1)].add_input(self.layers['r_ff_{}'.format(l-1)], -1)
                    # link
                    self.layers['r_fb_{}'.format(l-1)].add_input(self.layers['takelast_r_fb_{}'.format(l-1)])
                    self.layers['r_fb_{}'.format(l-1)].add_input(self.layers['p_{}'.format(l-1)])

            # feedback pass
            for l in range(L):
                self.layers['e_{}'.format(l)].add_input(self.layers['r_ff_{}'.format(l)])
                self.layers['e_{}'.format(l)].add_input(self.layers['p_{}'.format(l)])
                self.layers['r_ff_{}'.format(l+1)].add_input(self.layers['e_{}'.format(l)])
                self.layers['r_ff_{}'.format(l+1)].old_representation.add_input(self.layers['r_fb_{}'.format(l+1)])


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
