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
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.#
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
        receptive_pixels = 3  # 3
        n_features = 32
        feature_multiplier = configuration_dict['feature_multiplier']

        if "F" in configuration_dict['connectivity']:
            n_features = 64
        if "K" in configuration_dict['connectivity']:
            receptive_pixels = 5  # 5

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
                    bb.TimeConvolutionalLayerWithBatchNormalizationModule(
                        "conv0", self.net_params['bias_shapes'][0][-1],
                        is_training, 0.0, 1.0, 0.5,
                        self.net_params['activations'][0],
                        self.net_params['conv_filter_shapes'][0],
                        [1, 1, 1, 1], self.net_params['bias_shapes'][0],
                        self.net_params['global_weight_init_mean'],
                        self.net_params['global_weight_init_std'])
            else:
                self.layers["conv0"] = bb.TimeConvolutionalLayerModule(
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
                    is_training, beta_init=0.0, gamma_init=0.1,
                    ema_decay_rate=0.5, moment_axes=[0, 1, 2],
                    variance_epsilon=1e-3)

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
                    bb.TimeConvolutionalLayerWithBatchNormalizationModule(
                        "conv1", self.net_params['bias_shapes'][1][-1],
                        is_training, 0.0, 1.0, 0.5,
                        self.net_params['activations'][1],
                        self.net_params['conv_filter_shapes'][1],
                        [1, 1, 1, 1], self.net_params['bias_shapes'][1],
                        self.net_params['global_weight_init_mean'],
                        self.net_params['global_weight_init_std'])
            else:
                self.layers["conv1"] = bb.TimeConvolutionalLayerModule(
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
                    is_training, beta_init=0.0, gamma_init=0.1,
                    ema_decay_rate=0.5, moment_axes=[0, 1, 2],
                    variance_epsilon=1e-3)

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
                    self.layers["conv0"].preactivation.add_input(
                        self.layers["lateral0"], -1)
                    self.layers["lateral1"].add_input(
                        self.layers["conv1"])
                    self.layers["conv1"].preactivation.add_input(
                        self.layers["lateral1"], -1)
            if "T" in self.net_params['connectivity']:
                if self.net_params['batchnorm']:
                    self.layers["topdown0_batchnorm"].add_input(
                        self.layers["topdown0"])
                    self.layers["conv0"].preactivation.add_input(
                        self.layers["topdown0_batchnorm"], -1)
                    self.layers["topdown0"].add_input(
                        self.layers["conv1"])
                else:
                    self.layers["conv0"].preactivation.add_input(
                        self.layers["topdown0"], -1)
                    self.layers["topdown0"].add_input(
                        self.layers["conv1"])
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

