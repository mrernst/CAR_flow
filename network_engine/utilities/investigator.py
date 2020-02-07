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
import matplotlib.pyplot as plt
from tensorflow.python.training import checkpoint_utils as cu

# custom functions
# -----

import visualizer


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


def get_list_of_images(list_of_weights, stereo):
    for kernel in list_of_weights:
        kernel_name = kernel[0]
        kernel_value = kernel[1]
        kname = kernel_name.split('/')[1].split('_')[0] + '/kernels'
        receptive_pixels = kernel_value.shape[0].value
        if 'fc' in kname:
            pass
        elif 'conv0' in kname:
            if stereo:
                image.append(
                    tf.compat.v1.summary.image(kname, put_kernels_on_grid(
                        kname, tf.reshape(kernel_value,
                                          [2 * receptive_pixels,
                                              receptive_pixels, -1,
                                              network.
                                              net_params
                                              ['conv_filter_shapes'][0][-1]])),
                                     max_outputs=1))
            else:
                image.append(
                    tf.compat.v1.summary.image(kname, put_kernels_on_grid(
                        kname, kernel_value),
                        max_outputs=1))

        else:
            image.append(
                tf.compat.v1.summary.image(kname, put_kernels_on_grid(
                    kname, tf.reshape(
                        kernel_value, [receptive_pixels, receptive_pixels,
                                       1, -1])), max_outputs=1))

    return image


# model path from config
# -----

def get_model_paths_from_cfg():
    return ''

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

    if args.config_file:
        modelpath = get_model_paths_from_cfg(args.config_file)
    else:
        modelpath = '/Users/markus/Research/Code/saturn/experiments/001_noname_experiment/data/config0/BLT3_2l_fm1_d1.0_l20.0_bn1_bs100_lr0.003/mnist_0occ_Xp/28x28x1_grayscale_onehot/checkpoints'



    list_of_variables = cu.list_variables(modelpath)
    #get_list_of_images(list_of_variables, False)

    if True:
        conv0weights = cu.load_variable(modelpath, 'convolutional_layer_0/conv0_conv_var')
        viz = visualizer.put_kernels_on_grid(name='conv0_weights', kernel=tf.reshape(conv0weights,[2 * 3, 3, -1, 32]))
        viz_np = viz.numpy()
        plt.imshow(viz_np[0,:,:,0], cmap='gray')
        plt.show()

    else:
        conv0weights = tf.convert_to_tensor(conv0weights)
        viz = visualizer.put_kernels_on_grid(name='conv0_weights', kernel=conv0weights)
        viz_np = viz.numpy()
        plt.imshow(viz_np[0,:,:,0], cmap='gray')
        plt.show()

    conv1weights = cu.load_variable(modelpath, 'convolutional_layer_1/conv1_conv_var')
    conv1weights = tf.convert_to_tensor(conv1weights)
    viz = visualizer.put_kernels_on_grid(name='conv1_weights', kernel=conv1weights)
    viz_np = viz.numpy()
    plt.imshow(viz_np[0,:,:,0], cmap='gray')
    plt.show()

    lateral0weights = cu.load_variable(modelpath, 'lateral_layer_0/lateral0_var')
    lateral0weights = tf.convert_to_tensor(lateral0weights)
    viz = visualizer.put_kernels_on_grid(name='lateral0weights', kernel=lateral0weights)
    viz_np = viz.numpy()
    plt.imshow(viz_np[0,:,:,0], cmap='gray')
    plt.show()

    lateral1weights = cu.load_variable(modelpath, 'lateral_layer_1/lateral1_var')
    lateral1weights = tf.convert_to_tensor(lateral1weights)
    viz = visualizer.put_kernels_on_grid(name='lateral1weights', kernel=lateral1weights)
    viz_np = viz.numpy()
    plt.imshow(viz_np[0,:,:,0], cmap='gray')
    plt.show()

    topdown0weights = cu.load_variable(modelpath, 'topdown_layer_0/topdown0_var')
    topdown0weights = tf.convert_to_tensor(topdown0weights)
    viz = visualizer.put_kernels_on_grid(name='topdown0weights', kernel=topdown0weights)
    viz_np = viz.numpy()
    plt.imshow(viz_np[0,:,:,0], cmap='gray')
    plt.show()





# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
