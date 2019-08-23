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
import csv
import os
from utilities.tfrecord_handler import OSYCB_ENCODING
from utilities.networks.buildingblocks import softmax_cross_entropy, \
    sigmoid_cross_entropy

# helper functions
# -----

#TODO:
def handle_io_dirs(configuration_dict):
    """
    handle_io_dirs takes a dict configuration_dict and established the
    directory structure for the configured experiment. It returns paths to
    the checkpoints and the image_data.
    """
    pass
#TODO:
def get_image_files(tfrecord_dir,
        training_dir, validation_dir, test_dir, evaluation_dir):
    """

    """
    list_of_dirs = [training_dir, validation_dir, test_dir, evaluation_dir]
    list_of_files = []
    for dir in list_of_dirs:
        if dir:
            if dir == 'all':
                list_of_files.append()
            else:
                list_of_files.append()
        else:
            list_of_files.append()
    training, validation, testing, evaluation = list_of_files
    return training, validation, testing, evaluation


def infer_additional_parameters(configuration_dict):
    """
    infer_additional_parameters takes a dict configuration_dict and infers
    additional parameters on the grounds of dataset etc.
    """
    # define correct network parameters
    # -----

    if ('ycb' in configuration_dict['dataset']):
        configuration_dict['image_height'] = 240
        configuration_dict['image_width'] = 320
        configuration_dict['image_channels'] = 3
        configuration_dict['classes'] = 80
        configuration_dict['class_encoding'] = OSYCB_ENCODING
        if configuration_dict['downsampling'] == 'ds2':
            configuration_dict['image_height'] //= 2
            configuration_dict['image_width'] //= 2
        elif configuration_dict['downsampling'] == 'ds4':
            configuration_dict['image_height'] //= 4
            configuration_dict['image_width'] //= 4
    elif (('mnist' in configuration_dict['dataset']) and
            not('os' in configuration_dict['dataset'])):
        configuration_dict['image_height'] = 28
        configuration_dict['image_width'] = 28
        configuration_dict['image_channels'] = 1
        configuration_dict['classes'] = 10
        configuration_dict['class_encoding'] = \
            np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    else:
        configuration_dict['image_height'] = 32
        configuration_dict['image_width'] = 32
        configuration_dict['image_channels'] = 1
        configuration_dict['classes'] = 10
        configuration_dict['class_encoding'] = \
            np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    if configuration_dict['color'] == 'grayscale':
        configuration_dict['image_channels'] = 1
    if configuration_dict['stereo']:
        configuration_dict['image_channels'] *= 2

    # use sigmoid for n-hot task, otherwise softmax
    if configuration_dict['label_type'] == 'nhot':
        configuration_dict['crossentropy_fn'] = sigmoid_cross_entropy
    else:
        configuration_dict['crossentropy_fn'] = softmax_cross_entropy

    # change the image height and image width if the network is supposed
    # to crop the images
    if configuration_dict['cropped'] or configuration_dict['augmented']:
        configuration_dict['image_height'] = configuration_dict['image_width']\
            // 10 * 4
        configuration_dict['image_width'] = configuration_dict['image_width']\
            // 10 * 4

    return configuration_dict


def read_config_file(path_to_config_file):
    """
    read_config_file takes a string path_to_config_file and returns a
    dict config_dict with all the keys and values from the csv file.
    """
    config_dict = {}
    with open(path_to_config_file) as config_file:
        csvReader = csv.reader(config_file)
        for key, value in csvReader:
            config_dict[key] = value

    config_dict['config_file'] = path_to_config_file
    return convert_config_types(config_dict)


def convert_config_types(config_dictionary):
    for key, value in config_dictionary.items():
        try:
            if '.' in value:
                config_dictionary[key] = float(value)
            else:
                config_dictionary[key] = int(value)
        except(ValueError, TypeError):
            pass
    return config_dictionary


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


def print_tensor_info(tensor, name=None):
    """
    Takes a tf.tensor and returns name and shape for debugging purposes
    """
    name = name if name else tensor.name
    text = "[DEBUG] name = {}\tshape = {}"
    print(text.format(name, tensor.shape.as_list()))
    pass


def largest_indices(arr, n):
    """
    Returns the n largest indices from a numpy array.
    """
    flat_arr = arr.flatten()
    indices = np.argpartition(flat_arr, -n)[-n:]
    indices = indices[np.argsort(-flat_arr[indices])]
    return np.unravel_index(indices, arr.shape)


def print_misclassified_objects(cm, n_obj=5):
    """
    prints out the n_obj misclassified objects given a
    confusion matrix array cm.
    """
    np.fill_diagonal(cm, 0)
    maxind = largest_indices(cm, n_obj)
    most_misclassified = encoding[maxind[0]]
    classified_as = encoding[maxind[1]]
    print('most misclassified:', most_misclassified)
    print('classified as:', classified_as)
    pass

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
