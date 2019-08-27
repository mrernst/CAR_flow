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
import utilities.tfrecord_handler as tfrecord_handler
from utilities.networks.buildingblocks import softmax_cross_entropy, \
    sigmoid_cross_entropy
# helper functions
# -----


def get_output_directory(configuration_dict):
    """
    get_output_directory takes a dict configuration_dict and established the
    directory structure for the configured experiment. It returns paths to
    the checkpoints and the writer directories.
    """

    writer_directory = '{}/{}/{}/'.format(
        configuration_dict['output_dir'],
        configuration_dict['exp_name'],
        configuration_dict['name'])

    # architecture string
    architecture_string = ''
    architecture_string += '{}{}_{}l_fm{}_d{}_l2{}'.format(
        configuration_dict['connectivity'],
        configuration_dict['timedepth'],
        configuration_dict['network_depth'],
        configuration_dict['feature_mult'],
        configuration_dict['keep_prob'],
        configuration_dict['l2_lambda'])

    if configuration_dict['batchnorm']:
        architecture_string += '_bn1'
    else:
        architecture_string += '_bn0'
    architecture_string += '_bs{}'.format(configuration_dict['batchsize'])
    if configuration_dict['decaying_lrate']:
        architecture_string += '_lr{}-{}-{}'.format(
            configuration_dict['lr_eta'],
            configuration_dict['lr_delta'],
            configuration_dict['lr_d'])
    else:
        architecture_string += '_lr{}'.format(
            configuration_dict['learning_rate'])

    # data string
    data_string = ''
    if ('ycb' in configuration_dict['dataset']):
        data_string += "{}_{}occ_{}p".format(
            configuration_dict['dataset'],
            configuration_dict['n_occluders'],
            configuration_dict['occlusion_percentage'])
    else:
        data_string += "{}_{}occ_Xp".format(
            configuration_dict['dataset'],
            configuration_dict['n_occluders'])

    # format string
    format_string = ''
    format_string += '{}x{}x{}'.format(
        configuration_dict['image_height'],
        configuration_dict['image_width'],
        configuration_dict['image_channels'])
    format_string += "_{}_{}".format(
        configuration_dict.color,
        configuration_dict.label_type)

    writer_directory += "{}/{}/{}/".format(architecture_string,
                                           data_string, format_string)

    checkpoint_directory = writer_directory + 'checkpoints/'

    # make sure the directories exist, otherwise create them
    mkdir_p(checkpoint_directory)

    return writer_directory, checkpoint_directory


def get_input_directory(configuration_dict):
    """
    get_input_directory takes a dict configuration_dict and returns a path to
    the image_data and a parser for the tf_records files.
    """

    if configuration_dict['dataset'] == "osycb":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'osycb/tfrecord-files/{}occ/{}p/{}/'.format(
            configuration_dict['n_occluders'],
            configuration_dict['occlusion_percentage'],
            configuration_dict['downsampling'])
        parser = tfrecord_handler._osycb_parse_single
    elif configuration_dict['dataset'] == "osmnist":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'osmnist/tfrecord-files/{}occ/'.format(
            configuration_dict['n_occluders'])
        parser = tfrecord_handler._osmnist_parse_single
    elif configuration_dict['dataset'] == "mnist":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'mnist/tfrecord-files/{}occ/'.format(
            configuration_dict['n_occluders'])
        parser = tfrecord_handler._mnist_parse_single
    elif configuration_dict['dataset'] == "osdigit":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'digit_database/tfrecord-files/{}{}/'.format(
            configuration_dict['n_occluders']-1,
            configuration_dict['dataset'])
        parser = tfrecord_handler._osdigits_parse_single
    elif configuration_dict['dataset'] == "digit":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'digit_database/tfrecord-files/{}{}/'.format(
            configuration_dict['n_occluders']-1,
            configuration_dict['dataset'])
        parser = tfrecord_handler._digits_parse_single
    else:
        print("[INFO] Dataset not defined")
        pass

    return tfrecord_dir, parser


def get_image_files(training_dir, validation_dir, test_dir, evaluation_dir,
                    input_directory, dataset, n_occluders, downsampling):
    """
    get_image_files takes paths to tfrecord_dir, training_dir, validation_dir
    test_dir and evaluation_dir and returns the corresponding file names.
    """
    list_of_dirs = [training_dir, validation_dir, test_dir, evaluation_dir]
    list_of_types = ['train', 'validation', 'test', 'evaluation']
    list_of_files = []
    for dir in list_of_dirs:
        if dir:
            if dir == 'all':
                type = list_of_types[list_of_dirs.index(dir)]
                list_of_files.append(
                    tfrecord_handler.all_percentages_tfrecord_paths(
                        type, dataset, input_directory,
                        n_occluders, downsampling))
            else:
                list_of_files.append(
                    tfrecord_handler.tfrecord_auto_traversal(dir))
        else:
            list_of_files.append(tfrecord_handler.tfrecord_auto_traversal(
                dir + type + '/'))
    training, validation, testing, evaluation = list_of_files
    if len(testing) == 0:
        print('[INFO] No test-set found, using validation-set instead')
        testing += validation
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
        configuration_dict['class_encoding'] = tfrecord_handler.OSYCB_ENCODING
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
        configuration_dict['class_encoding'] = np.array(
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    else:
        configuration_dict['image_height'] = 32
        configuration_dict['image_width'] = 32
        configuration_dict['image_channels'] = 1
        configuration_dict['classes'] = 10
        configuration_dict['class_encoding'] = np.array(
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

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
