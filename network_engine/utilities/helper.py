#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# August 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# helper.py                                  oN88888UU[[[/;::-.        dP^
# helper functions / IO                     dNMMNN888UU[[[/;:--.   .o@P^
#                                          ,MMMMMMN888UU[[/;::-. o@^
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
# THE SOFTWARE.
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
import csv
import os
import sys
import errno
import importlib
import utilities.tfrecord_handler as tfrecord_handler
import utilities.tfevent_handler as tfevent_handler
from utilities.visualizer import put_kernels_on_grid, put_activations_on_grid
from utilities.networks.buildingblocks import softmax_cross_entropy, \
    sigmoid_cross_entropy
# helper functions
# -----


def compile_list_of_additional_summaries(network, time_depth,
                                         time_depth_beyond):
    add = []
    with tf.name_scope('histograms'):
        list_of_weights = network.get_all_weights()
        list_of_biases = network.get_all_biases()

        for kernel in list_of_weights:
            kname = kernel.name.split('/')[1].split('_')[0] + '_weights'
            add.append(tf.summary.histogram(kname, kernel))

    # with tf.name_scope('extras'):
    #     # this is the space to look at some interesting things that are not
    #     # necessarily defined in every network and accessible
    #
    #     add.append(tf.summary.histogram(
    #         'conv0_pre_activations_{}'.format(time_depth),
    #         network.layers["conv0"].preactivation.outputs[time_depth]))
    #     add.append(tf.summary.histogram(
    #         'conv0_activations_{}'.format(time_depth),
    #         network.layers["conv0"].outputs[time_depth]))
    #     add.append(tf.summary.histogram(
    #         'conv1_pre_activations_{}'.format(time_depth),
    #         network.layers["conv1"].preactivation.outputs[time_depth]))
    #     add.append(tf.summary.histogram(
    #         'conv1_activations_{}'.format(time_depth),
    #         network.layers["conv1"].outputs[time_depth]))
    #     add.append(tf.summary.histogram(
    #         'fc0_pre_activations_{}'.format(time_depth),
    #         network.layers["fc0"].preactivation.outputs[time_depth]))
    #
    #     # gets you information for every time-step and more information
    #     add += tfevent_handler.module_variable_summary(
    #         network.layers["conv0"].preactivation)
    #     add += tfevent_handler.module_variable_summary(
    #         network.layers["conv0"])
    #     add += tfevent_handler.module_variable_summary(
    #         network.layers["conv0"].preactivation)
    #     add += tfevent_handler.module_variable_summary(
    #         network.layers["conv0"])
    #     add += tfevent_handler.module_variable_summary(
    #         network.layers["fc0"])
    #
    #     with tf.name_scope('accuracy_and_error'):
    #         add += tfevent_handler.module_timedifference_summary(
    #             error, time_depth + time_depth_beyond)
    #         add += tfevent_handler.module_timedifference_summary(
    #             accuracy, time_depth + time_depth_beyond)
    #         add += tfevent_handler.module_timedifference_summary(
    #             network.layers["fc0"], time_depth)
    #         add += tfevent_handler.module_timedifference_summary(
    #             network.layers["conv0"], time_depth)
    #         add += tfevent_handler.module_timedifference_summary(
    #             network.layers["conv1"], time_depth)
    #
    #     with tf.name_scope('accuracy_and_error_beyond'):
    #         add += tfevent_handler.module_scalar_summary(error)
    #         add += tfevent_handler.module_scalar_summary(accuracy)
    #         add += tfevent_handler.module_scalar_summary(partial_accuracy)

    return add


def check_disqualified_strings(input_str, list_of_disqualified_str):
    """
    Given an str input_str and a list of str list_of_disqualified_str
    the function check_disqualified_strings returns true iff none of the
    strings in list_of_disqualified_str is part of the input_str
    """
    return_bool = True
    for dstr in list_of_disqualified_str:
        if dstr in input_str:
            return_bool = False
        else:
            pass

    return return_bool


def compile_list_of_image_summaries(network, stereo):

    image = []

    list_of_weights = network.get_all_weights()

    for kernel in list_of_weights:
        kname = kernel.name.split('/')[1].split('_')[0] + '/kernels'
        receptive_pixels = kernel.shape[0].value
        if 'fc' in kname:
            pass
        elif 'conv0' in kname:
            if stereo:
                image.append(
                    tf.summary.image(kname, put_kernels_on_grid(
                        kname, tf.reshape(kernel,
                                          [2 * receptive_pixels,
                                              receptive_pixels, -1,
                                              network.
                                              net_params
                                              ['conv_filter_shapes'][0][-1]])),
                                     max_outputs=1))
            else:
                image.append(
                    tf.summary.image(kname, put_kernels_on_grid(
                        kname, kernel),
                        max_outputs=1))

        else:
            image.append(
                tf.summary.image(kname, put_kernels_on_grid(
                    kname, tf.reshape(
                        kernel, [receptive_pixels, receptive_pixels,
                                 1, -1])), max_outputs=1))

    disqualifiers = ['pool', 'fc', 'batchnorm', 'dropout']
    for layer_key, layer_value in network.layers.items():
        if check_disqualified_strings(layer_key, disqualifiers):
            if 'conv' in layer_key:
                act_name = '{}/activations/timedifference'.format(layer_key)
                convt0 = layer_value.outputs[0]
                convtend =layer_value.outputs[len(network.outputs)-1]
                image.append(tf.summary.image(act_name,
                             put_activations_on_grid(
                                name=act_name,
                                V=convtend - convt0)))
                # batch mean of timedifference
                act_name = '{}/activations/timedifference/batchmean'.format(layer_key)
                image.append(tf.summary.image(act_name,
                             put_activations_on_grid(
                                name=act_name,
                                V=tf.reduce_mean(
                                    convtend,
                                    axis=0, keepdims=True) -
                                tf.reduce_mean(
                                    convt0, axis=0, keepdims=True))))

            for act_key, act_value in layer_value.outputs.items():
                act_name = '{}/activations/{}'.format(layer_key, act_key)
                image.append(tf.summary.image(act_name,
                             put_activations_on_grid(
                                name=act_name,
                                V=act_value)
                    )
                )

                act_name = '{}/activations/{}/batchmean'.format(layer_key, act_key)
                image.append(tf.summary.image(act_name,
                             put_activations_on_grid(
                                name=act_name,
                                V=tf.reduce_mean(
                                    act_value, axis=0, keepdims=True))
                    )
                )

    return image


def compile_list_of_train_summaries(network, loss, accuracy,
                                    partial_accuracy,
                                    time_depth):
    train = []

    # with tf.name_scope('accuracy_and_error'):
    #     train.append(tf.summary.scalar(
    #         error.name + "_{}".format(time_depth),
    #         error.outputs[time_depth]))
    #     train.append(tf.summary.scalar(
    #         accuracy.name + "_{}".format(time_depth),
    #         accuracy.outputs[time_depth]))
    #     train.append(tf.summary.scalar(
    #         partial_accuracy.name + "_{}".format(time_depth),
    #         partial_accuracy.outputs[time_depth]))

    with tf.name_scope('accuracy_and_error/'):
        train.append(tf.summary.scalar(
            'loss', loss.outputs[time_depth]))
        train.append(tf.summary.scalar(
            'accuracy', accuracy.outputs[time_depth]))
        train.append(tf.summary.scalar(
            'partial_accuracy', partial_accuracy.outputs[time_depth]))

    with tf.name_scope('weights_and_biases'):
        list_of_weights = network.get_all_weights()
        list_of_biases = network.get_all_biases()

        for kernel in list_of_weights:
            kname = kernel.name.split('/')[1].split('_')[0] + '_weights'
            train += tfevent_handler.variable_summaries(
                kernel, kname, weights=True)

        for bias in list_of_biases:
            bname = bias.name.split('/')[1].split('_')[0] + '_bias'
            train += tfevent_handler.variable_summaries(
                bias, bname)

    return train


def compile_list_of_test_summaries(testavg, loss, accuracy,
                                   partial_accuracy, time_depth,
                                   time_depth_beyond):
    test = []
    # TODO: try to write test and train to the same window
    with tf.name_scope('testtime/'):
        test.append(tf.summary.scalar(
            'loss', testavg.average_cross_entropy[time_depth]))
        test.append(tf.summary.scalar(
            'accuracy', testavg.average_accuracy[time_depth]))
        test.append(tf.summary.scalar(
            'partial_accuracy',
            testavg.average_partial_accuracy[time_depth]))

    # with tf.name_scope('accuracy_and_error/'):
    #     test.append(tf.summary.scalar(
    #         'loss', loss))
    #     test.append(tf.summary.scalar(
    #         'accuracy', accuracy))
    #     test.append(tf.summary.scalar(
    #         'partial_accuracy', partial_accuracy))

    with tf.name_scope('testtime_beyond'):
        for time in range(0, time_depth + time_depth_beyond + 1):
            test.append(tf.summary.scalar(
                'loss' + "_{}".format(time),
                testavg.average_cross_entropy[time]))
            test.append(tf.summary.scalar(
                'accuracy' + "_{}".format(time),
                testavg.average_accuracy[time]))
            test.append(tf.summary.scalar(
                'partial_accuracy' + "_{}".format(time),
                testavg.average_partial_accuracy[time]))

    return test


def get_and_merge_summaries(network, testavg, loss, accuracy, partial_accuracy,
                            time_depth, time_depth_beyond, stereo):

    test = compile_list_of_test_summaries(
        testavg, loss, accuracy, partial_accuracy,
        time_depth, time_depth_beyond)
    train = compile_list_of_train_summaries(
        network, loss, accuracy, partial_accuracy,
        time_depth)
    image = compile_list_of_image_summaries(
        network, stereo)
    add = compile_list_of_additional_summaries(
        network, time_depth, time_depth_beyond)

    return tf.summary.merge(test), \
        tf.summary.merge(train), \
        tf.summary.merge(image), \
        tf.summary.merge(add)


def get_output_directory(configuration_dict, flags):
    """
    get_output_directory takes a dict configuration_dict and established the
    directory structure for the configured experiment. It returns paths to
    the checkpoints and the writer directories.
    """

    cfg_name = flags.config_file.split('/')[-1].split('.')[0]
    writer_directory = '{}{}/{}/'.format(
        configuration_dict['output_dir'], cfg_name,
        flags.name)

    # architecture string
    architecture_string = ''
    architecture_string += '{}{}_{}l_fm{}_d{}_l2{}'.format(
        configuration_dict['connectivity'],
        configuration_dict['time_depth'],
        configuration_dict['network_depth'],
        configuration_dict['feature_multiplier'],
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
        configuration_dict['color'],
        configuration_dict['label_type'])

    writer_directory += "{}/{}/{}/".format(architecture_string,
                                           data_string, format_string)

    checkpoint_directory = writer_directory + 'checkpoints/'

    # make sure the directories exist, otherwise create them
    mkdir_p(checkpoint_directory)
    mkdir_p(checkpoint_directory + 'evaluation/')

    return writer_directory, checkpoint_directory


def get_input_directory(configuration_dict):
    """
    get_input_directory takes a dict configuration_dict and returns a path to
    the image_data and a parser for the tf_records files.
    """
    # TODO: Integrate CIFAR and FashionMNIST
    if configuration_dict['dataset'] == "osycb":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'osycb/tfrecord_files/{}occ/{}p/{}/'.format(
            configuration_dict['n_occluders'],
            configuration_dict['occlusion_percentage'],
            configuration_dict['downsampling'])
        parser = tfrecord_handler._osycb_parse_single
    elif configuration_dict['dataset'] == "osmnist":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'osmnist/tfrecord_files/{}occ/'.format(
            configuration_dict['n_occluders'])
        parser = tfrecord_handler._osmnist_parse_single
    elif configuration_dict['dataset'] == "osfashionmnist":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'osmnist/osfashionmnist/tfrecord_files/{}occ/'.format(
            configuration_dict['n_occluders'])
        parser = tfrecord_handler._osmnist_parse_single
    elif configuration_dict['dataset'] == "oskuzushijimnist":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'osmnist/oskuzushijimnist/tfrecord_files/{}occ/'.format(
            configuration_dict['n_occluders'])
        parser = tfrecord_handler._osmnist_parse_single
    elif configuration_dict['dataset'] == "osmnistwithcues":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'osmnistwithcues/tfrecord_files/{}occ/'.format(
            configuration_dict['n_occluders'])
        # TODO: A new folder structure like the following?
        # 'osmnist/tfrecord_files/fashion/cues/{}occ/'
        # 'osmnist/tfrecord_files/digits/random/{}occ/'
        parser = tfrecord_handler._osmnist_parse_single
    elif configuration_dict['dataset'] == "mnist":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'mnist/tfrecord_files/'
        parser = tfrecord_handler._mnist_parse_single
    elif configuration_dict['dataset'] == "fashionmnist":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'mnist/fashionmnist/tfrecord_files/'
        parser = tfrecord_handler._mnist_parse_single
    elif configuration_dict['dataset'] == "kuzushijimnist":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'mnist/kuzushijimnist/tfrecord_files/'
        parser = tfrecord_handler._mnist_parse_single
    elif configuration_dict['dataset'] == "cifar10":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'cifar10/tfrecord_files/'
        parser = tfrecord_handler._cifar10_parse_single
    elif configuration_dict['dataset'] == "osdigit":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'digit_database/tfrecord_files/{}{}/'.format(
            configuration_dict['n_occluders']-1,
            configuration_dict['dataset'])
        parser = tfrecord_handler._osdigits_parse_single
    elif configuration_dict['dataset'] == "digit":
        tfrecord_dir = configuration_dict['input_dir'] + \
            'digit_database/tfrecord_files/{}{}/'.format(
            configuration_dict['n_occluders']-1,
            configuration_dict['dataset'])
        parser = tfrecord_handler._digits_parse_single
    # deprecated datasets
    elif configuration_dict['dataset'] == "ycb1_single":
        parser = tfrecord_handler._ycb1_parse_single
        tfrecord_dir = ''
        print("[INFO] Dataset is deprecated, paths must be defined manually.")
    elif configuration_dict['dataset'] == "ycb1_sequence":
        parser = tfrecord_handler._ycb1_parse_sequence
        tfrecord_dir = ''
        print("[INFO] Dataset is deprecated, paths must be defined manually.")
    else:
        print("[INFO] Dataset not defined")
        sys.exit()

    return tfrecord_dir, parser


def get_image_files(tfrecord_dir, training_dir, validation_dir, test_dir,
                    evaluation_dir, input_directory, dataset, n_occluders,
                    downsampling):
    """
    get_image_files takes paths to tfrecord_dir, training_dir, validation_dir
    test_dir and evaluation_dir and returns the corresponding file names.
    """
    list_of_dirs = [training_dir, validation_dir, test_dir, evaluation_dir]
    list_of_types = ['train', 'validation', 'test', 'evaluation']
    list_of_files = []
    for i in range(len(list_of_dirs)):
        type = list_of_types[i]
        if list_of_dirs[i]:
            if list_of_dirs[i] == 'all':
                list_of_files.append(
                    tfrecord_handler.all_percentages_tfrecord_paths(
                        type=type, dataset=dataset, n_occluders=n_occluders,
                        downsampling=downsampling,
                        input_directory=input_directory))
            else:
                list_of_files.append(
                    tfrecord_handler.tfrecord_auto_traversal(list_of_dirs[i]))
        else:
            try:
                list_of_files.append(tfrecord_handler.tfrecord_auto_traversal(
                    tfrecord_dir + list_of_types[i] + '/'))
            except(FileNotFoundError):
                list_of_files.append('')
    training, validation, testing, evaluation = list_of_files
    if len(testing) == 0:
        print('[INFO] No test-set found, using validation-set instead')
        testing += validation
    elif len(validation) == 0:
        validation += testing
        print('[INFO] No validation-set found, using test-set instead')
    return training, validation, testing, evaluation


def infer_additional_parameters(configuration_dict):
    """
    infer_additional_parameters takes a dict configuration_dict and infers
    additional parameters on the grounds of dataset etc.
    """
    # define correct network parameters
    # -----

    # read the number of layers from the network file
    circuit = importlib.import_module(configuration_dict['network_module'])
    configuration_dict['network_depth'] = circuit.return_network_layers(
        configuration_dict['connectivity'])

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
    elif 'cifar10' in configuration_dict['dataset']:
        configuration_dict['image_height'] = 32
        configuration_dict['image_width'] = 32
        configuration_dict['image_channels'] = 3
        configuration_dict['classes'] = 10
        configuration_dict['class_encoding'] = np.array(
            ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
             'frog', 'horse', 'ship', 'truck'])
    else:
        configuration_dict['image_height'] = 32
        configuration_dict['image_width'] = 32
        configuration_dict['image_channels'] = 1
        configuration_dict['classes'] = 10
        configuration_dict['class_encoding'] = np.array(
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    if 'fashion' in configuration_dict['dataset']:
        configuration_dict['class_encoding'] = np.array(
            ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
    elif 'kuzushiji' in configuration_dict['dataset']:
        pass  # unicode characters not supported

    if configuration_dict['color'] == 'grayscale':
        configuration_dict['image_channels'] = 1
    if configuration_dict['stereo']:
        configuration_dict['image_channels'] *= 2

    # use sigmoid for n-hot task, otherwise softmax
    if configuration_dict['label_type'] == 'nhot':
        configuration_dict['crossentropy_fn'] = sigmoid_cross_entropy
    else:
        configuration_dict['crossentropy_fn'] = softmax_cross_entropy

    # to crop the images
    # store the original values
    configuration_dict['image_height_input'] = \
        configuration_dict['image_height']
    configuration_dict['image_width_input'] = \
        configuration_dict['image_width']

    # change the image height and image width if the network is supposed

    if configuration_dict['cropped'] or configuration_dict['augmented']:
        configuration_dict['image_height'] = \
            configuration_dict['image_width']\
            // 10 * 4
        configuration_dict['image_width'] = \
            configuration_dict['image_width']\
            // 10 * 4
    else:
        pass

    # overwrite the default time_depth if network is not recurrent
    if configuration_dict['connectivity'] in ['B', 'BK', 'BF', 'BD']:
        configuration_dict['time_depth'] = 0
        configuration_dict['time_depth_beyond'] = 0
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
            elif ('True' in value) or ('False' in value):
                config_dictionary[key] = value.lower() in \
                    ("yes", "true", "t", "1")
            elif 'None' in value:
                config_dictionary[key] = None
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


def print_misclassified_objects(cm, encoding, n_obj=5):
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


