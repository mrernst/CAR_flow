#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# August 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# engine.py                                  oN88888UU[[[/;::-.        dP^
# main program                              dNMMNN888UU[[[/;:--.   .o@P^
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
import matplotlib
import tfplot
import sys
import os
import re
import errno
import importlib

from tensorflow.contrib.tensorboard.plugins import projector

# TODO: Integrate class activation map helper into building blocks,
#  i.e. CAM Module

# custom libraries
# -----

import utilities.tfevent_handler as tfevent_handler
import utilities.tfrecord_handler as tfrecord_handler
import utilities.visualizer as visualizer
import utilities.helper as helper

import utilities.networks.buildingblocks as bb
import utilities.networks.preprocessor as preprocessor


# commandline arguments
# -----
class LearningRate(object):
    """docstring for LearningRate."""

    def __init__(self, lrate, eta, delta, d, global_epoch_variable):
        super(LearningRate, self).__init__()
        self.rate = tf.Variable(CONFIG['learning_rate'], trainable=False,
                                name='learning_rate')
        self.eta = tf.Variable(CONFIG['lr_eta'], trainable=False,
                               name='learning_rate_eta')
        self.delta = tf.Variable(CONFIG['lr_delta'], trainable=False,
                                 name='learning_rate_delta')
        self.d = tf.Variable(CONFIG['lr_d'], trainable=False,
                             name='learning_rate_d')

        self.divide_by_10 = tf.assign(self.rate, self.rate / 10,
                                      name='divide_by_10')

        # TODO initial learning rate should be specified in the setup
        self.decay_by_epoch = tf.assign(self.rate, self.eta * self.delta **
                                        (tf.cast(global_epoch_variable,
                                         tf.float32) /
                                            self.d), name='decay_by_epoch')

# commandline arguments
# -----

# FLAGS
tf.app.flags.DEFINE_boolean('testrun', False,
                            'simple configuration on local machine to test')
tf.app.flags.DEFINE_string('config_file', '/Users/markus/Research/Code/' +
                           'saturn/experiments/001_noname_experiment/' +
                           'files/config_files/config0.csv',
                           'path to the configuration file of the experiment')

FLAGS = tf.app.flags.FLAGS
CONFIG = helper.infer_additional_parameters(
    helper.read_config_file(FLAGS.config_file)
    )



# constants
# -----

INP_MIN = -1
INP_MAX = 1
DTYPE = tf.float32

# TODO fill this into the infer_additional_parameters
# use sigmoid for n-hot task, otherwise softmax


# define network io modules
# -----

foundation = importlib.import_module(CONFIG['network_module'])


inp = bb.NontrainableVariableModule("input", (CONFIG['batchsize'],
                                    CONFIG['image_height'],
                                    CONFIG['image_width'],
                                    CONFIG['image_channels']), dtype='uint8')
labels = bb.NontrainableVariableModule("input_labels", (CONFIG['batchsize'],
                                       CONFIG['classes']), dtype=DTYPE)

keep_prob = bb.PlaceholderModule(
    "keep_prob", shape=(), dtype=DTYPE)
is_training = bb.PlaceholderModule(
    "is_training", shape=(), dtype=tf.bool)


global_step = tf.Variable(0, trainable=False, name='global_step')
increment_global_step = tf.assign_add(
    global_step, 1, name='increment_global_step')

global_epoch = tf.Variable(0, trainable=False, name='global_epoch')
increment_global_epoch = tf.assign_add(
    global_epoch, 1, name='increment_global_epoch')

lrate = LearningRate(CONFIG['learning_rate'], CONFIG['lr_eta'],
                     CONFIG['lr_delta'], CONFIG['lr_d'], global_epoch)


# TODO:
# cropping augmenting and normalization should be part of a preprocessing
# network, because that is basically what we are doing here.
# solve it more elegantly, by passing the relevant config params to a bb class
# also include the calculation of image statistics in that submodule

inp_prep = preprocessor.PreprocessorNetwork(CONFIG['cropped'],
                                            CONFIG['augmented'],
                                            CONFIG['norm_by_stat'],
                                            INP_MAX,
                                            INP_MIN)
inp_prep.add_input(inp)


# handle input/output directies
# -----

# check directories
RESULT_DIRECTORY = FLAGS.output_dir + FLAGS.exp_name + '/'

if FLAGS.dataset == "ycb_db1":
    TFRECORD_DIRECTORY = '/home/aecgroup/aecdata/Textures/\
YCB_database1/tfrecord-files/single/{}/{}/'.format(
        FLAGS.ycb_object_distance, FLAGS.downsampling)
    parser = _db1_parse_single
elif FLAGS.dataset == "ycb_db2":
    TFRECORD_DIRECTORY = '/home/aecgroup/aecdata/Textures/\
YCB_database2/tfrecord-files/{}occ/{}p/{}/'.format(FLAGS.n_occluders, FLAGS.occlusion_percentage, FLAGS.downsampling)
    parser = _db2_parse_single
elif FLAGS.dataset == "ycb_db2b":
    TFRECORD_DIRECTORY = '/home/aecgroup/aecdata/Textures/\
YCB_database2/tfrecord-files/{}occ/10class/{}p/{}/'.format(FLAGS.n_occluders, FLAGS.occlusion_percentage, FLAGS.downsampling)
    parser = _db2b_parse_single
elif FLAGS.dataset == "ycb_db3":
    TFRECORD_DIRECTORY = '/home/aecgroup/aecdata/Textures/\
YCB_database3/tfrecord-files/{}occ/{}p/{}/'.format(FLAGS.n_occluders, FLAGS.occlusion_percentage, FLAGS.downsampling)
    parser = _db2_parse_single
elif FLAGS.dataset == "multimnist":
    TFRECORD_DIRECTORY = '/home/aecgroup/aecdata/Textures/\
multimnist/tfrecord-files/{}occ/'.format(FLAGS.n_occluders)
    parser = _multimnist_parse_single
elif ('stereo' in FLAGS.dataset):
    TFRECORD_DIRECTORY = '/home/aecgroup/aecdata/Textures/\
digit_database/tfrecord-files/{}/'.format(FLAGS.dataset)
    parser = _sdigits_parse_single
else:
    TFRECORD_DIRECTORY = '/home/aecgroup/aecdata/Textures/\
digit_database/tfrecord-files/{}/'.format(FLAGS.dataset)
    parser = _digits_parse_single


# result directory
RESULT_DIRECTORY = '{}/{}/{}/'.format(FLAGS.output_dir,
                                      FLAGS.exp_name, FLAGS.name)
# architecture string
ARCHITECTURE_STRING = ''
ARCHITECTURE_STRING += '{}{}_{}layer_fm{}_d{}'.format(
    FLAGS.connectivity, FLAGS.timedepth, FLAGS.network_depth, FLAGS.feature_mult, FLAGS.keep_prob)
if FLAGS.batchnorm:
    ARCHITECTURE_STRING += '_bn1'
else:
    ARCHITECTURE_STRING += '_bn0'
ARCHITECTURE_STRING += '_bs{}'.format(FLAGS.batchsize)
if FLAGS.decaying_lrate:
    ARCHITECTURE_STRING += '_lr{}-{}-{}'.format(
        FLAGS.lr_eta, FLAGS.lr_delta, FLAGS.lr_d)
else:
    ARCHITECTURE_STRING += '_lr{}'.format(FLAGS.learning_rate)
# data string
DATA_STRING = ''
if ('db2' in FLAGS.dataset):
    DATA_STRING += "{}_{}occ_{}p_{}cm".format(
        FLAGS.dataset, FLAGS.n_occluders, FLAGS.occlusion_percentage, FLAGS.ycb_object_distance)
elif ('db1' in FLAGS.dataset):
    DATA_STRING += "{}_0occ_0p_{}cm".format(
        FLAGS.dataset, FLAGS.ycb_object_distance)
elif ('stereo' in FLAGS.dataset):
    DATA_STRING += "sdigit_{}occ_Xp_50cm".format(int(FLAGS.dataset[0]) - 1)
elif ('digit' in FLAGS.dataset):
    DATA_STRING += "digit_{}occ_Xp_50cm".format(int(FLAGS.dataset[0]) - 1)
else:
    DATA_STRING += "{}_{}occ_Xp_Xcm".format(FLAGS.dataset, FLAGS.n_occluders)

# format string
FORMAT_STRING = ''
FORMAT_STRING += '{}x{}x{}'.format(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
if ('ycb' in FLAGS.dataset) and FLAGS.dataset != 'ycb_db2b':
    FORMAT_STRING += "_{}_{}".format(FLAGS.color, FLAGS.label_type)
else:
    FORMAT_STRING += "_grayscale_{}".format(FLAGS.label_type)
RESULT_DIRECTORY += "{}/{}/{}/".format(ARCHITECTURE_STRING,
                                       DATA_STRING, FORMAT_STRING)


CHECKPOINT_DIRECTORY = RESULT_DIRECTORY + 'checkpoints/'

# make sure the directories exist, otherwise create them
mkdir_p(CHECKPOINT_DIRECTORY)

# save config FLAGS to file
network_description = open(RESULT_DIRECTORY + "network_description.txt", "w")
network_description.write("Network Description File\n")
network_description.write("---\n\n")
for key, value in zip(FLAGS.__flags.keys(), FLAGS.__flags.values()):
    network_description.write("{}: {}\n".format(key, value))
network_description.close()


# get image data
# -----

# assign data_directories
if FLAGS.training_dir:
    if FLAGS.training_dir == 'all':
        training_filenames = all_percentages_tfrecord_paths(
            FLAGS, type='train')
    else:
        training_filenames = tfrecord_auto_traversal(FLAGS.training_dir)
else:
    training_filenames = tfrecord_auto_traversal(TFRECORD_DIRECTORY + 'train/')
if FLAGS.validation_dir:
    if FLAGS.validation_dir == 'all':
        validation_filenames = all_percentages_tfrecord_paths(
            FLAGS, type='validation')
    else:
        validation_filenames = tfrecord_auto_traversal(FLAGS.validation_dir)
else:
    validation_filenames = tfrecord_auto_traversal(
        TFRECORD_DIRECTORY + 'validation/')
if FLAGS.test_dir:
    if FLAGS.test_dir == 'all':
        test_filenames = all_percentages_tfrecord_paths(FLAGS, type='test')
    else:
        test_filenames = tfrecord_auto_traversal(FLAGS.test_dir)
else:
    test_filenames = tfrecord_auto_traversal(TFRECORD_DIRECTORY + 'test/')
if FLAGS.evaluation_dir:
    if FLAGS.test_dir == 'all':
        eval_filenames = all_percentages_tfrecord_paths(FLAGS, type='test')
    else:
        eval_filenames = tfrecord_auto_traversal(FLAGS.evaluation_dir)
else:
    eval_filenames = tfrecord_auto_traversal(
        TFRECORD_DIRECTORY + 'evaluation/')

if len(test_filenames) == 0:
    print('[INFO]: No test-set found, using validation-set instead')
    test_filenames += validation_filenames

# parse data from tf-record files
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(parser)

# TODO: Make this an environment variable, this gets forgotten way to often.
# prepare dataset-structure, make sure data is not shuffled for evaluation
if FLAGS.testrun:
    dataset = dataset.take(300)  # take smaller dataset for testing
if not(FLAGS.evaluate_ckpt):
    dataset = dataset.shuffle(buffer_size=FLAGS.buffer_size)

dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
iterator = dataset.make_initializable_iterator()
next_batch = iterator.get_next()

inp_left = next_batch[0]
inp_right = next_batch[1]

# preliminary support for grayscale within training
if FLAGS.color == 'grayscale':
    inp_left = tf.image.rgb_to_grayscale(inp_left)
    inp_right = tf.image.rgb_to_grayscale(inp_right)

if not FLAGS.stereo:
    inp_unknown = inp_left
else:
    inp_unknown = tf.concat([inp_left, inp_right], axis=3)

if FLAGS.label_type == "onehot":
    labels.variable = labels.variable.assign(next_batch[-1])
else:
    labels.variable = labels.variable.assign(next_batch[-2])

inp.variable = inp.variable.assign(inp_unknown)


# initialize classes with parameters
# -----

network = networks.MeNet2_constructor("MeNet", is_training.placeholder, keep_prob.placeholder,
                                      FLAGS, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES, net_params=None)
one_time_error = bb.ErrorModule("cross_entropy", CONFIG['crossentropy_fn'])
error = bb.TimeAddModule("add_error")
optimizer = bb.OptimizerModule("adam", tf.train.AdamOptimizer(lrate))
#optimizer = bb.OptimizerModule("momentum", tf.train.MomentumOptimizer(lrate, 0.9))
#optimizer = bb.OptimizerModule("GradientDescent", tf.train.GradientDescentOptimizer(lrate))
accuracy = bb.BatchAccuracyModule("accuracy")

network.add_input(inp_prep)
one_time_error.add_input(network)
one_time_error.add_input(labels)
error.add_input(one_time_error, 0)
error.add_input(error, -1)  # seems like a good idea, but is it necessary
optimizer.add_input(error)
accuracy.add_input(network)
accuracy.add_input(labels)

# L2 regularization
if FLAGS.l2_lambda != 0:
    lossL2 = bb.NontrainableVariableModule("lossL2", (), dtype=tf.float32)
    lossL2.variable = lossL2.variable.assign(tf.add_n([tf.nn.l2_loss(
        v) for v in tf.trainable_variables()]) * FLAGS.l2_lambda / (TIME_DEPTH + 1))
    #lossL2.variable = lossL2.variable.assign(tf.add_n([ tf.nn.l2_loss(v) for v in network.get_all_weights]) * FLAGS.l2_lambda)
    error.add_input(lossL2, 0)

error.create_output(TIME_DEPTH + TIME_DEPTH_BEYOND)
optimizer.create_output(TIME_DEPTH)
# accuracy.create_output(TIME_DEPTH)
# for time in range(TIME_DEPTH + 1,(TIME_DEPTH + TIME_DEPTH_BEYOND + 1)):
#  accuracy.create_output(time)
for time in range(0, (TIME_DEPTH + TIME_DEPTH_BEYOND + 1)):
    accuracy.create_output(time)

if FLAGS.label_type == 'nhot':
    accuracy = bb.NHotBatchAccuracyModule("accuracy", all_labels_true=True)
    accuracy_labels = bb.NHotBatchAccuracyModule(
        "accuracy_labels", all_labels_true=False)

    accuracy.add_input(network)
    accuracy.add_input(labels)
    accuracy_labels.add_input(network)
    accuracy_labels.add_input(labels)

    # accuracy.create_output(TIME_DEPTH)
    # accuracy_labels.create_output(TIME_DEPTH)
    # for time in range(TIME_DEPTH + 1,(TIME_DEPTH + TIME_DEPTH_BEYOND + 1)):
    #  accuracy.create_output(time)
    #  accuracy_labels.create_output(time)
    for time in range(0, (TIME_DEPTH + TIME_DEPTH_BEYOND + 1)):
        accuracy.create_output(time)
        accuracy_labels.create_output(time)


# Class Activation Mapping setup -> Put this into a module. You could review timesteps and so on.
# Might be useful and more understandable
# -----

if FLAGS.classactivation:
    top_conv = network.layers["dropoutc{}".format(
        FLAGS.network_depth - 1)].outputs[TIME_DEPTH]

    if FLAGS.batchnorm:
        fc_weights = network.layers['fc0'].input_module.weights
    else:
        fc_weights = network.layers['fc0'].weights

    class_activation_map, class_activation_map_resized = get_class_map(tf.argmax(
        network.outputs[TIME_DEPTH], axis=1), top_conv, fc_weights, IMAGE_WIDTH, BATCH_SIZE)

    cam_maxima = tf.reduce_max(
        class_activation_map,
        axis=[1, 2, 3], keep_dims=True
    )

    binary_cam = tf.to_int32(class_activation_map > 0.2 * cam_maxima)

    # get the segmentation maps for the target only
    segmap_left = next_batch[2][:, :, :, :1]
    segmap_right = next_batch[3][:, :, :, :1]

    # binarize and maxpooling
    maxpooled_segmap = segmap_left
    maxpooled_segmap = tf.to_int32(maxpooled_segmap > 0)
    maxpooled_segmap = tf.nn.max_pool(
        maxpooled_segmap, [1, 4, 4, 1], [1, 4, 4, 1], "SAME")

    # define a class_activation_map error
    segmentation_accuracy = 1. - tf.losses.absolute_difference(
        maxpooled_segmap, binary_cam, reduction=tf.losses.Reduction.MEAN)
else:
    class_activation_map_resized = tf.Variable(0., validate_shape=False)


# average accuracy and error at mean test-time (maybe solve more intelligently) + embedding stuff
# -----

THU_HEIGHT = 32
THU_WIDTH = int(IMAGE_WIDTH / IMAGE_HEIGHT * 32)

total_test_accuracy = {}
total_test_loss = {}
total_embedding_preclass = {}

for time in accuracy.outputs:
    total_test_accuracy[time] = tf.Variable(0., trainable=False)
    total_test_loss[time] = tf.Variable(0., trainable=False)
    #total_embedding_preclass[time] = tf.Variable(tf.zeros(shape=[0,np.prod(network.net_params['bias_shapes'][1])], dtype=tf.float32), validate_shape=False, name="preclass_{}".format(time), trainable=False)
    total_embedding_preclass[time] = tf.Variable(tf.zeros(shape=[0, int(np.prod(np.array(network.net_params['bias_shapes'][1]) / np.array(
        network.net_params['pool_strides'][1])))], dtype=tf.float32), validate_shape=False, name="preclass_{}".format(time), trainable=False)

count = tf.Variable(0., trainable=False)
embedding_labels = tf.Variable(tf.zeros(
    shape=0, dtype=tf.int64), validate_shape=False, name="preclass_labels", trainable=False)
embedding_thumbnails = tf.Variable(tf.zeros(shape=[0, THU_HEIGHT, THU_HEIGHT, IMAGE_CHANNELS],
                                            dtype=tf.int16), validate_shape=False, name="embedding_thumbnails", trainable=False)

update_total_test_accuracy = {}
update_total_test_loss = {}
update_embedding_preclass = {}

for time in accuracy.outputs:
    update_total_test_accuracy[time] = tf.assign_add(
        total_test_accuracy[time], accuracy.outputs[time])
    update_total_test_loss[time] = tf.assign_add(
        total_test_loss[time], error.outputs[time])
    #update_embedding_preclass[time] = tf.assign(total_embedding_preclass[time], tf.concat([total_embedding_preclass[time], tf.reshape(network.layers["conv1"].outputs[time], [BATCH_SIZE, -1])], axis=0), validate_shape=False)
    #update_embedding_preclass[time] = tf.assign(total_embedding_preclass[time], tf.concat([total_embedding_preclass[time], network.layers["flatpool0"].outputs[time]], axis=0), validate_shape=False)
    update_embedding_preclass[time] = tf.assign(total_embedding_preclass[time], tf.concat([total_embedding_preclass[time], tf.reshape(
        network.layers["dropoutc{}".format(FLAGS.network_depth - 1)].outputs[time], (BATCH_SIZE, -1))], axis=0), validate_shape=False)


update_count = tf.assign_add(count, 1.)
update_embedding_labels = tf.assign(embedding_labels, tf.concat(
    [embedding_labels, tf.argmax(labels.variable, axis=-1)], axis=0), validate_shape=False)
update_embedding_thumbnails = tf.assign(embedding_thumbnails, tf.concat([embedding_thumbnails, tf.cast(tf.image.resize_image_with_crop_or_pad(
    tf.image.resize_images(inp.variable, [THU_HEIGHT, THU_WIDTH]), THU_HEIGHT, THU_HEIGHT), dtype=tf.int16)], axis=0), validate_shape=False)

reset_total_test_accuracy = {}
reset_total_test_loss = {}
reset_embedding_preclass = {}

for time in accuracy.outputs:
    reset_total_test_loss[time] = tf.assign(total_test_loss[time], 0.)
    reset_total_test_accuracy[time] = tf.assign(total_test_accuracy[time], 0.)
    #reset_embedding_preclass[time] = tf.assign(total_embedding_preclass[time], tf.zeros(shape=[0,np.prod(network.net_params['bias_shapes'][1])], dtype=tf.float32), validate_shape=False)
    reset_embedding_preclass[time] = tf.assign(total_embedding_preclass[time], tf.zeros(shape=[0, int(np.prod(np.array(
        network.net_params['bias_shapes'][1]) / np.array(network.net_params['pool_strides'][1])))], dtype=tf.float32), validate_shape=False)

reset_count = tf.assign(count, 0.)
reset_embedding_labels = tf.assign(embedding_labels, tf.zeros(
    shape=0, dtype=tf.int64), validate_shape=False)
reset_embedding_thumbnails = tf.assign(embedding_thumbnails, tf.zeros(
    shape=[0, THU_HEIGHT, THU_HEIGHT, IMAGE_CHANNELS], dtype=tf.int16), validate_shape=False)

average_accuracy = {}
average_cross_entropy = {}
for time in accuracy.outputs:
    average_cross_entropy[time] = total_test_loss[time] / count
    average_accuracy[time] = total_test_accuracy[time] / count


# needed because tf 1.4 does not support grouping dict of tensor
if FLAGS.classactivation:
    total_test_segmentation_accuracy = {}
    update_total_test_segmentation_accuracy = {}
    reset_total_test_segmentation_accuracy = {}
    average_segmentation_accuracy = {}

    total_test_segmentation_accuracy[TIME_DEPTH] = tf.Variable(
        0., trainable=False)
    update_total_test_segmentation_accuracy[TIME_DEPTH] = tf.assign_add(
        total_test_segmentation_accuracy[TIME_DEPTH], segmentation_accuracy)
    reset_total_test_segmentation_accuracy[TIME_DEPTH] = tf.assign(
        total_test_segmentation_accuracy[TIME_DEPTH], 0.)
    average_segmentation_accuracy[TIME_DEPTH] = total_test_segmentation_accuracy[TIME_DEPTH] / count

    update_accloss = tf.stack((list(update_total_test_loss.values(
    )) + list(update_total_test_accuracy.values()) + list(update_total_test_segmentation_accuracy.values())))
    reset_accloss = tf.stack((list(reset_total_test_accuracy.values(
    )) + list(reset_total_test_loss.values()) + list(reset_total_test_segmentation_accuracy.values())))
else:
    update_accloss = tf.stack(
        (list(update_total_test_loss.values()) + list(update_total_test_accuracy.values())))
    reset_accloss = tf.stack(
        (list(reset_total_test_accuracy.values()) + list(reset_total_test_loss.values())))

update_emb = tf.group(tf.stack((list(update_embedding_preclass.values()))),
                      update_embedding_labels, update_embedding_thumbnails)
reset_emb = tf.group(tf.stack((list(reset_embedding_preclass.values()))),
                     reset_embedding_labels, reset_embedding_thumbnails)

# confusion matrix
total_confusion_matrix = tf.Variable(
    tf.zeros([CLASSES, CLASSES]), name="confusion_matrix", trainable=False)
update_confusion_matrix = tf.assign_add(total_confusion_matrix, tf.matmul(tf.transpose(
    tf.one_hot(tf.argmax(network.outputs[TIME_DEPTH], 1), CLASSES)), labels.outputs[TIME_DEPTH]))
reset_confusion_matrix = tf.assign(
    total_confusion_matrix, tf.zeros([CLASSES, CLASSES]))

update = tf.group(update_confusion_matrix, update_accloss, update_count)
reset = tf.group(reset_confusion_matrix, reset_accloss, reset_count)


# #input statistics for normalization
# input_statistics = {}
# update_input_statistics = {}
# reset_input_statistics = {}
#
# # not initialized to zero for graceful error handling
# input_statistics['N'] = tf.Variable(1., trainable=False)
# input_statistics['Sx'] = tf.Variable(tf.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]), trainable=False)
# input_statistics['Sxx'] = tf.Variable(tf.ones([1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]), trainable=False)
#
# update_input_statistics['N'] = tf.assign_add(input_statistics['N'], BATCH_SIZE)
# update_input_statistics['Sx'] = tf.assign_add(input_statistics['Sx'], tf.expand_dims(tf.reduce_sum(tf.cast(inp.variable, tf.float32), axis=0), 0))
# update_input_statistics['Sxx'] = tf.assign_add(input_statistics['Sxx'], tf.expand_dims(tf.reduce_sum(tf.square(tf.cast(inp.variable, tf.float32)), axis=0), 0))
#
# reset_input_statistics['N'] = tf.assign(input_statistics['N'], 0)
# reset_input_statistics['Sx'] = tf.assign(input_statistics['Sx'], tf.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]))
# reset_input_statistics['Sxx'] = tf.assign(input_statistics['Sxx'], tf.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]))


if FLAGS.label_type == 'nhot':
    total_test_accuracy_labels = {}
    update_total_test_accuracy_labels = {}
    reset_total_test_accuracy_labels = {}
    average_accuracy_labels = {}
    for time in accuracy_labels.outputs:
        total_test_accuracy_labels[time] = tf.Variable(0., trainable=False)
        update_total_test_accuracy_labels[time] = tf.assign_add(
            total_test_accuracy_labels[time], accuracy_labels.outputs[time])
        reset_total_test_accuracy_labels[time] = tf.assign(
            total_test_accuracy_labels[time], 0.)
        average_accuracy_labels[time] = total_test_accuracy_labels[time] / count

    # needed because tf 1.4 does not support grouping dict of tensor
    update_accloss = tf.stack((list(update_total_test_loss.values(
    )) + list(update_total_test_accuracy.values()) + list(update_total_test_accuracy_labels.values())))
    reset_accloss = tf.stack((list(reset_total_test_accuracy.values(
    )) + list(reset_total_test_loss.values()) + list(reset_total_test_accuracy_labels.values())))

    update = tf.group(update_confusion_matrix, update_accloss, update_count)
    reset = tf.group(reset_confusion_matrix, reset_accloss, reset_count)


# get information about which stimuli got classified correctly
if FLAGS.label_type == "onehot":
    bool_classification = tf.equal(
        tf.argmax(network.outputs[TIME_DEPTH], 1), tf.argmax(labels.variable, 1))
else:
    bool_classification = tf.reduce_all(tf.equal(tf.reduce_sum(tf.one_hot(tf.nn.top_k(network.outputs[TIME_DEPTH], k=tf.count_nonzero(
        labels.variable[-1], dtype=tf.int32)).indices, depth=tf.shape(labels.variable)[-1]), axis=-2), labels.variable), axis=-1)
    bcx1 = tf.nn.top_k(network.outputs[TIME_DEPTH], k=tf.count_nonzero(
        labels.variable[-1], dtype=tf.int32)).indices
    bcx2 = tf.nn.top_k(labels.variable, k=tf.count_nonzero(
        labels.variable[-1], dtype=tf.int32)).indices
    bool_classification = tf.stack([bcx1, bcx2])

# decide which parameters get written to tfevents
# -----

test_summaries = []
train_summaries = []
additional_summaries = []
image_summaries = []

with tf.name_scope('mean_test_time'):
    test_summaries.append(tf.summary.scalar(
        'testtime_avg_cross_entropy', average_cross_entropy[TIME_DEPTH]))
    test_summaries.append(tf.summary.scalar(
        'testtime_avg_accuracy', average_accuracy[TIME_DEPTH]))
    if FLAGS.label_type == 'nhot':
        test_summaries.append(tf.summary.scalar(
            'testtime_avg_accuracy_labels', average_accuracy_labels[TIME_DEPTH]))
    elif FLAGS.classactivation:
        test_summaries.append(tf.summary.scalar(
            'testtime_avg_segmentation_accuracy', average_segmentation_accuracy[TIME_DEPTH]))

with tf.name_scope('mean_test_time_beyond'):
    for time in accuracy.outputs:
        test_summaries.append(tf.summary.scalar(
            'testtime_avg_cross_entropy' + "_{}".format(time), average_cross_entropy[time]))
        test_summaries.append(tf.summary.scalar(
            'testtime_avg_accuracy' + "_{}".format(time), average_accuracy[time]))
        if FLAGS.label_type == 'nhot':
            test_summaries.append(tf.summary.scalar(
                'testtime_avg_accuracy_labels' + "_{}".format(time), average_accuracy_labels[time]))


with tf.name_scope('weights_and_biases'):
    if not FLAGS.batchnorm:
        train_summaries += variable_summaries(
            network.layers["conv0"].bias.outputs[TIME_DEPTH], 'conv0_bias_{}'.format(TIME_DEPTH))
        train_summaries += variable_summaries(
            network.layers["conv1"].bias.outputs[TIME_DEPTH], 'conv1_bias_{}'.format(TIME_DEPTH))
        train_summaries += variable_summaries(
            network.layers["fc0"].bias.outputs[TIME_DEPTH], 'fc0_bias_{}'.format(TIME_DEPTH))
    train_summaries += variable_summaries(
        network.layers["conv0"].conv.weights, 'conv0_kernel_{}'.format(TIME_DEPTH), weights=True)
    #train_summaries += variable_summaries(network.layers["pool0"].outputs[TIME_DEPTH], 'pool0_output_{}'.format(TIME_DEPTH))
    train_summaries += variable_summaries(
        network.layers["conv1"].conv.weights, 'conv1_kernel_{}'.format(TIME_DEPTH), weights=True)
    #train_summaries += variable_summaries(network.layers["pool1"].outputs[TIME_DEPTH], 'pool1_output_{}'.format(TIME_DEPTH))
    train_summaries += variable_summaries(
        network.layers["fc0"].input_module.weights, 'fc0_weights_{}'.format(TIME_DEPTH), weights=True)
    if "L" in FLAGS.connectivity:
        train_summaries += variable_summaries(
            network.layers["lateral0"].weights, 'lateral0_weights_{}'.format(TIME_DEPTH), weights=True)
        train_summaries += variable_summaries(
            network.layers["lateral1"].weights, 'lateral1_weights_{}'.format(TIME_DEPTH), weights=True)
        additional_summaries.append(tf.summary.histogram(
            'lateral0_weights_{}'.format(TIME_DEPTH), network.layers["lateral0"].weights))
        additional_summaries.append(tf.summary.histogram(
            'lateral1_weights_{}'.format(TIME_DEPTH), network.layers["lateral1"].weights))
    if "T" in FLAGS.connectivity:
        train_summaries += variable_summaries(
            network.layers["topdown0"].weights, 'topdown0_weights_{}'.format(TIME_DEPTH), weights=True)
        additional_summaries.append(tf.summary.histogram(
            'topdown0_weights_{}'.format(TIME_DEPTH), network.layers["topdown0"].weights))
    additional_summaries.append(tf.summary.histogram(
        'conv0_weights_{}'.format(TIME_DEPTH), network.layers["conv0"].conv.weights))
    additional_summaries.append(tf.summary.histogram(
        'conv1_weights_{}'.format(TIME_DEPTH), network.layers["conv1"].conv.weights))
    #additional_summaries.append(tf.summary.histogram('conv0_pre_activations_{}'.format(TIME_DEPTH), network.layers["conv0"].preactivation.outputs[TIME_DEPTH]))
    #additional_summaries.append(tf.summary.histogram('conv0_activations_{}'.format(TIME_DEPTH), network.layers["conv0"].outputs[TIME_DEPTH]))
    #additional_summaries.append(tf.summary.histogram('conv1_pre_activations_{}'.format(TIME_DEPTH), network.layers["conv1"].preactivation.outputs[TIME_DEPTH]))
    #additional_summaries.append(tf.summary.histogram('conv1_activations_{}'.format(TIME_DEPTH), network.layers["conv1"].outputs[TIME_DEPTH]))
    #additional_summaries.append(tf.summary.histogram('fc0_pre_activations_{}'.format(TIME_DEPTH), network.layers["fc0"].preactivation.outputs[TIME_DEPTH]))
    additional_summaries.append(tf.summary.histogram('fc0_weights{}'.format(
        TIME_DEPTH), network.layers["fc0"].input_module.weights[TIME_DEPTH]))

# comment this out to save space in the records and calculation time
# with tf.name_scope('activations'):
#   train_summaries += module_variable_summary(network.layers["conv0"].preactivation)
#   train_summaries += module_variable_summary(network.layers["conv0"])
#   train_summaries += module_variable_summary(network.layers["conv0"].preactivation)
#   train_summaries += module_variable_summary(network.layers["conv0"])
#   train_summaries += module_variable_summary(network.layers["fc0"])


# with tf.name_scope('time_differences'):
    #train_summaries += module_timedifference_summary(error, TIME_DEPTH + TIME_DEPTH_BEYOND)
    #train_summaries += module_timedifference_summary(accuracy, TIME_DEPTH + TIME_DEPTH_BEYOND)
    #train_summaries += module_timedifference_summary(network.layers["fc0"], TIME_DEPTH)
    #train_summaries += module_timedifference_summary(network.layers["conv0"], TIME_DEPTH)
    #train_summaries += module_timedifference_summary(network.layers["conv1"], TIME_DEPTH)


# comment this out to save space in the records and calculation time
# with tf.name_scope('accuracy_and_error_beyond'):
#  train_summaries += module_scalar_summary(error)
#  train_summaries += module_scalar_summary(accuracy)
#  if FLAGS.label_type == 'nhot':
#    train_summaries += module_scalar_summary(accuracy_labels)

with tf.name_scope('accuracy_and_error'):
    train_summaries.append(tf.summary.scalar(
        error.name + "_{}".format(TIME_DEPTH), error.outputs[TIME_DEPTH]))
    train_summaries.append(tf.summary.scalar(
        accuracy.name + "_{}".format(TIME_DEPTH), accuracy.outputs[TIME_DEPTH]))
    if FLAGS.label_type == 'nhot':
        train_summaries.append(tf.summary.scalar(
            accuracy_labels.name + "_{}".format(TIME_DEPTH), accuracy_labels.outputs[TIME_DEPTH]))
    elif FLAGS.classactivation:
        train_summaries.append(tf.summary.scalar(
            "segmentation_accuracy" + "_{}".format(TIME_DEPTH), segmentation_accuracy))


with tf.name_scope('images'):
    for lnr in range(FLAGS.network_depth - 1):
        image_summaries.append(tf.summary.image('conv{}/kernels'.format(lnr + 1), put_kernels_on_grid('conv{}/kernels'.format(lnr + 1), tf.reshape(
            network.layers["conv{}".format(lnr + 1)].conv.weights, [network.net_params['receptive_pixels'], network.net_params['receptive_pixels'], 1, -1])), max_outputs=1))
    if "T" in FLAGS.connectivity:
        for lnr in range(FLAGS.network_depth - 1):
            image_summaries.append(tf.summary.image('topdown{}/kernels'.format(lnr), put_kernels_on_grid('topdown{}/kernels'.format(lnr), tf.reshape(
                network.layers["topdown{}".format(lnr)].weights, [network.net_params['receptive_pixels'], network.net_params['receptive_pixels'], 1, -1])), max_outputs=1))
    if "L" in FLAGS.connectivity:
        for lnr in range(FLAGS.network_depth):
            image_summaries.append(tf.summary.image('lateral{}/kernels'.format(lnr), put_kernels_on_grid('lateral{}/kernels'.format(lnr), tf.reshape(
                network.layers["lateral{}".format(lnr)].weights, [network.net_params['receptive_pixels'], network.net_params['receptive_pixels'], 1, -1])), max_outputs=1))

    # this is more interesting, write down on case by case basis, cases: 1,2,3,6 Channels
    # 1 or 3 channels:
    if not FLAGS.stereo:
        image_summaries.append(tf.summary.image('conv0/kernels', put_kernels_on_grid(
            'conv0/kernels', network.layers["conv0"].conv.weights), max_outputs=1))
    else:
        image_summaries.append(tf.summary.image('conv0/kernels', put_kernels_on_grid('conv0/kernels', tf.reshape(network.layers["conv0"].conv.weights, [
                               network.net_params['receptive_pixels'], 2 * network.net_params['receptive_pixels'], -1, network.net_params['n_features']])), max_outputs=1))

    #image_summaries.append(tf.summary.image('conv0/activations', put_activations_on_grid('conv0/activations', network.layers["conv0"].outputs[TIME_DEPTH]), max_outputs=1))
    #image_summaries.append(tf.summary.image('sample_input', inp_prep.outputs[TIME_DEPTH][:,:,:,:1], max_outputs=1))


# start session, merge summaries, start writers
# -----

with tf.Session() as sess:

    train_merged = tf.summary.merge(train_summaries)
    test_merged = tf.summary.merge(test_summaries)
    add_merged = tf.summary.merge(additional_summaries)
    image_merged = tf.summary.merge(image_summaries)

    train_writer = tf.summary.FileWriter(
        RESULT_DIRECTORY + '/train', sess.graph)
    hist_writer = tf.summary.FileWriter(RESULT_DIRECTORY + '/train/hist')
    test_writer = tf.summary.FileWriter(RESULT_DIRECTORY + '/validation')
    image_writer = tf.summary.FileWriter(RESULT_DIRECTORY + '/validation/img')

    # debug writer for metadata etc.
    # debug_writer = tf.summary.FileWriter(RESULT_DIRECTORY + '/debug', sess.graph)
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)
    sess.run(tf.global_variables_initializer())

    # training and testing functions
    # -----

    def testing(train_it, flnames=validation_filenames, tag='Validation'):
        print(" " * 80 + "\r" + "[Validation]\tstarted", end="\r")
        sess.run([iterator.initializer, reset], feed_dict={filenames: flnames})
        while True:
            try:
                _, histograms, images, pic, cam = sess.run([update, add_merged, image_merged, inp.outputs[TIME_DEPTH], class_activation_map_resized], feed_dict={
                                                           keep_prob.placeholder: 1.0, is_training.placeholder: False})
            except (tf.errors.OutOfRangeError):
                break
        acc, loss, summary = sess.run(
            [average_accuracy[TIME_DEPTH], average_cross_entropy[TIME_DEPTH], test_merged])
        print(" " * 80 + "\r" +
              "[{}]\tloss: {:.5f}\tacc: {:.5f} \tstep: {}".format(tag, loss, acc, train_it))
        if not(FLAGS.restore_ckpt):
            test_writer.add_summary(summary, train_it)
            if FLAGS.visualization:
                hist_writer.add_summary(histograms, train_it)
                image_writer.add_summary(images, train_it)

                # pass additional confusion matrix to image_writer
                cm_figure = cm_to_figure(
                    total_confusion_matrix.eval(), encoding)
                image_writer.add_summary(tfplot.figure.to_summary(cm_figure,
                                         tag="dev/confusionmatrix"), train_it)

            # pass additional class activation maps of the last batch to image_writer
            if FLAGS.classactivation:
                cam_figure = cam_to_figure(cam, pic)
                image_writer.add_summary(tfplot.figure.to_summary(
                    cam_figure, tag="dev/saliencymap"), train_it)

        FLAGS.restore_ckpt = False
        return 0

    def training(train_it):
        sess.run(iterator.initializer, feed_dict={
                 filenames: training_filenames})
        while True:
            try:
                summary, histograms, loss, acc = sess.run([train_merged, add_merged, optimizer.outputs[TIME_DEPTH], accuracy.outputs[TIME_DEPTH]], feed_dict={
                                                          keep_prob.placeholder: FLAGS.keep_prob, is_training.placeholder: True})
                if (train_it % FLAGS.writeevery == 0):
                    train_writer.add_summary(summary, train_it)
                if FLAGS.verbose:
                    print(" " * 80 + "\r" + "[Training]\tloss: {:.5f}\tacc: {:.5f} \tstep: {}".format(
                        loss, acc, train_it), end="\r")
                train_it = increment_global_step.eval()

            except (tf.errors.OutOfRangeError):
                _ = increment_global_epoch.eval()
                if FLAGS.decaying_lrate:
                    _ = update_lrate.eval()
                    print(
                        " " * 80 + "\r" + "[INFO]: Learningrate updated to {:.5f}".format(lrate.eval()))
                break
        return train_it

    def evaluation(train_it, flnames=eval_filenames, tag='Evaluation'):
        print(" " * 80 + "\r" + "[Validation]\tstarted", end="\r")
        sess.run([iterator.initializer, reset, reset_emb],
                 feed_dict={filenames: flnames})

        list_of_output_samples = []
        list_of_output_times = []
        for time in network.outputs:
            list_of_output_times.append(tf.nn.softmax(network.outputs[time]))

        # delete bool_classification file if it already exists
        if os.path.exists(RESULT_DIRECTORY + 'checkpoints/' + 'evaluation/' + "bool_classification.txt"):
            os.remove(RESULT_DIRECTORY + 'checkpoints/' +
                      'evaluation/' + "bool_classification.txt")
        while True:
            try:
                _, _, histograms, images, bc, out = sess.run([update, update_emb, add_merged, image_merged, bool_classification, list_of_output_times], feed_dict={
                                                             keep_prob.placeholder: 1.0, is_training.placeholder: False})

                # save output of boolean comparison
                boolfile = open(RESULT_DIRECTORY + 'checkpoints/' +
                                'evaluation/' + "bool_classification.txt", "a")
                if FLAGS.label_type == "onehot":
                    for i in list(bc):
                        boolfile.write(str(int(i)) + '\n')
                else:
                    for i in range(len(bc[0])):
                        for el in bc[0][i]:
                            boolfile.write(
                                str(int(el in set(bc[1][i]))) + '\t')
                        boolfile.write('\n')
                boolfile.close()

                # temporary code to save output
                list_of_output_samples.append(out)

            except (tf.errors.OutOfRangeError):
                break
        if FLAGS.label_type == 'nhot':
            acc, loss, emb, emb_labels, emb_thu, summary = sess.run(
                [average_accuracy_labels[TIME_DEPTH], average_cross_entropy[TIME_DEPTH], total_embedding_preclass, embedding_labels, embedding_thumbnails, test_merged])
        else:
            acc, loss, emb, emb_labels, emb_thu, summary = sess.run(
                [average_accuracy[TIME_DEPTH], average_cross_entropy[TIME_DEPTH], total_embedding_preclass, embedding_labels, embedding_thumbnails, test_merged])
        print(" " * 80 + "\r" +
              "[{}]\tloss: {:.5f}\tacc: {:.5f} \tstep: {}".format(tag, loss, acc, train_it))

        #cm_summary = cm_to_tfsummary(total_confusion_matrix.eval(), encoding, tensor_name='dev/cm')
        # plot_confusion_matrix(cm)
        np.savez_compressed(RESULT_DIRECTORY + 'checkpoints/' + 'evaluation/' +
                            'softmax_output.npz', np.array(list_of_output_samples))
        # pass labels to write to metafile
        return emb, emb_labels, emb_thu

    def gather_input_stats(flnames=training_filenames):
        sess.run(iterator.initializer, feed_dict={filenames: flnames})
        print(" " * 80 + "\r" + "[Statistics]\tstarted", end="\r")
        sess.run([reset_input_statistics['N'],
                  reset_input_statistics['Sx'], reset_input_statistics['Sxx']])
        while True:
            try:
                N, Sx, Sxx = sess.run([update_input_statistics['N'], update_input_statistics['Sx'], update_input_statistics['Sxx']], feed_dict={
                                      is_training.placeholder: False, keep_prob.placeholder: 1.0})
            except (tf.errors.OutOfRangeError):
                sess.run([tf.assign(network.layers['pw_norm'].n, input_statistics['N']),
                          tf.assign(
                              network.layers['pw_norm'].sx, input_statistics['Sx']),
                          tf.assign(network.layers['pw_norm'].sxx, input_statistics['Sxx'])])
                # import matplotlib.pyplot as plt
                # a = sess.run(network.layers['pw_norm'].outputs[0], feed_dict = {is_training.placeholder:False, keep_prob.placeholder: 1.0})
                # for i in range(50):
                #   #plt.imshow(batch[0][i,:,:,0])
                #   plt.imshow(a[i,:,:,0])
                #   plt.show()
                break
            pass

    # continueing from restored checkpoint
    # -----

    if FLAGS.restore_ckpt:
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIRECTORY)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('[INFO]: Restored checkpoint successfully')
            # subtract epochs already done
            N_TRAIN_EPOCH -= global_epoch.eval()
            print('[INFO]: Continue training from last checkpoint: {} epochs remaining'.format(
                N_TRAIN_EPOCH))
            # sys.exit()
        else:
            print('[INFO]: No checkpoint found, starting experiment from scratch')
            FLAGS.restore_ckpt = False
            # sys.exit()

    # evaluating restored checkpoint
    # -----

    if FLAGS.evaluate_ckpt:
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIRECTORY)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            # make sure the directories exist, otherwise create them
            mkdir_p(CHECKPOINT_DIRECTORY + 'evaluation/')
            print('[INFO]: Restored checkpoint successfully, running evaluation')
            emb, emb_labels, emb_thu = evaluation(
                global_step.eval(), flnames=eval_filenames)  # test_filenames

            # visualize with tsne
            if True:
                saver.save(sess, RESULT_DIRECTORY + 'checkpoints/' + 'evaluation/' + FLAGS.name + FLAGS.connectivity
                           + FLAGS.dataset, global_step=global_step.eval())

                npnames = encoding
                lookat = np.zeros(emb_labels.shape, dtype=np.int32)
                lookat[-50:] = 1
                emb_labels = np.asarray(emb_labels, dtype=np.int32)

                # save labels to textfile to be read by tensorboard
                np.savetxt(RESULT_DIRECTORY + 'checkpoints/' + 'evaluation/' + "metadata.tsv", np.column_stack(
                    [emb_labels, npnames[emb_labels], lookat]), header="labels\tnames\tlookat", fmt=["%s", "%s", "%s"], delimiter="\t", comments='')
                # save thumbnails to sprite image
                save_sprite_image(RESULT_DIRECTORY + 'checkpoints/' + 'evaluation/' +
                                  'embedding_spriteimage.png', emb_thu[:, :, :, :])

                # configure metadata linking
                config = projector.ProjectorConfig()
                embeddings = {}
                # try to write down everything here
                for i in range(TIME_DEPTH + TIME_DEPTH_BEYOND + 1):
                    embeddings[i] = config.embeddings.add()
                    embeddings[i].tensor_name = total_embedding_preclass[i].name
                    embeddings[i].metadata_path = os.path.join(
                        RESULT_DIRECTORY + 'checkpoints/' + 'evaluation/', 'metadata.tsv')
                    embeddings[i].sprite.image_path = RESULT_DIRECTORY + \
                        'checkpoints/' + 'evaluation/' + 'embedding_spriteimage.png'
                    embeddings[i].sprite.single_image_dibb.extend(
                        [THU_HEIGHT, THU_HEIGHT])

                summary_writer = tf.summary.FileWriter(
                    RESULT_DIRECTORY + 'checkpoints/' + 'evaluation/')
                projector.visualize_embeddings(summary_writer, config)

            # plot_confusion_matrix(cm)
            #print_misclassified_objects(cm, 5)
            sys.exit()
            print('[INFO]: Continue training from last checkpoint')
        else:
            print('[INFO]: No checkpoint data found, exiting')
            sys.exit()

    # training loop
    # -----
    # prepare input normalization
    # gather_input_stats()
    train_it = global_step.eval()
    for i_train_epoch in range(N_TRAIN_EPOCH):
        if i_train_epoch % FLAGS.testevery == 0:
            _ = testing(train_it)
            saver.save(sess, CHECKPOINT_DIRECTORY + FLAGS.name + STEREO_STRING
                       + FLAGS.dataset_type, global_step=train_it)
        train_it = training(train_it)

    # final test (ideally on an independent testset)
    # -----

    testing(train_it, flnames=test_filenames, tag='Testing')
    saver.save(sess, CHECKPOINT_DIRECTORY + FLAGS.name + STEREO_STRING
               + FLAGS.dataset_type, global_step=train_it)

    train_writer.close()
    test_writer.close()
    image_writer.close()
    hist_writer.close()


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
