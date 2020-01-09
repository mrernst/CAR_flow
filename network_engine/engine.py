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
import utilities.afterburner as afterburner

# cross-platform development
from platform import system
IS_MACOSX = True if system() == 'Darwin' else False
PWD_STEM = "/Users/markus/Research/Code/" if IS_MACOSX else "/home/mernst/git/"

# commandline arguments
# -----

# FLAGS


tf.app.flags.DEFINE_boolean('testrun', False,
                            'simple configuration on local machine to test')
tf.app.flags.DEFINE_string('config_file', PWD_STEM +  #'/Users/markus/Research/Code/' + #'/home/mernst/git/'+
                           'saturn/experiments/001_noname_experiment/' +
                           'files/config_files/config0.csv',
                           'path to the configuration file of the experiment')
tf.app.flags.DEFINE_string('name', '',
                           'name of the run, i.e. iteration1')
tf.app.flags.DEFINE_boolean('restore_ckpt', True,
                            'restore model from last checkpoint')


FLAGS = tf.app.flags.FLAGS

CONFIG = helper.infer_additional_parameters(
    helper.read_config_file(FLAGS.config_file)
)


class LearningRate(object):
    """
    LearningRate interits from object. It stores internal variables for
    learning rate parameters and has tf methods to decay the learning rate
    or to divide it by 10
    """

    def __init__(self, lrate, eta, delta, d, global_epoch_variable):
        super(LearningRate, self).__init__()
        self.rate = tf.Variable(lrate, trainable=False,
                                name='learning_rate')
        self.eta = tf.Variable(eta, trainable=False,
                               name='learning_rate_eta')
        self.delta = tf.Variable(delta, trainable=False,
                                 name='learning_rate_delta')
        self.d = tf.Variable(d, trainable=False,
                             name='learning_rate_d')

        self.divide_by_10 = tf.compat.v1.assign(self.rate, self.rate / 10,
                                                name='divide_by_10')

        # TODO: initial learning rate should be specified in the setup
        self.decay_by_epoch = tf.compat.v1.assign(
            self.rate, self.eta * self.delta **
            (tf.cast(global_epoch_variable, tf.float32) /
                self.d), name='decay_by_epoch')


class EmbeddingObject(object):
    """
    EmbeddingObject inherits from object. It stores internal variables for
    thumbnailsize, thumbnails, labels and embedding and has tf methods to
    update and reset the embedding at testtime.
    """

    def __init__(self, thumbnailsize, image_height, image_width,
                 image_channels, batchsize, network_depth, network, accuracy):
        super(EmbeddingObject, self).__init__()
        self.thu_height = thumbnailsize
        self.thu_width = int(image_width / image_height * 32)

        self.total = {}
        update_embedding_preclass = {}
        reset_embedding_preclass = {}

        # these shapes are set to cover the standard networks
        # change for different embedding visualization
        self.labels = tf.Variable(tf.zeros(
            shape=0, dtype=tf.int64), validate_shape=False,
            name="preclass_labels", trainable=False)

        self.thumbnails = tf.Variable(tf.zeros(
            shape=[0, self.thu_height, self.thu_height, image_channels],
            dtype=tf.float32), validate_shape=False,
            name="embedding_thumbnails", trainable=False)

        update_embedding_labels = tf.compat.v1.assign(self.labels, tf.concat(
            [self.labels, tf.argmax(labels.variable, axis=-1)],
            axis=0), validate_shape=False)

        update_embedding_thumbnails = tf.compat.v1.assign(
            self.thumbnails, tf.concat(
                [self.thumbnails, tf.cast(
                    tf.image.resize_with_crop_or_pad(
                        tf.image.resize(
                            inp.variable, [self.thu_height, self.thu_width]),
                        self.thu_height, self.thu_height), dtype=tf.float32)],
                axis=0), validate_shape=False)

        reset_embedding_labels = tf.compat.v1.assign(self.labels, tf.zeros(
            shape=0, dtype=tf.int64),
            validate_shape=False)

        reset_embedding_thumbnails = tf.compat.v1.assign(
            self.thumbnails, tf.zeros(
                shape=[0, self.thu_height, self.thu_height, image_channels],
                dtype=tf.float32),
            validate_shape=False)

        # TODO: how to do timesteps here without having accuracy?

        for time in accuracy.outputs:
            self.total[time] = tf.Variable(
                tf.zeros(shape=[0, int(np.prod(np.array(
                    network.net_params['bias_shapes'][network_depth - 1]) /
                    np.array(
                    network.net_params['pool_strides'][network_depth - 1])))],
                    dtype=tf.float32), validate_shape=False,
                name="preclass_{}".format(time), trainable=False)

            update_embedding_preclass[time] = tf.compat.v1.assign(
                self.total[time],
                tf.concat([self.total[time],
                           tf.reshape(network.layers["dropoutc{}".format(
                               network_depth - 1)].outputs[time],
                    (batchsize, -1))], axis=0),
                validate_shape=False)

            reset_embedding_preclass[time] = tf.compat.v1.assign(
                self.total[time],
                tf.zeros(shape=[0, int(np.prod(np.array(
                    network.net_params['bias_shapes'][1]) /
                    np.array(network.net_params['pool_strides'][1])))],
                    dtype=tf.float32), validate_shape=False)

        self.update = tf.group(tf.stack(
            (list(update_embedding_preclass.values()))),
            update_embedding_labels, update_embedding_thumbnails)

        self.reset = tf.group(tf.stack(
            (list(reset_embedding_preclass.values()))),
            reset_embedding_labels, reset_embedding_thumbnails)


class TestAccuracy(object):
    """
    TestAccuracy inherits from object. It provides internal variables for
    accuracy and loss at different times and tf methods to update and reset
    during testtime.
    """

    def __init__(self, accuracy, error, label_type, partial_accuracy):
        super(TestAccuracy, self).__init__()

        self.count = tf.Variable(0., trainable=False)
        update_count = tf.compat.v1.assign_add(self.count, 1.)
        reset_count = tf.compat.v1.assign(self.count, 0.)

        total_test_accuracy = {}
        self.total_test_loss = {}

        update_total_test_accuracy = {}
        update_total_test_loss = {}

        reset_total_test_accuracy = {}
        reset_total_test_loss = {}

        self.average_accuracy = {}
        self.average_cross_entropy = {}

        for time in accuracy.outputs:
            total_test_accuracy[time] = tf.Variable(0., trainable=False)
            self.total_test_loss[time] = tf.Variable(0., trainable=False)

            update_total_test_accuracy[time] = tf.compat.v1.assign_add(
                total_test_accuracy[time], accuracy.outputs[time])

            update_total_test_loss[time] = tf.compat.v1.assign_add(
                self.total_test_loss[time], error.outputs[time])

            reset_total_test_loss[time] = tf.compat.v1.assign(
                self.total_test_loss[time], 0.)
            reset_total_test_accuracy[time] = tf.compat.v1.assign(
                total_test_accuracy[time], 0.)

            self.average_accuracy[time] = \
                total_test_accuracy[time] / self.count
            self.average_cross_entropy[time] = \
                self.total_test_loss[time] / self.count

        update_accloss = tf.stack(
            (list(update_total_test_loss.values()) + list(
                update_total_test_accuracy.values())))
        reset_accloss = tf.stack(
            (list(reset_total_test_accuracy.values()) + list(
                reset_total_test_loss.values())))

        if label_type == 'nhot':
            total_test_partial_accuracy = {}
            update_total_test_partial_accuracy = {}
            reset_total_test_partial_accuracy = {}

            self.average_partial_accuracy = {}

            for time in partial_accuracy.outputs:
                total_test_partial_accuracy[time] = tf.Variable(
                    0., trainable=False)
                update_total_test_partial_accuracy[time] = \
                    tf.compat.v1.assign_add(
                    total_test_partial_accuracy[time],
                    partial_accuracy.outputs[time])
                reset_total_test_partial_accuracy[time] = tf.compat.v1.assign(
                    total_test_partial_accuracy[time], 0.)
                self.average_partial_accuracy[time] = \
                    total_test_partial_accuracy[time] / self.count

            update_accloss = tf.stack((list(update_total_test_loss.values(
            )) + list(update_total_test_accuracy.values()) +
                list(update_total_test_partial_accuracy.values())))
            reset_accloss = tf.stack((list(reset_total_test_accuracy.values(
            )) + list(reset_total_test_loss.values()) +
                list(reset_total_test_partial_accuracy.values())))
        else:
            self.average_partial_accuracy = self.average_accuracy

        self.update = tf.group(update_accloss, update_count)
        self.reset = tf.group(reset_accloss, reset_count)


class ConfusionMatrix(object):
    """
    ConfusionMatrix inherits from object. It provides access to the
    total_confusion_matrix and tf methods to update and reset during
    testtime.
    """

    def __init__(self, network, labels, classes, time_depth):
        super(ConfusionMatrix, self).__init__()
        self.total = tf.Variable(
            tf.zeros([classes, classes]),
            name="confusion_matrix", trainable=False)

        update_confusion_matrix = tf.compat.v1.assign_add(
            self.total, tf.matmul(tf.transpose(
                tf.one_hot(tf.argmax(
                    network.outputs[time_depth], 1), classes)),
                labels.outputs[time_depth]))

        reset_confusion_matrix = tf.compat.v1.assign(
            self.total, tf.zeros([classes, classes]))

        self.update = tf.group(update_confusion_matrix)
        self.reset = tf.group(reset_confusion_matrix)


# constants
# -----

INP_MIN = -1
INP_MAX = 1
DTYPE = tf.float32

# TODO fill this into the infer_additional_parameters
# use sigmoid for n-hot task, otherwise softmax


# define network io modules
# -----

circuit = importlib.import_module(CONFIG['network_module'])


inp = bb.NontrainableVariableModule("input", (CONFIG['batchsize'],
                                              CONFIG['image_height_input'],
                                              CONFIG['image_width_input'],
                                              CONFIG['image_channels']),
                                    dtype=DTYPE)

labels = bb.NontrainableVariableModule("input_labels", (CONFIG['batchsize'],
                                                        CONFIG['classes']),
                                       dtype=DTYPE)

keep_prob = bb.PlaceholderModule(
    "keep_prob", shape=(), dtype=DTYPE)
is_training = bb.PlaceholderModule(
    "is_training", shape=(), dtype=tf.bool)


# TODO: This could be one class that can be incremented and maybe even account
# for batch accuracy.
global_step = tf.Variable(0, trainable=False, name='global_step')
increment_global_step = tf.compat.v1.assign_add(
    global_step, 1, name='increment_global_step')

global_epoch = tf.Variable(0, trainable=False, name='global_epoch')
increment_global_epoch = tf.compat.v1.assign_add(
    global_epoch, 1, name='increment_global_epoch')

lrate = LearningRate(CONFIG['learning_rate'],
                     CONFIG['lr_eta'],
                     CONFIG['lr_delta'],
                     CONFIG['lr_d'],
                     global_epoch)


# handle input/output directies
# -----

# check directories
TFRECORD_DIRECTORY, PARSER = helper.get_input_directory(CONFIG)
WRITER_DIRECTORY, CHECKPOINT_DIRECTORY = \
    helper.get_output_directory(CONFIG, FLAGS)

# get image data
# -----

# assign data_directories
training_filenames, validation_filenames, test_filenames,\
    evaluation_filenames = helper.get_image_files(TFRECORD_DIRECTORY,
                                                  CONFIG['training_dir'],
                                                  CONFIG['validation_dir'],
                                                  CONFIG['test_dir'],
                                                  CONFIG['evaluation_dir'],
                                                  CONFIG['input_dir'],
                                                  CONFIG['dataset'],
                                                  CONFIG['n_occluders'],
                                                  CONFIG['downsampling'])


# parse data from tf-record files
filenames = tf.compat.v1.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(PARSER)


if FLAGS.testrun:
    dataset = dataset.take(100)  # take smaller dataset for testing

dataset = dataset.shuffle(CONFIG['buffer_size'])  # bad for evaluation?
dataset = dataset.batch(CONFIG['batchsize'], drop_remainder=True)
iterator = dataset.make_initializable_iterator()
next_batch = iterator.get_next()

inp_left = next_batch[0]
inp_right = next_batch[1]

# preliminary support for grayscale within training
if CONFIG['color'] == 'grayscale' and not('mnist' in CONFIG['dataset']):
    inp_left = tf.image.rgb_to_grayscale(inp_left)
    inp_right = tf.image.rgb_to_grayscale(inp_right)

if not CONFIG['stereo']:
    inp_unknown = inp_left

    # define occlusion percentage
    if len(next_batch) > 4:
        occlusion_percentage = next_batch[-3]
    else:
        occlusion_percentage = tf.convert_to_tensor(
            np.repeat(CONFIG['occlusion_percentage'],
                      CONFIG['batchsize']))
else:
    inp_unknown = tf.concat([inp_left, inp_right], axis=3)

    if len(next_batch) > 4:
        occlusion_percentage = (next_batch[-3] + next_batch[-4])/2
    else:
        occlusion_percentage = tf.convert_to_tensor(
            np.repeat(CONFIG['occlusion_percentage'],
                      CONFIG['batchsize']))

if CONFIG['label_type'] == "onehot":
    labels.variable = labels.variable.assign(next_batch[-1])
else:
    labels.variable = labels.variable.assign(next_batch[-2])

inp.variable = inp.variable.assign(inp_unknown)


# initialize network classes
# -----

# preprocessor
# -----

# TODO:
# This is a dynamic preprocessor. Maybe it would make sense to write a static
# one and to write relevant files to disk to save computational ressources.

inp_prep = preprocessor.PreprocessorNetwork("preprocessor",
                                            INP_MIN,
                                            INP_MAX,
                                            CONFIG['cropped'],
                                            CONFIG['augmented'],
                                            CONFIG['norm_by_stat'],
                                            CONFIG['image_height'],
                                            CONFIG['image_width'],
                                            CONFIG['image_channels'],
                                            CONFIG['batchsize'],
                                            is_training.placeholder)
inp_prep.add_input(inp)


# network
# -----

network = circuit.constructor("rcnn",
                              CONFIG,
                              is_training.placeholder,
                              keep_prob.placeholder,
                              custom_net_parameters=None)

one_time_error = bb.ErrorModule("cross_entropy", CONFIG['crossentropy_fn'])
error = bb.TimeAddModule("add_error")
optimizer = bb.OptimizerModule("adam",
                               tf.compat.v1.train.AdamOptimizer(lrate.rate))
accuracy = bb.BatchAccuracyModule("accuracy")

network.add_input(inp_prep)
one_time_error.add_input(network)
one_time_error.add_input(labels)
error.add_input(one_time_error, 0)
error.add_input(error, -1)  # seems to work, but is this the right way..?
optimizer.add_input(error)
accuracy.add_input(network)
accuracy.add_input(labels)

# L2 regularization term
if CONFIG['l2_lambda'] != 0:
    lossL2 = bb.NontrainableVariableModule("lossL2", (), dtype=tf.float32)
    lossL2.variable = lossL2.variable.assign(
        tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()]) *
        CONFIG['l2_lambda'] / (CONFIG['time_depth'] + 1))
    error.add_input(lossL2, 0)
else:
    pass


# create outputs, i.e. unfold the network
error.create_output(CONFIG['time_depth'] + CONFIG['time_depth_beyond'])
optimizer.create_output(CONFIG['time_depth'])

for time in range(CONFIG['time_depth'] + CONFIG['time_depth_beyond'] + 1):
    accuracy.create_output(time)


if CONFIG['label_type'] == 'nhot':
    partial_accuracy = bb.NHotBatchAccuracyModule(
        "partial_accuracy", all_labels_true=False)
    accuracy = bb.NHotBatchAccuracyModule("accuracy", all_labels_true=True)

    accuracy.add_input(network)
    accuracy.add_input(labels)
    partial_accuracy.add_input(network)
    partial_accuracy.add_input(labels)

    for time in range(0, (CONFIG['time_depth'] +
                          CONFIG['time_depth_beyond'] + 1)):
        accuracy.create_output(time)
        partial_accuracy.create_output(time)


# get information about which stimuli got classified correctly
    bool_classification = tf.reduce_all(tf.equal(tf.reduce_sum(
        tf.one_hot(tf.nn.top_k(network.outputs[CONFIG['time_depth']],
                               k=tf.count_nonzero(
            labels.variable[-1], dtype=tf.int32)).indices,
            depth=tf.shape(labels.variable)[-1]), axis=-2),
        labels.variable), axis=-1)

    bcx1 = tf.nn.top_k(network.outputs[CONFIG['time_depth']],
                       k=tf.count_nonzero(
                       labels.variable[-1], dtype=tf.int32)).indices

    bcx2 = tf.nn.top_k(labels.variable, k=tf.count_nonzero(
        labels.variable[-1], dtype=tf.int32)).indices

    bool_classification = tf.stack([bcx1, bcx2])
else:
    bool_classification = tf.equal(
        tf.argmax(network.outputs[CONFIG['time_depth']], 1),
        tf.argmax(labels.variable, 1))

    partial_accuracy = accuracy

# average accuracy and error at mean test-time
# -----

# embedding object for storing high dimensional representation
embedding = EmbeddingObject(thumbnailsize=32,
                            image_height=CONFIG['image_height'],
                            image_width=CONFIG['image_width'],
                            image_channels=CONFIG['image_channels'],
                            batchsize=CONFIG['batchsize'],
                            network_depth=CONFIG['network_depth'],
                            network=network,
                            accuracy=accuracy)

# average test accuracy and error, defines partial_accuracy if not 'nhot'
testaverages = TestAccuracy(accuracy, error,
                            CONFIG['label_type'],
                            partial_accuracy)


# confusion matrix for tensorboard
confusion_matrix = ConfusionMatrix(network, labels,
                                   CONFIG['classes'],
                                   CONFIG['time_depth'])

# TODO: Update Confusion Matrix in the main loop,
# otherwise it will not be updated


# decide which parameters get written to tfevents
# -----

# write operations:
test_merged, train_merged, image_merged, add_merged = \
    helper.get_and_merge_summaries(network,
                                   testaverages,
                                   error,
                                   accuracy,
                                   partial_accuracy,
                                   CONFIG['time_depth'],
                                   CONFIG['time_depth_beyond'],
                                   CONFIG['stereo'])


# start session, merge summaries, start writers
# -----

with tf.compat.v1.Session() as sess:

    train_writer = tf.compat.v1.summary.FileWriter(
        WRITER_DIRECTORY + '/training', sess.graph)
    test_writer = tf.compat.v1.summary.FileWriter(
        WRITER_DIRECTORY + '/testing')
    add_writer = tf.compat.v1.summary.FileWriter(
        WRITER_DIRECTORY + '/testing/extra')
    image_writer = tf.compat.v1.summary.FileWriter(
        WRITER_DIRECTORY + '/testing/images')

    if FLAGS.testrun:
        # debug writer for metadata etc.
        debug_writer = tf.compat.v1.summary.FileWriter(
            WRITER_DIRECTORY + '/debug', sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    saver = tf.compat.v1.train.Saver(
        keep_checkpoint_every_n_hours=1, max_to_keep=2,
        save_relative_paths=True)
    sess.run(tf.compat.v1.global_variables_initializer())

    # training and testing functions
    # -----

    def evaluating(train_it, flnames=evaluation_filenames, tag='Evaluation'):
        print(" " * 80 + "\r" + "[{}]\tstarted".format(tag), end="\r")
        sess.run([iterator.initializer, testaverages.reset, embedding.reset],
                 feed_dict={filenames: flnames})

        output_array = np.array([])
        list_of_output_tensors = []
        list_of_bc_values = []
        list_of_occlusion_percentages = []

        for time in range(CONFIG['time_depth']+1):
            list_of_output_tensors.append(tf.nn.softmax(network.outputs[time]))

        # delete bool_classification file if it already exists
        # if os.path.exists(CHECKPOINT_DIRECTORY + 'evaluation/' +
        #                   "bool_classification.txt"):
        #     os.remove(WRITER_DIRECTORY + 'checkpoints/' +
        #               'evaluation/' + "bool_classification.txt")

        while True:
            try:
                _, _, extras, images, bc, occ, out = sess.run(
                    [testaverages.update, embedding.update, add_merged,
                        image_merged, bool_classification,
                        occlusion_percentage,
                        list_of_output_tensors],
                    feed_dict={keep_prob.placeholder: 1.0,
                               is_training.placeholder: False})

                # save output and bool_classification data
                out = np.swapaxes(np.array(out), 0, 1)
                if len(output_array) == 0:
                    output_array = out
                else:
                    output_array = np.concatenate(
                        [output_array, out], 0)


                list_of_occlusion_percentages += occ.tolist()
                if CONFIG['label_type'] == "onehot":
                    list_of_bc_values += bc.astype(np.int8).tolist()
                else:
                    for i in range(len(bc[0])):
                        for el in bc[0][i]:
                            list_of_bc_values += [int(el in set(bc[1][i]))]

            except (tf.errors.OutOfRangeError):
                break

        acc, loss, emb, emb_labels, emb_thu, summary = sess.run(
            [testaverages.average_partial_accuracy[CONFIG['time_depth']],
                testaverages.average_cross_entropy[CONFIG['time_depth']],
                embedding.total,
                embedding.labels,
                embedding.thumbnails,
                test_merged])

        print(" " * 80 + "\r" +
              "[{}]\tloss: {:.5f}\tacc: {:.5f} \tstep: {}"
              .format(tag, loss, acc, train_it))

        # pass labels to write to metafile
        return emb, emb_labels, emb_thu, \
            list_of_bc_values, list_of_occlusion_percentages, \
            output_array

    def write_embeddings_to_disk(emb, emb_labels, emb_thu, list_of_bc_values):
        saver.save(sess, CHECKPOINT_DIRECTORY +
                   'evaluation/' + CONFIG['exp_name'] +
                   CONFIG['connectivity'] + CONFIG['dataset'],
                   global_step=global_step.eval())

        # define custom labels for interesting stimuli
        lookat = np.zeros(emb_labels.shape, dtype=np.int32)
        # lookat[-50:] = 1

        emb_labels = np.asarray(emb_labels, dtype=np.int32)
        # save labels to textfile to be read by tensorboard
        np.savetxt(CHECKPOINT_DIRECTORY + 'evaluation/'
                   + "metadata.tsv",
                   np.column_stack(
                       [emb_labels,
                        CONFIG['class_encoding'][emb_labels],
                        lookat,
                        list_of_bc_values]),
                   header="labels\tnames\tlookat\tboolclass",
                   fmt=["%s", "%s", "%s", "%s"], delimiter="\t", comments='')

        # save thumbnails to sprite image
        visualizer.save_sprite_image(CHECKPOINT_DIRECTORY +
                                     'evaluation/' +
                                     'embedding_spriteimage.png',
                                     emb_thu[:, :, :, :])

        # configure metadata linking
        projector_config = projector.ProjectorConfig()
        embeddings_dict = {}

        for i in range(CONFIG['time_depth'] + CONFIG['time_depth_beyond'] + 1):
            embeddings_dict[i] = projector_config.embeddings.add()
            embeddings_dict[i].tensor_name = \
                embedding.total[i].name
            embeddings_dict[i].metadata_path = os.path.join(
                CHECKPOINT_DIRECTORY +
                'evaluation/', 'metadata.tsv')
            embeddings_dict[i].sprite.image_path = CHECKPOINT_DIRECTORY + \
                'evaluation/' + \
                'embedding_spriteimage.png'
            embeddings_dict[i].sprite.single_image_dim.extend(
                [embedding.thu_height, embedding.thu_height])

        tsne_writer = tf.compat.v1.summary.FileWriter(
            CHECKPOINT_DIRECTORY + 'evaluation/')
        projector.visualize_embeddings(
            tsne_writer, projector_config)
        pass

    def evaluate_data(projector_bool=False, flnames=evaluation_filenames):
        # checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIRECTORY)
        # if checkpoint and checkpoint.model_checkpoint_path:
        #     saver.restore(sess, checkpoint.model_checkpoint_path)
        # TODO: Do we really need checkpoint restore?
        # TODO: maybe evaluation directory is not needed? helper..
        # TODO: get the occlusion percentages here also
        emb, emb_labels, emb_thu, list_of_bc_values, \
            list_of_occlusion_percentages, output_array = evaluating(
                global_step.eval(), flnames=flnames)

        embedding_data = {}

        evaluation_data = \
            {'boolean_classification': np.array(list_of_bc_values),
             'occlusion_percentage': np.array(list_of_occlusion_percentages),
             'softmax_output': output_array}
        if projector_bool:
            write_embeddings_to_disk(emb, emb_labels, emb_thu,
                                     list_of_bc_values)
            embedding_data['embedding'] = emb
            embedding_data['labels'] = emb_labels
            embedding_data['thumbnails'] = emb_thu
        else:
            embedding_data = None
        # else:
        #     print('[INFO] No checkpoint data found, exiting')
        #     sys.exit()
        return evaluation_data, embedding_data

    def testing(train_it, flnames=validation_filenames, tag='Validation'):
        print(" " * 80 + "\r" + "[{}]\tstarted".format(tag), end="\r")
        sess.run(iterator.initializer, feed_dict={filenames: flnames})
        sess.run([testaverages.reset, confusion_matrix.reset])
        while True:
            try:
                _, _, extras, images = sess.run(
                    [testaverages.update, confusion_matrix.update,
                        add_merged, image_merged],
                    feed_dict={keep_prob.placeholder: 1.0,
                               is_training.placeholder: False})

            except(tf.errors.OutOfRangeError):
                break

        acc, loss, summary = sess.run(
            [testaverages.average_accuracy[CONFIG['time_depth']],
             testaverages.average_cross_entropy[CONFIG['time_depth']],
             test_merged])
        print(" " * 80 + "\r" +
              "[{}]\tloss: {:.5f}\tacc: {:.5f} \tstep: {}"
              .format(tag, loss, acc, train_it))

        if not(FLAGS.restore_ckpt):
            test_writer.add_summary(summary, train_it)

            if CONFIG['visualization']:
                add_writer.add_summary(extras, train_it)
                image_writer.add_summary(images, train_it)

                # pass additional confusion matrix to image_writer
                cm_figure = visualizer.cm_to_figure(
                    confusion_matrix.total.eval(), CONFIG['class_encoding'])
                image_writer.add_summary(
                    tfplot.figure.to_summary(
                        cm_figure, tag="confusionmatrix"), train_it)

                # helper.print_misclassified_objects(
                #     confusion_matrix.total.eval(),
                #     CONFIG['class_encoding'], 5)

        FLAGS.restore_ckpt = False
        return 0

    def training(train_it):
        sess.run(iterator.initializer, feed_dict={
                 filenames: training_filenames})
        while True:
            try:
                summary, extras, loss, acc = sess.run(
                    [train_merged, add_merged,
                        optimizer.outputs[CONFIG['time_depth']],
                        accuracy.outputs[CONFIG['time_depth']]],
                    feed_dict={keep_prob.placeholder: CONFIG['keep_prob'],
                               is_training.placeholder: True})

                if (train_it % CONFIG['write_every'] == 0):
                    train_writer.add_summary(summary, train_it)
                if CONFIG['verbose']:
                    print(" " * 80 + "\r" +
                          "[Training]\tloss: {:.5f}\tacc: {:.5f} \tstep: {}"
                          .format(loss, acc, train_it), end="\r")
                train_it = increment_global_step.eval()

            except (tf.errors.OutOfRangeError):
                _ = increment_global_epoch.eval()
                if CONFIG['decaying_lrate']:
                    _ = lrate.decay_by_epoch.eval()
                    print(" " * 80 +
                          "\r" + "[INFO] Learningrate updated to {:.5f}"
                          .format(lrate.rate.eval()))
                break

        return train_it

    # continueing from restored checkpoint
    # -----

    if FLAGS.restore_ckpt:
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIRECTORY)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('[INFO] Restored checkpoint successfully')
            # subtract epochs already done
            CONFIG['epochs'] -= global_epoch.eval()
            print('[INFO] Continue training from last checkpoint:' +
                  ' {} epochs remaining'
                  .format(CONFIG['epochs']))
            # sys.exit()
        else:
            print('[INFO] No checkpoint found,' +
                  'starting experiment from scratch')
            FLAGS.restore_ckpt = False
            # sys.exit()

    # training loop
    # -----

    # prepare input normalization on preprocessor
    if CONFIG['norm_by_stat']:
        inp_prep.gather_statistics(sess, iterator,
                                   training_filenames, filenames, is_training,
                                   show_image=True)

    train_it = global_step.eval()
    for i_train_epoch in range(CONFIG['epochs']):
        if i_train_epoch % CONFIG['test_every'] == 0:
            _ = testing(train_it)
            saver.save(sess, CHECKPOINT_DIRECTORY + CONFIG['exp_name'] +
                       FLAGS.name + CONFIG['dataset'], global_step=train_it)
        train_it = training(train_it)

    # final test (ideally on an independent testset)
    # -----
    testing(train_it, flnames=test_filenames, tag='Testing')
    saver.save(sess, CHECKPOINT_DIRECTORY + CONFIG['exp_name'] +
               FLAGS.name + CONFIG['dataset'], global_step=train_it)

    evaluation_data, embedding_data = \
        evaluate_data(projector_bool=CONFIG['projector'],
                      flnames=test_filenames[:2])  # reduced [:1]

    train_writer.close()
    test_writer.close()
    image_writer.close()
    add_writer.close()


# evaluation and afterburner
# -----

    essence = afterburner.DataEssence()
    essence.distill(path=WRITER_DIRECTORY, evaluation_data=evaluation_data,
                    embedding_data=None)  # embedding_data (save space)
    essence.write_to_file(filename=CONFIG['output_dir'] +
                          FLAGS.config_file.split('/')[-1].split('.')[0] +
                          '{}'.format(FLAGS.name) + '.pkl')
    essence.plot_essentials(CONFIG['output_dir'].rsplit('/', 2)[0] +
                            '/visualization/' +
                            FLAGS.config_file.split('/')[-1].split('.')[0] +
                            '{}'.format(FLAGS.name) + '.pdf')

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
