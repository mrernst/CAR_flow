#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# August 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# preprocessor.py                            oN88888UU[[[/;::-.        dP^
# preprocess files and                      dNMMNN888UU[[[/;:--.   .o@P^
# normalize according to input stats       ,MMMMMMN888UU[[/;::-. o@^
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


class PreprocessorNetwork(bb.ComposedModule):
    """
    PreprocessorNetwork inherits from ComposedModule. It is a input network
    that preprocesses the network input. It can crop parts of the image and
    applies normalization before the image is handed to the neural network.
    """

    def initialize_stats(self):
        # input statistics for normalization
        self.stats = {}
        self.update_stats = {}
        self.reset_stats = {}

        # not initialized to zero for graceful error handling
        self.stats['N'] = tf.Variable(1., trainable=False)

        self.stats['Sx'] = tf.Variable(
            tf.zeros([1, self.image_height, self.image_width,
                      self.image_channels]),
            trainable=False)

        self.stats['Sxx'] = tf.Variable(
            tf.ones([1, self.image_height, self.image_width,
                     self.image_channels]),
            trainable=False)

        self.update_stats['N'] = tf.assign_add(self.stats['N'], self.batchsize)
        self.update_stats['Sx'] = tf.assign_add(
            self.stats['Sx'], tf.expand_dims(
                tf.reduce_sum(
                    tf.cast(self.input_module.outputs[0], tf.float32),
                    axis=0), 0))
        self.update_stats['Sxx'] = tf.assign_add(
            self.stats['Sxx'], tf.expand_dims(
                tf.reduce_sum(tf.square(
                    tf.cast(self.input_module.outputs[0], tf.float32)),
                    axis=0), 0))

        self.reset_stats['N'] = tf.assign(self.stats['N'], 0)

        self.reset_stats['Sx'] = tf.assign(
            self.stats['Sx'],
            tf.zeros([1, self.image_height, self.image_width,
                     self.image_channels]))

        self.reset_stats['Sxx'] = tf.assign(
            self.stats['Sxx'],
            tf.zeros([1, self.image_height, self.image_width,
                     self.image_channels]))

        pass

    def gather_statistics(self, session, iterator, filenames, is_training,
                          show_average_image=False):
        initialize_stats(self)
        session.run(iterator.initializer, feed_dict={filenames: filenames})
        print(" " * 80 + "\r" + "[Statistics]\tstarted", end="\r")
        session.run([self.reset_stats['N'],
                    self.reset_stats['Sx'], self.reset_stats['Sxx']])
        while True:
            try:
                N, Sx, Sxx = session.run(
                    [self.update_stats['N'], self.update_stats['Sx'],
                        self.update_stats['Sxx']],
                    feed_dict={is_training.placeholder: False,
                               keep_prob.placeholder: 1.0})
            except (tf.errors.OutOfRangeError):
                session.run([tf.assign(
                    self.layers['inp_norm'].n, self.stats['N']),
                    tf.assign(self.layers['inp_norm'].sx,
                              self.stats['Sx']),
                    tf.assign(self.layers['inp_norm'].sxx,
                              self.stats['Sxx'])])
                if show_average_image:
                    import matplotlib.pyplot as plt
                    a = session.run(self.layers['inp_norm'].outputs[0],
                                    feed_dict={is_training.placeholder: False,
                                    keep_prob.placeholder: 1.0})
                    for i in range(10):
                        plt.imshow(a[i, :, :, 0])
                        plt.show()
                break
            pass

    def define_inner_modules(self, name, inp_min, inp_max, cropped_bool,
                             augmented_bool, norm_by_stat_bool,
                             image_height, image_width, image_channels,
                             batchsize, is_training):

        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.batchsize = batchsize

        # create all modules of the network
        # -----

        self.layers = {}

        # added a module that does nothing b/c it is not clear whether
        # any of the following operations is indeed triggered.
        self.layers['pass_through'] = bb.ActivationModule(
            'pass_through', tf.identity)

        # crop input if desired
        with tf.name_scope('image_manipulation'):
            if cropped_bool:
                self.layers['manipulated'] = bb.CropModule(
                    "input_manipulated", image_height, image_width)

            # augment data if desired
            elif augmented_bool:
                self.layers['manipulated'] = bb.AugmentModule(
                    "input_manipulated", is_training, image_width * 10 // 4)

            else:
                pass

        if norm_by_stat_bool:
            with tf.name_scope('pixel_wise_normalization'):
                self.layers["inp_norm"] = bb.PixelwiseNormalizationModule(
                    "inp_norm", [1, image_height, image_width, image_channels])
        else:
            with tf.name_scope('input_normalization'):
                self.layers["inp_norm"] = bb.NormalizationModule(
                    "inp_norm", inp_max, inp_min)
        #    connect all modules of the network in a meaningful way
        # -----

        with tf.name_scope('wiring_of_modules'):
            if cropped_bool or augmented_bool:
                self.layers["manipulated"].add_input(
                    self.layers['pass_through'])
                self.layers['inp_norm'].add_input(
                    self.layers['manipulated'])

            else:
                self.layers['inp_norm'].add_input(
                    self.layers['pass_through'])

        with tf.name_scope('input_output'):
            self.input_module = self.layers["pass_through"]
            self.output_module = self.layers["inp_norm"]


# TODO: Overwrite init function to have variables for the input Statistics
# TODO: Write Method that calcs those statistics given a session and filenames

# code from main engine:
#
#
# #input statistics for normalization
# self.stats = {}
# self.update_stats = {}
# self.reset_stats = {}
#
# # not initialized to zero for graceful error handling
# self.stats['N'] = tf.Variable(1., trainable=False)
# self.stats['Sx'] = tf.Variable(tf.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]), trainable=False)
# self.stats['Sxx'] = tf.Variable(tf.ones([1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]), trainable=False)
#
# self.update_stats['N'] = tf.assign_add(self.stats['N'], BATCH_SIZE)
# self.update_stats['Sx'] = tf.assign_add(self.stats['Sx'], tf.expand_dims(tf.reduce_sum(tf.cast(inp.variable, tf.float32), axis=0), 0))
# self.update_stats['Sxx'] = tf.assign_add(self.stats['Sxx'], tf.expand_dims(tf.reduce_sum(tf.square(tf.cast(inp.variable, tf.float32)), axis=0), 0))
#
# self.reset_stats['N'] = tf.assign(self.stats['N'], 0)
# self.reset_stats['Sx'] = tf.assign(self.stats['Sx'], tf.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]))
# self.reset_stats['Sxx'] = tf.assign(self.stats['Sxx'], tf.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]))
# ----
#
#
# def gather_input_stats(flnames=training_filenames):
#     sess.run(iterator.initializer, feed_dict={filenames: flnames})
#     print(" " * 80 + "\r" + "[Statistics]\tstarted", end="\r")
#     sess.run([self.reset_stats['N'],
#               self.reset_stats['Sx'], self.reset_stats['Sxx']])
#     while True:
#         try:
#             N, Sx, Sxx = sess.run([self.update_stats['N'], self.update_stats['Sx'], self.update_stats['Sxx']], feed_dict={
#                                   is_training.placeholder: False, keep_prob.placeholder: 1.0})
#         except (tf.errors.OutOfRangeError):
#             sess.run([tf.assign(network.layers['pw_norm'].n, self.stats['N']),
#                       tf.assign(
#                           network.layers['pw_norm'].sx, self.stats['Sx']),
#                       tf.assign(network.layers['pw_norm'].sxx, self.stats['Sxx'])])
#             # import matplotlib.pyplot as plt
#             # a = sess.run(network.layers['pw_norm'].outputs[0], feed_dict = {is_training.placeholder:False, keep_prob.placeholder: 1.0})
#             # for i in range(50):
#             #   #plt.imshow(batch[0][i,:,:,0])
#             #   plt.imshow(a[i,:,:,0])
#             #   plt.show()
#             break
#         pass


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
