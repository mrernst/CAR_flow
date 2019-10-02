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
import argparse
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data


# commandline arguments
# -----

# FLAGS

tf.app.flags.DEFINE_boolean('fashion', False,
                            'use fashion_mnist instead')
tf.app.flags.DEFINE_string('data_directory', "tfrecord_files/",
                            'Directory where TFRecords will be stored')

FLAGS = tf.app.flags.FLAGS

# custom functions
# -----


def _data_path(data_directory: str, name: str) -> str:
    """Construct a full path to a TFRecord file to be stored in the
    data_directory. Will also ensure the data directory exists

    Args:
        data_directory: The directory where the records will be stored
        name:           The name of the TFRecord

    Returns:
        The full path to the TFRecord file
    """
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    return os.path.join(data_directory, '{}.tfrecord'.format(name))


def _int64_feature(value: int) -> tf.train.Features.FeatureEntry:
    """Create a Int64List Feature

    Args:
        value: The value to store in the feature

    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value: str) -> tf.train.Features.FeatureEntry:
    """Create a BytesList Feature

    Args:
        value: The value to store in the feature

    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name: str, data_directory: str, num_shards: int = 1):
    """Convert the dataset into TFRecords on disk

    Args:
        data_set:       The MNIST data set to convert
        name:           The name of the data set
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
    """

    data_directory = data_directory + name + '/'
    print('Processing {} data'.format(name))

    images = data_set.images
    labels = data_set.labels

    num_examples, rows, cols, depth = data_set.images.shape

    def _process_examples(start_idx: int, end_index: int, filename: str):
        with tf.io.TFRecordWriter(filename) as writer:
            for index in range(start_idx, end_index):
                sys.stdout.write(
                    "\rProcessing sample {} of {}".format(index+1, num_examples))
                sys.stdout.flush()

                image_raw = images[index].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(rows),
                    'width': _int64_feature(cols),
                    'depth': _int64_feature(depth),
                    'label': _int64_feature(int(labels[index])),
                    'image_raw': _bytes_feature(image_raw)
                }))
                writer.write(example.SerializeToString())

    if num_shards == 1:
        _process_examples(0, data_set.num_examples,
                          _data_path(data_directory, name))
    else:
        total_examples = data_set.num_examples
        samples_per_shard = total_examples // num_shards

        for shard in range(num_shards):
            start_index = shard * samples_per_shard
            end_index = start_index + samples_per_shard
            _process_examples(start_index, end_index, _data_path(
                data_directory, '{}-{}'.format(name, shard+1)))

    print()


def convert_to_tf_record(data_directory: str):
    """Convert the TF MNIST Dataset to TFRecord formats

    Args:
        data_directory: The directory where the TFRecord files should be stored
    """
    fm_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    if FLAGS.fashion:
        mnist = input_data.read_data_sets(
            "/tmp/tensorflow/fashionmnist/input_data",
            reshape=False,
            source_url=fm_url
        )
    else:
        mnist = input_data.read_data_sets(
            "/tmp/tensorflow/mnist/input_data",
            reshape=False,
            )

    convert_to(mnist.validation, 'validation', data_directory, num_shards=10)
    convert_to(mnist.train, 'train', data_directory, num_shards=10)
    convert_to(mnist.test, 'test', data_directory, num_shards=10)


if __name__ == '__main__':
    if FLAGS.fashion:
        data_dir = '.fashionmnist/' + FLAGS.data_directory
    else:
        data_dir = './' + FLAGS.data_directory
    convert_to_tf_record(os.path.expanduser(FLAGS.data_directory))

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
