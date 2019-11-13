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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import errno


import pandas as pd
import numpy as np
import tensorflow as tf


# custom functions
# -----


OSYCB_ENCODING = np.array(['NULL_CLASS', '072-a_toy_airplane', '065-g_cups',
                           '063-b_marbles', '027_skillet', '036_wood_block',
                           '013_apple', '073-e_lego_duplo',
                           '028_skillet_lid', '017_orange',
                           '070-b_colored_wood_blocks', '015_peach',
                           '048_hammer', '063-a_marbles', '073-b_lego_duplo',
                           '035_power_drill', '054_softball',
                           '012_strawberry', '065-b_cups',
                           '072-c_toy_airplane', '062_dice',
                           '040_large_marker', '044_flat_screwdriver',
                           '037_scissors', '011_banana', '009_gelatin_box',
                           '014_lemon', '016_pear', '022_windex_bottle',
                           '065-c_cups', '072-d_toy_airplane',
                           '073-a_lego_duplo', '065-e_cups',
                           '003_cracker_box', '065-f_cups',
                           '070-a_colored_wood_blocks', '073-g_lego_duplo',
                           '033_spatula', '043_phillips_screwdriver',
                           '055_baseball', '073-d_lego_duplo', '029_plate',
                           '052_extra_large_clamp', '021_bleach_cleanser',
                           '065-a_cups', '019_pitcher_base', '018_plum',
                           '065-h_cups', '065-j_cups', '065-d_cups',
                           '025_mug', '032_knife', '065-i_cups',
                           '026_sponge', '071_nine_hole_peg_test',
                           '004_sugar_box', '056_tennis_ball',
                           '038_padlock', '053_mini_soccer_ball',
                           '059_chain', '061_foam_brick', '058_golf_ball',
                           '006_mustard_bottle', '073-f_lego_duplo',
                           '031_spoon', '051_large_clamp',
                           '072-b_toy_airplane', '050_medium_clamp',
                           '072-e_toy_airplane', '042_adjustable_wrench',
                           '010_potted_meat_can', '024_bowl',
                           '073-c_lego_duplo', '007_tuna_fish_can',
                           '008_pudding_box', '057_racquetball',
                           '030_fork', '002_master_chef_can',
                           '077_rubiks_cube', '005_tomato_soup_can'])

tf.app.flags.DEFINE_string('train_directory', './',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', './',
                           'Validation data directory')

# tf.app.flags.DEFINE_string('pdstruct_file',
#                            '/Users/markus/Desktop/test.gzip',
#                            'Pandas Dataframe Pickle file')

# tf.app.flags.DEFINE_string('output_directory',
#                            '/Users/markus/Desktop/ycbtest/',
#                            'where the tfrecord files will be stored')

tf.app.flags.DEFINE_string('input_directory',
                           '/Users/markus/mountpoint/aecgroup/aecdata/Results_python/markus/OS-YCB/YCB_database2/',
                           'where the image data is actually stored')
tf.app.flags.DEFINE_string('pdstruct_file', '/home/aecgroup/aecdata/Textures/occluded/datasets/osycb/dataframes/OSYCB_2occ_allperc_combined.gzip',
                           'Pandas Dataframe Pickle file')
tf.app.flags.DEFINE_string('output_directory',
                           '/home/aecgroup/aecdata/Textures/occluded/datasets/osycb/',
                           'where the tfrecord files will be stored')




tf.app.flags.DEFINE_string(
    'name_modifier', 'ds4', 'string that gets attached to the filename of the tfrecord-files for better discrimination')


tf.app.flags.DEFINE_integer('train_shards', 10,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 0,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 10,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_integer('object_distance', 50,
                            'Distance from the camera to the object shown in the image')

tf.app.flags.DEFINE_boolean('export', False,
                            'export to jpeg files')
tf.app.flags.DEFINE_boolean('central_crop', False,
                            'central crop of the image')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# TODO modify input of function to include other class-lists, maybe duplicate

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


def _write_to_file(img_enc_left, img_enc_right, label, count):
    mkdir_p(FLAGS.output_directory + "/export/left/label_{}/".format(label))
    mkdir_p(FLAGS.output_directory + "/export/right/label_{}/".format(label))

    f = open(FLAGS.output_directory + "/export/left/label_{}/{}.jpeg".format(label, count), "wb+")
    f.write(img_enc_left)
    f.close()
    f = open(FLAGS.output_directory + "/export/right/label_{}/{}.jpeg".format(label, count), "wb+")
    f.write(img_enc_right)
    f.close()

def _convert_to_example(filename_l, image_buffer_l, filename_r, image_buffer_r, label, text, occ1_text, occ2_text, occ3_text, occ1_label, occ2_label, occ3_label, height, width):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      text: string, unique human-readable, e.g. 'dog'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),

        'image/class/occ1_label': _int64_feature(occ1_label),
        'image/class/occ1_text': _bytes_feature(tf.compat.as_bytes(occ1_text)),
        'image/class/occ2_label': _int64_feature(occ2_label),
        'image/class/occ2_text': _bytes_feature(tf.compat.as_bytes(occ2_text)),
        'image/class/occ3_label': _int64_feature(occ3_label),
        'image/class/occ3_text': _bytes_feature(tf.compat.as_bytes(occ3_text)),
        # TODO: Add different classes that are only present in the pd datastruct
        # such as occlusion, eye_occlusion, eye_position etc.
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/left/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename_l))),
        'image/left/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer_l)),
        'image/right/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename_r))),
        'image/right/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer_r))}))
    return example


#TODO replace the _convert_example function with this:
def make_tf_example(image_string_left, image_string_right, labels,
                    occlusion_percentage_left, occlusion_percentage_right,
                    segmentation_string_left, segmentation_string_right):
    """ Make tf-examples from image strings and labels"""
    feature_dict = \
        {'image_left': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_string_left])),
            'image_right': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_string_right])),
            'occlusion_left': tf.train.Feature(
            float_list=tf.train.FloatList(value=[occlusion_percentage_left])),
            'occlusion_right': tf.train.Feature(
            float_list=tf.train.FloatList(value=[occlusion_percentage_right])),
            'occlusion_avg': tf.train.Feature(
            float_list=tf.train.FloatList(value=[
                (occlusion_percentage_left + occlusion_percentage_right)/2])),
            'segmap_left': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[segmentation_string_left])),
            'segmap_right': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[segmentation_string_right]))
         }

    for i in range(len(labels)):
        feature_dict['label{}'.format(i+1)] = \
            tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]]))

    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

        # Initializes function that converts rgb to grayscale
        self._jpeg_data = tf.placeholder(dtype=tf.string)
        image_j = tf.image.rgb_to_grayscale(
            tf.image.decode_jpeg(self._jpeg_data, channels=3))
        self._rgb_to_grayscale = tf.image.encode_jpeg(
            image_j, format='grayscale', quality=100)


        self._crop_data = tf.placeholder(dtype=tf.string)
        image_decoded = tf.image.decode_jpeg(self._crop_data, channels=3)
        self.cropped = tf.image.encode_jpeg(tf.image.resize_with_crop_or_pad(
            image_decoded,
            32,
            32), format='rgb', quality=100)

        self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
        self._encode_jpeg = tf.image.encode_jpeg(
            self._encode_jpeg_data, format='rgb', quality=100)

        self._encode_png_data = tf.placeholder(dtype=tf.uint8)
        self._encode_png = tf.image.encode_png(
            self._encode_png_data)

    def central_crop(self, image_data, target_height, target_width):
        # Initializes function that converts rgb to grayscale
        image = self._sess.run(self.cropped,
                               feed_dict={self._crop_data: image_data})

        return image

    def encode_jpeg(self, array):
        # Initializes function that converts rgb to grayscale
        image = self._sess.run(self._encode_jpeg,
                               feed_dict={self._encode_jpeg_data: array})

        return image

    def encode_png(self, array):
        # Initializes function that converts rgb to grayscale
        image = self._sess.run(self._encode_png,
                               feed_dict={self._encode_png_data: array})

        return image

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def rgb_to_grayscale(self, imagedata):
        imagedata = self._sess.run(self._rgb_to_grayscale,
                                   feed_dict={self._jpeg_data: image_data})
        return imagedata


def _is_png(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    if FLAGS.central_crop:
        image_data = coder.central_crop(image_data, width//10*4, width//10*4)

    return image_data, height, width

# TODO: this needs to have additional inputs, too.


def _process_segmentation_map(filename_l, height, width, coder):
    from PIL import Image
    import numpy as np
    from scipy.misc import imresize
    import matplotlib.pyplot as plt

    segmentation_file = filename_l.rsplit('_left', 1)[0]+'.npz'
    segmaps = np.load(segmentation_file)

    segmap_l = segmaps['segmentation_left']
    segmap_r = segmaps['segmentation_right']

    segmap_l = imresize(segmap_l, size=(height, width))
    segmap_r = imresize(segmap_r, size=(height, width))
    bin_segmap_l = np.array(segmap_l > 0, dtype=int)
    bin_segmap_r = np.array(segmap_r > 0, dtype=int)


    # construct binary maps
    bin_segmap_l[:, :, 0] = bin_segmap_l[:, :, 0] - \
        bin_segmap_l[:, :, 1] - bin_segmap_l[:, :, 2]
    bin_segmap_l[:, :, 1] = bin_segmap_l[:, :, 1] - bin_segmap_l[:, :, 2]

    bin_segmap_r[:, :, 0] = bin_segmap_r[:, :, 0] - \
        bin_segmap_r[:, :, 1] - bin_segmap_r[:, :, 2]
    bin_segmap_r[:, :, 1] = bin_segmap_r[:, :, 1] - bin_segmap_r[:, :, 2]

    segmap_l = np.multiply(bin_segmap_l, np.array(segmap_l > 0, dtype=int)*255)
    segmap_r = np.multiply(bin_segmap_r, np.array(segmap_r > 0, dtype=int)*255)

    jpeg_left = coder.encode_jpeg(segmap_l)
    jpeg_right = coder.encode_jpeg(segmap_r)
    #
    # jpeg_left = coder.encode_png(segmap_l)
    # jpeg_right = coder.encode_png(segmap_r)

    if FLAGS.central_crop:
        segmap_l = coder.central_crop(jpeg_left, width//10*4, width//10*4)
        segmap_r = coder.central_crop(jpeg_right, width//10*4, width//10*4)
        return segmap_l, segmap_r, height/10*4, width/10*4
    else:
        return segmap_l, segmap_r, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, occ_texts, occ_labels, occ_percs, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d.tfrecord' % (
            name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(
            shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename_l = filenames[0][i]
            filename_r = filenames[1][i]
            label = labels[i]
            text = texts[i]
            occ1_text = occ_texts[0][i]
            occ2_text = occ_texts[1][i]
            occ3_text = occ_texts[2][i]
            occ1_label = occ_labels[0][i]
            occ2_label = occ_labels[1][i]
            occ3_label = occ_labels[2][i]
            occ_left = occ_percs[0][i]
            occ_right = occ_percs[1][i]
            occ_avg = occ_percs[2][i]

            image_buffer_l, height, width = _process_image(filename_l, coder)
            image_buffer_r, _, _ = _process_image(filename_r, coder)

            # process segmentation_maps
            seg_buffer_l, seg_buffer_r, height, width = _process_segmentation_map(filename_l, height, width, coder)

            if FLAGS.export:
                _write_to_file(image_buffer_l, image_buffer_r, label,
                               shard_counter*FLAGS.train_shards+counter)
            else:
                #example = _convert_to_example(filename_l, image_buffer_l, filename_r, image_buffer_r, label,
                #                              text, occ1_text, occ2_text, occ3_text, occ1_label, occ2_label, occ3_label, height, width)
                example = make_tf_example(image_buffer_l, image_buffer_r, (label, occ1_label, occ2_label, occ3_label), occ_left, occ_right, seg_buffer_l, seg_buffer_r)
                writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, occ_texts, occ_labels, occ_percs, num_shards):
    """Process and save list of images as TFRecord of Example protos.

    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    assert len(filenames[0]) == len(texts)
    assert len(filenames[0]) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(
        0, len(filenames[0]), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' %
          (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, occ_texts, occ_labels, occ_percs, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames[0])))
    sys.stdout.flush()

def _find_image_files_in_struct(pdDataFrameDir, distance):
    # get the information instead from the datastruct
    pdDataFrame = pd.read_pickle(pdDataFrameDir)

    # make selection of what you actually want to extract -> fine pictures
    pdDataFrame_l = pdDataFrame[pdDataFrame.scale ==
                                'fine'][pdDataFrame.eye == 'left']
    pdDataFrame_r = pdDataFrame[pdDataFrame.scale ==
                                'fine'][pdDataFrame.eye == 'right']

    filenames_l = pdDataFrame_l.filepath.values.tolist()
    filenames_r = pdDataFrame_r.filepath.values.tolist()
    texts = pdDataFrame_l.object_in_focus.values
    occ1_texts = pdDataFrame_l.first_occluder.values
    occ2_texts = pdDataFrame_l.second_occluder.values
    occ3_texts = pdDataFrame_l.third_occluder.values

    occs_avg = pdDataFrame_l.occlusion.values
    occs_left = pdDataFrame_l.occlusion_left.values
    occs_right = pdDataFrame_l.occlusion_right.values

    # # labels of focussed object
    # unique_labels = pdDataFrame_l.object_in_focus.unique().tolist()
    # labels = []
    # # Leave label index 0 empty as a background class.
    # label_index = 1
    # for text in unique_labels:
    #   labels.extend([label_index] * texts[texts == text].shape[0])
    #   label_index += 1
    #
    # insert any sorting you want (important for comparison)
    unique_labels = OSYCB_ENCODING[1:]  # OSYCB_ENCODING[0] is NULLCLASS
    labels = np.zeros_like(texts)
    occ1_labels = np.zeros_like(occ1_texts)
    occ2_labels = np.zeros_like(occ2_texts)
    occ3_labels = np.zeros_like(occ3_texts)
    # Leave label index 0 empty as a background class.
    label_index = 1
    for text in unique_labels:
        labels[np.where(texts == 'G' + text)[0]] = label_index

        occ1_labels[np.where(occ1_texts == 'G' + text)[0]] = label_index
        occ2_labels[np.where(occ2_texts == 'G' + text)[0]] = label_index
        occ3_labels[np.where(occ3_texts == 'G' + text)[0]] = label_index
        label_index += 1

    # remap the texts arrays to a list
    texts = texts.tolist()
    occ1_texts = occ1_texts.tolist()
    occ2_texts = occ2_texts.tolist()
    occ3_texts = occ3_texts.tolist()

    labels = labels.tolist()
    occ1_labels = occ1_labels.tolist()
    occ2_labels = occ2_labels.tolist()
    occ3_labels = occ3_labels.tolist()

    occs_avg = occs_avg.tolist()
    occs_left = occs_left.tolist()
    occs_right = occs_right.tolist()

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames_l)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames_l = [filenames_l[i] for i in shuffled_index]
    filenames_r = [filenames_r[i] for i in shuffled_index]

    occ1_texts = [occ1_texts[i] for i in shuffled_index]
    occ2_texts = [occ2_texts[i] for i in shuffled_index]
    occ3_texts = [occ3_texts[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]

    occ1_labels = [occ1_labels[i] for i in shuffled_index]
    occ2_labels = [occ2_labels[i] for i in shuffled_index]
    occ3_labels = [occ3_labels[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    occs_avg = [occs_avg[i] for i in shuffled_index]
    occs_left = [occs_left[i] for i in shuffled_index]
    occs_right = [occs_right[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels.' %
          (len(filenames_l), len(unique_labels)))

    # workaround to cleanup filenames
    filenames_l = [FLAGS.input_directory + filename.split('/', 7)[-1] for filename in filenames_l]
    filenames_r = [FLAGS.input_directory + filename.split('/', 7)[-1] for filename in filenames_r]

    # store some variables in tuples
    occ_texts = (occ1_texts, occ2_texts, occ3_texts)
    occ_labels = (occ1_labels, occ2_labels, occ3_labels)
    filenames = (filenames_l, filenames_r)
    occ_percs = (occs_left, occs_right, occs_avg)
    return filenames, texts, labels, occ_texts, occ_labels, occ_percs


def _process_dataset_from_struct(name, pdDataFrameDir, num_shards):
    """Process a complete data set and save it as a TFRecord.

    Args:
      name: string, unique identifier specifying the data set.
      pdDataFrameDir: string, path of the pandas DataFrame struct generated by        the YCB_database generator
      num_shards: integer number of shards for this data set.
    """
    object_distance = float(FLAGS.object_distance)
    filenames, texts, labels, occ_texts, occ_labels, occ_percs = _find_image_files_in_struct(
        pdDataFrameDir, object_distance)
    _process_image_files(name + str(FLAGS.object_distance) + FLAGS.name_modifier,
                         filenames, texts, labels, occ_texts, occ_labels, occ_percs, num_shards)


def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.validation_shards')
    print('Saving results to %s' % FLAGS.output_directory)

    # Run it!
    _process_dataset_from_struct('validation', FLAGS.pdstruct_file,
                                 FLAGS.validation_shards)
    _process_dataset_from_struct('train', FLAGS.pdstruct_file,
                                 FLAGS.train_shards)


if __name__ == '__main__':
    tf.app.run()


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
