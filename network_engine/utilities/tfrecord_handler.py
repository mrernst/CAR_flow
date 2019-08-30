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
import os  # handle system path and filenames
from random import shuffle


# -----------------------
# list and load tfrecords
# -----------------------

def list_tfrecord_file(file_list, dirpath):
    """
    list_tfrecord_file takes a list filenames a string dirpath
    and returns a list of paths to tfrecords.
    """
    tfrecord_list = []
    for i in range(len(file_list)):
        if (dirpath == './'):
            current_file_abs_path = os.path.abspath(file_list[i])
        else:
            current_file_abs_path = dirpath + file_list[i]
        if current_file_abs_path.endswith(".tfrecord"):
            tfrecord_list.append(current_file_abs_path)
        else:
            pass
    return tfrecord_list


def tfrecord_auto_traversal(dirpath='./', shuffled=True):
    """
    tfrecord_auto_traversal takes a directory and returns a list
    of all files in that directory that end in '.tfrecord'.
    """
    current_folder_filename_list = os.listdir(dirpath)
    if current_folder_filename_list is not None:
        print("%s files were found under given directory. " %
              len(current_folder_filename_list))
        print("Only files ending with '*.tfrecord' " +
              "will be loaded!")
        tfrecord_list = list_tfrecord_file(
            current_folder_filename_list, dirpath)
        if len(tfrecord_list) != 0:
            for list_index in range(len(tfrecord_list)):
                print(tfrecord_list[list_index])
        else:
            print("Cannot find any tfrecord files, please check path.")
    if shuffled:
        shuffle(tfrecord_list)
    return tfrecord_list


# combine tfrecords from different paths
# -----

# TODO: Needs to be rewritten
def all_percentages_tfrecord_paths(type, dataset, n_occluders, downsampling,
                                   input_directory, shuffled=True):
    """
    all_percentages_tfrecord_paths takes FLAGS and returns all corresponding
    tfrecords the specific dataset.
    """
    filenames = []
    if dataset == "osycb":
        for occp in [20, 40, 60, 80]:
            tfrecord_directory = input_directory + \
                'osycb/tfrecord-files/{}occ/{}p/{}/'.format(
                    n_occluders, occp,  downsampling)
            filenames += tfrecord_auto_traversal(
                tfrecord_directory + type + '/')
    elif 'osdigit' in dataset:
        for occp in [3, 4, 5]:
            TFRECORD_DIRECTORY = input_directory + \
                'digit_database/tfrecord-files/{}{}/'.format(occp, dataset)
            filenames += tfrecord_auto_traversal(
                TFRECORD_DIRECTORY + type + '/')
    if shuffled:
        shuffle(filenames)
    return filenames


# ---------------
# tfrecord parser
# ---------------

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


def _osycb_parse_single(example_proto):
    features = {
        "image/left/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/right/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/left/filename": tf.io.FixedLenFeature([], tf.string),
        "image/right/filename": tf.io.FixedLenFeature([], tf.string),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
        "image/class/text": tf.io.FixedLenFeature([], tf.string),
        "image/class/occ1_label": tf.io.FixedLenFeature([], tf.int64),
        "image/class/occ1_text": tf.io.FixedLenFeature([], tf.string),
        "image/class/occ2_label": tf.io.FixedLenFeature([], tf.int64),
        "image/class/occ2_text": tf.io.FixedLenFeature([], tf.string),
        "image/class/occ3_label": tf.io.FixedLenFeature([], tf.int64),
        "image/class/occ3_text": tf.io.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.io.parse_single_example(example_proto, features)
    one_hot = tf.one_hot(parsed_features["image/class/label"], 80)

    occ1_one_hot = tf.one_hot(parsed_features["image/class/occ1_label"], 80)
    occ2_one_hot = tf.one_hot(parsed_features["image/class/occ2_label"], 80)
    occ3_one_hot = tf.one_hot(parsed_features["image/class/occ3_label"], 80)
    n_hot = one_hot + occ1_one_hot + occ2_one_hot + occ3_one_hot

    image_encoded_l = parsed_features["image/left/encoded"]
    image_encoded_r = parsed_features["image/right/encoded"]

    image_decoded_l = tf.image.decode_jpeg(image_encoded_l, channels=3)
    image_decoded_r = tf.image.decode_jpeg(image_encoded_r, channels=3)

    return image_decoded_l, image_decoded_r, n_hot, one_hot


def _osdigits_parse_single(example_proto):
    features = {
        "image/left/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/right/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/colorspace": tf.io.FixedLenFeature([], tf.string),
        "image/channels": tf.io.FixedLenFeature([], tf.int64),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
        # "image/class/target": tf.io.FixedLenFeature([3], tf.int64),
        "image/class/binary_target": tf.io.FixedLenFeature([10], tf.int64)}

    parsed_features = tf.io.parse_single_example(example_proto, features)
    no_classes = 10
    channels = parsed_features["image/channels"]
    one_hot = tf.one_hot(parsed_features["image/class/label"], no_classes)
    n_hot = tf.cast(parsed_features["image/class/binary_target"], tf.float32)
    image_encoded_l = parsed_features["image/left/encoded"]
    image_encoded_r = parsed_features["image/right/encoded"]
    image_decoded_l = tf.image.decode_jpeg(image_encoded_l, channels=1)
    image_decoded_r = tf.image.decode_jpeg(image_encoded_r, channels=1)

    return image_decoded_l, image_decoded_r, n_hot, one_hot


def _digits_parse_single(example_proto):
    features = {
        "image/left/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/colorspace": tf.io.FixedLenFeature([], tf.string),
        "image/channels": tf.io.FixedLenFeature([], tf.int64),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
        # "image/class/target": tf.io.FixedLenFeature([3], tf.int64),
        "image/class/binary_target": tf.io.FixedLenFeature([10], tf.int64)}

    parsed_features = tf.io.parse_single_example(example_proto, features)
    no_classes = 10
    channels = parsed_features["image/channels"]
    one_hot = tf.one_hot(parsed_features["image/class/label"], no_classes)
    n_hot = tf.cast(parsed_features["image/class/binary_target"], tf.float32)
    image_encoded_l = parsed_features["image/left/encoded"]
    image_decoded_l = tf.image.decode_jpeg(image_encoded_l, channels=1)

    return image_decoded_l, image_decoded_l, n_hot, one_hot


def _osmnist_parse_single(example_proto):
    features = {
        "image_left": tf.io.FixedLenFeature([], tf.string),
        "image_right": tf.io.FixedLenFeature([], tf.string),
        "label1": tf.io.FixedLenFeature([], tf.int64),
        "label2": tf.io.FixedLenFeature([], tf.int64),
        "label3": tf.io.FixedLenFeature([], tf.int64),
        "occlusion_left": tf.io.FixedLenFeature([], tf.float32),
        "occlusion_right": tf.io.FixedLenFeature([], tf.float32),
        "occlusion_avg": tf.io.FixedLenFeature([], tf.float32),
        "segmap_left": tf.io.FixedLenFeature([], tf.string),
        "segmap_right": tf.io.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    no_classes = 10
    one_hot = tf.one_hot(parsed_features["label1"], no_classes)

    occ1_one_hot = tf.one_hot(parsed_features["label2"], no_classes)
    occ2_one_hot = tf.one_hot(parsed_features["label3"], no_classes)
    n_hot = one_hot + occ1_one_hot + occ2_one_hot

    images_encoded_l = parsed_features["image_left"]
    images_encoded_r = parsed_features["image_right"]

    image_decoded_l = tf.image.decode_png(images_encoded_l)
    image_decoded_r = tf.image.decode_png(images_encoded_r)

    segmaps_encoded_l = parsed_features["segmap_left"]
    segmaps_encoded_r = parsed_features["segmap_right"]

    segmap_decoded_l = tf.image.decode_png(segmaps_encoded_l)
    segmap_decoded_r = tf.image.decode_png(segmaps_encoded_r)

    return image_decoded_l, image_decoded_r, segmap_decoded_l, \
        segmap_decoded_r, n_hot, one_hot


def _mnist_parse_single(example_proto):
    features = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image_raw": tf.io.FixedLenFeature([], tf.string),

    }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    no_classes = 10
    one_hot = tf.one_hot(parsed_features["label"], no_classes)
    n_hot = one_hot

    images_encoded = parsed_features["image_raw"]
    height = parsed_features['height']
    width = parsed_features['width']
    depth = parsed_features['depth']
    image_shape = tf.stack([height, width, depth])

    image_decoded = tf.decode_raw(images_encoded, tf.float32)
    image = tf.reshape(image_decoded, image_shape)
    return image, image, n_hot, one_hot


def decode_bytebatch(raw_bytes):
    return tf.image.decode_jpeg(raw_bytes, channels=3)

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
