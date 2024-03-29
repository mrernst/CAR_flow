#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# August 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# tfrecord_handler.py                        oN88888UU[[[/;::-.        dP^
# manages everything to do                  dNMMNN888UU[[[/;:--.   .o@P^
# with tfrecord reading                    ,MMMMMMN888UU[[/;::-. o@^
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
                'osycb/tfrecord_files/{}occ/{}p/{}/'.format(
                    n_occluders, occp, downsampling)
            filenames += tfrecord_auto_traversal(
                tfrecord_directory + type + '/')
    elif 'osdigit' in dataset:
        for occp in [3, 4, 5]:
            TFRECORD_DIRECTORY = input_directory + \
                'digit_database/tfrecord_files/{}{}/'.format(occp, dataset)
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

    image_decoded_l = tf.cast(
        tf.image.decode_jpeg(image_encoded_l, channels=3), tf.float32)
    image_decoded_r = tf.cast(
        tf.image.decode_jpeg(image_encoded_r, channels=3), tf.float32)

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

    image_decoded_l = tf.cast(
        tf.image.decode_png(images_encoded_l), tf.float32)
    image_decoded_r = tf.cast(
        tf.image.decode_png(images_encoded_r), tf.float32)

    segmaps_encoded_l = parsed_features["segmap_left"]
    segmaps_encoded_r = parsed_features["segmap_right"]

    segmap_decoded_l = tf.image.decode_png(segmaps_encoded_l)
    segmap_decoded_r = tf.image.decode_png(segmaps_encoded_r)

    occlusion_l = parsed_features["occlusion_left"]
    occlusion_r = parsed_features["occlusion_left"]

    return image_decoded_l, image_decoded_r, segmap_decoded_l, \
        segmap_decoded_r, occlusion_l, occlusion_r, n_hot, one_hot


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


def _cifar10_parse_single(example_proto):
    features = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "format": tf.io.FixedLenFeature([], tf.string),

    }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    no_classes = 10
    one_hot = tf.one_hot(parsed_features["label"], no_classes)
    n_hot = one_hot

    images_encoded = parsed_features["image_raw"]
    height = parsed_features['height']
    width = parsed_features['width']
    image_shape = tf.stack([height, width, 3])

    image_decoded = tf.image.decode_png(images_encoded)
    image = tf.cast(image_decoded, tf.float32)
    return image, image, n_hot, one_hot


def decode_bytebatch(raw_bytes):
    return tf.image.decode_jpeg(raw_bytes, channels=3)



# -----------------
# deprecated datasets
# -----------------


def _ycb1_parse_sequence(example_proto):
    # Define how to parse the example
    context_features = {
    'sequence/length': tf.FixedLenFeature([], dtype=tf.int64),
    'sequence/height': tf.FixedLenFeature([], dtype=tf.int64),
    'sequence/width': tf.FixedLenFeature([], dtype=tf.int64),
    'sequence/objectdistance': tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        'image/left/filename': tf.FixedLenSequenceFeature([], dtype=tf.string),
        'image/right/filename': tf.FixedLenSequenceFeature([], dtype=tf.string),

        'image/left/encoded': tf.FixedLenSequenceFeature([], dtype=tf.string),
        'image/right/encoded': tf.FixedLenSequenceFeature([], dtype=tf.string),

        'image/class/label': tf.FixedLenSequenceFeature([], dtype=tf.int64),
        'image/class/text': tf.FixedLenSequenceFeature([], dtype=tf.string),

        'image/objectroll': tf.FixedLenSequenceFeature([], dtype=tf.float32),
        'image/objectpitch': tf.FixedLenSequenceFeature([], dtype=tf.float32),
        'image/objectyaw': tf.FixedLenSequenceFeature([], dtype=tf.float32),

        'image/colorspace': tf.FixedLenSequenceFeature([], dtype=tf.string),
        'image/channels': tf.FixedLenSequenceFeature([], dtype=tf.int64),
        'image/format': tf.FixedLenSequenceFeature([], dtype=tf.string)

    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_features,
        sequence_features=sequence_features
        )
    labels = sequence_parsed["image/class/label"]
    one_hot = tf.one_hot(labels, 80)
    images_encoded_lf = sequence_parsed["image/left/encoded"]
    images_encoded_rf = sequence_parsed["image/right/encoded"]

    images_decoded_lf = tf.map_fn(decode_bytebatch, images_encoded_lf,
                                  dtype=tf.uint8)
    images_decoded_rf = tf.map_fn(decode_bytebatch, images_encoded_rf,
                                  dtype=tf.uint8)


    return images_decoded_lf, images_decoded_rf, one_hot, one_hot


def _ycb1_parse_single(example_proto):
    features = {
        "image/left/encoded": tf.FixedLenFeature([], tf.string),
        "image/right/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/left/filename": tf.FixedLenFeature([], tf.string),
        "image/right/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),
        "image/class/text": tf.FixedLenFeature([], tf.string),}


    parsed_features = tf.parse_single_example(example_proto, features)
    labels = parsed_features["image/class/label"]
    one_hot = tf.one_hot(parsed_features["image/class/label"], 80)
    image_encoded_l = parsed_features["image/left/encoded"]
    image_encoded_r = parsed_features["image/right/encoded"]
    image_decoded_l = tf.cast(
        tf.image.decode_jpeg(image_encoded_l, channels=3), tf.float32)
    image_decoded_r = tf.cast(
        tf.image.decode_jpeg(image_encoded_r, channels=3), tf.float32)

    return image_decoded_l, image_decoded_r, one_hot, one_hot


