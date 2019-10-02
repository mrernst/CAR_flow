#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# August 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# build_os-mnist.py                          oN88888UU[[[/;::-.        dP^
# dataset generator:                        dNMMNN888UU[[[/;:--.   .o@P^
# occluded stereo multi MNIST             ,MMMMMMN888UU[[/;::-. o@^
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
from pdb import set_trace
import cv2

# commandline arguments
# -----

# FLAGS


tf.app.flags.DEFINE_boolean('fashion', False,
                            'use fashion_mnist instead')
tf.app.flags.DEFINE_integer('n_proliferation', 10,
                            'number of generated samples per sample')
tf.app.flags.DEFINE_boolean('centered_target', False,
                            'center target in the middle, additional cue')

FLAGS = tf.app.flags.FLAGS

# constants and robot parameters
# -----

FOC_DIST = 0.5
N_MAX_OCCLUDERS = 3
OCC_DIST_TO_FOC = np.zeros([N_MAX_OCCLUDERS])
OCC_DIST = np.zeros([N_MAX_OCCLUDERS])
OCC_SHIFT = np.zeros([N_MAX_OCCLUDERS+1])
SCALING_ARRAY = np.ones([N_MAX_OCCLUDERS+1, 2])

# custom functions
# -----


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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


def pad_to32(img, shape=[32, 32]):
    '''shape=[h, w]'''
    result = np.zeros(shape)
    y_offset = int(round(0.025*((28+(shape[0]-28)/2))))
    result[2-y_offset:30-y_offset, 2:30] = img
    return result


def pad_to56(img, shape=[56, 56]):
    '''shape=[h, w]'''
    result = np.zeros(shape)
    result[14:42, 14:42] = img
    return result


def pad_to42(img, shape=[42, 42]):
    '''shape=[h, w]'''
    result = np.zeros(shape)
    result[7:35, 7:35] = img
    return result


def random_crop(img, lshape=56, cshape=32, occludernumber=2,
                vfloor=True, forced_offset=0):
    halfpoint = ((lshape - cshape) // 2)
    sizediff = (lshape - cshape)

    i, j = np.random.choice(np.concatenate([
        np.arange(0 + occludernumber, halfpoint - forced_offset),
        np.arange(halfpoint + forced_offset,
                  sizediff + 1 - occludernumber)]), 2)
    if vfloor:
        i = halfpoint - int(round(((OCC_SHIFT[occludernumber]/2) + 0.025) *
                            ((28+(cshape-28)/2))))  # (28+(32-28)/2)
    jr = j + int(round(OCC_SHIFT[occludernumber]*((28+(cshape-28)/2))))
    return img[i: i + cshape, j: j + cshape], \
        img[i: i + cshape, jr: jr + cshape]


def distancetoangle(object_distance):
    ''' distancetoangle takes the object_distance defined by initializing an
    object and returns the angle needed to adjust vergence of the robot'''
    X_EYES_POSITION = 0.062335
    Y_EYES_DISTANCE = 0.034000 + 0.034000
    new_x = object_distance - X_EYES_POSITION
    return (-np.arctan(Y_EYES_DISTANCE /
                       np.sqrt(new_x**2 + Y_EYES_DISTANCE**2)) * 360 /
                      (2 * np.pi))


for i in range(N_MAX_OCCLUDERS):
    OCC_DIST[i] = np.arange(0.4, 0.2, (0.4 - 0.2) / (-1 * N_MAX_OCCLUDERS))[i]
    OCC_DIST_TO_FOC[i] = FOC_DIST - \
        np.arange(0.4, 0.2, (0.4 - 0.2) / (-1 * N_MAX_OCCLUDERS))[i]

for i in range(N_MAX_OCCLUDERS):
    OCC_SHIFT[i+1] = 2*(OCC_DIST_TO_FOC[i] *
                        np.tan(-1.*distancetoangle(FOC_DIST) * (2 * np.pi) /
                        360.))
    SCALING_ARRAY[i+1] = (FOC_DIST / OCC_DIST[i], FOC_DIST / OCC_DIST[i])


# builder class
# -----

class OSMNISTBuilder(object):
    def __init__(self, n_proliferation=10, num_class=10,
                 shape=[32, 32, 1], centered_target=True, fashion=False):
        self.fashion = fashion
        self.num_class = num_class
        self.centered_target = centered_target
        self.n_per_class, self.remainder = divmod(
            n_proliferation, num_class - 1)
        self.n_proliferation = n_proliferation
        # a simple TF graph here
        self.np_img = tf.compat.v1.placeholder(tf.uint8, shape=shape)
        self.png_img = tf.image.encode_png(self.np_img)
        self.jpeg_img = tf.image.encode_jpeg(self.np_img)

        self.np_rgb_img = tf.compat.v1.placeholder(tf.uint8, shape=[32, 32, 3])
        self.png_rgb_img = tf.image.encode_png(self.np_rgb_img)

        self.sess = None
        self.tfr_writer = None

    def build(self, oFilename, target='training'):
        ''' build training or testing set '''
        if target == 'training':
            x, _, array_size = self._load_mnist()
            N_OUTPUT = self.n_proliferation*array_size[0]

        elif target == 'testing':
            _, x, array_size = self._load_mnist()
            N_OUTPUT = self.n_proliferation*array_size[1]

        else:
            raise ValueError('Only `training` and `testing` are supported.')

        c = 1
        self.sess = tf.compat.v1.Session()
        tfr_writer = tf.io.TFRecordWriter(oFilename)
        for i in range(self.num_class):
            x_digit_i = x[i]
            other_class = set(range(self.num_class)) - set([i])
            for xi in x_digit_i:
                for j in other_class:
                    Nj = x[j].shape[0]
                    index = np.random.choice(range(Nj), self.n_per_class,
                                             replace=False)
                    imgs_from_that_class = x[j][index]

                    for xo in imgs_from_that_class:
                        l = np.random.choice(list(other_class-set([j])))
                        Nl = x[l].shape[0]
                        index = np.random.choice(range(Nl))
                        xl = x[l][index]
                        combined_array = np.concatenate([xi, xo, xl], -1)
                        labels = np.array([i, j, l])
                        merged_image_left, merged_image_right,\
                            occlusion_percentage_left,\
                            occlusion_percentage_right,\
                            segmentation_map_left,\
                            segmentation_map_right = \
                            self._resize_pad_crop_merge(combined_array, labels)
                        self._save(merged_image_left, merged_image_right,
                                   labels, occlusion_percentage_left,
                                   occlusion_percentage_right,
                                   segmentation_map_left,
                                   segmentation_map_right,
                                   tfr_writer)
                        print('\rProcessing {:08d}/{:08d}...'
                              .format(c, N_OUTPUT), end='')
                        c += 1

                for _ in range(self.remainder):
                    j = np.random.choice(list(other_class))
                    Nj = x[j].shape[0]
                    index = np.random.choice(range(Nj))
                    xo = x[j][index]
                    l = np.random.choice(list(other_class-set([j])))
                    Nl = x[l].shape[0]
                    index = np.random.choice(range(Nl))
                    xl = x[l][index]
                    combined_array = np.concatenate([xi, xo, xl], -1)
                    labels = np.array([i, j, l])
                    merged_image_left, merged_image_right,\
                        occlusion_percentage_left,\
                        occlusion_percentage_right,\
                        segmentation_map_left,\
                        segmentation_map_right = \
                        self._resize_pad_crop_merge(combined_array, labels)
                    self._save(merged_image_left, merged_image_right,
                               labels, occlusion_percentage_left,
                               occlusion_percentage_right,
                               segmentation_map_left,
                               segmentation_map_right,
                               tfr_writer)
                    print('\rProcessing {:08d}/{:08d}...'
                          .format(c, N_OUTPUT), end='')
                    c += 1
        print()
        self.sess.close()
        tfr_writer.close()

    def _load_mnist(self):
        if self.fashion:
            (x, y), (x_t, y_t) = tf.keras.datasets.fashion_mnist.load_data()
        else:
            (x, y), (x_t, y_t) = tf.keras.datasets.mnist.load_data()

        if len(x.shape) == 3:
            x = np.expand_dims(x, -1)
            x_t = np.expand_dims(x_t, -1)

            # diminish test set for testing
            # x_t, y_t = x_t[:100], y_t[:100]
            # x, y = x[:100], y[:100]

        array_size = (x.shape[0], x_t.shape[0])
        x = [x[y == i] for i in range(self.num_class)]
        x_t = [x_t[y_t == i] for i in range(self.num_class)]
        return x, x_t, array_size

    def _resize_pad_crop_merge(self, combined_array, labels):
        # cv2.resize(img, dsize=(int(0.35*28), int(0.35*28)),
        #            interpolation=cv2.INTER_CUBIC)
        # instead of random_cropping only do a padded version of xi
        # so xi is in the middle of the canvas
        combined_array_left = np.zeros([32, 32, 4])
        combined_array_right = np.zeros([32, 32, 4])

        # pad the target
        if self.centered_target:
            combined_array_left[:, :, 0] = pad_to32(combined_array[:, :, 0])
            combined_array_right[:, :, 0] = combined_array_left[:, :, 0]
        else:
            combined_array_left[:, :, 0], combined_array_right[:, :, 0] = \
                random_crop(pad_to42(combined_array[:, :, 0]),
                            lshape=42, cshape=32,
                            occludernumber=0, vfloor=self.centered_target)
        unoccluded_target = combined_array_left[:, :, 0]
        target_pixels = unoccluded_target[unoccluded_target != 0].shape[0]

        # Shuffle the occluders
        for dig in range(1, len(labels)):
            combined_array_left[:, :, dig], combined_array_right[:, :, dig] = \
                random_crop(pad_to42(combined_array[:, :, dig]),
                            lshape=42, cshape=32,
                            occludernumber=dig, vfloor=self.centered_target)

        # Make sure there is a notion of occlusion and order
        for dig in range(len(labels)-1):
            # combined_array_left[:,:,dig]\
            #   [combined_array_left[:,:,dig+1] != 0] = 0
            combined_array_left[:, :, dig][np.max(
                combined_array_left[:, :, dig+1:], axis=-1,
                keepdims=False) != 0] = 0
            # combined_array_right[:,:,dig]\
            #   [combined_array_right[:,:,dig+1] != 0] = 0
            combined_array_right[:, :, dig][np.max(
                combined_array_right[:, :, dig+1:], axis=-1,
                keepdims=False) != 0] = 0

        # the following are actually the segmentation maps
        segmap_left = combined_array_left[:, :, 0:3]
        segmap_right = combined_array_right[:, :, 0:3]
        # but maybe the target pixels are enough
        # target_left = combined_array_left[:, :, 0:1]
        # target_right = combined_array_right[:, :, 0:1]

        # calculate occlusion percentage
        occlusion_percentage_left = \
            (target_pixels -
                combined_array_left[:, :, 0]
                [combined_array_left[:, :, 0] != 0]
                .shape[0])/target_pixels
        occlusion_percentage_right = \
            (target_pixels -
                combined_array_right[:, :, 0]
                [combined_array_right[:, :, 0] != 0]
                .shape[0])/target_pixels
        # merge the two images
        combined_img_left = np.max(combined_array_left, -1, keepdims=True)
        combined_img_right = np.max(combined_array_right, -1, keepdims=True)
        return combined_img_left, combined_img_right,\
            occlusion_percentage_left, occlusion_percentage_right,\
            segmap_left, segmap_right

    def _save(self, merged_image_left, merged_image_right,
              labels, occlusion_percentage_left, occlusion_percentage_right,
              segmentation_map_left, segmentation_map_right, writer):

        png_encoded_left = self.sess.run(
          self.png_img, feed_dict={self.np_img: merged_image_left})
        png_encoded_right = self.sess.run(
          self.png_img, feed_dict={self.np_img: merged_image_right})

        # jpeg_encoded = self.sess.run(
        # self.jpeg_img, feed_dict={self.np_img: combined_img})

        png_encoded_map_left = self.sess.run(
          self.png_rgb_img, feed_dict={self.np_rgb_img: segmentation_map_left})
        png_encoded_map_right = self.sess.run(
          self.png_rgb_img,
          feed_dict={self.np_rgb_img: segmentation_map_right})

        ex = make_tf_example(png_encoded_left, png_encoded_right,
                             labels, occlusion_percentage_left,
                             occlusion_percentage_right, png_encoded_map_left,
                             png_encoded_map_right)
        writer.write(ex.SerializeToString())


# ------------
# main program
# ------------

if __name__ == '__main__':
    # TODO: replace n_proliferation with shards
    builder = OSMNISTBuilder(
        centered_target=FLAGS.centered_target,
        n_proliferation=FLAGS.n_proliferation,
        fashion=FLAGS.fashion)

    datasetname = 'os'
    if FLAGS.fashion:
        datasetname += 'fashion'
    datasetname += 'mnist'
    if FLAGS.centered_target:
        datasetname += 'centered'

    builder.build('./{}_train.tfrecord'.format(datasetname), 'training')
    builder.build('./{}_test.tfrecord'.format(datasetname), 'testing')

# TODO: add the option to make os-mnist without centering the target digit
# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
