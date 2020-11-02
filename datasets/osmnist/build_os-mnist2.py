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
import os
import errno
from scipy.ndimage import zoom


# commandline arguments
# -----

# FLAGS


tf.app.flags.DEFINE_boolean('fashion', False,
                            'use fashion_mnist instead')
tf.app.flags.DEFINE_boolean('kuzushiji', False,
                            'Use kuzushiji mnist instead')
tf.app.flags.DEFINE_integer('n_proliferation', 10,
                            'number of generated samples per sample')
tf.app.flags.DEFINE_integer('n_shards', 1,
                            'number of files for the dataset')
tf.app.flags.DEFINE_boolean('centered_target', False,
                            'center target in the middle, additional cue')
tf.app.flags.DEFINE_boolean('testrun', False,
                            'small dataset for testing purposes')
tf.app.flags.DEFINE_boolean('export', False,
                            'export to jpeg files')
tf.app.flags.DEFINE_boolean('zoom', False,
                            'scale objects by distance')

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

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def get_zoom(z_tar, true_size = .6):
    canvas_size = 1 * 2 * z_tar # tan(45) = 1
    return true_size/canvas_size
    

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


def _write_to_file(img_enc_left, img_enc_right, labels, target, count):
    mkdir_p("./export/{}/left/label_{}/".format(target, labels[0]))
    mkdir_p("./export/{}/right/label_{}/".format(target, labels[0]))

    f = open("./export/{}/left/label_{}/{}.png".format(target,
             labels[0], count), "wb+")
    f.write(img_enc_left)
    f.close()
    f = open("./export/{}/right/label_{}/{}.png".format(target,
             labels[0], count), "wb+")
    f.write(img_enc_right)
    f.close()


def pad_to32(img, shape=[32, 32]):
    '''shape=[h, w]'''
    result = np.zeros(shape)
    y_offset = int(round(0.025*((28+(shape[0]-28)/2))))
    result[2-y_offset:30-y_offset, 2:30] = img
    return result


def pad_to64(img, shape=[64, 64]):
    '''shape=[h, w]'''
    result = np.zeros(shape)
    result[18:46, 18:46] = img
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
    OCC_DIST[i] = np.arange(0.4, 0.1, -0.1)[i]
    OCC_DIST_TO_FOC[i] = FOC_DIST - \
       np.arange(0.4, 0.1, -0.1)[i]

for i in range(N_MAX_OCCLUDERS):
    OCC_SHIFT[i+1] = 2*(OCC_DIST_TO_FOC[i] *
                        np.tan(-1.*distancetoangle(FOC_DIST) * (2 * np.pi) /
                        360.))
    SCALING_ARRAY[i+1] = (FOC_DIST / OCC_DIST[i], FOC_DIST / OCC_DIST[i])


# builder class
# -----

class OSMNISTBuilder(object):
    def __init__(self, n_proliferation=10, num_class=10,
                 shape=[32, 32, 1], centered_target=True,
                 fashion=False, kuzushiji=False):
        self.fashion = fashion
        self.kuzushiji = kuzushiji
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
        self.jpeg_rgb_img = tf.image.encode_jpeg(self.np_rgb_img)

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
                        # marker to go on: condition_satisfied = True
                        merged_image_left, merged_image_right,\
                            occlusion_percentage_left,\
                            occlusion_percentage_right,\
                            segmentation_map_left,\
                            segmentation_map_right = \
                            self._resize_pad_crop_merge_rec_cond(
                                combined_array, labels, cond=[0.2, 0.8])
                        self._save(merged_image_left, merged_image_right,
                                   labels, occlusion_percentage_left,
                                   occlusion_percentage_right,
                                   segmentation_map_left,
                                   segmentation_map_right,
                                   tfr_writer, target, c)
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
                        self._resize_pad_crop_merge_rec_cond(
                            combined_array, labels, cond=[0.2, 0.8])
                    self._save(merged_image_left, merged_image_right,
                               labels, occlusion_percentage_left,
                               occlusion_percentage_right,
                               segmentation_map_left,
                               segmentation_map_right,
                               tfr_writer, target, c)
                    print('\rProcessing {:08d}/{:08d}...'
                          .format(c, N_OUTPUT), end='')
                    c += 1
        print()
        self.sess.close()
        tfr_writer.close()

    def _load_mnist(self):

        if self.fashion:
            (x, y), (x_t, y_t) = tf.keras.datasets.fashion_mnist.load_data()
        elif self.kuzushiji:
            from tensorflow.examples.tutorials.mnist import input_data
            km_url = 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/'
            mnist = input_data.read_data_sets(
                "/tmp/tensorflow/kuzushijimnist/input_data",
                reshape=False,
                source_url=km_url
            )
            x = (mnist.train.images * 255).astype('uint8')
            y = mnist.train.labels
            x_t = (mnist.test.images * 255).astype('uint8')
            y_t = mnist.test.labels
        else:
            (x, y), (x_t, y_t) = tf.keras.datasets.mnist.load_data()

        if len(x.shape) == 3:
            x = np.expand_dims(x, -1)
            x_t = np.expand_dims(x_t, -1)

        # diminish test set for testing
        if FLAGS.testrun:
            x_t, y_t = x_t[:100], y_t[:100]
            x, y = x[:100], y[:100]

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
            combined_array_left[:, :, 0] = pad_to32(clipped_zoom(combined_array[:, :, 0], get_zoom(FOC_DIST)))
            combined_array_right[:, :, 0] = combined_array_left[:, :, 0]
        else:
            combined_array_left[:, :, 0], combined_array_right[:, :, 0] = \
                random_crop(pad_to64(clipped_zoom(combined_array[:, :, 0], get_zoom(FOC_DIST))),
                            lshape=64, cshape=32,
                            occludernumber=0, vfloor=self.centered_target)
        unoccluded_target = combined_array_left[:, :, 0]
        target_pixels = unoccluded_target[unoccluded_target != 0].shape[0]

        # Shuffle the occluders
        for dig in range(1, len(labels)):
            combined_array_left[:, :, dig], combined_array_right[:, :, dig] = \
                random_crop(pad_to64(clipped_zoom(combined_array[:, :, dig], get_zoom(OCC_DIST[dig-1]))),
                            lshape=64, cshape=32,
                            occludernumber=dig, vfloor=self.centered_target) # i think zooming has to happen here
        
        
        
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

    def _resize_pad_crop_merge_rec_cond(
            self, combined_array, labels, cond=[0., 1.]):
        combined_img_left, combined_img_right,\
            occlusion_percentage_left,\
            occlusion_percentage_right,\
            segmap_left,\
            segmap_right = \
            self._resize_pad_crop_merge(combined_array, labels)
        # find out if occlusion is between .2 and .8
        o_avg = \
            (occlusion_percentage_left + occlusion_percentage_right)/2
        if (o_avg > cond[0]) and (o_avg < cond[1]):
            return combined_img_left, combined_img_right,\
                occlusion_percentage_left, occlusion_percentage_right,\
                segmap_left, segmap_right
        else:
            return self._resize_pad_crop_merge_rec_cond(
                combined_array, labels, cond)

    def _save(self, merged_image_left, merged_image_right,
              labels, occlusion_percentage_left, occlusion_percentage_right,
              segmentation_map_left, segmentation_map_right, writer, target,
              count):
        if FLAGS.export:
            encoded_left = self.sess.run(
                self.png_img, feed_dict={self.np_img: merged_image_left})
            encoded_right = self.sess.run(
                self.png_img, feed_dict={self.np_img: merged_image_right})
            _write_to_file(encoded_left, encoded_right,
                           labels, target, count)

        else:
            png_encoded_left = self.sess.run(
              self.png_img, feed_dict={self.np_img: merged_image_left})
            png_encoded_right = self.sess.run(
              self.png_img, feed_dict={self.np_img: merged_image_right})

            png_encoded_map_left = self.sess.run(
              self.png_rgb_img,
              feed_dict={self.np_rgb_img: segmentation_map_left})
            png_encoded_map_right = self.sess.run(
              self.png_rgb_img,
              feed_dict={self.np_rgb_img: segmentation_map_right})

            ex = make_tf_example(png_encoded_left, png_encoded_right,
                                 labels, occlusion_percentage_left,
                                 occlusion_percentage_right,
                                 png_encoded_map_left,
                                 png_encoded_map_right)
            writer.write(ex.SerializeToString())

# ------------
# main program
# ------------


if __name__ == '__main__':
    # TODO: replace n_proliferation with shards
    datasetname = 'os'
    if FLAGS.fashion:
        datasetname += 'fashion'
        pathmodifier = './osfashionmnist/'
    elif FLAGS.kuzushiji:
        datasetname += 'kuzushiji'
        pathmodifier = './oskuzushijimnist/'
    else:
        pathmodifier = './'
    datasetname += 'mnist'
    if FLAGS.centered_target:
        datasetname += 'centered'

    path = '{}tfrecord_files/2occ/'.format(pathmodifier)
    mkdir_p(path + 'train/')
    mkdir_p(path + 'test/')
    mkdir_p(path + 'validation/')

    FLAGS.n_proliferation //= FLAGS.n_shards
    builder = OSMNISTBuilder(
        centered_target=FLAGS.centered_target,
        n_proliferation=FLAGS.n_proliferation,
        fashion=FLAGS.fashion,
        kuzushiji=FLAGS.kuzushiji)

    for i in range(FLAGS.n_shards):
        builder.build(
            '{}/train/{}_train{}.tfrecord'.format(
                path, datasetname, i), 'training')
        builder.build(
            '{}/test/{}_test{}.tfrecord'.format(
                path, datasetname, i), 'testing')


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
