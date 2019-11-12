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

from PIL import Image
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

id = 'G006_mustard_bottle20p_id2_0_14_1'
# id = 'G006_mustard_bottle20p_id2_0_23_1'
# id = 'G006_mustard_bottle20p_id2_10_6_3'
# id = 'G006_mustard_bottle60p_id2_74_27_1'
# id = 'G006_mustard_bottle80p_id2_41_27_4'

segmap = np.load('demo_images/{}.npz'.format(id))
image_l = Image.open('demo_images/{}_left_downsampled4.jpeg'.format(id))
image_r = Image.open('demo_images/{}_right_downsampled4.jpeg'.format(id))

image_l = np.array(image_l)
image_r = np.array(image_r)

segmap_l = imresize(segmap['segmentation_left'], size=(60, 80))
segmap_r = imresize(segmap['segmentation_right'], size=(60, 80))


segmap_l = segmap_l[14:46, 24:56, :]
segmap_r = segmap_r[14:46, 24:56, :]

image_l = image_l[14:46, 24:56, :]
image_r = image_r[14:46, 24:56, :]

plt.imshow(np.concatenate([image_l, image_r], axis=1), cmap='gray')
plt.savefig('stereo.png')
plt.imshow(np.concatenate([segmap_l, segmap_r], axis=1))
plt.savefig('stereo_segmap.png')


bin_segmap_l = np.array(segmap_l > 0, dtype=int)
bin_segmap_r = np.array(segmap_r > 0, dtype=int)


# construct binary maps
bin_segmap_l[:, :, 0] = bin_segmap_l[:, :, 0] - \
    bin_segmap_l[:, :, 1] - bin_segmap_l[:, :, 2]
bin_segmap_l[:, :, 1] = bin_segmap_l[:, :, 1] - bin_segmap_l[:, :, 2]

bin_segmap_r[:, :, 0] = bin_segmap_r[:, :, 0] - \
    bin_segmap_r[:, :, 1] - bin_segmap_r[:, :, 2]
bin_segmap_r[:, :, 1] = bin_segmap_r[:, :, 1] - bin_segmap_r[:, :, 2]

plt.imshow(np.concatenate([np.multiply(
    bin_segmap_l, segmap_l), np.multiply(bin_segmap_r, segmap_r)], axis=1))
plt.savefig('stereo_bin_segmap.png')


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
