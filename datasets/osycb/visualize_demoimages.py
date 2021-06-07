#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# August 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# visualize_demoimages.py                    oN88888UU[[[/;::-.        dP^
# Produce Demo-Images for                   dNMMNN888UU[[[/;:--.   .o@P^
# publication                              ,MMMMMMN888UU[[/;::-. o@^
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

