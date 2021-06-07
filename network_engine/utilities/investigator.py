#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# January 2020                                   _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# investigator.py                            oN88888UU[[[/;::-.        dP^
# convolutional weight analysis             dNMMNN888UU[[[/;:--.   .o@P^
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
# Copyright 2020 Markus Ernst
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.
#
# _____________________________________________________________________________
#
# The investigator
#
#           .\"\"\"-.
#          /      \\
#          |  _..--'-.
#          >.`__.-\"\";\"`
#         / /(     ^\\
#         '-`)     =|-.
#          /`--.'--'   \\ .-.
#        .'`-._ `.\\    | J /
#       /      `--.|   \\__/


# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.python.training import checkpoint_utils as cu

# custom functions
# -----

import visualizer


# commandline arguments
# -----

parser = argparse.ArgumentParser()
parser.add_argument(
     "-cfgf",
     "--config_file",
     type=str,
     default=None,
     help='path to config file')
parser.add_argument(
     "-cfgdir",
     "--config_dir",
     type=str,
     default=None,
     help='path to config directory')
parser.add_argument(
     "-mem",
     "--memory",
     type=int,
     default=20,
     help='memory to be reserved (GB)')
args = parser.parse_args()


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


def get_list_of_images(list_of_weights, stereo):
    for kernel in list_of_weights:
        kernel_name = kernel[0]
        kernel_value = kernel[1]
        kname = kernel_name.split('/')[1].split('_')[0] + '/kernels'
        receptive_pixels = kernel_value.shape[0].value
        if 'fc' in kname:
            pass
        elif 'conv0' in kname:
            if stereo:
                image.append(
                    tf.compat.v1.summary.image(kname, put_kernels_on_grid(
                        kname, tf.reshape(kernel_value,
                                          [2 * receptive_pixels,
                                              receptive_pixels, -1,
                                              network.
                                              net_params
                                              ['conv_filter_shapes'][0][-1]])),
                                     max_outputs=1))
            else:
                image.append(
                    tf.compat.v1.summary.image(kname, put_kernels_on_grid(
                        kname, kernel_value),
                        max_outputs=1))

        else:
            image.append(
                tf.compat.v1.summary.image(kname, put_kernels_on_grid(
                    kname, tf.reshape(
                        kernel_value, [receptive_pixels, receptive_pixels,
                                       1, -1])), max_outputs=1))

    return image



# tsne functions
# -----

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y

# model path from config
# -----

def get_model_paths_from_cfg():
    return ''

# Store weight matrices in a dict of arrays
# -----


# -----------------
# statistical analysis
# -----------------

# PCA
# -----

# Distribution and Histograms
# -----

# Correlation of Data
# -----

# Fourier
# -----

# Overall Statistics
# -----

# Comparison
# -----

if __name__ == __main__:

    if args.config_file:
        modelpath = get_model_paths_from_cfg(args.config_file)
    else:
        modelpath = '/Users/markus/Research/Code/saturn/experiments/001_noname_experiment/data/config0/BLT3_2l_fm1_d1.0_l20.0_bn1_bs100_lr0.003/mnist_0occ_Xp/28x28x1_grayscale_onehot/checkpoints'



    list_of_variables = cu.list_variables(modelpath)
    #get_list_of_images(list_of_variables, False)

    if True:
        conv0weights = cu.load_variable(modelpath, 'convolutional_layer_0/conv0_conv_var')
        viz = visualizer.put_kernels_on_grid(name='conv0_weights', kernel=tf.reshape(conv0weights,[2 * 3, 3, -1, 32]))
        viz_np = viz.numpy()
        plt.imshow(viz_np[0,:,:,0], cmap='gray')
        plt.show()

    else:
        conv0weights = tf.convert_to_tensor(conv0weights)
        viz = visualizer.put_kernels_on_grid(name='conv0_weights', kernel=conv0weights)
        viz_np = viz.numpy()
        plt.imshow(viz_np[0,:,:,0], cmap='gray')
        plt.show()

    conv1weights = cu.load_variable(modelpath, 'convolutional_layer_1/conv1_conv_var')
    conv1weights = tf.convert_to_tensor(conv1weights)
    viz = visualizer.put_kernels_on_grid(name='conv1_weights', kernel=conv1weights)
    viz_np = viz.numpy()
    plt.imshow(viz_np[0,:,:,0], cmap='gray')
    plt.show()

    lateral0weights = cu.load_variable(modelpath, 'lateral_layer_0/lateral0_var')
    lateral0weights = tf.convert_to_tensor(lateral0weights)
    viz = visualizer.put_kernels_on_grid(name='lateral0weights', kernel=lateral0weights)
    viz_np = viz.numpy()
    plt.imshow(viz_np[0,:,:,0], cmap='gray')
    plt.show()

    lateral1weights = cu.load_variable(modelpath, 'lateral_layer_1/lateral1_var')
    lateral1weights = tf.convert_to_tensor(lateral1weights)
    viz = visualizer.put_kernels_on_grid(name='lateral1weights', kernel=lateral1weights)
    viz_np = viz.numpy()
    plt.imshow(viz_np[0,:,:,0], cmap='gray')
    plt.show()

    topdown0weights = cu.load_variable(modelpath, 'topdown_layer_0/topdown0_var')
    topdown0weights = tf.convert_to_tensor(topdown0weights)
    viz = visualizer.put_kernels_on_grid(name='topdown0weights', kernel=topdown0weights)
    viz_np = viz.numpy()
    plt.imshow(viz_np[0,:,:,0], cmap='gray')
    plt.show()

