#!/usr/bin/python
#
# Project Saturn
# _____________________________________________________________________________
#
#                                                                         _.oo.
# August 2019                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# afterburner.py                             oN88888UU[[[/;::-.        dP^
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

# custom functions
# -----
import utilities.visualizer as visualizer
import utilities.tfevent_handler as tfevent_handler


# commandline arguments
# -----
tf.app.flags.DEFINE_string('path_to_experiment', '/Users/markus/Research/' +
                           'Code/saturn/experiments/001_noname_experiment/',
                           'path to the experiment, i.e. the data_essences')

FLAGS = tf.app.flags.FLAGS


class DataEssence(object):
    """docstring for DataEssence."""

    def __init__(self):
        super(DataEssence, self).__init__()
        self.essence = {}
        self.plot = {}

    def generate(self, config_dict):
        self.datacore = {}
        pass

    def write_to_file(self, dir):
        pass

    def read_from_file(self, path_to_essence):
        pass

    def distill(self, evaluation_data, embedding_data=None):
        if embedding_data:
            pass
        else:
            pass

        _read_tfevents()
        pass



    def _read_tfevents(arg):
        pass
    # # save output of boolean comparison
    # boolfile = open(WRITER_DIRECTORY + 'checkpoints/' +
    #                 'evaluation/' + "bool_classification.txt", "a")
    #
    # if CONFIG['label_type'] == "onehot":
    #     for i in list(bc):
    #         boolfile.write(str(int(i)) + '\n')
    # else:
    #     for i in range(len(bc[0])):
    #         for el in bc[0][i]:
    #             boolfile.write(
    #                 str(int(el in set(bc[1][i]))) + '\t')
    #         boolfile.write('\n')
    # boolfile.close()


    # np.savez_compressed(WRITER_DIRECTORY + 'checkpoints/' + 'evaluation/' +
    #                     'softmax_output.npz',
    #                     np.array(list_of_output_values))

    def plot_essentials(arg):
        pass


class EssenceCollection(object):
    """docstring for EssenceCollection."""

    def __init__(self, arg):
        super(EssenceCollection, self).__init__()
        self.collection = self.collect_data_essences()

    def collect_data_essences(path_to_experiment):
        # gather and read all files in files/essence/
        mkdir_p(path_to_experiment + 'files/essence/')
        collection = {}
        return collection

    def plot_essentials():
        pass


# TODO: Think about how to implement this, low priority
def freeze_model(path_to_model, path_to_checkpoint):
    pass


def unfreeze_model(path_to_model, path_to_checkpoint):
    pass


evaluation_filenames = []
evaluate_ckpt = False


if __name__ == '__main__':
    print('[INFO] afterburner running, collecting data')
    ess_coll = EssenceCollection(FLAGS.path_to_experiment)
    ess_coll.plot_essentials()


# _____________________________________________________________________________
# Description:
#
# This program is supposed to be invoked after the experiment is over,
# it includes functions to create a numpy pickle file with all the important
# data for plotting.
# If invoked as a main program
# the afterburner tries to combine the experiments into useful comparison plots
# _____________________________________________________________________________

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
