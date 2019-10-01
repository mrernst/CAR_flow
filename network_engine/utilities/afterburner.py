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
import pickle
import os
from matplotlib import pyplot as plt

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

    def write_to_file(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.essence, output, pickle.HIGHEST_PROTOCOL)

    def read_from_file(self, filename):
        with open(filename, 'rb') as input:
            self.essence = pickle.load(input)

    def distill(self, path, evaluation_data, embedding_data=None):
        self.essence = self._read_tfevents(path)

        needless_keys = [l for l in list(self.essence.keys()) if '/' in l]
        for key in needless_keys:
            self.essence.pop(key, None)

        self.essence['evaluation'] = evaluation_data
        if embedding_data:
            self.essence['embedding'] = embedding_data

    def _read_tfevents(self, path):
        em = tfevent_handler.read_multiple_runs(path)
        df = tfevent_handler.convert_em_to_df(em)
        return df

    def plot_essentials(self, savefile):
        # start figure
        fig, axes = plt.subplots(4, 3, sharex='all', figsize=[7, 9])
        # plot images onto figure
        self._plot_traintest_lcurve(axes)
        self._plot_timebased_lcurve(axes)
        self._plot_parameter_lcurve(fig)

        # save figure
        fig.suptitle(savefile.rsplit('/')[-1])
        fig.savefig(savefile)
        pass

    def _plot_traintest_lcurve(self, axes):
        training_acc = \
            self.essence['training']['accuracy_and_error/accuracy'].tolist()
        training_partacc = \
            self.essence['training']['accuracy_and_error/partial_accuracy']\
            .tolist()
        training_loss = \
            self.essence['training']['accuracy_and_error/loss'].tolist()
        training_steps = \
            self.essence['training']['step'].tolist()

        testing_acc = \
            self.essence['testing']['testtime/accuracy'].tolist()
        testing_partacc = \
            self.essence['testing']['testtime/partial_accuracy'].tolist()
        testing_loss = \
            self.essence['testing']['testtime/loss'].tolist()
        testing_steps = \
            self.essence['testing']['step'].tolist()

        axes[0, 0].plot(training_steps, training_acc)
        axes[0, 0].plot(testing_steps, testing_acc)
        axes[0, 0].set_title('accuracy')

        axes[0, 2].plot(training_steps, training_partacc)
        axes[0, 2].plot(testing_steps, testing_partacc)
        axes[0, 2].set_title('partial accuracy')

        axes[0, 1].plot(training_steps, training_loss)
        axes[0, 1].plot(testing_steps, testing_loss)
        axes[0, 1].set_title('loss')

        pass

    def _plot_timebased_lcurve(self, axes):
        sorted_list_of_keys = self.essence['testing'].keys().tolist()
        sorted_list_of_keys.sort()
        timedepth = int(sorted_list_of_keys[-1].rsplit('_')[-1])

        sorted_list_of_keys = \
            [key for key in sorted_list_of_keys if 'testtime_beyond' in key]
        sorted_list_of_keys = \
            [key.rsplit('_', 1)[0] for key in sorted_list_of_keys]
        sorted_list_of_keys = list(set(sorted_list_of_keys))
        sorted_list_of_keys.sort()
        for i in range(len(sorted_list_of_keys)):
            for j in range(timedepth+1):
                axes[1, i].plot(self.essence['testing']['step'].tolist(),
                                self.essence['testing']['{}_{}'.format(
                                    sorted_list_of_keys[i], j)].tolist(),
                                label=sorted_list_of_keys[i].split('/')[-1])
            axes[1, i].set_title(sorted_list_of_keys[i].split('/')[-1])
        pass

    def _plot_parameter_lcurve(self, fig):
        sorted_list_of_keys = self.essence['training'].keys().tolist()
        sorted_list_of_keys.sort()
        sorted_list_of_keys = \
            [key for key in sorted_list_of_keys if 'weights_and_biases' in key]
        sorted_list_of_keys = \
            [key.rsplit('/', 1)[0] for key in sorted_list_of_keys]
        sorted_list_of_keys = list(set(sorted_list_of_keys))
        sorted_list_of_keys = \
            [key for key in sorted_list_of_keys if '_weights' in key]
        sorted_list_of_keys.sort()
        # for i in range(len(sorted_list_of_keys)):
        for i, ax in enumerate(fig.axes[6:]):
            if i < len(sorted_list_of_keys):
                ax.plot(
                    self.essence['training']['step'].tolist(),
                    self.essence['training']['{}/mean_1'.format(
                        sorted_list_of_keys[i])].tolist(),
                    label=sorted_list_of_keys[i].split('/')[1],
                    color='tab:blue')
                ax.plot(
                    self.essence['training']['step'].tolist(),
                    self.essence['training']['{}/min_1'.format(
                        sorted_list_of_keys[i])].tolist(),
                    label=sorted_list_of_keys[i].split('/')[1],
                    color='tab:blue')
                ax.plot(
                    self.essence['training']['step'].tolist(),
                    self.essence['training']['{}/max_1'.format(
                        sorted_list_of_keys[i])].tolist(),
                    label=sorted_list_of_keys[i].split('/')[1],
                    color='tab:blue')
                ax.fill_between(
                    self.essence['training']['step'],
                    self.essence['training']['{}/mean_1'.format(
                        sorted_list_of_keys[i])] +
                    self.essence['training']['{}/stddev_1'.format(
                        sorted_list_of_keys[i])],
                    self.essence['training']['{}/mean_1'.format(
                        sorted_list_of_keys[i])] -
                    self.essence['training']['{}/stddev_1'.format(
                        sorted_list_of_keys[i])],
                    color='tab:blue',
                    alpha=0.3
                )
                ax.set_title(
                    sorted_list_of_keys[i].split('/')[1].split('_')[1])
        pass

    # TODO: write these visualizer functions
    def visualize_softmax_output(self):
        pass

    def vizualize_tsne_analysis(self):
        pass


class EssenceCollection(object):
    """docstring for EssenceCollection."""

    def __init__(self, arg):
        super(EssenceCollection, self).__init__()
        self.collection = self.collect_data_essences()

    def collect_data_essences(self, path_to_experiment):
        # gather and read all files in files/essence/
        collection = {}
        essence = DataEssence()
        path_to_data = path_to_experiment + "/data"
        for file in os.listdir(path_to_data):
            if file.endswith(".pkl"):
                essence.read_from_file(os.path.join(path_to_data, file))
                collection[file.split('.')[0]] = essence.essence
                # delete file
                # os.rm(os.path.join(path_to_data, file))
        return collection

    def write_to_file(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.essence, output, pickle.HIGHEST_PROTOCOL)

    def read_from_file(self, filename):
        with open(filename, 'rb') as input:
            self.essence = pickle.load(input)

    def plot_essentials(self, filename):
        pass

    def _plot_barplot_comparison(self):
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
    ess_coll.write_to_file(FLAGS.path_to_experiment + '/data/')

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
