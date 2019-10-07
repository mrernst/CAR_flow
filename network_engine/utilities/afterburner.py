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
    def __init__(self, remove_files=False):
        super(EssenceCollection, self).__init__()
        path_to_file = os.path.realpath(__file__)
        self.path_to_experiment = path_to_file.rsplit('/', 3)[0]
        self.collection = self.collect_data_essences(
            self.path_to_experiment, remove_files)

    def collect_data_essences(self, path_to_experiment, remove_files):
        # gather and read all files in files/essence/
        collection = {}
        essence = DataEssence()
        path_to_data = path_to_experiment + "/data/"
        for file in os.listdir(path_to_data):
            if file.endswith(".pkl") and file.startswith("conf"):
                config_name = file.split('.')[0].rsplit('i', 1)[0]
                iteration_number = file.split('.')[0].rsplit('i', 1)[-1]
                essence.read_from_file(os.path.join(path_to_data, file))
                if config_name in collection.keys():
                    collection[config_name][iteration_number] = essence.essence
                else:
                    collection[config_name] = {}
                    collection[config_name][iteration_number] = essence.essence
                # delete file
                if remove_files:
                    os.remove(os.path.join(path_to_data, file))
                    print("[INFO] File '{}' deleted.".format(file))
                else:
                    pass
        return collection

    def write_to_file(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.collection, output, pickle.HIGHEST_PROTOCOL)

    def read_from_file(self, filename):
        with open(filename, 'rb') as input:
            self.collection = pickle.load(input)

    def plot_barplot_comparison(self, filename):
        plt.imshow(np.random.random([100,100]))
        plt.savefig(filename)
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
    ess_coll = EssenceCollection(remove_files=False)

    ess_coll.plot_barplot_comparison(
        ess_coll.path_to_experiment +
            '/visualization/{}_comparision.pdf'.format(
            ess_coll.path_to_experiment.rsplit('/')[-1]))
    ess_coll.write_to_file(
        ess_coll.path_to_experiment +
        '/data/{}.pkl'.format(
            ess_coll.path_to_experiment.rsplit('/')[-1]))




# ------ plotting to be incorporated

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.style
import matplotlib as mpl
from matplotlib import collections
from matplotlib.patches import Rectangle
from scipy import stats

COLORSCHEME = "color"
PROJECTDIR = '/home/aecgroup/aecdata/Results_python/markus/stereommnist_all/'


from collections import namedtuple
from cycler import cycler


if COLORSCHEME == "monochrome":
    mpl.style.use('classic')
    #lines.lineStyles.keys()
    #markers.MarkerStyle.markers.keys()
    # Create cycler object. Use any styling from above you please
    monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':', '=.']) * cycler('marker', ['^',',', '.']))


    #bar_cycle = (cycler('hatch', ['///', '--', '...','\///', 'xxx', '\\\\']) * cycler('color', 'w')*cycler('zorder', [10]))
    bar_cycle = (cycler('hatch', ['/', '--', '...','\///', 'xx', '\\']) * cycler('color', 'w')*cycler('zorder', [10]))
    bar_cycle = (cycler('hatch', ['/', '//', '///','////']) * cycler('color', 'w')*cycler('zorder', [10]))
    #bar_cycle = (cycler('hatch', ['', '', '','', '', '']) * cycler('color', 'w')*cycler('zorder', [10]))

    styles = bar_cycle()

else:
    mpl.style.use('default')
    #mpl.rcParams['errorbar.capsize'] = 3
    #mpl.rcParams['hatch.color'] = "white"
    #hatches = ["/////", '\\\\\\\\\\' , "/////"]
    hatches = ["////","/////","/////","/////"]

    import seaborn as sns
    current_palette = sns.cubehelix_palette(5, start=2.15, rot=-.005)
    current_palette = sns.color_palette("Blues")
    current_palette = sns.color_palette("colorblind")
    sns.set_palette(current_palette)

sys.path.insert(0, '/home/mernst/git/aec-py/network/helper/')

from read_tfevent import convert_em_to_df, read_multiple_runs



HOT_ENC = ["nhot"]
HOT_ENC = ["onehot"]

HOT_LABELS = ["_labels"]

PERCENTAGE = ["20p", "40p", "60p", "80p"]
PERCENTAGE = ["Xp"]
BATCHNORM = ["BN"]
INPUT = ["monocular", "binocular"]# ["monocular", "binocular", "binocular"] #["monocular", "monocular", "binocular", "binocular"]

ARCHARRAY = ['B0', 'BF0', 'BK0', 'BT3', 'BL3', 'BLT3']
ARCHNAMES = ['B0', 'B-F0', 'B-K0', 'BT3', 'BL3', 'BLT3']
COLOR = ['grayscale','grayscale']
#ARCHARRAY = ['B0']
OCCARRAY = [1]
ITERATIONS = 5

EVALITER = 1

#ITERATIONS = 1
for inp in range(len(INPUT)):
  for h in range(len(HOT_ENC)):

    rects = []
    n_groups = len(ARCHARRAY)
    n_bars = len(PERCENTAGE)
    data = np.zeros([ITERATIONS, len(PERCENTAGE), n_groups])
    means = np.zeros([len(PERCENTAGE), n_groups])
    stderror = np.zeros([len(PERCENTAGE), n_groups])

    mcnemar_data = {}

    i = 0
    j = 0
    for it in range(ITERATIONS):
      for i in range(n_bars):
        for j in range(n_groups):
          DATADIR = PROJECTDIR + "data/iteration{}/without_offset/{}_2layer_fm1_d1.0_bn1_bs400_lr0.003/multimnist_2occ_{}_Xcm/32x32_{}_grayscale_{}/validation/".format((it + 1), ARCHARRAY[j], (PERCENTAGE)[i], INPUT[inp], HOT_ENC[h])
          EVALDIR = PROJECTDIR + "data/iteration{}/without_offset/{}_2layer_fm1_d1.0_bn1_bs400_lr0.003/multimnist_2occ_{}_Xcm/32x32_{}_grayscale_{}/checkpoints/evaluation/".format((it + 1), ARCHARRAY[j], (PERCENTAGE)[i], INPUT[inp], HOT_ENC[h])
          if it==(EVALITER-1):
            mcnemar_data[it,i,j] = np.loadtxt(EVALDIR + 'bool_classification.txt', dtype=bool)
            if mcnemar_data[it,i,j].all() or (~mcnemar_data[it,i,j]).all():
              print('found one {},{},{},{}'.format(ARCHARRAY[j], INPUT[inp], COLOR[inp], HOT_ENC[h]))

            #DATADIR = PROJECTDIR + "data/iteration{}/default/{}_2layer_fm2_d1.0_bn1_bs100_lr0.1-0.1-40.0/db2_2occ_{}_50cm/32x32_{}_color_{}/validation/".format((it + 1), ARCHARRAY[j], (PERCENTAGE)[i], INPUT[inp], HOT_ENC[h])
            #print(DATADIR)
            #DATADIR = "/home/aecgroup/aecdata/Results_python/markus/SpNet_{}lr0.003{}_YCBdb2_i{}/2occ/{}/ds4/{}/{}/color/validation/".format(ARCHARRAY[j],(BATCHNORM)[0],  (it + 1), (PERCENTAGE)[i], HOT_ENC[h], INPUT[inp])
          multiplexer = read_multiple_runs(DATADIR)
          dataframe_dict = convert_em_to_df(multiplexer)
          if HOT_ENC[h]=='nhot':
              data[it,i, j] += 1 - dataframe_dict['.']['mean_test_time/testtime_avg_accuracy_labels'].values[-1]
              chancelevel = 1 - 0.0375
          else:
              data[it,i, j] += 1 - dataframe_dict['.']['mean_test_time/testtime_avg_accuracy'].values[-1]
              chancelevel = 1 - 0.0125

    means = np.mean(data, axis=0)
    stderror = np.std(data, axis=0) / np.sqrt(ITERATIONS)

    #########################
    # printout to fill tables
    print(inp, HOT_ENC[h])
    print((means).round(3))
    print((stderror).round(3))
    #########################


    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.8 #0.2

    opacity = 1
    error_config = {'ecolor': '0.3'}
    current_palette = sns.dark_palette(sns.color_palette('colorblind')[0], 5, reverse=True)
    current_palette = sns.dark_palette(sns.color_palette('Set3')[0], 5, reverse=True)
    current_palette = sns.dark_palette('LightGray', 5, reverse=True)


    for k in range(n_bars):
      if COLORSCHEME=="monochrome":
        rects.append(ax.bar(index[:3] + k*bar_width, means[k][:3], bar_width,
                     alpha=opacity, **next(styles),
                     yerr=stderror[k][:3], error_kw=error_config,
                     label='{}'.format('feedforward'), linewidth=1))
      else:
        rects.append(ax.bar(index[:3] + k*bar_width, means[k][:3], bar_width,
                     alpha=opacity,
                     yerr=stderror[k][:3], error_kw=error_config,
                     label='{}'.format('feedforward'), linewidth=0, color=current_palette[k])) #, edgecolor=current_palette[(k+1)], hatch=hatches[k]))
    current_palette = sns.dark_palette(sns.color_palette('colorblind')[1], 5, reverse=True)
    current_palette = sns.dark_palette(sns.color_palette('Set3')[2], 5, reverse=True)
    current_palette = sns.dark_palette('Black', 5, reverse=False)
    light_palette = sns.dark_palette('LightGray', 5, reverse=True)
    dark_palette = sns.dark_palette('Black', 5, reverse=False)


    for k in range(n_bars):
      if COLORSCHEME=="monochrome":
        rects.append(ax.bar(index[3:] + k*bar_width, means[k][3:], bar_width,
                     alpha=opacity, **next(styles),
                     yerr=stderror[k][3:], error_kw=error_config,
                     label='{}'.format('recurrent'), linewidth=1))
      else:
        rects.append(ax.bar(index[3:] + k*bar_width, means[k][3:], bar_width,
                     alpha=opacity,
                     yerr=stderror[k][3:], error_kw=error_config,
                     label='{}'.format('recurrent'), linewidth=0, color=current_palette[k])) #, edgecolor=current_palette[(k+1)], hatch=hatches[k]))



    ax.set_xlabel('Network architecture')
    ax.set_ylabel('Error rate')
    ax.set_title('Stereo MultiMNIST (32x32), {}, {}, {}'.format(INPUT[inp], COLOR[inp],HOT_ENC[h]), fontsize=10)
    # ax.set_xticks(index + bar_width * 1)
    ax.set_ylim([0,ax.get_ylim()[1]*1])#*1.25])
    ##ax.set_ylim([0,ax.get_ylim()[1]])
    ax.set_xticks(index + bar_width * (len(PERCENTAGE)/2. - 0.5))
    ax.set_xticklabels([arch[:-1] for arch in ARCHARRAY])
    ax.legend(frameon=True, facecolor='white', edgecolor='white', framealpha=1.0)
    ax.grid(axis='y', zorder=0, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
    ax.spines['left'].set_visible(False)

    ax.axhline(y=chancelevel, xmin=0, xmax=6, color='black')
    ax.text(0.2, chancelevel+0.005, r'chancelevel')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(frameon=True, facecolor='white', edgecolor='white', framealpha=1.0, bbox_to_anchor=(1, .9), loc='center left')

    #mcnemar's table
    for i in range(len(PERCENTAGE)):
      qstar = 0.05
      pval = np.ones([len(ARCHARRAY),len(ARCHARRAY)])
      chi_table = np.ones([len(ARCHARRAY),len(ARCHARRAY)])
      significance_table = np.zeros([len(ARCHARRAY),len(ARCHARRAY)])
      vertexlist = []
      for k in range(len(ARCHARRAY)):
        for j in range(len(ARCHARRAY)):
          if k!=j and k>j:
            #print(i,j,k,mcnemar_data[(EVALITER-1),i,k].shape, mcnemar_data[(EVALITER-1),i,j].shape)
            mcnemar_table = 2*mcnemar_data[(EVALITER-1),i,k] -  mcnemar_data[(EVALITER-1),i,j]
            if (h == 'nhot'):
              ak = np.sum(np.array([e for e in mcnemar_table == 1.]), axis=1)
              bk = np.sum(np.array([e for e in mcnemar_table == 2.]), axis=1)
              ck = np.sum(np.array([e for e in mcnemar_table == -1.]), axis=1)
              dk = np.sum(np.array([e for e in mcnemar_table == 0]), axis=1)
              chi_sq_denom = np.sum(((bk - ck)/mcnemar_table.shape[-1]))**2
              chi_sq_count = np.sum(((bk - ck)/mcnemar_table.shape[-1])**2)
              chi_sq = chi_sq_denom/chi_sq_count
            else:
              a = mcnemar_table[mcnemar_table == 1.].shape[0]
              b = mcnemar_table[mcnemar_table == 2.].shape[0]
              c = mcnemar_table[mcnemar_table == -1.].shape[0]
              d = mcnemar_table[mcnemar_table == 0].shape[0]
              chi_sq = ((b-c)**2)/(b+c)
            pval[k,j] = 1 - stats.chi2.cdf(chi_sq, 1) #stats.chi2.pdf(chi_sq , 1) #
            chi_table[k,j] = chi_sq

      print(np.round(pval,4))
      print(np.round(chi_table,4))
      sorted_pvals = np.sort(pval[pval<1])
      bjq = np.arange(1,len(sorted_pvals)+1)/len(sorted_pvals)*qstar

      for k in range(len(ARCHARRAY)):
        for j in range(len(ARCHARRAY)):
          if k!=j and k>j:
            if pval[k,j] in sorted_pvals[sorted_pvals - bjq <= 0]:
              significance_table[k,j] = 1

              if j<3 and k>2 and (means[i][j] < means[i][k]):
                vertexlist.append([[j+0.25,k-0.5],[j+0.5,k-0.25],[j+0.5,k-0.5]])
            else:
              significance_table[k,j] = 0


      ax = plt.axes([0.8, 0.3, .2, .2])
      ax.matshow(significance_table, cmap='Greys')
      #ax.set_title("McNemar's Test {} {}".format(PERCENTAGE[i],h))
      ax.set_xticklabels([''] + [arch[:-1] for arch in ARCHARRAY], fontsize=8)#fontsize=65)
      ax.tick_params(labelbottom='on',labeltop='off', top='off', right='off')
      ax.set_yticklabels([arch[:-1] for arch in ARCHARRAY], fontsize=8) #fontsize=65)
      ax.set_xlim([-0.5, len(ARCHARRAY)-1-0.5])
      ax.set_ylim([len(ARCHARRAY)-0.5, -0.5+1])
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)

      ax.axhline(y=2.5, xmin=-0.5, xmax=2.5, color='white', linewidth=1)
      ax.axvline(x=2.5, ymin=-0.5, ymax=2.5, color='white', linewidth=1)
      for vertices in vertexlist:
        pc = collections.PolyCollection((vertices,), color='white', edgecolor="none")
        ax.add_collection(pc)

      ax.add_patch(Rectangle((-2.5, 6.9), 1, 1, fill='black', color='black', alpha=1, clip_on=False))
      ax.text(-2.5,9.4, 'Significant difference \n(two-sided McNemar test, \nexpected FDR=0.05)', fontsize=8, horizontalalignment='left', verticalalignment='center')
      plt.savefig(PROJECTDIR + 'visualization/plots/YCBdb2_trainwithall_barplot_{}_{}_{}_bw.pdf'.format(HOT_ENC[h],COLOR[inp], INPUT[inp]))
      plt.show()



      fig, ax = plt.subplots(figsize=(12,12))
      ax.matshow(significance_table, cmap='Greys')
      #ax.set_title("McNemar's Test {} {}".format(PERCENTAGE[i],h))
      ax.set_xticklabels([''] + [arch[:-1] for arch in ARCHARRAY],fontsize=65)
      ax.tick_params(labelbottom='on',labeltop='off', top='off', right='off')
      ax.set_yticklabels([arch[:-1] for arch in ARCHARRAY],fontsize=65)
      ax.set_xlim([-0.5, len(ARCHARRAY)-1-0.5])
      ax.set_ylim([len(ARCHARRAY)-0.5, -0.5+1])
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.axhline(y=2.5, xmin=-0.5, xmax=2.5, color='white')
      ax.axvline(x=2.5, ymin=-0.5, ymax=2.5, color='white')
      for vertices in vertexlist:
        pc = collections.PolyCollection((vertices,), color='white', edgecolor="none")
        ax.add_collection(pc)
      #ax.set_ylabel('Network Architecture')
      plt.tight_layout()
    plt.savefig(PROJECTDIR + 'visualization/plots/trainwithall_{}{}_{}_{}_mcnemar_fixed_lrate.pdf'.format(OCCARRAY[i],HOT_ENC[h],COLOR[inp],INPUT[inp]))
    plt.show()


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
