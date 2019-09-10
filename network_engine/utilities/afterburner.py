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
from utilities.visualizer import *


class DataEssence(object):
    """docstring for DataEssence."""

    def __init__(self):
        super(DataEssence, self).__init__()
        self.essence = {}

    def generate(self, config_dict):
        self.datacore = {}
        pass

    def write_to_file(self, path_to_essence):
        pass

    def distill(self, list_of_essence_file_paths):
        pass

    def plot_essentials(arg):
        pass

    def freeze_model(self, path_to_model, path_to_checkpoint):
        pass


evaluation_filenames = []
evaluate_ckpt = False


def evaluation(train_it, flnames=evaluation_filenames, tag='Evaluation'):
    print(" " * 80 + "\r" + "[{}}]\tstarted".format(tag), end="\r")
    sess.run([iterator.initializer, testaverages.reset, embedding.reset],
             feed_dict={filenames: flnames})

    # TODO: this should be part of the afterburner, saved to python native
    # get the full output distribution, rewrite the evaluation function
    # all of this including the tsne is afterburner stuff
    list_of_output_samples = []
    list_of_output_times = []
    for time in network.outputs:
        list_of_output_times.append(tf.nn.softmax(network.outputs[time]))

    # delete bool_classification file if it already exists

    if os.path.exists(WRITER_DIRECTORY + 'checkpoints/' + 'evaluation/' +
                      "bool_classification.txt"):

        os.remove(WRITER_DIRECTORY + 'checkpoints/' +
                  'evaluation/' + "bool_classification.txt")
    while True:
        try:
            _, _, extras, images, bc, out = sess.run(
                [update, update_emb, add_merged, image_merged,
                    bool_classification, list_of_output_times],
                feed_dict={keep_prob.placeholder: 1.0,
                           is_training.placeholder: False})

            # save output of boolean comparison
            boolfile = open(WRITER_DIRECTORY + 'checkpoints/' +
                            'evaluation/' + "bool_classification.txt", "a")

            if CONFIG['label_type'] == "onehot":
                for i in list(bc):
                    boolfile.write(str(int(i)) + '\n')
            else:
                for i in range(len(bc[0])):
                    for el in bc[0][i]:
                        boolfile.write(
                            str(int(el in set(bc[1][i]))) + '\t')
                    boolfile.write('\n')
            boolfile.close()

            # temporary code to save output
            list_of_output_samples.append(out)

        except (tf.errors.OutOfRangeError):
            break

    acc, loss, emb, emb_labels, emb_thu, summary = sess.run(
        [testaverages.average_partial_accuracy[CONFIG['time_depth']],
            testaverages.average_cross_entropy[CONFIG['time_depth']],
            embedding.total,
            embedding.labels,
            embedding.thumbnails,
            test_merged])

    print(" " * 80 + "\r" +
          "[{}]\tloss: {:.5f}\tacc: {:.5f} \tstep: {}"
          .format(tag, loss, acc, train_it))

    np.savez_compressed(WRITER_DIRECTORY + 'checkpoints/' + 'evaluation/' +
                        'softmax_output.npz',
                        np.array(list_of_output_samples))

    # pass labels to write to metafile
    return emb, emb_labels, emb_thu


# evaluating restored checkpoint
# -----

if evaluate_ckpt:
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIRECTORY)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)

        # make sure the directories exist, otherwise create them
        mkdir_p(CHECKPOINT_DIRECTORY + 'evaluation/')
        print('[INFO] Restored checkpoint successfully,' +
              ' running evaluation')
        emb, emb_labels, emb_thu = evaluation(
            global_step.eval(), flnames=evaluation_filenames)

        # TODO:: Write a dedicated tsne writer function
        # visualize with tsne
        if False:
            saver.save(sess, WRITER_DIRECTORY + 'checkpoints/' +
                       'evaluation/' + CONFIG['name'] +
                       CONFIG['connectivity'] + CONFIG['dataset'],
                       global_step=global_step.eval())

            npnames = CONFIG['class_encoding']
            lookat = np.zeros(emb_labels.shape, dtype=np.int32)
            lookat[-50:] = 1
            emb_labels = np.asarray(emb_labels, dtype=np.int32)

            # save labels to textfile to be read by tensorboard
            np.savetxt(WRITER_DIRECTORY + 'checkpoints/' + 'evaluation/'
                       + "metadata.tsv",
                       np.column_stack([emb_labels, npnames[emb_labels],
                                       lookat]),
                       header="labels\tnames\tlookat",
                       fmt=["%s", "%s", "%s"], delimiter="\t", comments='')

            # save thumbnails to sprite image
            save_sprite_image(WRITER_DIRECTORY + 'checkpoints/' +
                              'evaluation/' +
                              'embedding_spriteimage.png',
                              emb_thu[:, :, :, :])

            # configure metadata linking
            projector_config = projector.ProjectorConfig()
            embeddings_dict = {}
            # try to write down everything here
            for i in range(TIME_DEPTH + TIME_DEPTH_BEYOND + 1):
                embeddings_dict[i] = projector_config.embeddings_dict.add()
                embeddings_dict[i].tensor_name = \
                    total_embedding_preclass[i].name
                embeddings_dict[i].metadata_path = os.path.join(
                    WRITER_DIRECTORY + 'checkpoints/' +
                    'evaluation/', 'metadata.tsv')
                embeddings_dict[i].sprite.image_path = WRITER_DIRECTORY + \
                    'checkpoints/' + 'evaluation/' + \
                    'embedding_spriteimage.png'
                embeddings_dict[i].sprite.single_image_dibb.extend(
                    [embedding.thu_height, embedding.thu_height])

            tnse_writer = tf.compat.v1.summary.FileWriter(
                WRITER_DIRECTORY + 'checkpoints/' + 'evaluation/')
            projector.visualize_embeddings(
                tsne_writer, projector_config)

        sys.exit()
        print('[INFO] Continue training from last checkpoint')
    else:
        print('[INFO] No checkpoint data found, exiting')
        sys.exit()


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
