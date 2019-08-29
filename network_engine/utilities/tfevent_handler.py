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

# standard libraries
# -----
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import sys
import os

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_multiplexer


# custom functions
# -----
from utilities.visualizer import make_cmap


# ---------------------
# checkpoint management
# ---------------------


def read_from_ckpt(listofnamesandshapes, checkpoint):
    tf.reset_default_graph()
    t = {}
    t_ev = {}
    for tup in listofnamesandshapes:
        t[tup[0]] = tf.get_variable(tup[0], tup[1])
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        print("[INFO] Model restored")
        for tup in listofnamesandshapes:
            t_ev[tup[0]] = t[tup[0]].eval()
        return t_ev


def restore_from_dir(sess, folder_path, raise_if_not_found=False,
                     copy_mismatched_shapes=False):
    """
    restore_from_dir is the default restoration function. It takes a session
    and folder_path and restores the checkpoint for further training. It
    returns the iteration number start_iter.
    """
    start_iter = 0
    ckpt = tf.train.get_checkpoint_state(folder_path)
    if ckpt and ckpt.model_checkpoint_path:
        print('[INFO]: Restoring')
        start_iter = restore(sess, ckpt.model_checkpoint_path,
                             raise_if_not_found, copy_mismatched_shapes)
    else:
        if raise_if_not_found:
            raise Exception(
                '[Error]: No checkpoint to restore in %s' % folder_path)
        else:
            print('[Error]: No checkpoint to restore in %s' % folder_path)
    return start_iter


def graceful_restore(session, save_file, raise_if_not_found=False,
                     copy_mismatched_shapes=False):
    """
    graceful_restore is a replacement for restore_from_dir. It should not be
    the default restoration system, but it is useful when you modified the
    model and want to restore as much of the data as possible.
    """
    if not os.path.exists(save_file) and raise_if_not_found:
        raise Exception('File %s not found' % save_file)
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0])
                        for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    var_name_to_var = {var.name: var for var in tf.global_variables()}
    restore_vars = []
    restored_var_names = set()
    restored_var_new_shape = []
    print('[Restoring]:')
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        for var_name, saved_var_name in var_names:
            if 'global_step' in var_name:
                restored_var_names.add(saved_var_name)
                continue
            curr_var = var_name_to_var[var_name]
            try:
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
                    print(str(saved_var_name) + ' -> \t' + str(var_shape) +
                                                ' = ' +
                          str(int(np.prod(var_shape) * 4 / 10**6)) + 'MB')
                    restored_var_names.add(saved_var_name)
                else:
                    print('Shape mismatch for var', saved_var_name,
                          'expected', var_shape,
                          'got', saved_shapes[saved_var_name])
                    restored_var_new_shape.append(
                        (saved_var_name, curr_var,
                         reader.get_tensor(saved_var_name)))
                    print('bad things')
            except (ValueError):
                continue

    ignored_var_names = sorted(
        list(set(saved_shapes.keys()) - restored_var_names))
    print('\n')
    if len(ignored_var_names) == 0:
        print('[INFO]: Restored all variables')
    else:
        print('[INFO]: Did not restore:' + '\n\t'.join(ignored_var_names))

    if len(restore_vars) > 0:
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

    if len(restored_var_new_shape) > 0 and copy_mismatched_shapes:
        print('[INFO]: Trying to restore misshapen variables')
        assign_ops = []
        for name, kk, vv in restored_var_new_shape:
            copy_sizes = np.minimum(kk.get_shape().as_list(), vv.shape)
            slices = [slice(0, cs) for cs in copy_sizes]
            print('copy shape', name, kk.get_shape().as_list(),
                  '->', copy_sizes.tolist())
            new_arr = session.run(kk)
            new_arr[slices] = vv[slices]
            assign_ops.append(tf.assign(kk, new_arr))
        session.run(assign_ops)
        print('[INFO]: Copying unmatched weights done')
    print('[INFO]: Restored %s' % save_file)
    try:
        start_iter = int(save_file.split('-')[-1])
    except ValueError:
        print('[INFO]: Could not parse start iter, assuming 0')
        start_iter = 0
    return start_iter

# ---------------
# write summaries
# ---------------


def variable_summaries(var, name, weights=False):
    """
    Attach an overview of summaries to a Tensor
    for TensorBoard visualization.
    """
    summaries = []
    with tf.name_scope('summaries_{}'.format(name)):
        mean = tf.reduce_mean(var)
        summaries.append(tf.compat.v1.summary.scalar('mean', mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        summaries.append(tf.compat.v1.summary.scalar(
            'stddev', stddev))
        summaries.append(tf.compat.v1.summary.scalar(
            'max', tf.reduce_max(var)))
        summaries.append(tf.compat.v1.summary.scalar(
            'min', tf.reduce_min(var)))
        if weights:
            # exitatory = tf.nn.relu(var)
            # inhibitory = tf.nn.relu(tf.negative(var))
            # summaries += variable_summaries(exitatory, name + '/exitatory')
            # summaries += variable_summaries(inhibitory, name + '/inhibitory')
            summaries.append(tf.compat.v1.summary.scalar('norm', tf.norm(var)))
            # summaries.append(tf.summary.histogram('histogram', var))
    return summaries


def module_scalar_summary(module):
    """
    module_scalar_summary takes a module as its argument and returns a list
    of summary objects to be added to a list or merged
    """
    return [tf.compat.v1.summary.scalar(module.name + "_{}".format(timestep),
            module.outputs[timestep]) for timestep in module.outputs]


def module_variable_summary(module):
    """
    module_variable_summary takes a module as its argument and returns a
    list of summary objects to be added to a list or merged
    """
    return [variable_summaries(module.outputs[timestep],
                               module.name + "_{}".format(timestep))
            for timestep in module.outputs]


def module_timedifference_summary(module, timedepth):
    """
    module_timedifference_summary takes a module as its argument and
    returns a list of summary objects to be added to a list or merged
    """
    if timedepth == 0:
        ret = []
    else:
        ret = [variable_summaries(
            module.outputs[timestep + 1] - module.outputs[timestep],
            module.name + '_{}minus{}'.format(timestep + 1, timestep))
            for timestep in range(timedepth)]
    return ret


# ---------------
# read summaries
# ---------------


def read_single_summary(path_to_tfevent, chist=0, img=0, audio=0, scalars=0,
                        hist=0):
    ea = event_accumulator.EventAccumulator(path_to_tf_event, size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: chist,
            event_accumulator.IMAGES: img,
            event_accumulator.AUDIO: audio,
            event_accumulator.SCALARS: scalars,
            event_accumulator.HISTOGRAMS: hist,
                                            })
    ea.Reload()
    ea.Tags()
    df = pd.DataFrame(ea.Scalars('mean_test_time/average_cross_entropy'))
    return ea


def read_multiple_runs(path_to_project, chist=0, img=0, audio=0, scalars=0,
                       hist=0):
    # use with event_multiplexer (multiplexes different events together
    # useful for retraining I guess...)
    em = event_multiplexer.EventMultiplexer(size_guidance={
        event_accumulator.COMPRESSED_HISTOGRAMS: chist,
        event_accumulator.IMAGES: img,
        event_accumulator.AUDIO: audio,
        event_accumulator.SCALARS: scalars,
        event_accumulator.HISTOGRAMS: hist,
    })
    em.AddRunsFromDirectory(path_to_project)
    # load data
    em.Reload()
    return em


def convert_em_to_df(multiplexer):
    df_dict = {}
    for run in multiplexer.Runs().keys():
        # create fresh empty dataframe
        run_df = pd.DataFrame()
        for tag in multiplexer.Runs()[run]["scalars"]:
            tag_df = pd.DataFrame(multiplexer.Scalars(run, tag))
            tag_df = tag_df.drop(tag_df.columns[[0]], axis=1)
            run_df[tag] = tag_df.value
            run_df["step"] = tag_df.step
        df_dict[run] = run_df
    return df_dict


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
