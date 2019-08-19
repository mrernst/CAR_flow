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
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_multiplexer


# custom functions
# -----
from visualizer import make_cmap


# ---------------
# data management
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


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
