#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Modified: Amir Alansary <amiralansary@gmail.com>
# Modified: Arjit Jain <thearjitjain@gmail.com>

import time
import numpy as np
from tqdm import tqdm
import multiprocessing
from six.moves import queue

from tensorpack.utils import logger

from tensorpack.callbacks.base import Callback
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs
from tensorpack.utils.concurrency import StoppableThread, ShareSessionThread

import os
import SimpleITK as sitk

###############################################################################
weight = {
'AC':0.99,
'PC':0.95,
'BPons':0.95,
'RP':0.95,
'VN4':0.95,
'genu':0.95,
'LE':0.95,
'RE':0.95,
'CM':0.05,
'SMV':0.90,
'dens_axis':0.90,
'rostrum':0.90,
'l_caud_head':0.90,
'r_caud_head':0.90,
'optic_chiasm':0.90,
'r_corp':0.90,
'l_corp':0.90,
'l_front_pole':0.80,
'r_front_pole':0.80,
'r_lat_ext':0.90,
'l_lat_ext':0.90,
'r_occ_pole':0.90,
'l_occ_pole':0.90,
'r_prim_ext':0.90,
'l_prim_ext':0.90,
'mid_prim_inf':0.90,
'mid_prim_sup':0.90,
'r_sup_ext':0.65,
'l_sup_ext':0.65,
'r_temp_pole':0.90,
'l_temp_pole':0.90,
'm_ax_inf':0.90,
'm_ax_sup':0.90,
'mid_basel':0.85,
'left_cereb':0.85,
'RP_front':0.85,
'rostrum_front':0.85,
'r_inner_corpus':0.90,
'l_inner_corpus':0.90,
'mid_lat':0.70,
'mid_sup':0.70,
'left_lateral_inner_ear':0.85,
'right_lateral_inner_ear':0.85,
'top_left':0.80,
'top_right':0.80,
'callosum_left':0.60,
'callosum_right':0.60,
'lat_left':0.60,
'lat_right':0.60,
'lat_ven_left':0.60,
'lat_ven_right':0.60
}

def play_one_episode(env, func, render=False, agents=2):
    def predict(s, agents):
        """
        Run a full episode, mapping observation to action, WITHOUT 0.001 greedy.
    :returns sum of rewards
        """
        # pick action with best predicted Q-value
        acts = np.zeros((agents,))
        for i in range(0, agents):
            s[i] = s[i][None, :, :, :]
        q_values = func(*s)
        for i in range(0, agents):
            q_values[i] = q_values[i].flatten()
            acts[i] = np.argmax(q_values[i])

        return acts, q_values

    obs = env.reset()
    obs = list(obs)
    sum_r = np.zeros((agents,))
    filenames_list = []
    distError_list = []
    locations_list = []
    isOver = [False] * agents
    while True:
        acts, q_values = predict(obs, agents)
        obs, r, isOver, info = env.step(acts, q_values, isOver)
        obs = list(obs)
        if render:
            env.render()

        for i in range(0, agents):
            if not isOver[i]:
                sum_r[i] += r[i]
            if np.all(isOver):
                filenames_list.append(info["filename_{}".format(i)])
                distError_list.append(info["distError_{}".format(i)])
                locations_list.append(info["location_{}".format(i)])
        if np.all(isOver):
            return sum_r, filenames_list, distError_list, q_values, locations_list


###############################################################################

def play_n_episodes(
    player,
    predfunc,
    nr,
    render=False,
    agents=2,
    fidname="fiducial_name",
    infDir="../inference",
):
    """wraps play_one_episode, playing a single episode at a time and logs results
    used when playing demos."""
    if infDir[-1] == "/":
        infDir = infDir[:-1]
    if not os.path.isdir(infDir):
        os.mkdir(infDir)
    logger.info("Start Playing ... ")
    dists = np.zeros((agents, nr))
    weights = []
    fnames = []
    logs = []
    for k in range(nr):
        # if k != 0:
        #     player.restart_episode()
        score, filename, distance_error, q_values, location = play_one_episode(
            player, predfunc, render=render, agents=agents
        )
        img = sitk.ReadImage(infDir + "/" + os.path.basename(filename[0]))
        fcsv_new = open(
            infDir + "/" + os.path.basename(filename[0])[:-7] + '_lmks.fcsv', "w"
        )
        fcsv_new.write(
            "# Markups fiducial file version = 4.10\n# CoordinateSystem = 0\n# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
        )
        weighted_error = 0
        denominator = 0
        for i in range(0, agents):
            dists[i, k] = distance_error[i]
            weighted_error += weight[fidname[i]] * distance_error[i]
            denominator += weight[fidname[i]]
            logger.info(
                "{}/{} - {} - AGENT {} - score {} - distError {} - q_values {} - location {}".format(
                    k + 1,
                    nr,
                    filename[i],
                    i,
                    score[i],
                    distance_error[i],
                    q_values[i],
                    location[i],
                )
            )
            logs.append([distance_error[i], filename[i], i])
            physical = img.TransformContinuousIndexToPhysicalPoint(
                (float(location[i][0]), float(location[i][1]), float(location[i][2]))
            )
            fcsv_new.write(
                "vtkMRMLMarkupsFiducialNode_{},{},{},{},0,0,0,1,1,1,0,{},,\n".format(
                    i, -physical[0], -physical[1], physical[2], fidname[i]
                )
            )
        weighted_error /= denominator #Sum of all the weights
        fcsv_new.close()
        weights.append(weighted_error)
        fnames.append(filename[0])
    weighted = np.column_stack((np.array(weights), np.array(fnames)))
    np.save(infDir + "/errorAnalysis.npy", np.array(logs))
    np.save(infDir + '/weightedErrorAnalysis.npy', np.array(weighted))
    for i in range(0, agents):
        mean_dists = np.mean(dists[i])
        var_dist = np.var(dists[i])
        max_dist = np.max(dists[i])
        logger.info("MEAN DISTANCE OF AGENT {} is {}".format(i, mean_dists))
        logger.info("VARIANCE DISTANCE OF AGENT {} is {}".format(i, var_dist))
        logger.info("MAX DISTANCE OF AGENT {} is {}".format(i, max_dist))
    logger.info("MEAN OF WEIGHTED ERROR is {}".format(np.mean(weights)))
    logger.info("VARIANCE OF WEIGHTED ERROR is {}".format(np.var(weights)))
    logger.info("MAX  OF WEIGHTED ERROR  is {}".format(np.max(weights)))
###############################################################################


def eval_with_funcs(predictors, nr_eval, get_player_fn, files_list=None, agents=2):
    """
    Args:
        predictors ([PredictorBase])

    Runs episodes in parallel, returning statistics about the model performance.
    """

    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue, distErrorQueue, agents=2):
            super(Worker, self).__init__()
            self.agents = agents
            self._func = func
            self.q = queue
            self.q_dist = distErrorQueue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(
                    task=False, files_list=files_list, agents=self.agents
                )
                while not self.stopped():
                    try:
                        sum_r, filename, dist, q_values, location = play_one_episode(
                            player, self.func, agents=self.agents
                        )
                    except RuntimeError:
                        return
                    for i in range(0, self.agents):
                        self.queue_put_stoppable(self.q, sum_r[i])
                        self.queue_put_stoppable(self.q_dist, dist[i])

    q = queue.Queue()
    q_dist = queue.Queue()

    threads = [Worker(f, q, q_dist, agents=agents) for f in predictors]

    # start all workers
    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()
    dist_stat = StatCounter()

    # show progress bar w/ tqdm
    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        r = q.get()
        stat.feed(r)
        dist = q_dist.get()
        dist_stat.feed(dist)

    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        r = q.get()
        stat.feed(r)

    while q_dist.qsize():
        dist = q_dist.get()
        dist_stat.feed(dist)

    if stat.count > 0:
        return (stat.average, stat.max, dist_stat.average, dist_stat.max)
    return (0, 0, 0, 0)


###############################################################################


def eval_model_multithread(pred, nr_eval, get_player_fn, files_list):
    """
    Args:
        pred (OfflinePredictor): state -> Qvalue

    Evaluate pretrained models, or checkpoints of models during training
    """
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    with pred.sess.as_default():
        mean_score, max_score, mean_dist, max_dist = eval_with_funcs(
            [pred] * NR_PROC, nr_eval, get_player_fn, files_list
        )
    logger.info(
        "Average Score: {}; Max Score: {}; Average Distance: {}; Max Distance: {}".format(
            mean_score, max_score, mean_dist, max_dist
        )
    )


###############################################################################


class Evaluator(Callback):
    def __init__(
        self,
        nr_eval,
        input_names,
        output_names,
        get_player_fn,
        files_list=None,
        agents=2,
    ):
        self.files_list = files_list
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn
        self.agents = agents

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 2)
        self.pred_funcs = [
            self.trainer.get_predictor(self.input_names, self.output_names)
        ] * NR_PROC

    def _trigger(self):
        """triggered by Trainer"""
        mean_score, max_score, mean_dist, max_dist = eval_with_funcs(
            self.pred_funcs,
            self.eval_episode,
            self.get_player_fn,
            self.files_list,
            agents=self.agents,
        )

        # log scores
        self.trainer.monitors.put_scalar("mean_score", mean_score)
        self.trainer.monitors.put_scalar("max_score", max_score)
        self.trainer.monitors.put_scalar("mean_distance", mean_dist)
        self.trainer.monitors.put_scalar("max_distance", max_dist)


###############################################################################
