#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: medical.py
# Author: Amir Alansary <amiralansary@gmail.com>
# Modified: Arjit Jain <thearjitjain@gmail.com>

import csv
import itertools


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
import copy
import os
import sys
import six
import random
import threading
import numpy as np
from tensorpack import logger
from collections import Counter, defaultdict, deque, namedtuple

import cv2
import math
import time
from PIL import Image
import subprocess
import shutil

import gym
from gym import spaces

try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'.")

from tensorpack.utils.utils import get_rng
from tensorpack.utils.stats import StatCounter

from IPython.core.debugger import set_trace
from dataReader import *

_ALE_LOCK = threading.Lock()

Rectangle = namedtuple("Rectangle", ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])


# ===================================================================
# =================== 3d medical environment ========================
# ===================================================================

meanFiducialLocations = np.round([[150.7702, 138.60277954, 151.98192137],
       [150.74588707, 162.10524866, 110.23203827],
       [151.09621394, 156.29945211, 144.78881849],
       [180.83112872,  82.83708274, 131.29505951],
       [151.13853325, 165.22111792, 149.20116468],
       [119.85870404,  83.46380745, 131.34063267],
       [150.99089833, 159.52185415, 134.44216521],
       [151.55070037, 170.83357517, 129.71503149],
       [150.93878646, 175.14151264, 131.3957788 ],
       [150.38895892, 185.02951062, 122.06681438],
       [173.80787289, 188.94281011, 161.22261882],
       [127.15455103, 186.64361216, 161.80933774],
       [151.93073548, 157.20220841,  81.4353914 ],
       [150.59105531, 122.38810986, 163.5195688 ],
       [165.49572885, 120.02350303, 160.85346251],
       [183.31368314, 180.90842714, 111.32023007],
       [161.14224195,  80.39792361, 179.32363185],
       [151.79010064, 153.47446946, 170.62079148],
       [214.38790092, 176.2824004 , 147.0723954 ],
       [162.74941969, 238.88180688, 140.35430103],
       [183.30568753, 174.82067126, 122.66866385],
       [161.29664229, 169.47159828, 217.22514084],
       [186.85032027, 114.29973643, 133.83023224],
       [171.33925287, 148.17630543, 138.50647418],
       [128.53853775, 147.51666101, 137.35989143],
       [184.34108122, 148.59714047, 135.76772275],
       [118.43539672, 144.21874083, 133.07198722],
       [162.19982716, 174.10511567, 124.08372292],
       [208.46878039, 140.3457483 , 108.45904222],
       [151.6218516 , 182.21440122, 164.35046143],
       [150.1336313 , 129.85575415, 176.01411505],
       [151.29735792, 146.9260105 , 156.38491841],
       [150.86096144, 150.86383804, 143.96060059],
       [151.35081922, 187.46337068, 122.42567749],
       [151.13254747, 203.78583404, 135.81749554],
       [151.32575415, 191.66619639, 149.14032132],
       [150.65423831, 134.3909411 , 138.37864457],
       [134.56719961, 118.87153816, 163.84388299],
       [120.87916644, 182.75201017, 110.23060053],
       [140.55058493,  79.47563927, 178.40777421],
       [149.0524632 , 153.42689793, 171.06462178],
       [ 86.27817145, 167.34519222, 145.36055114],
       [133.97973733, 237.35662212, 141.78197246],
       [119.05957563, 172.48246284, 122.56024558],
       [141.57094056, 185.89693817, 215.72659647],
       [114.58058444, 113.07033697, 132.23603307],
       [ 94.07960264, 142.56579797, 108.18640964],
       [150.70425898, 128.437913  , 157.51355033],
       [150.52220298, 131.35622781, 169.277802  ],
       [169.44072143, 107.3632511 , 200.86611963],
       [131.37129757, 106.97741528, 201.17642023]]).astype(int)

class MedicalPlayer(gym.Env):
    """Class that provides 3D medical image environment.
    This is just an implementation of the classic "agent-environment loop".
    Each time-step, the agent chooses an action, and the environment returns
    an observation and a reward."""

    def __init__(
        self,
        directory=None,
        viz=False,
        task=False,
        files_list=None,
        screen_dims=(27, 27, 27),
        history_length=20,
        multiscale=True,
        max_num_frames=0,
        saveGif=False,
        saveVideo=False,
        agents=2,
        fiducials=None,
        infDir="../inference",
    ):
        """
        :param train_directory: environment or game name
        :param viz: visualization
            set to 0 to disable
            set to +ve number to be the delay between frames to show
            set to a string to be the directory for storing frames
        :param screen_dims: shape of the frame cropped from the image to feed
            it to dqn (d,w,h) - defaults (27,27,27)
        :param nullop_start: start with random number of null ops
        :param location_history_length: consider lost of lives as end of
            episode (useful for training)
        :max_num_frames: maximum numbe0r of frames per episode.
        """
        # self.csvfile = 'DQN_fetal_US_agents_2_400k_RC_LC_CRP.csv'
        #
        # # if os.path.exists(self.csvfile): sys.exit('csv file exists')
        #
        # if task!='train':
        #     with open(self.csvfile, 'w') as outcsv:
        #         fields = ["filename", "dist_error"]
        #         writer = csv.writer(outcsv)
        #         writer.writerow(map(lambda x: x, fields))
        #
        # x = [0.5, 0.25, 0.75]
        # y = [0.5, 0.25, 0.75]
        # z = [0.5, 0.25, 0.75]
        # self.start_points = []
        # for combination in itertools.product(x, y, z):
        #     if 0.5 in combination: self.start_points.append(combination)
        # self.start_points = itertools.cycle(self.start_points)
        # self.count_points = 0
        # self.total_loc = []
        ######################################################################

        super(MedicalPlayer, self).__init__()
        # number of agents
        self.agents = agents
        self.fiducials = fiducials
        # inits stat counters
        self.reset_stat()
        # counter to limit number of steps per episodes
        self.cnt = 0
        # maximum number of frames (steps) per episodes
        self.max_num_frames = max_num_frames
        # stores information: terminal, score, distError
        self.info = None
        # option to save display as gif
        self.saveGif = saveGif
        self.saveVideo = saveVideo
        # training flag
        self.task = task
        # image dimension (2D/3D)
        self.screen_dims = screen_dims
        self.dims = len(self.screen_dims)
        # multi-scale agent
        self.multiscale = multiscale

        # init env dimensions
        if self.dims == 2:
            self.width, self.height = screen_dims
        else:
            self.width, self.height, self.depth = screen_dims

        with _ALE_LOCK:
            self.rng = get_rng(self)
            # visualization setup
            if isinstance(viz, six.string_types):  # check if viz is a string
                assert os.path.isdir(viz), viz
                viz = 0
            if isinstance(viz, int):
                viz = float(viz)
            self.viz = viz
            if self.viz and isinstance(self.viz, float):
                self.viewer = None
                self.gif_buffer = []

        # get action space and minimal action set
        self.action_space = spaces.Discrete(6)  # change number actions here
        self.actions = self.action_space.n
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.screen_dims, dtype=np.uint8
        )
        # history buffer for storing last locations to check oscilations
        self._history_length = history_length
        self._loc_history = []
        self._qvalues_history = []
        # stat counter to store current score or accumlated reward
        self.current_episode_score = []
        self.rectangle = []
        for i in range(0, self.agents):
            self.current_episode_score.append(StatCounter())
            self._loc_history.append([(0,) * self.dims] * self._history_length)
            self._qvalues_history.append([(0,) * self.actions] * self._history_length)
            self.rectangle.append(
                Rectangle(0, 0, 0, 0, 0, 0)
            )  # initialize rectangle limits from input image coordinates

        # add your data loader here
        if self.task == "play":
            self.files = filesListBrainMRLandmark(
                files_list,
                returnLandmarks=False,
                eval=True,
                fiducials=fiducials,
                infDir=infDir, agents=self.agents,
            )
        else:
            if self.task == "eval":
                self.files = filesListBrainMRLandmark(
                    files_list,
                    returnLandmarks=True,
                    fiducials=fiducials,
                    eval=True,
                    infDir=infDir, agents=self.agents,
                )
            else:
                self.files = filesListBrainMRLandmark(
                    files_list,
                    returnLandmarks=True,
                    fiducials=fiducials,
                    eval=False,
                    infDir=infDir,agents=self.agents,
                )

        # prepare file sampler
        self.filepath = None
        self._image = None
        self._target_loc = None
        self.spacing = None
        self.sampled_files = self.files.sample_circular()
        # reset buffer, terminal, counters, and init new_random_game
        self._restart_episode()

    def reset(self):
        # with _ALE_LOCK:
        self._restart_episode()
        return self._current_state()

    def _restart_episode(self):
        """
        restart current episoide
        """
        self.terminal = [False] * self.agents
        self.reward = np.zeros((self.agents,))
        self.cnt = 0  # counter to limit number of steps per episodes
        self.num_games.feed(1)
        self._loc_history = []
        self._qvalues_history = []
        for i in range(0, self.agents):
            self.current_episode_score[i].reset()

            self._loc_history.append([(0,) * self.dims] * self._history_length)
            # list of q-value lists
            self._qvalues_history.append([(0,) * self.actions] * self._history_length)

        self.new_random_game()

    def new_random_game(self):
        """
        load image,
        set dimensions,
        randomize start point,
        init _screen, qvals,
        calc distance to goal
        """
        self.terminal = [False] * self.agents

        self.viewer = None

        #
        # if self.task!='train':
        #     #######################################################################
        #     ## generate results for yuwanwei landmark miccai2018 paper
        #     ## save results in csv file
        #     if self.count_points == 0:
        #         print('\n============== new game ===============\n')
        #         # save results
        #         if self.total_loc:
        #             with open(self.csvfile, 'a') as outcsv:
        #                 fields = [self.filename, self.cur_dist]
        #                 writer = csv.writer(outcsv)
        #                 writer.writerow(map(lambda x: x, fields))
        #             self.total_loc = []
        #         # sample a new image
        #         self._image, self._target_loc, self.filepath, self.spacing = next(self.sampled_files)
        #         scale = next(self.start_points)
        #         self.count_points += 1
        #     else:
        #         self.count_points += 1
        #         logger.info('count_points {}'.format(self.count_points))
        #         scale = next(self.start_points)
        #
        #     x_temp = int(scale[0] * self._image[0].dims[0])
        #     y_temp = int(scale[1] * self._image[0].dims[1])
        #     z_temp = int(scale[2] * self._image[0].dims[2])
        #     logger.info('starting point {}-{}-{}'.format(x_temp, y_temp, z_temp))
        #     #######################################################################
        # else:
        self._image, self._target_loc, self.filepath, self.spacing = next(
            self.sampled_files
        )
        # multiscale (e.g. start with 3 -> 2 -> 1)
        # scale can be thought of as sampling stride
        if self.multiscale:
            ## brain
            self.action_step = 9
            self.xscale = 3
            self.yscale = 3
            self.zscale = 3
            ## cardiac
            # self.action_step =   6
            # self.xscale = 2
            # self.yscale = 2
            # self.zscale = 2
        else:
            self.action_step = 1
            self.xscale = 1
            self.yscale = 1
            self.zscale = 1
        # image volume size
        self._image_dims = self._image[0].dims

        #######################################################################
        # ## select random starting point
        # # add padding to avoid start right on the border of the image
        # if self.task == "train":
        #     skip_thickness = (
        #         (int)(self._image_dims[0] / 5),
        #         (int)(self._image_dims[1] / 5),
        #         (int)(self._image_dims[2] / 5),
        #     )
        # else:
        #     skip_thickness = (
        #         int(self._image_dims[0] / 4),
        #         int(self._image_dims[1] / 4),
        #         int(self._image_dims[2] / 4),
        #     )
        #
        # # if self.task == 'train':
        # x = []
        # y = []
        # z = []
        # for i in range(0, self.agents):
        #     x.append(
        #         self.rng.randint(
        #             0 + skip_thickness[0], self._image_dims[0] - skip_thickness[0]
        #         )
        #     )
        #     y.append(
        #         self.rng.randint(
        #             0 + skip_thickness[1], self._image_dims[1] - skip_thickness[1]
        #         )
        #     )
        #     z.append(
        #         self.rng.randint(
        #             0 + skip_thickness[2], self._image_dims[2] - skip_thickness[2]
        #         )
        #     )
        # # else:
        # #     x=[]
        # #     y=[]
        # #     z=[]
        # #     for i in range(0,self.agents):
        # #         x.append(x_temp)
        # #         y.append(y_temp)
        # #         z.append(z_temp)
        #
        #######################################################################

        self._location = []
        self._start_location = []
        for i in self.fiducials:
            self._location.append(tuple(meanFiducialLocations[i]))
            self._start_location.append(tuple(meanFiducialLocations[i]))
        self._qvalues = [[0] * self.actions] * self.agents
        self._screen = self._current_state()

        if self.task == "play":
            self.cur_dist = [0] * self.agents
        else:
            self.cur_dist = []
            for i in range(0, self.agents):
                self.cur_dist.append(
                    self.calcDistance(
                        self._location[i], self._target_loc[i], self.spacing
                    )
                )

    def calcDistance(self, points1, points2, spacing=(1, 1, 1)):
        """ calculate the distance between two points in mm"""
        spacing = np.array(spacing)
        points1 = spacing * np.array(points1)
        points2 = spacing * np.array(points2)
        return np.linalg.norm(points1 - points2)

    def step(self, act, q_values, isOver):
        """The environment's step function returns exactly what we need.
        Args:
          act:
        Returns:
          observation (object):
            an environment-specific object representing your observation of
            the environment. For example, pixel data from a camera, joint angles
            and joint velocities of a robot, or the board state in a board game.
          reward (float):
            amount of reward achieved by the previous action. The scale varies
            between environments, but the goal is always to increase your total
            reward.
          done (boolean):
            whether it's time to reset the environment again. Most (but not all)
            tasks are divided up into well-defined episodes, and done being True
            indicates the episode has terminated. (For example, perhaps the pole
            tipped too far, or you lost your last life.)
          info (dict):
            diagnostic information useful for debugging. It can sometimes be
            useful for learning (for example, it might contain the raw
            probabilities behind the environment's last state change). However,
            official evaluations of your agent are not allowed to use this for
            learning.
        """
        for i in range(0, self.agents):
            if isOver[i]:
                act[i] = 10
        self._qvalues = q_values
        current_loc = self._location
        next_location = copy.deepcopy(current_loc)

        self.terminal = [False] * self.agents
        go_out = [False] * self.agents
        ###################### agent 1 movement #####################################
        for i in range(0, self.agents):
            # UP Z+ -----------------------------------------------------------
            if act[i] == 0:
                next_location[i] = (
                    current_loc[i][0],
                    current_loc[i][1],
                    round(current_loc[i][2] + self.action_step),
                )
                if next_location[i][2] >= self._image_dims[2]:
                    # print(' trying to go out the image Z+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True

            # FORWARD Y+ ---------------------------------------------------------
            if act[i] == 1:
                next_location[i] = (
                    current_loc[i][0],
                    round(current_loc[i][1] + self.action_step),
                    current_loc[i][2],
                )
                if next_location[i][1] >= self._image_dims[1]:
                    # print(' trying to go out the image Y+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # RIGHT X+ -----------------------------------------------------------
            if act[i] == 2:
                next_location[i] = (
                    round(current_loc[i][0] + self.action_step),
                    current_loc[i][1],
                    current_loc[i][2],
                )
                if next_location[i][0] >= self._image_dims[0]:
                    # print(' trying to go out the image X+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # LEFT X- -----------------------------------------------------------
            if act[i] == 3:
                next_location[i] = (
                    round(current_loc[i][0] - self.action_step),
                    current_loc[i][1],
                    current_loc[i][2],
                )
                if next_location[i][0] <= 0:
                    # print(' trying to go out the image X- ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # BACKWARD Y- ---------------------------------------------------------
            if act[i] == 4:
                next_location[i] = (
                    current_loc[i][0],
                    round(current_loc[i][1] - self.action_step),
                    current_loc[i][2],
                )
                if next_location[i][1] <= 0:
                    # print(' trying to go out the image Y- ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # DOWN Z- -----------------------------------------------------------
            if act[i] == 5:
                next_location[i] = (
                    current_loc[i][0],
                    current_loc[i][1],
                    round(current_loc[i][2] - self.action_step),
                )
                if next_location[i][2] <= 0:
                    # print(' trying to go out the image Z- ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # ---------------------------------------------------------------------
        #############################################################################

        # ---------------------------------------------------------------------
        # punish -1 reward if the agent tries to go out
        if self.task != "play":
            for i in range(0, self.agents):
                if go_out[i]:
                    self.reward[i] = -1
                else:
                    self.reward[i] = self._calc_reward(
                        current_loc[i], next_location[i], agent=i
                    )

        # update screen, reward ,location, terminal
        self._location = next_location
        self._screen = self._current_state()

        # terminate if the distance is less than 1 during trainig
        if self.task == "train":
            for i in range(0, self.agents):
                if self.cur_dist[i] <= 1:
                    self.terminal[i] = True
                    self.num_success[i].feed(1)

        # terminate if maximum number of steps is reached
        self.cnt += 1
        if self.cnt >= self.max_num_frames:
            for i in range(0, self.agents):
                self.terminal[i] = True

        # update history buffer with new location and qvalues
        if self.task != "play":
            for i in range(0, self.agents):
                self.cur_dist[i] = self.calcDistance(
                    self._location[i], self._target_loc[i], self.spacing
                )

        self._update_history()

        # check if agent oscillates
        if self._oscillate:
            self._location = self.getBestLocation()
            # self._location=[item for sublist in temp for item in sublist]
            self._screen = self._current_state()

            if self.task != "play":
                for i in range(0, self.agents):
                    self.cur_dist[i] = self.calcDistance(
                        self._location[i], self._target_loc[i], self.spacing
                    )

            # multi-scale steps
            if self.multiscale:
                if self.xscale > 1:
                    self.xscale -= 1
                    self.yscale -= 1
                    self.zscale -= 1
                    self.action_step = int(self.action_step / 3)
                    self._clear_history()
                # terminate if scale is less than 1
                else:

                    for i in range(0, self.agents):
                        self.terminal[i] = True
                        if self.cur_dist[i] <= 1:
                            self.num_success[i].feed(1)

            else:

                for i in range(0, self.agents):
                    self.terminal[i] = True
                    if self.cur_dist[i] <= 1:
                        self.num_success[i].feed(1)
        # render screen if viz is on
        with _ALE_LOCK:
            if self.viz:
                if isinstance(self.viz, float):
                    self.display()

        distance_error = self.cur_dist
        for i in range(0, self.agents):
            self.current_episode_score[i].feed(self.reward[i])

        info = {}
        for i in range(0, self.agents):
            info["score_{}".format(i)] = self.current_episode_score[i].sum
            info["gameOver_{}".format(i)] = self.terminal[i]
            info["distError_{}".format(i)] = distance_error[i]
            info["filename_{}".format(i)] = self.filepath[i]
            info["location_{}".format(i)] = self._location[i]

            #######################################################################
            ## generate results for yuwanwei landmark miccai2018 paper

        # if all(self.terminal):
        #     logger.info(info)
        #     self.total_loc.append(self._location)
        #     if not (self.count_points == 5):
        #         self._restart_episode()
        #     else:
        #         mean_location = np.mean(self.total_loc, axis=0)
        #         for i in range(0,self.agents):
        #             logger.info('agent {}  \n mean_location{}'.format(i, mean_location[i]))
        #             if self.task != 'play':
        #                 self.cur_dist[i] = self.calcDistance(mean_location[i],
        #                                               self._target_loc[i],
        #                                               self.spacing[i])
        #                 logger.info('agent {} , final distance error {} \n'.format(i,self.cur_dist[i]))
        #         self.count_points = 0
        #######################################################################

        return self._current_state(), self.reward, self.terminal, info

    def getBestLocation(self):
        """ get best location with best qvalue from last for locations
        stored in history
        """
        best_location = []
        for i in range(0, self.agents):
            last_qvalues_history = self._qvalues_history[i][-4:]
            last_loc_history = self._loc_history[i][-4:]
            best_qvalues = np.max(last_qvalues_history, axis=1)
            best_idx = best_qvalues.argmax()
            best_location.append(last_loc_history[best_idx])
        #
        # last_qvalues_history=[]
        # last_loc_history=[]
        # best_qvalues=[]
        # best_idx=[]
        #
        # for i in range(0,self.agents):
        #     last_qvalues_history.append(self._qvalues_history[i][-4:])
        #     last_loc_history.append( self._loc_history[i][-4:])
        #     best_qvalues.append(np.max(last_qvalues_history[i], axis=1))
        #     best_idx.append(best_qvalues[i].argmin())
        #     best_location.append(last_loc_history[best_idx[i]])

        return best_location

    def _clear_history(self):
        """ clear history buffer with current state
        """
        self._loc_history = []
        self._qvalues_history = []
        for i in range(0, self.agents):
            self._loc_history.append([(0,) * self.dims] * self._history_length)
            self._qvalues_history.append([(0,) * self.actions] * self._history_length)

    def _update_history(self):
        """ update history buffer with current state
        """
        # update location history
        for i in range(0, self.agents):
            self._loc_history[i][:-1] = self._loc_history[i][1:]
            self._loc_history[i][-1] = self._location[i]

            # update q-value history
            self._qvalues_history[i][:-1] = self._qvalues_history[i][1:]
            self._qvalues_history[i][-1] = np.ravel(self._qvalues[i])

    def _current_state(self):
        """
        crop image data around current location to update what network sees.
        update rectangle

        :return: new state
        """
        # initialize screen with zeros - all background

        screen = np.zeros(
            (self.agents, self.screen_dims[0], self.screen_dims[1], self.screen_dims[2])
        ).astype(self._image[0].data.dtype)

        for i in range(0, self.agents):
            # screen uses coordinate system relative to origin (0, 0, 0)
            screen_xmin, screen_ymin, screen_zmin = 0, 0, 0
            screen_xmax, screen_ymax, screen_zmax = self.screen_dims

            # extract boundary locations using coordinate system relative to "global" image
            # width, height, depth in terms of screen coord system

            if self.xscale % 2:
                xmin = self._location[i][0] - int(self.width * self.xscale / 2) - 1
                xmax = self._location[i][0] + int(self.width * self.xscale / 2)
                ymin = self._location[i][1] - int(self.height * self.yscale / 2) - 1
                ymax = self._location[i][1] + int(self.height * self.yscale / 2)
                zmin = self._location[i][2] - int(self.depth * self.zscale / 2) - 1
                zmax = self._location[i][2] + int(self.depth * self.zscale / 2)
            else:
                xmin = self._location[i][0] - round(self.width * self.xscale / 2)
                xmax = self._location[i][0] + round(self.width * self.xscale / 2)
                ymin = self._location[i][1] - round(self.height * self.yscale / 2)
                ymax = self._location[i][1] + round(self.height * self.yscale / 2)
                zmin = self._location[i][2] - round(self.depth * self.zscale / 2)
                zmax = self._location[i][2] + round(self.depth * self.zscale / 2)

            ###########################################################

            # check if they violate image boundary and fix it
            if xmin < 0:
                xmin = 0
                screen_xmin = screen_xmax - len(np.arange(xmin, xmax, self.xscale))
            if ymin < 0:
                ymin = 0
                screen_ymin = screen_ymax - len(np.arange(ymin, ymax, self.yscale))
            if zmin < 0:
                zmin = 0
                screen_zmin = screen_zmax - len(np.arange(zmin, zmax, self.zscale))
            if xmax > self._image_dims[0]:
                xmax = self._image_dims[0]
                screen_xmax = screen_xmin + len(np.arange(xmin, xmax, self.xscale))
            if ymax > self._image_dims[1]:
                ymax = self._image_dims[1]
                screen_ymax = screen_ymin + len(np.arange(ymin, ymax, self.yscale))
            if zmax > self._image_dims[2]:
                zmax = self._image_dims[2]
                screen_zmax = screen_zmin + len(np.arange(zmin, zmax, self.zscale))

            # crop image data to update what network sees
            # image coordinate system becomes screen coordinates
            # scale can be thought of as a stride
            screen[
                i,
                screen_xmin:screen_xmax,
                screen_ymin:screen_ymax,
                screen_zmin:screen_zmax,
            ] = self._image[i].data[
                xmin : xmax : self.xscale,
                ymin : ymax : self.yscale,
                zmin : zmax : self.zscale,
            ]

            ###########################################################
            # update rectangle limits from input image coordinates
            # this is what the network sees
            self.rectangle[i] = Rectangle(xmin, xmax, ymin, ymax, zmin, zmax)

        return screen

    def get_plane(self, z=0, agent=0):
        return self._image[agent].data[:, :, z]

    def _calc_reward(self, current_loc, next_loc, agent):
        """ Calculate the new reward based on the decrease in euclidean distance to the target location
        """

        curr_dist = self.calcDistance(
            current_loc, self._target_loc[agent], self.spacing
        )
        next_dist = self.calcDistance(
            next_loc, self._target_loc[agent], self.spacing
        )
        dist = curr_dist - next_dist

        return dist

    @property
    def _oscillate(self):
        """ Return True if the agent is stuck and oscillating
        """
        counter = []
        freq = []
        for i in range(0, self.agents):
            counter.append(Counter(self._loc_history[i]))
            freq.append(counter[i].most_common())

            if freq[i][0][0] == (0, 0, 0):
                if freq[i][1][1] > 3:
                    return True
                else:
                    return False
            elif freq[i][0][1] > 3:
                return True

    def get_action_meanings(self):
        """ return array of integers for actions"""
        ACTION_MEANING = {
            1: "UP",  # MOVE Z+
            2: "FORWARD",  # MOVE Y+
            3: "RIGHT",  # MOVE X+
            4: "LEFT",  # MOVE X-
            5: "BACKWARD",  # MOVE Y-
            6: "DOWN",  # MOVE Z-
        }
        return [ACTION_MEANING[i] for i in self.actions]

    @property
    def getScreenDims(self):
        """
        return screen dimensions
        """
        return (self.width, self.height, self.depth)

    def lives(self):
        return None

    def reset_stat(self):
        """ Reset all statistics counter"""
        self.stats = defaultdict(list)
        self.num_games = StatCounter()
        self.num_success = [StatCounter()] * int(self.agents)

    def display(self, return_rgb_array=False):
        # pass
        for i in range(0, self.agents):
            # get dimensions
            current_point = self._location[i]
            target_point = None
            if self.task != "play":
                target_point = self._target_loc[i]
            # print("_location", self._location)
            # print("_target_loc", self._target_loc)
            # print("current_point", current_point)
            # print("target_point", target_point)
            # get image and convert it to pyglet
            plane = self.get_plane(current_point[2], agent=i)  # z-plane
            # plane = np.squeeze(self._current_state()[:,:,13])
            img = cv2.cvtColor(plane, cv2.COLOR_GRAY2RGB)  # congvert to rgb
            # rescale image
            # INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
            scale_x = 2
            scale_y = 2
            #
            img = cv2.resize(
                img,
                (int(scale_x * img.shape[1]), int(scale_y * img.shape[0])),
                interpolation=cv2.INTER_LINEAR,
            )
            # skip if there is a viewer open
            if (not self.viewer) and self.viz:
                from viewer import SimpleImageViewer

                self.viewer = SimpleImageViewer(
                    arr=img, scale_x=1, scale_y=1, filepath=self.filepath[i] + str(i)
                )
                self.gif_buffer = []
            # display image
            self.viewer.draw_image(img)
            # draw current point
            self.viewer.draw_circle(
                radius=scale_x * 1,
                pos_x=scale_x * current_point[0],
                pos_y=scale_y * current_point[1],
                color=(0.0, 0.0, 1.0, 1.0),
            )
            # draw a box around the agent - what the network sees ROI
            self.viewer.draw_rect(
                scale_x * self.rectangle[i].xmin,
                scale_y * self.rectangle[i].ymin,
                scale_x * self.rectangle[i].xmax,
                scale_y * self.rectangle[i].ymax,
            )
            self.viewer.display_text(
                "Agent " + str(i),
                color=(204, 204, 0, 255),
                x=scale_x * self.rectangle[i].xmin - 15,
                y=scale_y * self.rectangle[i].ymin,
            )
            # display info
            text = "Spacing " + str(self.xscale)
            self.viewer.display_text(
                text, color=(204, 204, 0, 255), x=10, y=self._image_dims[1] - 80
            )

            # ---------------------------------------------------------------------

            if self.task != "play":
                # draw a transparent circle around target point with variable radius
                # based on the difference z-direction
                diff_z = scale_x * abs(current_point[2] - target_point[2])
                self.viewer.draw_circle(
                    radius=diff_z,
                    pos_x=scale_x * target_point[0],
                    pos_y=scale_y * target_point[1],
                    color=(1.0, 0.0, 0.0, 0.2),
                )
                # draw target point
                self.viewer.draw_circle(
                    radius=scale_x * 1,
                    pos_x=scale_x * target_point[0],
                    pos_y=scale_y * target_point[1],
                    color=(1.0, 0.0, 0.0, 1.0),
                )
                # display info
                color = (0, 204, 0, 255) if self.reward[i] > 0 else (204, 0, 0, 255)
                text = "Error " + str(round(self.cur_dist[i], 3)) + "mm"
                self.viewer.display_text(text, color=color, x=10, y=20)

            # ---------------------------------------------------------------------

            # render and wait (viz) time between frames
            self.viewer.render()
            # time.sleep(self.viz)
            # save gif
            if self.saveGif:
                image_data = (
                    pyglet.image.get_buffer_manager()
                    .get_color_buffer()
                    .get_image_data()
                )
                data = image_data.get_data("RGB", image_data.width * 3)
                arr = np.array(bytearray(data)).astype("uint8")
                arr = np.flip(
                    np.reshape(arr, (image_data.height, image_data.width, -1)), 0
                )
                im = Image.fromarray(arr)
                self.gif_buffer.append(im)

                if not self.terminal[i]:
                    gifname = self.filepath[0] + ".gif"
                    self.viewer.saveGif(gifname, arr=self.gif_buffer, duration=self.viz)
            if self.saveVideo:
                dirname = "tmp_video"
                if self.cnt <= 1:
                    if os.path.isdir(dirname):
                        logger.warn(
                            """Log directory {} exists! Use 'd' to delete it. """.format(
                                dirname
                            )
                        )
                        act = (
                            input("select action: d (delete) / q (quit): ")
                            .lower()
                            .strip()
                        )
                        if act == "d":
                            shutil.rmtree(dirname, ignore_errors=True)
                        else:
                            raise OSError("Directory {} exits!".format(dirname))
                    os.mkdir(dirname)

                frame = dirname + "/" + "%04d" % self.cnt + ".png"
                pyglet.image.get_buffer_manager().get_color_buffer().save(frame)
                if self.terminal[i]:
                    resolution = (
                        str(3 * self.viewer.img_width)
                        + "x"
                        + str(3 * self.viewer.img_height)
                    )
                    save_cmd = [
                        "ffmpeg",
                        "-f",
                        "image2",
                        "-framerate",
                        "30",
                        "-pattern_type",
                        "sequence",
                        "-start_number",
                        "0",
                        "-r",
                        "6",
                        "-i",
                        dirname + "/%04d.png",
                        "-s",
                        resolution,
                        "-vcodec",
                        "libx264",
                        "-b:v",
                        "2567k",
                        self.filepath[i] + ".mp4",
                    ]
                    subprocess.check_output(save_cmd)
                    shutil.rmtree(dirname, ignore_errors=True)


# =============================================================================
# ================================ FrameStack =================================
# =============================================================================
class FrameStack(gym.Wrapper):
    """used when not training. wrapper for Medical Env"""

    def __init__(self, env, k, agents=2):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.agents = agents
        self.k = k  # history length
        # self.frames=[]
        # for i in range(0,self.agents):
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self._base_dim = len(shp)
        new_shape = shp + (k,)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        ob = tuple(ob)
        # for i in range(0, self.agents):
        for _ in range(self.k - 1):
            self.frames.append(np.zeros_like(ob))
        self.frames.append(ob)
        return self._observation()

    def step(self, act, q_values, isOver):
        for i in range(0, self.agents):
            if isOver[i]:
                act[i] = 15
        current_st, reward, terminal, info = self.env.step(act, q_values, isOver)
        # for i in range(0,self.agents):
        current_st = tuple(current_st)
        self.frames.append(current_st)
        return self._observation(), reward, terminal, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=-1)


# =============================================================================
# ================================== notes ====================================
# =============================================================================
"""

## Notes from landmark detection Siemens paper
# states  -> ROI - center current pos - size (2D 60x60) (3D 26x26x26)
# actions -> move (up, down, left, right)
# rewards -> delta(d) relative distance change after executing a move (action)

# re-sample -> isotropic (2D 2mm) (3D 1mm)

# gamma = 0.9 , replay memory size P = 100000 , learning rate = 0.00025
# net : 3 conv+pool - 3 FC+dropout (3D kernels for 3d data)

# navigate till oscillation happen (terminate when infinite loop)

# location is a high-confidence landmark -> if the expected reward from this location is max(q*(s_target,a))<1 the agent is closer than one pixel

# object is not in the image: oscillation occurs at points where max(q)>4


## Other Notes:

    DeepMind's original DQN paper
        used frame skipping (for fast playing/learning) and
        applied pixel-wise max to consecutive frames (to handle flickering).

    so an input to the neural network is consisted of four frame;
        [max(T-1, T), max(T+3, T+4), max(T+7, T+8), max(T+11, T+12)]

    ALE provides mechanism for frame skipping (combined with adjustable random action repeat) and color averaging over skipped frames. This is also used in simple_dqn's ALEEnvironment

    Gym's Atari Environment has built-in stochastic frame skipping common to all games. So the frames returned from environment are not consecutive.

    The reason behind Gym's stochastic frame skipping is, as mentioned above, to make environment stochastic. (I guess without this, the game will be completely deterministic?)
    cf. in original DQN and simple_dqn same randomness is achieved by having agent performs random number of dummy actions at the beginning of each episode.

    I think if you want to reproduce the behavior of the original DQN paper, the easiest will be disabling frame skip and color averaging in ALEEnvironment then construct the mechanism on agent side.


"""
