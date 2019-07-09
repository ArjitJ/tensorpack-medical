#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Amir Alansary <amiralansary@gmail.com>
# Modified: Arjit Jain <thearjitjain@gmail.com>


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)


import os
import argparse

import tensorflow as tf
from medical import MedicalPlayer, FrameStack
from tensorpack.input_source import QueueInput
from tensorpack_medical.models.conv3d import Conv3D
from common import Evaluator, eval_model_multithread, play_n_episodes
from DQNModel import Model3D as DQNModel
from expreplay import ExpReplay

from tensorpack import (
    PredictConfig,
    OfflinePredictor,
    get_model_loader,
    logger,
    TrainConfig,
    ModelSaver,
    PeriodicTrigger,
    ScheduledHyperParamSetter,
    ObjAttrParam,
    HumanHyperParamSetter,
    argscope,
    RunOp,
    LinearWrap,
    FullyConnected,
    PReLU,
    SimpleTrainer,
    launch_train_with_config,
)


###############################################################################
# BATCH SIZE USED IN NATURE PAPER IS 32 - MEDICAL IS 256
BATCH_SIZE = 48
# BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
# IMAGE_SIZE = (45, 45, 45)
IMAGE_SIZE = (27, 27, 27)
# how many frames to keep
# in other words, how many observations the network can see
FRAME_HISTORY = 4
# the frequency of updating the target network
UPDATE_FREQ = 4
# DISCOUNT FACTOR - NATURE (0.99) - MEDICAL (0.9)
GAMMA = 0.9  # 0.99
# REPLAY MEMORY SIZE - NATURE (1e6) - MEDICAL (1e5 view-patches)
MEMORY_SIZE = 1e5  # 6
# consume at least 1e6 * 27 * 27 * 27 bytes
INIT_MEMORY_SIZE = MEMORY_SIZE // 20  # 5e4
# each epoch is 100k played frames
# STEPS_PER_EPOCH = 10000 // UPDATE_FREQ * 10
# STEPS_PER_EPOCH = 3677
# num training epochs in between model evaluations
EPOCHS_PER_EVAL = 6
# the number of episodes to run during evaluation
EVAL_EPISODE = 2598

###############################################################################


def get_player(
    directory=None,
    files_list=None,
    viz=False,
    task="play",
    saveGif=False,
    saveVideo=False,
    agents=2,
    fiducials=None,
    infDir="../inference",
):
    # in atari paper, max_num_frames = 30000
    env = MedicalPlayer(
        screen_dims=IMAGE_SIZE,
        viz=viz,
        saveGif=saveGif,
        saveVideo=saveVideo,
        task=task,
        files_list=files_list,
        agents=agents,
        max_num_frames=1500,
        fiducials=fiducials,
        infDir=infDir,
    )
    if task != "train":
        # in training, env will be decorated by ExpReplay, and history
        # is taken care of in expreplay buffer
        # otherwise, FrameStack modifies self.step to save observations into a queue
        env = FrameStack(env, FRAME_HISTORY, agents=agents)
    return env


def get_player_fn(*myargs, **kwargs):
    return get_player(fiducials=args.fiducials, infDir=args.inferDir, *myargs, **kwargs)


###############################################################################


class Model(DQNModel):
    def __init__(self, agents=2):
        super(Model, self).__init__(
            IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA, agents
        )

    def _get_DQN_prediction(self, images):
        """ image: [0,255]

        :returns predicted Q values"""
        # normalize image values to [0, 1]

        agents = len(images)

        Q_list = []

        with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True):

            for i in range(0, agents):
                images[i] = images[i] / 255.0
                with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True):

                    if i == 0:
                        conv_0 = tf.layers.conv3d(
                            images[i],
                            name="conv0",
                            filters=32,
                            kernel_size=[5, 5, 5],
                            strides=[1, 1, 1],
                            padding="same",
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                2.0
                            ),
                            bias_initializer=tf.zeros_initializer(),
                        )
                        max_pool_0 = tf.layers.max_pooling3d(
                            conv_0, 2, 2, name="max_pool0"
                        )
                        conv_1 = tf.layers.conv3d(
                            max_pool_0,
                            name="conv1",
                            filters=32,
                            kernel_size=[5, 5, 5],
                            strides=[1, 1, 1],
                            padding="same",
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                2.0
                            ),
                            bias_initializer=tf.zeros_initializer(),
                        )
                        max_pool1 = tf.layers.max_pooling3d(
                            conv_1, 2, 2, name="max_pool1"
                        )
                        conv_2 = tf.layers.conv3d(
                            max_pool1,
                            name="conv2",
                            filters=64,
                            kernel_size=[4, 4, 4],
                            strides=[1, 1, 1],
                            padding="same",
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                2.0
                            ),
                            bias_initializer=tf.zeros_initializer(),
                        )
                        max_pool2 = tf.layers.max_pooling3d(
                            conv_2, 2, 2, name="max_pool2"
                        )
                        conv3 = tf.layers.conv3d(
                            max_pool2,
                            name="conv3",
                            filters=64,
                            kernel_size=[3, 3, 3],
                            strides=[1, 1, 1],
                            padding="same",
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                2.0
                            ),
                            bias_initializer=tf.zeros_initializer(),
                        )
                    else:
                        conv_0 = tf.layers.conv3d(
                            images[i],
                            name="conv0",
                            reuse=True,
                            filters=32,
                            kernel_size=[5, 5, 5],
                            strides=[1, 1, 1],
                            padding="same",
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                2.0
                            ),
                            bias_initializer=tf.zeros_initializer(),
                        )
                        max_pool_0 = tf.layers.max_pooling3d(
                            conv_0, 2, 2, name="max_pool0"
                        )
                        conv_1 = tf.layers.conv3d(
                            max_pool_0,
                            name="conv1",
                            reuse=True,
                            filters=32,
                            kernel_size=[5, 5, 5],
                            strides=[1, 1, 1],
                            padding="same",
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                2.0
                            ),
                            bias_initializer=tf.zeros_initializer(),
                        )
                        max_pool1 = tf.layers.max_pooling3d(
                            conv_1, 2, 2, name="max_pool1"
                        )
                        conv_2 = tf.layers.conv3d(
                            max_pool1,
                            name="conv2",
                            reuse=True,
                            filters=64,
                            kernel_size=[4, 4, 4],
                            strides=[1, 1, 1],
                            padding="same",
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                2.0
                            ),
                            bias_initializer=tf.zeros_initializer(),
                        )
                        max_pool2 = tf.layers.max_pooling3d(
                            conv_2, 2, 2, name="max_pool2"
                        )
                        conv3 = tf.layers.conv3d(
                            max_pool2,
                            name="conv3",
                            reuse=True,
                            filters=64,
                            kernel_size=[3, 3, 3],
                            strides=[1, 1, 1],
                            padding="same",
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                2.0
                            ),
                            bias_initializer=tf.zeros_initializer(),
                        )

                ### now for the dense layers##
                if "Dueling" not in self.method:
                    fc0 = FullyConnected("fc0_{}".format(i), conv3, 512, nl=tf.nn.relu)
                    fc1 = FullyConnected("fc1_{}".format(i), fc0, 256, nl=tf.nn.relu)
                    fc2 = FullyConnected("fc2_{}".format(i), fc1, 128, nl=tf.nn.relu)
                    Q = FullyConnected(
                        "fct_{}".format(i), fc2, self.num_actions, nl=tf.identity
                    )
                    Q_list.append(tf.identity(Q, name="Qvalue_{}".format(i)))

                else:
                    fc0 = FullyConnected("fc0V_{}".format(i), conv3, 512, nl=tf.nn.relu)
                    fc1 = FullyConnected("fc1V_{}".format(i), fc0, 256, nl=tf.nn.relu)
                    fc2 = FullyConnected("fc2V_{}".format(i), fc1, 128, nl=tf.nn.relu)
                    V = FullyConnected("fctV_{}".format(i), fc2, 1, nl=tf.identity)

                    fcA0 = FullyConnected(
                        "fc0A_{}".format(i), conv3, 512, nl=tf.nn.relu
                    )
                    fcA1 = FullyConnected("fc1A_{}".format(i), fcA0, 256, nl=tf.nn.relu)
                    fcA2 = FullyConnected("fc2A_{}".format(i), fcA1, 128, nl=tf.nn.relu)
                    A = FullyConnected(
                        "fctA_{}".format(i), fcA2, self.num_actions, nl=tf.identity
                    )

                    Q = tf.add(A, V - tf.reduce_mean(A, 1, keepdims=True))
                    Q_list.append(tf.identity(Q, name="Qvalue_{}".format(i)))

        return Q_list


###############################################################################
def get_initial_value(epoch):
    schedule = [(0, 1), (10, 0.1), (320, 0.01)]
    laste, lastv = None, None
    for e, v in schedule:
        if e == epoch:
            return v
        if e > epoch:
            break
        laste, lastv = e, v
    if laste is None or laste == e:
        return None
    v = (epoch - laste) * 1.0 / (e - laste) * (v - lastv) + lastv
    return v


def get_config(
    files_list,
    input_names=["state_1", "state_2"],
    output_names=["Qvalue_1", "Qvalue_2"],
    agents=2,
    last=0,
    fiducials=None,
    val_files = None
):
    """This is only used during training."""
    expreplay = ExpReplay(
        predictor_io_names=(input_names, output_names),
        player=get_player(
            task="train",
            files_list=files_list,
            agents=agents,
            fiducials=fiducials,
            infDir=args.inferDir,
        ),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=get_initial_value(last),
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY,
        agents=agents,
    )

    return TrainConfig(
        # dataflow=expreplay,
        data=QueueInput(expreplay),
        model=Model(agents=agents),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                # update target network every 10k steps
                every_k_steps=10000 // UPDATE_FREQ,
            ),
            expreplay,
            ScheduledHyperParamSetter("learning_rate", [(60, 4e-4), (100, 2e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, "exploration"),
                # 1->0.1 in the first million steps
                [(0, 1), (10, 0.1), (320, 0.01)],
                interp="linear",
            ),
            PeriodicTrigger(
                Evaluator(
                    nr_eval=EVAL_EPISODE,
                    input_names=input_names,
                    output_names=output_names,
                    files_list=val_files,
                    get_player_fn=get_player_fn,
                    agents=agents,
                ),
                every_k_epochs=EPOCHS_PER_EVAL,
            ),
            HumanHyperParamSetter("learning_rate"),
        ],
        steps_per_epoch=args.step,
        max_epoch=1000,
    )


###############################################################################
fidNumToName = {
    0: "AC",
    1: "BPons",
    2: "CM",
    3: "LE",
    4: "PC",
    5: "RE",
    6: "RP",
    7: "RP_front",
    8: "SMV",
    9: "VN4",
    10: "callosum_left",
    11: "callosum_right",
    12: "dens_axis",
    13: "genu",
    14: "l_caud_head",
    15: "l_corp",
    16: "l_front_pole",
    17: "l_inner_corpus",
    18: "l_lat_ext",
    19: "l_occ_pole",
    20: "l_prim_ext",
    21: "l_sup_ext",
    22: "l_temp_pole",
    23: "lat_left",
    24: "lat_right",
    25: "lat_ven_left",
    26: "lat_ven_right",
    27: "left_cereb",
    28: "left_lateral_inner_ear",
    29: "m_ax_inf",
    30: "m_ax_sup",
    31: "mid_basel",
    32: "mid_lat",
    33: "mid_prim_inf",
    34: "mid_prim_sup",
    35: "mid_sup",
    36: "optic_chiasm",
    37: "r_caud_head",
    38: "r_corp",
    39: "r_front_pole",
    40: "r_inner_corpus",
    41: "r_lat_ext",
    42: "r_occ_pole",
    43: "r_prim_ext",
    44: "r_sup_ext",
    45: "r_temp_pole",
    46: "right_lateral_inner_ear",
    47: "rostrum",
    48: "rostrum_front",
    49: "top_left",
    50: "top_right",
}

fidNametoNum = {v: k for k, v in fidNumToName.items()}

###############################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", help="comma separated list of GPU(s) to use.")
    parser.add_argument("--load", help="load model")
    parser.add_argument(
        "--task",
        help='task to perform. Must load a pretrained model if task is "play" or "eval"',
        choices=["play", "eval", "train"],
        default="train",
    )
    parser.add_argument(
        "--algo",
        help="algorithm",
        choices=["DQN", "Double", "Dueling", "DuelingDouble"],
        default="DQN",
    )
    parser.add_argument(
        "--files",
        type=argparse.FileType("r"),
        nargs="+",
        help="""Filepath to the text file that contains list of images.
                                Each line of this file is a full path to an image scan.
                                For (task == train or eval) there should be two input files ['images', 'landmarks']""",
    )
    parser.add_argument(
        "--saveGif",
        help="save gif image of the game",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--saveVideo", help="save video of the game", action="store_true", default=False
    )
    parser.add_argument(
        "--logDir",
        help="store logs in this directory during training",
        default="train_log",
    )
    parser.add_argument(
        "--name", help="name of current experiment for logs", default="dev"
    )
    parser.add_argument(
        "--lastEpoch",
        type=int,
        help="if loading a model, specify the last epoch you trained it on",
        default=0,
    )
    parser.add_argument(
        "--agents", help="Number of agents to train together", default=2
    )
    parser.add_argument("--fiducials", nargs="+")
    parser.add_argument(
        "--inferDir", help="directory to save the inferences", default="../inference"
    )
    parser.add_argument("--step", help="steps per epoch", default=2500)
    parser.add_argument("--valFiles",
                        type=argparse.FileType("r"),
                        nargs="+",
                        help="""For Validation. Filepath to the text file that contains list of images.
                                Each line of this file is a full path to an image scan.
                                For (task == train or eval) there should be two input files ['images', 'landmarks']""", default=None)


    args = parser.parse_args()
    try:
        args.fiducials = [int(i) for i in args.fiducials]
    except:
        args.fiducials = [fidNametoNum[i] for i in args.fiducials]
        args.fiducials = [int(i) for i in args.fiducials]
    args.fiducials = sorted(args.fiducials)
    args.agents = len(args.fiducials)
    if args.valFiles is None:
        args.valFiles = args.files
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # check input files
    if args.task == "play":
        error_message = """Wrong input files {} for {} task - should be 1 \'images.txt\' """.format(
            len(args.files), args.task
        )
        assert len(args.files) == 1
    else:
        error_message = """Wrong input files {} for {} task - should be 2 [\'images.txt\', \'landmarks.txt\'] """.format(
            len(args.files), args.task
        )
        assert len(args.files) == 2, error_message

    # args.agents=int(args.agents)

    METHOD = args.algo
    # load files into env to set num_actions, num_validation_files
    init_player = MedicalPlayer(
        files_list=args.files,
        screen_dims=IMAGE_SIZE,
        task=args.task,
        agents=args.agents,
        fiducials=args.fiducials,
        infDir=args.inferDir,
    )
    NUM_ACTIONS = init_player.action_space.n
    num_files = init_player.files.num_files

    ##########################################################
    # initialize states and Qvalues for the various agents
    state_names = []
    qvalue_names = []
    for i in range(0, args.agents):
        state_names.append("state_{}".format(i))
        qvalue_names.append("Qvalue_{}".format(i))

    ############################################################

    if args.task != "train":
        assert args.load is not None
        pred = OfflinePredictor(
            PredictConfig(
                model=Model(agents=args.agents),
                session_init=get_model_loader(args.load),
                input_names=state_names,
                output_names=qvalue_names,
            )
        )
        # demo pretrained model one episode at a time
        if args.task == "play":
            play_n_episodes(
                get_player(
                    files_list=args.files,
                    viz=0,
                    saveGif=args.saveGif,
                    saveVideo=args.saveVideo,
                    task="play",
                    agents=args.agents,
                    fiducials=args.fiducials,
                    infDir=args.inferDir,
                ),
                pred,
                num_files,
                agents=args.agents,
                fidname=[fidNumToName[i] for i in args.fiducials],
                infDir=args.inferDir,
            )
        # run episodes in parallel and evaluate pretrained model
        elif args.task == "eval":
            play_n_episodes(
                get_player(
                    files_list=args.files,
                    viz=0,
                    saveGif=args.saveGif,
                    saveVideo=args.saveVideo,
                    task="eval",
                    agents=args.agents,
                    fiducials=args.fiducials,
                    infDir=args.inferDir,
                ),
                pred,
                num_files,
                agents=args.agents,
                fidname=[fidNumToName[i] for i in args.fiducials],
                infDir=args.inferDir,
            )
    else:  # train model
        logger_dir = os.path.join(args.logDir, args.name)
        logger.set_logger_dir(logger_dir)
        config = get_config(
            args.files,
            input_names=state_names,
            output_names=qvalue_names,
            agents=args.agents,
            last=args.lastEpoch,
            fiducials=args.fiducials,
            val_files=args.valFiles
        )
        if args.load:  # resume training from a saved checkpoint
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
