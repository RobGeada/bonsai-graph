import datetime
import time

import torch.nn as nn
import torch.optim as optim

from bonsai_graph.graphnet import *



import glob
import os

import numpy as np
import scipy.signal
import torch

import bonsai_graph.utils.tensor_utils as utils
from bonsai_graph.manager import CitationGNNManager

logger = utils.get_logger()


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


history = []


def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


class Trainer(object):
    """Manage the training process"""

    def __init__(self, kwargs):
        """
        Constructor for training algorithm.
        Build sub-model manager and controller.
        Build optimizer and cross entropy loss for controller.
        Args:
            args: From command line, picked up by `argparse`.
        """
        self.kwargs = kwargs
        self.cuda = True
        self.epoch = 0
        self.start_epoch = 0
        self.submodel_manager = CitationGNNManager(self.kwargs)

    def form_gnn_info(self, gnn):
        if self.args.search_mode == "micro":
            actual_action = {}
            if self.args.predict_hyper:
                actual_action["action"] = gnn[:-4]
                actual_action["hyper_param"] = gnn[-4:]
            else:
                actual_action["action"] = gnn
                actual_action["hyper_param"] = [0.005, 0.8, 5e-5, 128]
            return actual_action
        return gnn

    def train(self):
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """

        for self.epoch in range(self.start_epoch, self.kwargs['max_epoch']):
            # 1. Training the shared parameters of the child graphnas
            self.train_shared(max_step=self.args.shared_initial_step)
            # 2. Training the controller parameters theta
            self.train_controller()
            # 3. Derive architectures
            self.derive(sample_num=self.args.derive_num_sample)



    def train_shared(self, max_step=50, gnn_list=None):
        """
        Args:
            max_step: Used to run extra training steps as a warm-up.
            gnn: If not None, is used instead of calling sample().
        """
        if max_step == 0:  # no train shared
            return

        print("*" * 35, "training model", "*" * 35)
        gnn_list = gnn_list if gnn_list else self.controller.sample(max_step)

        for gnn in gnn_list:
            gnn = self.form_gnn_info(gnn)
            try:
                _, val_score = self.submodel_manager.train(gnn, format=self.args.format)
                logger.info(f"{gnn}, val_score:{val_score}")
            except RuntimeError as e:
                if 'CUDA' in str(e):  # usually CUDA Out of Memory
                    print(e)
                else:
                    raise e

        print("*" * 35, "training over", "*" * 35)
