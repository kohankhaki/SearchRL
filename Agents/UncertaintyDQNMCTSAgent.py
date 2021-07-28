from ssl import RAND_pseudo_bytes
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from abc import abstractmethod
import random

import Utils as utils
import Config as config
from DataStructures.Node_Torch import Node_Torch as Node
from Agents.UncertaintyDQNAgent import UncertaintyDQNAgent
from Agents.MCTSAgent_Torch import MCTSAgent_Torch as MCTSAgent
from Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN, StateActionVFNN_het, rnd_network
from Networks.ValueFunctionNN.StateValueFunction import StateVFNN
from Networks.RepresentationNN.StateRepresentation import StateRepresentation
import pickle


episodes_only_dqn = config.episodes_only_dqn
episodes_only_mcts = config.episodes_only_mcts

class UncertaintyDQNMCTSAgent_InitialValue(UncertaintyDQNAgent, MCTSAgent):
    name = "UncertaintyDQNMCTSAgent_InitialValue"

    def __init__(self, params={}):
        UncertaintyDQNAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)

        self.episode_counter = -1
    
    def start(self, observation, info=None):
        self.episode_counter += 1
        if self.episode_counter < episodes_only_dqn:
            action = UncertaintyDQNAgent.start(self, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.start(self, observation)
        else:
            if self.episode_counter % 2 == 0:
                action = UncertaintyDQNAgent.start(self, observation)
            else:
                action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter < episodes_only_dqn:
            action = UncertaintyDQNAgent.step(self, reward, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.step(self, reward, observation)
        else:
            if self.episode_counter % 2 == 0:
                action = UncertaintyDQNAgent.step(self, reward, observation)
            else:
                action = MCTSAgent.step(self, reward, observation)
        return action

    def end(self, reward):
        if self.episode_counter < episodes_only_dqn:
            UncertaintyDQNAgent.end(self, reward)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            MCTSAgent.end(self, reward)
        else:
            if self.episode_counter % 2 == 0:
                UncertaintyDQNAgent.end(self, reward)
            else:
                MCTSAgent.end(self, reward)


    def expansion(self, node):
        for a in range(self.num_actions):
            action_index = torch.tensor([a]).unsqueeze(0)
            next_state, is_terminal, reward, _ = self.model(node.get_state(),
                                                              action_index)  # with the assumption of deterministic model
            # if np.array_equal(next_state, node.get_state()):
            #     continue
            value, uncertainty = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                         value=value, uncertainty=uncertainty)
            node.add_child(child)
        child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, node.get_childs())))
        softmax_denominator = np.sum(np.exp(-child_uncertainties))
        for child in node.get_childs():
            child_value = child.get_avg_value() * np.exp(-child.uncertainty) / softmax_denominator
            child.set_initial_value(child_value)

    def get_initial_value(self, state):
        value, uncertainty = self.getStateValueUncertainty(state, 'ensemble')
        return value.item(), uncertainty.item()