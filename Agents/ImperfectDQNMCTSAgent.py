from Agents.RealBaseDynaAgent import RealBaseDynaAgent
import numpy as np
import torch
import random
import gc
import Utils as utils
from torch.utils.tensorboard import SummaryWriter
from ete3 import Tree, TreeStyle, TextFace, add_face_to_node

from Agents.BaseAgent import BaseAgent
from Agents.RealBaseDynaAgent import RealBaseDynaAgent
from Agents.MCTSAgent import MCTSAgent
from DataStructures.Node import Node
from profilehooks import timecall, profile, coverage


class ImperfectMCTSAgent(RealBaseDynaAgent, MCTSAgent):
    name = "ImperfectMCTSAgent"

    def __init__(self, params={}):
        self.model_loss = []
        self.time_step = 0
        # self.writer = SummaryWriter()

        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon']

        self.transition_buffer = []
        self.transition_buffer_size = 2**12


        self.model_type = params['model']['type']
        self._model = {self.model_type:dict(network=None,
                                            layers_type=params['model']['layers_type'],
                                            layers_features=params['model']['layers_features'],
                                            action_layer_num=params['model']['action_layer_num'],
                                            # if one more than layer numbers => we will have num of actions output
                                            batch_size=16,
                                            step_size=params['model_stepsize'],
                                            training=True)}
        self.num_ensembles = 5

        self._sr = dict(network=None,
                        layers_type=[],
                        layers_features=[],
                        batch_size=None,
                        step_size=None,
                        batch_counter=None,
                        training=False)

        self.reward_function = params['reward_function']
        self.device = params['device']
        if params['goal'] is not None:
            self.goal = torch.from_numpy(params['goal']).float().to(self.device)

        self.num_steps = 0
        self.num_terminal_steps = 0

        self.is_pretrained = False

        # MCTS parameters
        self.C = params['c']
        self.num_iterations = params['num_iteration']
        self.num_rollouts = params['num_simulation']
        self.rollout_depth = params['simulation_depth']
        self.keep_subtree = False
        self.keep_tree = False
        self.root = None    

        self.transition_dynamics = params['transition_dynamics']

    def start(self, observation):
        '''
        :param observation: numpy array -> (observation shape)
        :return: action : numpy array
        '''
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)


        self.prev_state = self.getStateRepresentation(observation)
        temp_prev_action = torch.tensor([[0]], device=self.device, dtype=torch.long)
        if self._model[self.model_type]['network'] is None and self._model[self.model_type]['training']:
            self.initModel(self.prev_state, temp_prev_action)


        if self.keep_tree and self.root is None:
            self.root = Node(None, observation)
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
            print(self.subtree_node.get_avg_value())
        else:
            self.subtree_node = Node(None, observation)
            self.expansion(self.subtree_node)

        # self.render_tree()
        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        action_ind = self.getActionIndex(action)
        self.prev_action = torch.tensor([[action_ind]], device=self.device, dtype=torch.long)

        return action



    def step(self, reward, observation):
        self.time_step += 1

        if not self.keep_subtree:
            self.subtree_node = Node(None, observation)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.state = self.getStateRepresentation(observation)
        action_ind = self.getActionIndex(action)
        self.action = torch.tensor([[action_ind]], device=self.device, dtype=torch.long)
        reward = torch.tensor([reward], device=self.device)

        # store the new transition in buffer
        self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     self.state,
                                                     self.action, False, self.time_step, 0))
        # train/plan with model
        if self._model[self.model_type]['training']:
            if len(self.transition_buffer) >= self._model[self.model_type]['batch_size']:
                self.trainModel()
        self.plan()

        self.updateStateRepresentation()

        self.prev_state = self.getStateRepresentation(observation)
        self.prev_action = self.action  # another option:** we can again call self.policy function **

        return action


    def end(self, reward):
        reward = torch.tensor([reward], device=self.device)

        self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     None,
                                                     None, True, self.time_step, 0))
        if self._model[self.model_type]['training']:
            if len(self.transition_buffer) >= self._model[self.model_type]['batch_size']:
                self.trainModel()      
        self.updateStateRepresentation()


    def true_model(self, state, action):
        action_index = self.getActionIndex(action)
        torch_action_index = torch.tensor([action_index], device=self.device).unsqueeze(0)
        torch_state = self.getStateRepresentation(state)
        torch_next_state = torch.round(self.modelRollout(torch_state, torch_action_index)[0])

        transition = self.transition_dynamics[int(state[0]), int(state[1]), action_index]
        next_state, is_terminal, reward = transition[0:2], transition[2], transition[3]

        print(next_state, ' --- ', torch_next_state)
        return next_state, is_terminal, reward