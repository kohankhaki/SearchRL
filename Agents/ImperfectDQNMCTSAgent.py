import math
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from ete3 import Tree, TreeStyle, TextFace, add_face_to_node
from torch.utils.tensorboard import SummaryWriter

import Utils as utils
from Agents.MCTSAgent_Torch import MCTSAgent_Torch as MCTSAgent
from Agents.RealBaseDynaAgent import RealBaseDynaAgent
from DataStructures.Node_Torch import Node_Torch as Node
from Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN

is_gridWorld = False
is_cartpole = False
class ImperfectMCTSAgent_gridworld(RealBaseDynaAgent, MCTSAgent):
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
            self.goal_np = params['goal']

        self.num_steps = 0
        self.num_terminal_steps = 0

        self.is_model_pretrained = True

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
            self.root = Node(None, self.prev_state[0])
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, self.prev_state[0])
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.prev_action = torch.tensor([[action]], device=self.device)
        return self.action_list[action]

    def step(self, reward, observation):
        self.time_step += 1

        self.state = self.getStateRepresentation(observation)
        if not self.keep_subtree:
            self.subtree_node = Node(None, self.state[0])
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.action = torch.tensor([[action]], device=self.device, dtype=torch.long)
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

        return self.action_list[action]


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

    def model(self, state, action_index):
        with torch.no_grad():
            state0 = int(state[0].item())
            state1 = int(state[1].item())
            action_index = action_index[0, 0].item()
            next_state = self.saved_model[state0, state1, action_index]
            uncertainty = self.saved_uncertainty[state0, state1, action_index][0].item()
            next_state, reward, is_terminal = self.get_transition(next_state)
            #if want not rounded next_state, replace next_state with _  
        
        return next_state, is_terminal, reward, uncertainty

    def model_np(self, state, action_index):
        # state = np.clip(state, 0, 8)
        next_state = self.saved_model_np[int(state[0]), int(state[1]), action_index]
        next_state, reward, is_terminal = self.get_transition_np(next_state)
        uncertainty = self.saved_model_uncertainty_np[int(state[0]), int(state[1]), action_index][0]
        return next_state, is_terminal, reward, uncertainty

    def get_torch_action_index(self, action_index):
        return torch.tensor([action_index], device=self.device).unsqueeze(0)

    def get_transition(self, state):
        if is_gridWorld:
            state = torch.clamp(state, 0, 8)
            state = torch.round(state)
            reward = None
            is_terminal = None
            if torch.equal(state, self.goal):
                reward = 0
                is_terminal = True
            else:
                reward = -1
                is_terminal = False
        else:
            pass

        return state, reward, is_terminal

    def get_transition_np(self, state):
        state = np.clip(state, 0, 8)
        state = np.round(state)
        reward = None
        is_terminal = None
        if np.array_equal(state, self.goal_np):
            reward = 0
            is_terminal = True
        else:
            reward = -1
            is_terminal = False
        return state, reward, is_terminal

    def trainModel(self):
        if self.is_model_pretrained:
            return
        RealBaseDynaAgent.trainModel(self)
        self.saveModel()
        self.saveModelNp()

    def initModel(self, state, action):
        return 
        RealBaseDynaAgent.initModel(self, state, action)
        if self.is_model_pretrained:
            RealBaseDynaAgent.loadModelFile(self, "r0_stepsize0.000244140625")
        self.saved_model = torch.zeros([9, 9, 4, 2], device=self.device)
        self.saved_uncertainty = torch.zeros([9, 9, 4, 1], device=self.device)
        self.saved_model_np = np.zeros([9, 9, 4, 2])
        self.saved_model_uncertainty_np = np.zeros([9, 9, 4, 1])
        self.saveModel()
        self.saveModelNp()


    # def true_model(self, state, action_index):
    #     transition = self.transition_dynamics[int(state[0]), int(state[1]), action_index]
    #     next_state, is_terminal, reward = transition[0:2], transition[2], transition[3]
    #     return next_state, is_terminal, reward


    def saveModel(self):
        return
        with torch.no_grad():
            for s0 in range(9):
                for s1 in range(9):
                    for a in range(4):
                        state = torch.tensor([[s0, s1]], device=self.device)
                        action_index = torch.tensor([[a]], device=self.device)
                        next_state, uncertainty = self.modelRollout(state, action_index)
                        self.saved_model[s0, s1, a] = next_state[0]
                        self.saved_uncertainty[s0, s1, a] = torch.tensor([uncertainty], device=self.device)

    def saveModelNp(self):
        return
        with torch.no_grad():
            for s0 in range(9):
                for s1 in range(9):
                    for a in range(4):
                        state = torch.tensor([[s0, s1]], device=self.device)
                        action_index = torch.tensor([[a]], device=self.device)
                        next_state, uncertainty = self.modelRollout(state, action_index)
                        self.saved_model_np[s0, s1, a] = next_state[0].cpu().numpy()    
                        self.saved_model_uncertainty_np[s0, s1, a] = uncertainty

class ImperfectMCTSAgent(RealBaseDynaAgent, MCTSAgent):
    name = "ImperfectMCTSAgent"

    def __init__(self, params={}):
        self.model_loss = []
        self.time_step = 0
        self.writer = SummaryWriter()
        self.writer_iterations = 0


        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon_min']

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
        self.true_model = params['true_fw_model']
        if params['goal'] is not None:
            self.goal = torch.from_numpy(params['goal']).float().to(self.device)
            self.goal_np = params['goal']

        self.num_steps = 0
        self.num_terminal_steps = 0

        self.is_model_pretrained = False

        # MCTS parameters
        self.C = params['c']
        self.num_iterations = params['num_iteration']
        self.num_rollouts = params['num_simulation']
        self.rollout_depth = params['simulation_depth']
        self.keep_subtree = False
        self.keep_tree = False
        self.root = None    


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
            self.root = Node(None, self.prev_state)
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, self.prev_state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.prev_action = torch.tensor([[action]], device=self.device)
        return self.action_list[action]

    def step(self, reward, observation):
        self.time_step += 1

        self.state = self.getStateRepresentation(observation)
        if not self.keep_subtree:
            self.subtree_node = Node(None, self.state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        
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

        return self.action_list[action]


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

    def model(self, state, action_index):
        with torch.no_grad():
            next_state, uncertainty = self.modelRollout(state, action_index)
            true_next_state, is_terminal, reward = self.true_model(state[0], action_index.item())
            #if want not rounded next_state, replace next_state with _  
        
            if is_cartpole:
                x, x_dot, theta, theta_dot = next_state[0]
                theta_threshold_radians = 12 * 2 * math.pi / 360
                x_threshold = 2.4
                is_terminal = bool(
                    x < -x_threshold
                    or x > x_threshold
                    or theta < -theta_threshold_radians
                    or theta > theta_threshold_radians
                )
                if not is_terminal:
                    reward = 1.0
                else:
                    reward = 0.0
        return next_state, is_terminal, reward, 0

    def trainModel(self):
        if self.is_model_pretrained:
            return
        RealBaseDynaAgent.trainModel(self)

    def initModel(self, state, action): 
        RealBaseDynaAgent.initModel(self, state, action)
        if self.is_model_pretrained:
            RealBaseDynaAgent.loadModelFile(self, "LearnedModel/HeteroscedasticLearnedModel/TestCartpole_stepsize0.001_network2")

class ImperfectMCTSAgentUncertainty(RealBaseDynaAgent, MCTSAgent):
    name = "ImperfectMCTSAgentUncertainty"
    rollout_idea = None # None, 1
    selection_idea = None # None, 1
    backpropagate_idea = None # None, 1

    assert rollout_idea in [None, 1] and selection_idea in [None, 1] and backpropagate_idea in [None, 1]# add the idea to assertion list too

    def __init__(self, params={}):
        self.model_loss = []
        self.time_step = 0
        self.writer = SummaryWriter()
        self.writer_iterations = 0

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
        self.true_model = params['true_fw_model']
        if params['goal'] is not None:
            self.goal = torch.from_numpy(params['goal']).float().to(self.device)
            self.goal_np = params['goal']

        self.num_steps = 0
        self.num_terminal_steps = 0

        self.is_model_pretrained = True

        # MCTS parameters
        self.C = params['c']
        self.num_iterations = params['num_iteration']
        self.num_rollouts = params['num_simulation']
        self.rollout_depth = params['simulation_depth']
        self.keep_subtree = False
        self.keep_tree = False
        self.root = None    


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
            self.root = Node(None, self.prev_state)
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, self.prev_state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.prev_action = torch.tensor([[action]], device=self.device)
        return self.action_list[action]

    def step(self, reward, observation):
        self.time_step += 1

        self.state = self.getStateRepresentation(observation)
        if not self.keep_subtree:
            self.subtree_node = Node(None, self.state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        
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

        return self.action_list[action]


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

    def model(self, state, action_index):
        with torch.no_grad():
            next_state, uncertainty = self.modelRollout(state, action_index)
            true_next_state, is_terminal, reward = self.true_model(state[0], action_index.item())
            true_next_state = torch.from_numpy(true_next_state).to(self.device)
            #if want not rounded next_state, replace next_state with _  
            true_uncertainty = torch.mean(torch.pow(true_next_state - next_state[0], 2))

            # uncertainty_err = (true_uncertainty - uncertainty)**2
            # self.writer.add_scalar('Uncertainty_err', uncertainty_err, self.writer_iterations)
            # self.writer_iterations += 1

            if is_cartpole:
                x, x_dot, theta, theta_dot = next_state[0]
                theta_threshold_radians = 12 * 2 * math.pi / 360
                x_threshold = 2.4
                is_terminal = bool(
                    x < -x_threshold
                    or x > x_threshold
                    or theta < -theta_threshold_radians
                    or theta > theta_threshold_radians
                )
                if not is_terminal:
                    reward = 1.0
                else:
                    reward = 0.0
        return next_state, is_terminal, reward, true_uncertainty

    def rollout(self, node):
        if ImperfectMCTSAgentUncertainty.rollout_idea == 1:
            sum_returns = 0
            for _ in range(self.num_rollouts):
                depth = 0
                is_terminal = node.is_terminal
                state = node.get_state()

                gamma_prod = 1
                single_return = 0
                sum_uncertainty = 0
                return_list = []
                weight_list = []
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    next_state, is_terminal, reward, uncertainty = self.model(state, action_index)
                    uncertainty = uncertainty.item()
                    single_return += reward * gamma_prod
                    gamma_prod *= self.gamma
                    sum_uncertainty += uncertainty

                    return_list.append(single_return)
                    weight_list.append(sum_uncertainty)
                    depth += 1
                    state = next_state
                return_list = np.asarray(return_list)
                weight_list = np.asarray(weight_list)

                weights = np.exp(-weight_list) / np.sum(np.exp(-weight_list))
                if len(weights) > 0:
                    uncertain_return = np.average(return_list, weights=weights)
                else: # the starting node is a terminal state
                    uncertain_return = 0
                sum_returns += uncertain_return
            return sum_returns / self.num_rollouts
        else:
            return MCTSAgent.rollout(self, node)

    def selection(self):
        if ImperfectMCTSAgentUncertainty.backpropagate_idea == 1:
            selected_node = self.subtree_node
            while len(selected_node.get_childs()) > 0:
                max_uct_value = -np.inf
                child_values = list(map(lambda n: n.get_weighted_avg_value()+n.reward_from_par, selected_node.get_childs()))
                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, selected_node.get_childs())))
                max_child_value = max(child_values)
                min_child_value = min(child_values)
                softmax_denominator = np.sum(np.exp(-child_uncertainties))
                for ind, child in enumerate(selected_node.get_childs()):
                    if child.num_visits == 0:
                        selected_node = child
                        break
                    else:
                        child_value = child_values[ind]
                        child_uncertainty = child_uncertainties[ind]
                        if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                            child_value = (child_value - min_child_value) / (max_child_value - min_child_value)

                        child_uncertainty = np.exp(-child_uncertainty) / softmax_denominator

                        uct_value = child_value + \
                                    self.C * ((child.parent.num_visits / child.num_visits) ** 0.5) - child_uncertainty
                        # print("old:", uct_value - child_uncertainty, "  new:",uct_value, "  unc:", child_uncertainty)

                    if max_uct_value < uct_value:
                        max_uct_value = uct_value
                        selected_node = child
            return selected_node
        elif ImperfectMCTSAgentUncertainty.selection_idea == 1:
            selected_node = self.subtree_node
            while len(selected_node.get_childs()) > 0:
                max_uct_value = -np.inf
                child_values = list(map(lambda n: n.get_avg_value()+n.reward_from_par, selected_node.get_childs()))
                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, selected_node.get_childs())))
                max_child_value = max(child_values)
                min_child_value = min(child_values)
                softmax_denominator = np.sum(np.exp(child_uncertainties))
                for ind, child in enumerate(selected_node.get_childs()):
                    if child.num_visits == 0:
                        selected_node = child
                        break
                    else:
                        child_value = child_values[ind]
                        child_uncertainty = child_uncertainties[ind]
                        if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                            child_value = (child_value - min_child_value) / (max_child_value - min_child_value)

                        child_uncertainty = np.exp(child_uncertainty) / softmax_denominator

                        uct_value = child_value + \
                                    self.C * ((child.parent.num_visits / child.num_visits) ** 0.5) - child_uncertainty
                        # print("old:", uct_value - child_uncertainty, "  new:",uct_value, "  unc:", child_uncertainty)

                    if max_uct_value < uct_value:
                        max_uct_value = uct_value
                        selected_node = child
            return selected_node
        else:
            return MCTSAgent.selection(self)
    
    def expansion(self, node):
        for a in range(self.num_actions):
            action_index = torch.tensor([a]).unsqueeze(0)
            next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                            action_index)  # with the assumption of deterministic model
            # if np.array_equal(next_state, node.get_state()):
            #     continue
            value = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                        value=value, uncertainty=uncertainty.item())
            node.add_child(child)

    def backpropagate(self, node, value):
        if ImperfectMCTSAgentUncertainty.backpropagate_idea == 1:
            while node is not None:
                node.add_to_values(value)
                node.inc_visits()
                
                siblings = node.parent.get_childs()
                siblings_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, siblings)))
                softmax_denominator = np.sum(np.exp(-siblings_uncertainties))
                value *= self.gamma 
                value += node.reward_from_par
                value *= np.exp(-node.uncertainty) / softmax_denominator
                node = node.parent
        else:
            while node is not None:
                node.add_to_values(value)
                node.inc_visits()
                value *= self.gamma
                value += node.reward_from_par
                node = node.parent

    def trainModel(self):
        if self.is_model_pretrained:
            return
        RealBaseDynaAgent.trainModel(self)

    def initModel(self, state, action): 
        RealBaseDynaAgent.initModel(self, state, action)
        if self.is_model_pretrained:
            RealBaseDynaAgent.loadModelFile(self, "LearnedModel/HeteroscedasticLearnedModel/TestCartpole_stepsize0.01_network4")

class ImperfectMCTSAgentUncertaintyHandDesignedModel_cartpole(RealBaseDynaAgent, MCTSAgent):
    name = "ImperfectMCTSAgentUncertaintyHandDesignedModel"
    rollout_idea = 1 # None, 1
    selection_idea = 1 # None, 1
    backpropagate_idea = None # None, 1

    assert rollout_idea in [None, 1] and selection_idea in [None, 1] and backpropagate_idea in [None, 1]# add the idea to assertion list too

    def __init__(self, params={}):
        self.model_corruption = params['model_corruption']
        self.model_loss = []
        self.time_step = 0

        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon_min']

        self.transition_buffer = []
        self.transition_buffer_size = 2**12


        self.num_ensembles = 5


        self._sr = dict(network=None,
                        layers_type=[],
                        layers_features=[],
                        batch_size=None,
                        step_size=None,
                        batch_counter=None,
                        training=False)

        self.device = params['device']
        self.true_model = params['true_fw_model']

        self.num_steps = 0
        self.num_terminal_steps = 0

        self.is_model_pretrained = True

        # MCTS parameters
        self.C = params['c']
        self.num_iterations = params['num_iteration']
        self.num_rollouts = params['num_simulation']
        self.rollout_depth = params['simulation_depth']
        self.keep_subtree = False
        self.keep_tree = False
        self.root = None    


    def start(self, observation):
        '''
        :param observation: numpy array -> (observation shape)
        :return: action : numpy array
        '''
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)

        self.prev_state = self.getStateRepresentation(observation)

        temp_prev_action = torch.tensor([[0]], device=self.device, dtype=torch.long)

        if self.keep_tree and self.root is None:
            self.root = Node(None, self.prev_state)
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, self.prev_state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.prev_action = torch.tensor([[action]], device=self.device)
        return self.action_list[action]

    def step(self, reward, observation):
        self.time_step += 1

        self.state = self.getStateRepresentation(observation)
        if not self.keep_subtree:
            self.subtree_node = Node(None, self.state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        
        reward = torch.tensor([reward], device=self.device)

        # store the new transition in buffer
        self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     self.state,
                                                     self.action, False, self.time_step, 0))

        self.updateStateRepresentation()

        self.prev_state = self.getStateRepresentation(observation)
        self.prev_action = self.action  # another option:** we can again call self.policy function **

        return self.action_list[action]

    def end(self, reward):
        reward = torch.tensor([reward], device=self.device)

        self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     None,
                                                     None, True, self.time_step, 0))
              
        self.updateStateRepresentation()

    def model(self, state, action_index):
        with torch.no_grad():
            true_next_state, is_terminal, reward = self.true_model(state[0], action_index.item())
            true_next_state = torch.from_numpy(true_next_state).to(self.device)
            if self.model_corruption == 0:
                corrupted_next_state = true_next_state.unsqueeze(0)
            else:
                corrupted_next_state = torch.normal(mean=true_next_state, std=self.model_corruption).unsqueeze(0)

            #if want not rounded next_state, replace next_state with _  
            true_uncertainty = torch.mean(torch.pow(true_next_state - corrupted_next_state[0], 2))

            # uncertainty_err = (true_uncertainty - uncertainty)**2
            # self.writer.add_scalar('Uncertainty_err', uncertainty_err, self.writer_iterations)
            # self.writer_iterations += 1

            if is_cartpole:
                x, x_dot, theta, theta_dot = corrupted_next_state[0]
                theta_threshold_radians = 12 * 2 * math.pi / 360
                x_threshold = 2.4
                is_terminal = bool(
                    x < -x_threshold
                    or x > x_threshold
                    or theta < -theta_threshold_radians
                    or theta > theta_threshold_radians
                )
                if not is_terminal:
                    reward = 1.0
                else:
                    reward = 0.0
        return corrupted_next_state, is_terminal, reward, true_uncertainty

    def rollout(self, node):
        if ImperfectMCTSAgentUncertainty.rollout_idea == 1:
            sum_returns = 0
            for _ in range(self.num_rollouts):
                depth = 0
                is_terminal = node.is_terminal
                state = node.get_state()

                gamma_prod = 1
                single_return = 0
                sum_uncertainty = 0
                return_list = []
                weight_list = []
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    next_state, is_terminal, reward, uncertainty = self.model(state, action_index)
                    uncertainty = uncertainty.item()
                    single_return += reward * gamma_prod
                    gamma_prod *= self.gamma
                    sum_uncertainty += uncertainty

                    return_list.append(single_return)
                    weight_list.append(sum_uncertainty)
                    depth += 1
                    state = next_state
                return_list = np.asarray(return_list)
                weight_list = np.asarray(weight_list)

                weights = np.exp(-weight_list) / np.sum(np.exp(-weight_list))
                if len(weights) > 0:
                    uncertain_return = np.average(return_list, weights=weights)
                else: # the starting node is a terminal state
                    uncertain_return = 0
                sum_returns += uncertain_return
            return sum_returns / self.num_rollouts
        else:
            return MCTSAgent.rollout(self, node)

    def selection(self):
        if ImperfectMCTSAgentUncertainty.backpropagate_idea == 1:
            selected_node = self.subtree_node
            while len(selected_node.get_childs()) > 0:
                max_uct_value = -np.inf
                child_values = list(map(lambda n: n.get_weighted_avg_value()+n.reward_from_par, selected_node.get_childs()))
                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, selected_node.get_childs())))
                max_child_value = max(child_values)
                min_child_value = min(child_values)
                softmax_denominator = np.sum(np.exp(child_uncertainties))
                for ind, child in enumerate(selected_node.get_childs()):
                    if child.num_visits == 0:
                        selected_node = child
                        break
                    else:
                        child_value = child_values[ind]
                        child_uncertainty = child_uncertainties[ind]
                        if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                            child_value = (child_value - min_child_value) / (max_child_value - min_child_value)

                        child_uncertainty = np.exp(child_uncertainty) / softmax_denominator

                        uct_value = child_value + \
                                    self.C * ((child.parent.num_visits / child.num_visits) ** 0.5) - child_uncertainty
                        # print("old:", uct_value - child_uncertainty, "  new:",uct_value, "  unc:", child_uncertainty)

                    if max_uct_value < uct_value:
                        max_uct_value = uct_value
                        selected_node = child
            return selected_node
        
        elif ImperfectMCTSAgentUncertainty.selection_idea == 1:
            selected_node = self.subtree_node
            while len(selected_node.get_childs()) > 0:
                max_uct_value = -np.inf
                child_values = list(map(lambda n: n.get_avg_value()+n.reward_from_par, selected_node.get_childs()))
                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, selected_node.get_childs())))
                max_child_value = max(child_values)
                min_child_value = min(child_values)
                softmax_denominator = np.sum(np.exp(child_uncertainties))
                for ind, child in enumerate(selected_node.get_childs()):
                    if child.num_visits == 0:
                        selected_node = child
                        break
                    else:
                        child_value = child_values[ind]
                        child_uncertainty = child_uncertainties[ind]
                        if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                            child_value = (child_value - min_child_value) / (max_child_value - min_child_value)

                        child_uncertainty = np.exp(child_uncertainty) / softmax_denominator

                        uct_value = child_value + \
                                    self.C * ((child.parent.num_visits / child.num_visits) ** 0.5) - child_uncertainty
                        # print("old:", uct_value - child_uncertainty, "  new:",uct_value, "  unc:", child_uncertainty)

                    if max_uct_value < uct_value:
                        max_uct_value = uct_value
                        selected_node = child
            return selected_node
        else:
            return MCTSAgent.selection(self)
    
    def expansion(self, node):
        for a in range(self.num_actions):
            action_index = torch.tensor([a]).unsqueeze(0)
            next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                            action_index)  # with the assumption of deterministic model
            # if np.array_equal(next_state, node.get_state()):
            #     continue
            value = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                        value=value, uncertainty=uncertainty.item())
            node.add_child(child)

    def backpropagate(self, node, value):
        if ImperfectMCTSAgentUncertainty.backpropagate_idea == 1:
            while node is not None:
                node.add_to_values(value)
                node.inc_visits()
                
                siblings = node.parent.get_childs()
                siblings_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, siblings)))
                softmax_denominator = np.sum(np.exp(-siblings_uncertainties))
                value *= self.gamma 
                value += node.reward_from_par
                value *= np.exp(-node.uncertainty) / softmax_denominator
                node = node.parent
        else:
            while node is not None:
                node.add_to_values(value)
                node.inc_visits()
                value *= self.gamma
                value += node.reward_from_par
                node = node.parent


class ImperfectMCTSAgentUncertaintyHandDesignedModel_gridworld(RealBaseDynaAgent, MCTSAgent):
    name = "ImperfectMCTSAgentUncertaintyHandDesignedModel"
    rollout_idea = None  # None, 1
    selection_idea = None  # None, 1
    backpropagate_idea = None  # None, 1

    assert rollout_idea in [None, 1] and selection_idea in [None, 1] and backpropagate_idea in [None,
                                                                                                1]  # add the idea to assertion list too

    def __init__(self, params={}):
        self.model_corruption = params['model_corruption']
        self.model_loss = []
        self.time_step = 0

        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon_min']

        self.transition_buffer = []
        self.transition_buffer_size = 2 ** 12

        self.num_ensembles = 5

        self._sr = dict(network=None,
                        layers_type=[],
                        layers_features=[],
                        batch_size=None,
                        step_size=None,
                        batch_counter=None,
                        training=False)

        self.device = params['device']
        self.true_model = params['true_fw_model']
        self.corrupt_model = params['corrupted_fw_model']

        self.num_steps = 0
        self.num_terminal_steps = 0

        self.is_model_pretrained = True

        # MCTS parameters
        self.C = params['c']
        self.num_iterations = params['num_iteration']
        self.num_rollouts = params['num_simulation']
        self.rollout_depth = params['simulation_depth']
        self.keep_subtree = False
        self.keep_tree = False
        self.root = None
        self.tau = params['tau']

    def start(self, observation, info=None):
        '''
        :param observation: numpy array -> (observation shape)
        :return: action : numpy array
        '''
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)

        self.prev_state = self.getStateRepresentation(observation)

        temp_prev_action = torch.tensor([[0]], device=self.device, dtype=torch.long)

        if self.keep_tree and self.root is None:
            self.root = Node(None, self.prev_state)
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, self.prev_state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
            # self.render_tree()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.prev_action = torch.tensor([[action]], device=self.device)

        return action

    def step(self, reward, observation, info=None):
        self.time_step += 1
        self.state = self.getStateRepresentation(observation)
        if not self.keep_subtree:
            self.subtree_node = Node(None, self.state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
            # self.render_tree()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree
        self.action = torch.tensor([[action]], device=self.device, dtype=torch.long)

        reward = torch.tensor([reward], device=self.device)

        # store the new transition in buffer
        self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     self.state,
                                                     self.action, False, self.time_step, 0))

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

        self.updateStateRepresentation()

    def model(self, state, action):
        with torch.no_grad():
            true_next_state, is_terminal, reward = self.true_model(state[0], action[0])
            true_next_state = torch.from_numpy(true_next_state).to(self.device)
            corrupted_next_state, _, _ = self.corrupt_model(state[0], action[0])
            corrupted_next_state = torch.from_numpy(corrupted_next_state).unsqueeze(0)

            # if want not rounded next_state, replace next_state with _
            true_uncertainty = torch.mean(torch.pow(true_next_state - corrupted_next_state[0], 2).float())
            # uncertainty_err = (true_uncertainty - uncertainty)**2
            # self.writer.add_scalar('Uncertainty_err', uncertainty_err, self.writer_iterations)
            # self.writer_iterations += 1

        return corrupted_next_state, is_terminal, reward, true_uncertainty

    def rollout(self, node):
        if ImperfectMCTSAgentUncertaintyHandDesignedModel_gridworld.rollout_idea == 1:
            sum_returns = 0
            for _ in range(self.num_rollouts):
                depth = 0
                is_terminal = node.is_terminal
                state = node.get_state()

                gamma_prod = 1
                single_return = 0
                sum_uncertainty = 0
                return_list = []
                weight_list = []
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, uncertainty = self.model(state, action)
                    uncertainty = uncertainty.item()
                    single_return += reward * gamma_prod
                    gamma_prod *= self.gamma
                    sum_uncertainty += uncertainty

                    return_list.append(single_return)
                    weight_list.append(sum_uncertainty)
                    depth += 1
                    state = next_state
                return_list = np.asarray(return_list)
                weight_list = np.asarray(weight_list)

                weights = np.exp(weight_list) / np.sum(np.exp(weight_list))
                # print(return_list)
                # print(weight_list)
                # print(weights)
                # print(np.average(return_list, weights=weights), np.average(return_list))
                if len(weights) > 0:
                    uncertain_return = np.average(return_list, weights=weights)
                else:  # the starting node is a terminal state
                    uncertain_return = 0
                sum_returns += uncertain_return
            return sum_returns / self.num_rollouts
        else:
            sum_returns = 0
            for i in range(self.num_rollouts):
                depth = 0
                single_return = 0
                is_terminal = node.is_terminal
                state = node.get_state()
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, _ = self.model(state, action)
                    single_return += reward
                    depth += 1
                    state = next_state

                sum_returns += single_return
            return sum_returns / self.num_rollouts

    def selection(self):
        if ImperfectMCTSAgentUncertaintyHandDesignedModel_gridworld.backpropagate_idea == 1:
            selected_node = self.subtree_node
            while len(selected_node.get_childs()) > 0:
                max_uct_value = -np.inf
                child_values = list(
                    map(lambda n: n.get_weighted_avg_value() + n.reward_from_par, selected_node.get_childs()))
                max_child_value = max(child_values)
                min_child_value = min(child_values)

                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, selected_node.get_childs())))
                softmax_denominator = np.sum(np.exp(child_uncertainties / self.tau))
                softmax_uncertainties = np.exp(child_uncertainties / self.tau) / softmax_denominator

                for ind, child in enumerate(selected_node.get_childs()):
                    if child.num_visits == 0:
                        selected_node = child
                        break
                    else:

                        child_value = child_values[ind]
                        child_value = child.get_avg_value() + child.reward_from_par
                        child_uncertainty = child_uncertainties[ind]
                        softmax_uncertainty = softmax_uncertainties[ind]

                        if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                            child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                        elif min_child_value == max_child_value:
                            child_value = 0.5

                        uct_value = (child_value + \
                                     self.C * ((np.log(child.parent.num_visits) / child.num_visits) ** 0.5)) * \
                                    (1 - softmax_uncertainty)
                        # print("old:", uct_value - child_uncertainty, "  new:",uct_value, "  unc:", child_uncertainty)
                    if max_uct_value < uct_value:
                        max_uct_value = uct_value
                        selected_node = child
            return selected_node

        elif ImperfectMCTSAgentUncertaintyHandDesignedModel_gridworld.selection_idea == 1:
            selected_node = self.subtree_node
            while len(selected_node.get_childs()) > 0:
                max_uct_value = -np.inf
                child_values = list(map(lambda n: n.get_avg_value() + n.reward_from_par, selected_node.get_childs()))
                max_child_value = max(child_values)
                min_child_value = min(child_values)

                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, selected_node.get_childs())))
                softmax_denominator = np.sum(np.exp(child_uncertainties / self.tau))
                softmax_uncertainties = np.exp(child_uncertainties / self.tau)/softmax_denominator

                for ind, child in enumerate(selected_node.get_childs()):
                    if child.num_visits == 0:
                        selected_node = child
                        break
                    else:

                        child_value = child_values[ind]
                        child_value = child.get_avg_value() + child.reward_from_par
                        child_uncertainty = child_uncertainties[ind]
                        softmax_uncertainty = softmax_uncertainties[ind]

                        if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                            child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                        elif min_child_value == max_child_value:
                            child_value = 0.5

                        uct_value = (child_value + \
                                     self.C * ((np.log(child.parent.num_visits) / child.num_visits) ** 0.5)) *\
                                    (1 - softmax_uncertainty)
                        # print("old:", uct_value - child_uncertainty, "  new:",uct_value, "  unc:", child_uncertainty)
                    if max_uct_value < uct_value:
                        max_uct_value = uct_value
                        selected_node = child
            return selected_node
        else:
            return MCTSAgent.selection(self)

    def expansion(self, node):
        for a in self.action_list:
            action = torch.tensor(a).unsqueeze(0)
            next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                                      action)  # with the assumption of deterministic model
            # if np.array_equal(next_state, node.get_state()):
            #     continue
            value = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                         value=value, uncertainty=uncertainty.item())
            node.add_child(child)

    def backpropagate(self, node, value):
        if ImperfectMCTSAgentUncertaintyHandDesignedModel_gridworld.backpropagate_idea == 1:
            while node is not None:
                node.add_to_values(value)
                node.inc_visits()
                if node.parent is not None:
                    siblings = node.parent.get_childs()
                    siblings_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, siblings)))
                    softmax_denominator = np.sum(np.exp(siblings_uncertainties))
                    value *= self.gamma
                    value += node.reward_from_par
                    value *= np.exp(node.uncertainty) / softmax_denominator
                node = node.parent
        else:
            while node is not None:
                node.add_to_values(value)
                node.inc_visits()
                value *= self.gamma
                value += node.reward_from_par
                node = node.parent

    def render_tree(self):
        def my_layout(node):
            F = TextFace(node.name, tight_text=True)
            add_face_to_node(F, node, column=0, position="branch-right")

        t = Tree()
        ts = TreeStyle()
        ts.show_leaf_name = False
        queue = [(self.subtree_node, None)]
        while queue:
            node, parent = queue.pop(0)
            uct_value = 0
            child_value = None
            child_uncertainty = 0
            if node.parent is not None:
                # child_values = list(map(lambda n: n.get_avg_value() + n.reward_from_par, node.parent.get_childs()))
                child_values = list(
                    map(lambda n: n.get_weighted_avg_value() + n.reward_from_par, node.parent.get_childs()))
                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, node.parent.get_childs())))
                softmax_denominator = np.sum(np.exp(child_uncertainties / self.tau))
                max_child_value = max(child_values)
                min_child_value = min(child_values)

                child_value = node.get_avg_value() + node.reward_from_par
                child_uncertainty = np.exp(node.uncertainty / self.tau) / softmax_denominator
                if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                    child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                elif min_child_value == max_child_value:
                    child_value = 0.5

                if node.num_visits == 0:
                    uct_value = np.inf
                else:
                    uct_value = (child_value + \
                                self.C * ((np.log(node.parent.num_visits) / node.num_visits) ** 0.5) ) *\
                                (1 - child_uncertainty)




            # node_face = str(node.get_state()) + "," + str(node.num_visits) + "," + str(node.get_avg_value()) \
            #             + "," + str(node.is_terminal) + "," + str(uct_value) + "," + str(node.uncertainty)
            node_face = str(node.get_state()[0].numpy()) + "," + str(node.num_visits) + "," + str(node.get_avg_value()) \
                        + "," + str(round(uct_value, 3)) + "," + str(child_uncertainty)
            if parent is None:
                p = t.add_child(name=node_face)
            else:
                p = parent.add_child(name=node_face)
            for child in node.get_childs():
                queue.append((child, p))

        ts.layout_fn = my_layout
        # t.render('t.png', tree_style=ts)
        # print(t.get_ascii(show_internal=Tree))
        t.show(tree_style=ts)

class ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld(RealBaseDynaAgent, MCTSAgent):
    name = "ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld"
    rollout_idea = None  # None, 1
    selection_idea = None  # None, 1
    backpropagate_idea = None  # None, 1
    expansion_idea = 1
    assert rollout_idea in [None, 1, 2, 3, 4, 5] and \
           selection_idea in [None, 1, 2] and \
           backpropagate_idea in [None, 1]  \
           and expansion_idea in [None, 1]

    def __init__(self, params={}):
        self.episode_counter = 0
        self.model_corruption = params['model_corruption']
        self.model_loss = []
        self.time_step = 0

        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon_min']

        self.transition_buffer = []
        self.transition_buffer_size = 2 ** 12


        self.policy_values = 'q'  # 'q' or 's' or 'qs'
        self.policy_values = params['vf']['type']  # 'q' or 's' or 'qs'

        self._vf = {'q': dict(network=None,
                              num_ensembles=1,
                              layers_type=params['vf']['layers_type'],
                              layers_features=params['vf']['layers_features'],
                              action_layer_num=params['vf']['action_layer_num'],
                              # if one more than layer numbers => we will have num of actions output
                              batch_size=16,
                              step_size=params['max_stepsize'],
                              training=True),}
        self.is_pretrained = False
        self._target_vf = dict(network=None,
                               counter=0,
                               layers_num=None,
                               action_layer_num=None,
                               update_rate=10,
                               type=None)

        self._sr = dict(network=None,
                        layers_type=[],
                        layers_features=[],
                        batch_size=None,
                        step_size=None,
                        batch_counter=None,
                        training=False)

        self.device = params['device']
        self.true_model = params['true_fw_model']
        self.corrupt_model = params['corrupted_fw_model']

        self.num_steps = 0
        self.num_terminal_steps = 0

        self.is_model_pretrained = True

        # MCTS parameters
        self.C = params['c']
        self.num_iterations = params['num_iteration']
        self.num_rollouts = params['num_simulation']
        self.rollout_depth = params['simulation_depth']
        self.keep_subtree = False
        self.keep_tree = False
        self.root = None
        self.tau = params['tau']

    def start(self, observation, info=None):
        '''
        :param observation: numpy array -> (observation shape)
        :return: action : numpy array
        '''
        self.episode_counter += 1
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)

        self.prev_state = self.getStateRepresentation(observation)

        if self._vf['q']['network'] is None and self._vf['q']['training']:
            self.init_q_value_function_network(self.prev_state)  # a general state action VF for all actions
        self.setTargetValueFunction(self._vf['q'], 'q')

        if self.keep_tree and self.root is None:
            self.root = Node(None, self.prev_state)
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, self.prev_state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
            # self.render_tree()
        action, sub_tree = self.choose_action()
        action_index = self.getActionIndex(action)

        self.subtree_node = sub_tree
        self.prev_action = torch.tensor([action_index]).unsqueeze(0)
        return action

    def step(self, reward, observation, info=None):
        self.time_step += 1
        self.state = self.getStateRepresentation(observation)
        if not self.keep_subtree:
            self.subtree_node = Node(None, self.state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
            # self.render_tree()
        action, sub_tree = self.choose_action()
        action_index = self.getActionIndex(action)
        self.subtree_node = sub_tree
        self.action = torch.tensor([action_index]).unsqueeze(0)

        reward = torch.tensor([reward], device=self.device)


        self.updateStateRepresentation()
        self.prev_state = self.getStateRepresentation(observation)
        self.prev_action = self.action
        return action

    def end(self, reward):
        reward = torch.tensor([reward], device=self.device)
        self.updateStateRepresentation()



    def policy(self, state):
        '''
        :param state: torch -> (1, state_shape)
        :return: action: index torch
        '''
        if random.random() < self.epsilon:
            ind = torch.tensor([[random.randrange(self.num_actions)]],
                               device=self.device, dtype=torch.long)
            return ind
        with torch.no_grad():
            v = []
            if self.policy_values == 'q':
                ensemble_values = [self._vf['q']['network'][i](state) for i in range(self._vf['q']['num_ensembles'])]
                avg_values = np.sum(ensemble_values) / self._vf['q']['num_ensembles']
                ind = avg_values.max(1)[1].view(1, 1)
                return ind
            else:
                raise ValueError('policy is not defined')

    def init_q_value_function_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''

        nn_state_shape = state.shape
        self._vf['q']['network'] = []
        self.optimizer = []
        for i in range(self._vf['q']['num_ensembles']):
            self._vf['q']['network'].append(StateActionVFNN(nn_state_shape, self.num_actions,
                                                   self._vf['q']['layers_type'],
                                                   self._vf['q']['layers_features'],
                                                   self._vf['q']['action_layer_num']).to(self.device))
            self.optimizer.append(optim.Adam(self._vf['q']['network'][i].parameters(), lr=self._vf['q']['step_size']))

    def setTargetValueFunction(self, vf, type):
        if self._target_vf['network'] is None:
            nn_state_shape = self.prev_state.shape
            self._target_vf['network'] = StateActionVFNN(
                nn_state_shape,
                self.num_actions,
                vf['layers_type'],
                vf['layers_features'],
                vf['action_layer_num']).to(self.device)
        random_ensemble = np.random.randint(0, vf['num_ensembles'])
        self._target_vf['network'].load_state_dict(vf['network'][random_ensemble].state_dict())  # copy weights and stuff

    def updateValueFunction(self, transition_batch, vf_type):

        batch = utils.transition(*zip(*transition_batch))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None,
                      batch.state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.state
                                           if s is not None])
        prev_state_batch = torch.cat(batch.prev_state)
        prev_action_batch = torch.cat(batch.prev_action)
        reward_batch = torch.cat(batch.reward)

        # BEGIN DQN
        next_state_values = torch.zeros(self._vf['q']['batch_size'], device=self.device)
        next_state_values[non_final_mask] = self._target_vf['network'](non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        for i in range(self._vf['q']['num_ensembles']):
            state_action_values = self._vf['q']['network'][i](prev_state_batch).gather(1, prev_action_batch)
            loss = F.mse_loss(state_action_values,
                              expected_state_action_values.unsqueeze(1))
            self.optimizer[i].zero_grad()
            loss.backward()
            self.optimizer[i].step()
        # END DQN

        self._target_vf['counter'] += 1

    def choose_action(self):
        max_value = -np.inf
        max_action_list = []
        max_child_list = []
        for child in self.subtree_node.get_childs():
            value = child.num_visits
            if value > max_value:
                max_value = value
                max_action_list = [child.get_action_from_par()]
                max_child_list = [child]
            elif value == max_value:
                max_action_list.append(child.get_action_from_par())
                max_child_list.append(child)
        random_ind = random.randint(0, len(max_action_list) - 1)
        return max_action_list[random_ind], max_child_list[random_ind]

    def model(self, state, action):
        with torch.no_grad():
            true_next_state, is_terminal, reward = self.true_model(state[0], action[0])
            true_next_state = torch.from_numpy(true_next_state).to(self.device)
            corrupted_next_state, _, _ = self.corrupt_model(state[0], action[0])
            corrupted_next_state = torch.from_numpy(corrupted_next_state).unsqueeze(0)

            # if want not rounded next_state, replace next_state with _
            true_uncertainty = torch.mean(torch.pow(true_next_state - corrupted_next_state[0], 2).float())
            # uncertainty_err = (true_uncertainty - uncertainty)**2
            # self.writer.add_scalar('Uncertainty_err', uncertainty_err, self.writer_iterations)
            # self.writer_iterations += 1

        return corrupted_next_state, is_terminal, reward, true_uncertainty

    def rollout(self, node):
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld.rollout_idea == 1:
            sum_returns = 0
            for _ in range(self.num_rollouts):
                depth = 0
                is_terminal = node.is_terminal
                state = node.get_state()

                gamma_prod = 1
                single_return = 0
                sum_uncertainty = 0
                return_list = []
                weight_list = []
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, uncertainty = self.model(state, action)
                    uncertainty = uncertainty.item()
                    single_return += reward * gamma_prod
                    gamma_prod *= self.gamma
                    sum_uncertainty += uncertainty

                    return_list.append(single_return)
                    weight_list.append(sum_uncertainty)
                    depth += 1
                    state = next_state
                return_list = np.asarray(return_list)
                weight_list = np.asarray(weight_list)

                weights = np.exp(weight_list) / np.sum(np.exp(weight_list))
                # print(return_list)
                # print(weight_list)
                # print(weights)
                # print(np.average(return_list, weights=weights), np.average(return_list))
                if len(weights) > 0:
                    uncertain_return = np.average(return_list, weights=weights)
                else:  # the starting node is a terminal state
                    uncertain_return = 0
                sum_returns += uncertain_return
            return sum_returns / self.num_rollouts

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld.rollout_idea == 2:
            sum_returns = 0
            for _ in range(self.num_rollouts):
                depth = 0
                is_terminal = node.is_terminal
                state = node.get_state()

                gamma_prod = 1
                single_return = 0
                sum_uncertainty = 0
                return_list = []
                weight_list = []
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, uncertainty = self.model(state, action)
                    uncertainty = uncertainty.item()
                    single_return += reward * gamma_prod
                    gamma_prod *= self.gamma
                    sum_uncertainty += uncertainty
                    depth += 1

                    bootstrap_value = 0
                    if not is_terminal:
                        with torch.no_grad():
                            s_rep = next_state
                            if np.array_equal(next_state, [[0,0]]):
                                s_value = -8
                            elif np.array_equal(next_state, [[0,1]]):
                                s_value = -7
                            elif np.array_equal(next_state, [[0,2]]):
                                s_value = -6
                            elif np.array_equal(next_state, [[0,3]]):
                                s_value = -5
                            elif np.array_equal(next_state, [[0,4]]):
                                s_value = -4
                            elif np.array_equal(next_state, [[0,5]]):
                                s_value = -3
                            elif np.array_equal(next_state, [[0,6]]):
                                s_value = -2
                            elif np.array_equal(next_state, [[0,7]]):
                                s_value = -1
                            elif np.array_equal(next_state, [[1,0]]):
                                s_value = -9
                            elif np.array_equal(next_state, [[1,7]]):
                                s_value = 0
                            elif np.array_equal(next_state, [[2,0]]):
                                s_value = -8
                            elif np.array_equal(next_state, [[2,1]]):
                                s_value = -7
                            elif np.array_equal(next_state, [[2,2]]):
                                s_value = -6
                            elif np.array_equal(next_state, [[2,3]]):
                                s_value = -5
                            elif np.array_equal(next_state, [[2,4]]):
                                s_value = -4
                            elif np.array_equal(next_state, [[2,5]]):
                                s_value = -3
                            elif np.array_equal(next_state, [[2,6]]):
                                s_value = -2
                            elif np.array_equal(next_state, [[2,7]]):
                                s_value = -1
                            else:
                                print("unknown state", next_state)
                                exit()

                            # ensemble_values = np.asarray([self._vf['q']['network'][i](s_rep).numpy() for i in
                            #                               range(self._vf['q']['num_ensembles'])])
                            # avg_values = np.mean(ensemble_values, axis=0)
                            # s_value = np.mean(avg_values)
                        bootstrap_value = gamma_prod * s_value

                    return_list.append(single_return + bootstrap_value)
                    weight_list.append(sum_uncertainty)
                    state = next_state

                return_list = np.asarray(return_list)
                weight_list = np.asarray(weight_list)
                if self.episode_counter > 30:
                    weights = np.exp(-weight_list) / np.sum(np.exp(-weight_list))
                else:
                    weights = np.exp(weight_list) / np.sum(np.exp(weight_list))
                if len(weights) > 0:
                    uncertain_return = np.average(return_list, weights=weights)
                else:  # the starting node is a terminal state
                    uncertain_return = 0
                sum_returns += uncertain_return
            return sum_returns / self.num_rollouts

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld.rollout_idea == 3:
            sum_returns = 0
            for i in range(self.num_rollouts):
                depth = 0
                single_return = 0
                is_terminal = node.is_terminal
                state = node.get_state()
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, uncertainty = self.model(state, action)
                    uncertainty = uncertainty.item()
                    single_return += reward * uncertainty
                    depth += 1
                    state = next_state

                sum_returns += single_return
            return sum_returns / self.num_rollouts

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld.rollout_idea == 4:
            sum_returns = 0
            for i in range(self.num_rollouts):
                depth = 0
                is_terminal = node.is_terminal
                state = node.get_state()

                gamma_prod = 1
                single_return = 0
                sum_uncertainty = 0
                return_list = []
                weight_list = []
                while not is_terminal and depth < self.rollout_depth:
                    action_index = self.policy(state)
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, uncertainty = self.model(state, action)
                    uncertainty = uncertainty.item()
                    single_return += reward * gamma_prod
                    gamma_prod *= self.gamma
                    sum_uncertainty += uncertainty
                    depth += 1
                    bootstrap_value = 0
                    if not is_terminal:
                        with torch.no_grad():
                            s_rep = next_state
                            ensemble_values = np.asarray([self._vf['q']['network'][i](s_rep).numpy() for i in
                                                          range(self._vf['q']['num_ensembles'])])
                            avg_values = np.mean(ensemble_values, axis=0)
                            s_value = np.mean(avg_values)
                        bootstrap_value = gamma_prod * s_value
                    return_list.append(single_return + bootstrap_value)
                    weight_list.append(sum_uncertainty)


                    reward = torch.tensor([reward], device=self.device)
                    if not is_terminal:
                        self.updateTransitionBuffer(utils.transition(state,
                                                                     action_index,
                                                                     reward,
                                                                     next_state,
                                                                     None, False, self.time_step, 0))
                    else:
                        self.updateTransitionBuffer(utils.transition(state,
                                                                     action_index,
                                                                     reward,
                                                                     None,
                                                                     None, False, self.time_step, 0))
                    if self._vf['q']['training']:
                        if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                                    transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                                    self.updateValueFunction(transition_batch, 'q')
                    if self._target_vf['counter'] >= self._target_vf['update_rate']:
                        self.setTargetValueFunction(self._vf['q'], 'q')

                    state = next_state

                return_list = np.asarray(return_list)
                weight_list = np.asarray(weight_list)
                weights = np.exp(-weight_list) / np.sum(np.exp(-weight_list))

                if len(weights) > 0:
                    uncertain_return = np.average(return_list, weights=weights)
                else:  # the starting node is a terminal state
                    uncertain_return = 0
                sum_returns += uncertain_return
            return sum_returns / self.num_rollouts

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld.rollout_idea == 5:
            uncertainty_list = []
            return_list = []
            for i in range(self.num_rollouts):
                depth = 0
                single_return = 0
                sum_uncertainty = 0
                gamma_prod = 1
                is_terminal = node.is_terminal
                state = node.get_state()
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, uncertainty = self.model(state, action)
                    sum_uncertainty += gamma_prod * uncertainty
                    gamma_prod *= self.gamma
                    single_return += reward
                    depth += 1
                    state = next_state
                uncertainty_list.append(sum_uncertainty)
                return_list.append(single_return)

            uncertainty_list = np.asarray(uncertainty_list)
            softmax_uncertainty_list = np.exp(-uncertainty_list / self.tau) / np.sum(np.exp(-uncertainty_list / self.tau))
            weighted_avg = np.average(return_list, weights=softmax_uncertainty_list)
            return weighted_avg

        else:
            sum_returns = 0
            for i in range(self.num_rollouts):
                depth = 0
                single_return = 0
                is_terminal = node.is_terminal
                state = node.get_state()
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, _ = self.model(state, action)
                    single_return += reward
                    depth += 1
                    state = next_state

                sum_returns += single_return
            return sum_returns / self.num_rollouts

    def selection(self):
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld.backpropagate_idea == 1:
            selected_node = self.subtree_node
            while len(selected_node.get_childs()) > 0:
                max_uct_value = -np.inf
                child_values = list(
                    map(lambda n: n.get_weighted_avg_value() + n.reward_from_par, selected_node.get_childs()))
                max_child_value = max(child_values)
                min_child_value = min(child_values)

                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, selected_node.get_childs())))
                softmax_denominator = np.sum(np.exp(child_uncertainties / self.tau))
                softmax_uncertainties = np.exp(child_uncertainties / self.tau) / softmax_denominator

                for ind, child in enumerate(selected_node.get_childs()):
                    if child.num_visits == 0:
                        selected_node = child
                        break
                    else:

                        child_value = child_values[ind]
                        child_value = child.get_avg_value() + child.reward_from_par
                        child_uncertainty = child_uncertainties[ind]
                        softmax_uncertainty = softmax_uncertainties[ind]

                        if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                            child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                        elif min_child_value == max_child_value:
                            child_value = 0.5

                        uct_value = (child_value + \
                                     self.C * ((np.log(child.parent.num_visits) / child.num_visits) ** 0.5))
                        # print("old:", uct_value - child_uncertainty, "  new:",uct_value, "  unc:", child_uncertainty)
                    if max_uct_value < uct_value:
                        max_uct_value = uct_value
                        selected_node = child
            return selected_node

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld.selection_idea == 1:
            selected_node = self.subtree_node
            while len(selected_node.get_childs()) > 0:
                max_uct_value = -np.inf
                child_values = list(map(lambda n: n.get_avg_value() + n.reward_from_par, selected_node.get_childs()))
                max_child_value = max(child_values)
                min_child_value = min(child_values)

                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, selected_node.get_childs())))
                softmax_denominator = np.sum(np.exp(child_uncertainties / self.tau))
                softmax_uncertainties = np.exp(child_uncertainties / self.tau)/softmax_denominator

                for ind, child in enumerate(selected_node.get_childs()):
                    if child.num_visits == 0:
                        selected_node = child
                        break
                    else:

                        child_value = child_values[ind]
                        child_value = child.get_avg_value() + child.reward_from_par
                        child_uncertainty = child_uncertainties[ind]
                        softmax_uncertainty = softmax_uncertainties[ind]

                        if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                            child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                        elif min_child_value == max_child_value:
                            child_value = 0.5

                        uct_value = (child_value + \
                                     self.C * ((np.log(child.parent.num_visits) / child.num_visits) ** 0.5)) *\
                                    (1 - softmax_uncertainty)
                        # print("old:", uct_value - child_uncertainty, "  new:",uct_value, "  unc:", child_uncertainty)
                    if max_uct_value < uct_value:
                        max_uct_value = uct_value
                        selected_node = child
            return selected_node

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld.selection_idea == 2:
            selected_node = self.subtree_node
            while len(selected_node.get_childs()) > 0:
                max_uct_value = -np.inf
                child_values = list(map(lambda n: n.get_avg_value() + n.reward_from_par, selected_node.get_childs()))
                max_child_value = max(child_values)
                min_child_value = min(child_values)
                for ind, child in enumerate(selected_node.get_childs()):
                    if child.num_visits == 0:
                        selected_node = child
                        break
                    else:
                        child_value = child_values[ind]
                        if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                            child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                        uct_value = child_value + \
                                    self.C * ((np.log(child.parent.num_visits) / child.num_visits) ** 0.5)
                    if max_uct_value < uct_value:
                        max_uct_value = uct_value
                        selected_node = child

                state = selected_node.parent.get_state()
                next_state = selected_node.get_state()
                reward = torch.tensor([selected_node.reward_from_par], device=self.device)
                is_terminal = selected_node.is_terminal
                action_index = torch.tensor([self.getActionIndex(selected_node.action_from_par)]).unsqueeze(0)
                if not is_terminal:
                    self.updateTransitionBuffer(utils.transition(state,
                                                                 action_index,
                                                                 reward,
                                                                 next_state,
                                                                 None, False, self.time_step, 0))
                else:
                    self.updateTransitionBuffer(utils.transition(state,
                                                                 action_index,
                                                                 reward,
                                                                 None,
                                                                 None, False, self.time_step, 0))

            return selected_node
        else:
            return MCTSAgent.selection(self)

    def expansion(self, node):
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld.expansion_idea == 1:
            for a in self.action_list:
                action = torch.tensor(a).unsqueeze(0)
                next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                                          action)  # with the assumption of deterministic model
                # if np.array_equal(next_state, node.get_state()):
                #     continue
                value = self.get_initial_value(next_state)
                child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                             value=value, uncertainty=uncertainty.item())
                node.add_child(child)

                state = node.get_state()
                reward = torch.tensor([reward], device=self.device)
                action_index = torch.tensor([self.getActionIndex(a)]).unsqueeze(0)
                if not is_terminal:
                    transition = utils.transition(state,
                                                  action_index,
                                                  reward,
                                                  next_state,
                                                  None, False, self.time_step, 0)
                else:
                    transition = utils.transition(state,
                                                  action_index,
                                                  reward,
                                                  None,
                                                  None, False, self.time_step, 0)
                is_already_there = False
                # for t in self.transition_buffer:
                #     if np.array_equal(t.prev_state, transition.state) and \
                #             np.array_equal(t.prev_action, transition.prev_action):
                #         is_already_there = True
                if not is_already_there:
                    self.updateTransitionBuffer(transition)

                if self._vf['q']['training']:
                    if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                        transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                        self.updateValueFunction(transition_batch, 'q')
                if self._target_vf['counter'] >= self._target_vf['update_rate']:
                    self.setTargetValueFunction(self._vf['q'], 'q')

        else:
            for a in self.action_list:
                action = torch.tensor(a).unsqueeze(0)
                next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                                          action)  # with the assumption of deterministic model
                # if np.array_equal(next_state, node.get_state()):
                #     continue
                value = self.get_initial_value(next_state)
                child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                             value=value, uncertainty=uncertainty.item())
                node.add_child(child)

    def backpropagate(self, node, value):
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction_gridworld.backpropagate_idea == 1:
            while node is not None:
                node.add_to_values(value)
                node.inc_visits()
                if node.parent is not None:
                    siblings = node.parent.get_childs()
                    siblings_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, siblings)))
                    softmax_denominator = np.sum(np.exp(siblings_uncertainties))
                    value *= self.gamma
                    value += node.reward_from_par
                    value *= np.exp(node.uncertainty) / softmax_denominator
                node = node.parent
        else:
            while node is not None:
                node.add_to_values(value)
                node.inc_visits()
                value *= self.gamma
                value += node.reward_from_par
                node = node.parent

    def render_tree(self):
        def my_layout(node):
            F = TextFace(node.name, tight_text=True)
            add_face_to_node(F, node, column=0, position="branch-right")

        t = Tree()
        ts = TreeStyle()
        ts.show_leaf_name = False
        queue = [(self.subtree_node, None)]
        while queue:
            node, parent = queue.pop(0)
            uct_value = 0
            child_value = None
            child_uncertainty = 0
            if node.parent is not None:
                # child_values = list(map(lambda n: n.get_avg_value() + n.reward_from_par, node.parent.get_childs()))
                child_values = list(
                    map(lambda n: n.get_weighted_avg_value() + n.reward_from_par, node.parent.get_childs()))
                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, node.parent.get_childs())))
                softmax_denominator = np.sum(np.exp(child_uncertainties / self.tau))
                max_child_value = max(child_values)
                min_child_value = min(child_values)

                child_value = node.get_avg_value() + node.reward_from_par
                child_uncertainty = np.exp(node.uncertainty / self.tau) / softmax_denominator
                if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                    child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                elif min_child_value == max_child_value:
                    child_value = 0.5

                if node.num_visits == 0:
                    uct_value = np.inf
                else:
                    uct_value = (child_value + \
                                self.C * ((np.log(node.parent.num_visits) / node.num_visits) ** 0.5) ) *\
                                (1 - child_uncertainty)




            # node_face = str(node.get_state()) + "," + str(node.num_visits) + "," + str(node.get_avg_value()) \
            #             + "," + str(node.is_terminal) + "," + str(uct_value) + "," + str(node.uncertainty)
            node_face = str(node.get_state()[0].numpy()) + "," + str(node.num_visits) + "," + str(node.get_avg_value()) \
                        + "," + str(round(uct_value, 3)) + "," + str(child_uncertainty)
            if parent is None:
                p = t.add_child(name=node_face)
            else:
                p = parent.add_child(name=node_face)
            for child in node.get_childs():
                queue.append((child, p))

        ts.layout_fn = my_layout
        # t.render('t.png', tree_style=ts)
        # print(t.get_ascii(show_internal=Tree))
        t.show(tree_style=ts)

class ImperfectMCTSAgentIdeas(RealBaseDynaAgent, MCTSAgent):
    name = "ImperfectMCTSAgentIdeas"

    def __init__(self, params={}):
        self.model_loss = []
        self.time_step = 0

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
            self.goal_np = params['goal']

        self.num_steps = 0
        self.num_terminal_steps = 0

        self.is_model_pretrained = True

        # MCTS parameters
        self.C = params['c']
        self.num_iterations = params['num_iteration']
        self.num_rollouts = params['num_simulation']
        self.rollout_depth = params['simulation_depth']
        self.keep_subtree = False
        self.keep_tree = False
        self.root = None    

        self.transition_dynamics = params['transition_dynamics']

        # self.corrupt_prob = 0.25
        # self.corrupt_step = 7

        self.expansion_remove_child_prob = 0.

        self.rollout_uncertainty_action = False

        self.use_uncertainty_selection = False
        
        if is_gridWorld:
            self.transition_dynamics = params['transition_dynamics']
        else:
            self.true_model = params['true_fw_model']

    def start(self, observation):
        '''
        :param observation: numpy array -> (observation shape)
        :return: action : numpy array
        '''
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)

        self.prev_state = self.getStateRepresentation(observation)
        # temp_prev_action = torch.tensor([[0]], device=self.device, dtype=torch.long)
        # if self._model[self.model_type]['network'] is None and self._model[self.model_type]['training']:
        #     self.initModel(self.prev_state, temp_prev_action)

        self.initModel()

        if self.keep_tree and self.root is None:
            self.root = Node(None, self.prev_state[0])
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, self.prev_state[0])
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.prev_action = torch.tensor([[action]], device=self.device)
        return self.action_list[action]

    def step(self, reward, observation):
        self.time_step += 1

        self.state = self.getStateRepresentation(observation)

        self.saveModelNp()

        if not self.keep_subtree:
            self.subtree_node = Node(None, self.state[0])
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree

        self.action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        reward = torch.tensor([reward], device=self.device)

        # store the new transition in buffer
        self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     self.state,
                                                     self.action, False, self.time_step, 0))
        # # train/plan with model
        # if self._model[self.model_type]['training']:
        #     if len(self.transition_buffer) >= self._model[self.model_type]['batch_size']:
        #         self.trainModel()
        
        # self.plan()

        self.updateStateRepresentation()

        self.prev_state = self.getStateRepresentation(observation)
        self.prev_action = self.action  # another option:** we can again call self.policy function **

        return self.action_list[action]


    def end(self, reward):
        reward = torch.tensor([reward], device=self.device)

        self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     None,
                                                     None, True, self.time_step, 0))
        
        # if self._model[self.model_type]['training']:
        #     if len(self.transition_buffer) >= self._model[self.model_type]['batch_size']:
        #         self.trainModel()      
        self.updateStateRepresentation()

    def initModel(self):
        return
        self.saved_model_np = np.zeros([9, 9, 4, 2], dtype=int)
        self.saved_model_uncertainty_np = np.zeros([9, 9, 4, 1])
        self.saved_model_reward_np = np.zeros([9, 9, 4, 1])
        self.saved_model_is_terminal_np = np.zeros([9, 9, 4, 1])
        self.saveModelNp()

    def true_model(self, state, action_index):
        transition = self.transition_dynamics[int(state[0]), int(state[1]), action_index]
        next_state, is_terminal, reward = transition[0:2], transition[2], transition[3]
        return next_state, is_terminal, reward

    def saveModelNp(self):
        return
        with torch.no_grad():
            for s0 in range(9):
                for s1 in range(9):
                    for a in range(4):
                        # state = torch.tensor([[s0, s1]], device=self.device)
                        # action_index = torch.tensor([[a]], device=self.device)
                        next_state, is_terminal, reward, uncertainty = self.create_model([s0, s1], a)
                        self.saved_model_np[s0, s1, a] = next_state    
                        self.saved_model_uncertainty_np[s0, s1, a] = uncertainty
                        self.saved_model_reward_np[s0, s1, a] = reward
                        self.saved_model_is_terminal_np[s0, s1, a] = is_terminal

    def create_model(self, state, action_index):
        return
        # state_np = state.cpu().numpy()
        # action_index = action_index.cpu().numpy()[0, 0]
        true_next_state_np, is_terminal, reward = self.true_model(state, action_index)
        next_state = true_next_state_np
        r = random.random()
        if r < self.corrupt_prob:
            for _ in range(self.corrupt_step):
                action_index = random.randint(0, self.num_actions - 1)
                transition = self.transition_dynamics[int(state[0]), int(state[1]), action_index]
                next_state, is_terminal, reward = transition[0:2], transition[2], transition[3]
                state = next_state
        difference = true_next_state_np - next_state
        difference = np.sum(difference ** 2)
        uncertainty = math.sqrt(difference)
        # next_state = torch.from_numpy(next_state).to(self.device)
        return next_state, is_terminal, reward, uncertainty

    def model(self, state, action_index):
        # print(state, action_index)
        next_state, is_terminal, reward = self.true_model(state, action_index)

        # next_state = self.saved_model_np[state[0], state[1], action_index]
        # is_terminal = self.saved_model_is_terminal_np[state[0], state[1], action_index]
        # reward = self.saved_model_reward_np[state[0], state[1], action_index]
        # uncertainty = self.saved_model_uncertainty_np[state[0], state[1], action_index]
        return next_state, is_terminal, reward, 0


    def expansion(self, node):
        self.use_uncertainty_expansion_type = 'remove_most_uncertain'
        if self.use_uncertainty_expansion_type == 'remove_most_uncertain':
            expected_children = []
            max_uncertainty = -np.inf
            max_child_uncertainty_index = -1
            for a in range(self.num_actions):
                # action_index = torch.tensor([a]).unsqueeze(0)
                next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                                a)  # with the assumption of deterministic model
                value = self.get_initial_value(next_state)
                child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                            value=value, uncertainty=uncertainty)
                expected_children.append(child)
                if uncertainty > max_uncertainty:
                    max_child_uncertainty_index = a
                    max_uncertainty = uncertainty
            if random.random() < self.expansion_remove_child_prob:
                expected_children.pop(max_child_uncertainty_index)
            for i in range(len(expected_children)):
                node.add_child(expected_children[i])

    def rollout(self, node):
        sum_returns = 0
        for _ in range(self.num_rollouts):
            depth = 0
            single_return = 0
            is_terminal = node.is_terminal
            state = node.get_state()

            path_reward_list = []
            path_uncertainty_list = []
            while not is_terminal and depth < self.rollout_depth:
                if not self.rollout_uncertainty_action:
                    action_index = random.randint(0, self.num_actions - 1)
                else:
                    uncertainty_list = []
                    sum = 0.0
                    for a_index in range(self.num_actions):
                        # a_index = torch.tensor([a_index]).unsqueeze(0)
                        next_state, is_terminal, reward, uncertainty = self.model(state, a_index)
                        # print(next_state)
                        uncertainty_list.append(1 / (uncertainty + 10 ** -6))
                        sum += 1 / (uncertainty + 10 ** -6)
                    uncertainty_list = [x / sum for x in uncertainty_list]
                    action_index = np.random.choice(self.num_actions, 1, p = uncertainty_list)[0]
                # action_index = torch.tensor([action_index]).unsqueeze(0)
                next_state, is_terminal, reward, uncertainty = self.model(state, action_index)
                path_reward_list.append(reward)
                path_uncertainty_list.append(uncertainty)
                depth += 1
                state = next_state
            normalized_return = 0
            for i in range(len(path_reward_list) - 1, 0, -1):
                single_return += path_reward_list[i]
                # normalized_return += path_reward_list[i]
                # normalized_uncertainty = (4 - path_uncertainty_list[i]) / 100 + 0.96
                # normalized_uncertainty = 1
                # single_return *= normalized_uncertainty
            sum_returns += single_return
        return sum_returns / self.num_rollouts


    def selection(self):
        selected_node = self.subtree_node
        while len(selected_node.get_childs()) > 0:
            max_uct_value = -np.inf
            child_values = list(map(lambda n: n.get_avg_value()+n.reward_from_par, selected_node.get_childs()))
            child_uncertainties = list(map(lambda n: n.get_uncertainty(), selected_node.get_childs()))
            max_child_value = max(child_values)
            min_child_value = min(child_values)
            max_child_uncertainty = max(child_uncertainties)
            min_child_uncertainty = min(child_uncertainties)
            for ind, child in enumerate(selected_node.get_childs()):
                if child.num_visits == 0:
                    selected_node = child
                    break
                else:
                    child_value = child_values[ind]
                    child_uncertainty = child_uncertainties[ind]
                    if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                        child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                    if max_child_uncertainty != min_child_uncertainty:
                        child_uncertainty = (child_uncertainty - min_child_uncertainty) / (max_child_uncertainty - min_child_uncertainty)
                    uct_value = child_value + \
                                self.C * ((child.parent.num_visits / child.num_visits) ** 0.5)
                    if self.use_uncertainty_selection:
                        uct_value -= (2 ** 1) * (child_uncertainty)
                if max_uct_value < uct_value:
                    max_uct_value = uct_value
                    selected_node = child
        return selected_node


    def convert_uncertainty(self, uncertainty):
        self.max_uncertainty = 4
        self.min_uncertainty = 0
        self.uncertainty_conversion_type = 'linear'
        self.uncertainty_conversion_range = 0.1
        self.uncertainty_conversion_rate = 0.5

        uncertainty = (self.max_uncertainty - uncertainty) / (self.max_uncertainty - self.min_uncertainty)
        if self.uncertainty_conversion_type == 'linear':
            uncertainty = uncertainty
        elif self.uncertainty_conversion_type == 'power':
            uncertainty = uncertainty ** self.uncertainty_conversion_rate

        return uncertainty * self.uncertainty_conversion_range
