from Agents.RealBaseDynaAgent import RealBaseDynaAgent
import numpy as np
import torch
import math 
import random
import gc
import Utils as utils
from torch.utils.tensorboard import SummaryWriter
from ete3 import Tree, TreeStyle, TextFace, add_face_to_node

from Agents.BaseAgent import BaseAgent
from Agents.RealBaseDynaAgent import RealBaseDynaAgent
from Agents.MCTSAgent_Torch import MCTSAgent_Torch as MCTSAgent
from DataStructures.Node_Torch import Node_Torch as Node

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

    # @timecall(immediate=False)
    def model(self, state, action_index):
        #prev
        # next_state = self.modelRollout(state, action_index)[0]
        # # next_state = torch.clamp(next_state, min=0, max=8)
        # next_state, reward, is_terminal = self.get_transition(next_state)
        #prev
        with torch.no_grad():
            state0 = int(state[0].item())
            state1 = int(state[1].item())
            action_index = action_index[0, 0].item()
            next_state = self.saved_model[state0, state1, action_index]
            uncertainty = self.saved_uncertainty[state0, state1, action_index][0].item()
            next_state, reward, is_terminal = self.get_transition(next_state)
            #if want not rounded next_state, replace next_state with _  
        
        return next_state, is_terminal, reward, uncertainty

    # @timecall(immediate=False)
    def rollout(self, node):
        sum_returns = 0
        for i in range(self.num_rollouts):
            depth = 0
            single_return = 0
            is_terminal = node.is_terminal
            state = node.get_state().cpu().numpy()
            while not is_terminal and depth < self.rollout_depth:
                action_index = random.randint(0, self.num_actions - 1)
                next_state, is_terminal, reward, uncertainry = self.model_np(state, action_index)
                single_return += reward
                depth += 1
                state = next_state
            sum_returns += single_return
        return sum_returns / self.num_rollouts

    def model_np(self, state, action_index):
        # state = np.clip(state, 0, 8)
        next_state = self.saved_model_np[int(state[0]), int(state[1]), action_index]
        next_state, reward, is_terminal = self.get_transition_np(next_state)
        uncertainty = self.saved_model_uncertainty_np[int(state[0]), int(state[1]), action_index][0]
        return next_state, is_terminal, reward, uncertainty

    def get_torch_action_index(self, action_index):
        return torch.tensor([action_index], device=self.device).unsqueeze(0)

    def get_transition(self, state):
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
        RealBaseDynaAgent.initModel(self, state, action)
        if self.is_model_pretrained:
            RealBaseDynaAgent.loadModelFile(self, "M32x16_r1e2999s100_heter.p")
        self.saved_model = torch.zeros([9, 9, 4, 2], device=self.device)
        self.saved_uncertainty = torch.zeros([9, 9, 4, 1], device=self.device)
        self.saved_model_np = np.zeros([9, 9, 4, 2])
        self.saved_model_uncertainty_np = np.zeros([9, 9, 4, 1])
        self.saveModel()
        self.saveModelNp()

    def saveModel(self):
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
        with torch.no_grad():
            for s0 in range(9):
                for s1 in range(9):
                    for a in range(4):
                        state = torch.tensor([[s0, s1]], device=self.device)
                        action_index = torch.tensor([[a]], device=self.device)
                        next_state, uncertainty = self.modelRollout(state, action_index)
                        self.saved_model_np[s0, s1, a] = next_state[0].cpu().numpy()    
                        self.saved_model_uncertainty_np[s0, s1, a] = uncertainty

    def expansion(self, node):
        for a in range(self.num_actions):
            action_index = torch.tensor([a]).unsqueeze(0)
            next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                              action_index)  # with the assumption of deterministic model
            value = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                         value=value, uncertainty=uncertainty)
            node.add_child(child)




class ImperfectMCTSAgentUncertainty(ImperfectMCTSAgent):
    name = "ImperfectMCTSAgentUncertainty"

    def __init__(self, params={}):
        ImperfectMCTSAgent.__init__(self, params)

    def start(self, observation):
        return ImperfectMCTSAgent.start(self, observation)

    def step(self, reward, observation):
        return ImperfectMCTSAgent.step(self, reward, observation)

    def end(self, reward):
        ImperfectMCTSAgent.end(self, reward)

    def expansion(self, node):
        expected_children = []
        max_uncertainty = -np.inf
        max_child_uncertainty_index = -1
        for a in range(self.num_actions):
            action_index = torch.tensor([a]).unsqueeze(0)
            next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                              action_index)  # with the assumption of deterministic model
            value = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                         value=value, uncertainty=uncertainty)
            expected_children.append(child)
            if uncertainty > max_uncertainty:
                max_child_uncertainty_index = a
                max_uncertainty = uncertainty
        if random.random() < 0.0:
            expected_children.pop(max_child_uncertainty_index)
        for i in range(len(expected_children)):
            node.add_child(expected_children[i])


    def rollout(self, node):
        sum_returns = 0
        for i in range(self.num_rollouts):
            depth = 0
            single_return = 0
            is_terminal = node.is_terminal
            state = node.get_state().cpu().numpy()
            while not is_terminal and depth < self.rollout_depth:
                uncertainty_list = []
                action_index_list = []
                sum = 0.0
                for a_index in range(self.num_actions):
                    next_state, is_terminal, reward, uncertainry = self.model_np(state, a_index)
                    uncertainty_list.append(1 - uncertainry)
                    sum += 1 - uncertainry
                uncertainty_list /= sum
                action_index = np.random.choice(4, 1, p = uncertainty_list)[0]
                next_state, is_terminal, reward, uncertainry = self.model_np(state, action_index)
                single_return += reward
                depth += 1
                state = next_state
            sum_returns += single_return
        return sum_returns / self.num_rollouts


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

        self.corrupt_prob = 0.25
        self.corrupt_step = 3

        self.expansion_remove_child_prob = 0.0

        self.rollout_random_action = False


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

    def true_model(self, state, action_index):
        transition = self.transition_dynamics[int(state[0]), int(state[1]), action_index]
        next_state, is_terminal, reward = transition[0:2], transition[2], transition[3]
        return next_state, is_terminal, reward

    def model(self, state, action_index):
        state_np = state.cpu().numpy()
        action_index = action_index.cpu().numpy()[0, 0]
        true_next_state_np, is_terminal, reward = self.true_model(state_np, action_index)
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
        next_state = torch.from_numpy(next_state).to(self.device)
        return next_state, is_terminal, reward, uncertainty

    def expansion(self, node):
        expected_children = []
        max_uncertainty = -np.inf
        max_child_uncertainty_index = -1
        for a in range(self.num_actions):
            action_index = torch.tensor([a]).unsqueeze(0)
            next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                              action_index)  # with the assumption of deterministic model
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
            while not is_terminal and depth < self.rollout_depth:
                uncertainty_list = []
                if self.rollout_random_action:
                    action_index = random.randint(0, self.num_actions - 1)
                else:
                    sum = 0.0
                    for a_index in range(self.num_actions):
                        a_index = torch.tensor([a_index]).unsqueeze(0)
                        next_state, is_terminal, reward, uncertainry = self.model(state, a_index)
                        uncertainty_list.append(1 / (uncertainry + 10 ** -6))
                        sum += 1 / (uncertainry + 10 ** -6)
                    uncertainty_list = [x / sum for x in uncertainty_list]
                    action_index = np.random.choice(self.num_actions, 1, p = uncertainty_list)[0]
                action_index = torch.tensor([action_index]).unsqueeze(0)
                next_state, is_terminal, reward, _ = self.model(state, action_index)
                single_return += reward
                depth += 1
                state = next_state
            sum_returns += single_return
        return sum_returns / self.num_rollouts