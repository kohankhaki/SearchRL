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

class ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction(RealBaseDynaAgent, MCTSAgent):
    name = "ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction"
    rollout_idea = None  # None, 1
    selection_idea = 1  # None, 1
    backpropagate_idea = None  # None, 1
    expansion_idea = None
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
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.rollout_idea == 1:
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

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.rollout_idea == 2:
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

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.rollout_idea == 3:
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

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.rollout_idea == 4:
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

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.rollout_idea == 5:
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
                rollout_path = []
                depth = 0
                single_return = 0
                is_terminal = node.is_terminal
                state = node.get_state()
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    try:
                        next_state, is_terminal, reward, _ = self.model(state, action)
                        rollout_path.append([state, next_state, is_terminal])
                    except:
                        for i in rollout_path:
                            with open("log.txt", "a") as file:
                                file.write("rolloutpath_state:"+str(i[0]))
                                file.write("rolloutpath_nextstate:"+ str(i[1]))
                                file.write("rolloutpath_terminal:"+str(i[2]))
                                file.write("____________")
                        exit(0)
                    single_return += reward
                    depth += 1
                    state = next_state

                sum_returns += single_return
            return sum_returns / self.num_rollouts

    def selection(self):
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.backpropagate_idea == 1:
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

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.selection_idea == 1:
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

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.selection_idea == 2:
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
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.expansion_idea == 1:
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
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.backpropagate_idea == 1:
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
