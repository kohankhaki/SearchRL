import random
import torch
import numpy as np

from Agents.MCTSAgent import MCTSAgent
from Agents.BaseDynaAgent import BaseDynaAgent
from DataStructures.Node import Node
import Utils as utils


# MCTS uses DQN values for nodes initialization
class DQNMCTSAgent_InitialValue(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_InitialValue"

    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        self.episode_counter = -1

    def start(self, observation):
        self.episode_counter += 1
        if self.episode_counter % 2 == 0:
            action = BaseDynaAgent.start(self, observation)
        else:
            action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter % 2 == 0:
            action = BaseDynaAgent.step(self, reward, observation)
        else:
            action = MCTSAgent.step(self, reward, observation)

        return action

    def end(self, reward):
        BaseDynaAgent.end(self, reward)

    def get_initial_value(self, state):
        state_representation = self.getStateRepresentation(state)
        value = self.getStateActionValue(state_representation)
        return value.item()


# MCTS uses DQN values for bootstrap
class DQNMCTSAgent_Bootstrap(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_Bootstrap"

    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        self.episode_counter = -1

    def start(self, observation):
        self.episode_counter += 1
        if self.episode_counter % 2 == 0:
            action = BaseDynaAgent.start(self, observation)
        else:
            action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter % 2 == 0:
            action = BaseDynaAgent.step(self, reward, observation)
        else:
            action = MCTSAgent.step(self, reward, observation)

        return action

    def end(self, reward):
        BaseDynaAgent.end(self, reward)

    def rollout(self, node):
        sum_returns = 0
        for i in range(self.num_rollouts):
            depth = 0
            single_return = 0
            is_terminal = False
            state = node.get_state()
            while not is_terminal and depth < self.rollout_depth:
                a = random.choice(self.action_list)
                next_state, is_terminal, reward = self.true_model(state, a)
                single_return += reward
                depth += 1
                state = next_state
            if not is_terminal:
                state_representation = self.getStateRepresentation(state)
                bootstrap_value = self.getStateActionValue(state_representation)
                single_return += bootstrap_value.item()
            sum_returns += single_return

        return sum_returns / self.num_rollouts


# DQN uses Tree made by MCTS selection tree
class DQNMCTSAgent_UseTreeSelection(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_UseTreeSelection"

    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        self.episode_counter = -1

    def start(self, observation):
        self.episode_counter += 1
        if self.episode_counter % 2 == 0:
            action = BaseDynaAgent.start(self, observation)
        else:
            action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter % 2 == 0:
            self.time_step += 1

            self.state = self.getStateRepresentation(observation)
            self.action = self.policy(self.state)

            # update target
            if self._target_vf['counter'] >= self._target_vf['update_rate']:
                self.setTargetValueFunction(self._vf['q'], 'q')
                # self.setTargetValueFunction(self._vf['s'], 's')

            # update value function with the buffer
            if self._vf['q']['training']:
                if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                    transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                    self.updateValueFunction(transition_batch, 'q')
            if self._vf['s']['training']:
                if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                    transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                    self.updateValueFunction(transition_batch, 's')

            # train/plan with model
            self.trainModel()
            self.plan()

            self.updateStateRepresentation()

            self.prev_state = self.getStateRepresentation(observation)
            self.prev_action = self.action  # another option:** we can again call self.policy function **

            action = self.action_list[self.prev_action.item()]
        else:
            action = MCTSAgent.step(self, reward, observation)

        return action

    def end(self, reward):
        if self._vf['q']['training']:
            if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')
        if self._vf['s']['training']:
            if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 's')

        self.trainModel()
        self.updateStateRepresentation()

    def selection(self):
        selected_node = self.subtree_node
        buffer_prev_state = None
        buffer_prev_action = None
        buffer_reward = None
        while len(selected_node.get_childs()) > 0:
            max_uct_value = -np.inf
            child_values = list(map(lambda n: n.get_avg_value(), selected_node.get_childs()))
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
                                self.C * ((selected_node.num_visits / child.num_visits) ** 0.5)
                if max_uct_value < uct_value:
                    max_uct_value = uct_value
                    selected_node = child


            if selected_node.is_terminal:
                buffer_prev_state = self.getStateRepresentation(selected_node.parent.state)
                prev_action_index = self.getActionIndex(selected_node.action_from_par)
                buffer_prev_action = torch.tensor([prev_action_index], device=self.device).view(1, 1)
                buffer_reward = torch.tensor([selected_node.reward_from_par], device=self.device)
                buffer_state = None
                buffer_action = None
                is_terminal = True
                self.updateTransitionBuffer(utils.transition(buffer_prev_state,
                                                         buffer_prev_action,
                                                         buffer_reward,
                                                         buffer_state,
                                                         buffer_action, is_terminal, self.time_step, 0))



            buffer_state = self.getStateRepresentation(selected_node.parent.state)
            action_index = self.getActionIndex(selected_node.action_from_par)
            buffer_action = torch.tensor([action_index], device=self.device).view(1, 1)
            reward = selected_node.reward_from_par
            # store the new transition in buffer


            if buffer_prev_state is not None:
                if selected_node.parent.is_terminal:
                    buffer_state = None
                    buffer_action = None
                self.updateTransitionBuffer(utils.transition(buffer_prev_state,
                                                         buffer_prev_action,
                                                         buffer_reward,
                                                         buffer_state,
                                                         buffer_action, selected_node.parent.is_terminal, self.time_step, 0))
                # print('--------------Transition---------------')
                # print(buffer_prev_state)
                # print(buffer_prev_action)
                # print(buffer_reward)
                # print(buffer_state)
                # print(buffer_action)
                if selected_node.parent.is_terminal:
                    print(selected_node.parent.is_terminal)
            buffer_prev_state = buffer_state
            buffer_prev_action = buffer_action
            buffer_reward = torch.tensor([reward], device=self.device)

        return selected_node


# DQN uses Tree made by MCTS expansion transitions
class DQNMCTSAgent_UseTreeExpansion(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_UseTreeExpansion"

    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        self.episode_counter = -1

    def start(self, observation):
        self.episode_counter += 1
        if self.episode_counter % 2 == 0:
            action = BaseDynaAgent.start(self, observation)
        else:
            action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter % 2 == 0:
            self.time_step += 1

            self.state = self.getStateRepresentation(observation)
            self.action = self.policy(self.state)

            # update target
            if self._target_vf['counter'] >= self._target_vf['update_rate']:
                self.setTargetValueFunction(self._vf['q'], 'q')
                # self.setTargetValueFunction(self._vf['s'], 's')

            # update value function with the buffer
            if self._vf['q']['training']:
                if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                    transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                    self.updateValueFunction(transition_batch, 'q')
            if self._vf['s']['training']:
                if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                    transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                    self.updateValueFunction(transition_batch, 's')

            # train/plan with model
            self.trainModel()
            self.plan()

            self.updateStateRepresentation()

            self.prev_state = self.getStateRepresentation(observation)
            self.prev_action = self.action  # another option:** we can again call self.policy function **

            action = self.action_list[self.prev_action.item()]
        else:
            action = MCTSAgent.step(self, reward, observation)

        return action

    def end(self, reward):
        if self._vf['q']['training']:
            if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')
        if self._vf['s']['training']:
            if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 's')

        self.trainModel()
        self.updateStateRepresentation()

    def expansion(self, node):
        for a in self.action_list:
            next_state, is_terminal, reward = self.true_model(node.get_state(),
                                                              a)  # with the assumption of deterministic model
            # if np.array_equal(next_state, node.get_state()):
            #     continue
            value = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                         value=value)
            node.add_child(child)

            buffer_prev_state = self.getStateRepresentation(node.get_state())
            act_ind = self.getActionIndex(a)
            buffer_prev_action = torch.tensor([act_ind], device=self.device).view(1, 1)
            buffer_reward = torch.tensor([reward], device=self.device)
            buffer_state = None
            buffer_action = None
            if not is_terminal:
                buffer_state = self.getStateRepresentation(next_state)
                buffer_action = self.policy(buffer_state)
            self.updateTransitionBuffer(utils.transition(buffer_prev_state,
                                                         buffer_prev_action,
                                                         buffer_reward,
                                                         buffer_state,
                                                         buffer_action, is_terminal,
                                                         self.time_step, 0))

