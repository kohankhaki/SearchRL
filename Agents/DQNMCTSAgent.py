import random
import torch
import numpy as np
import pickle

from Agents.MCTSAgent import MCTSAgent
from Agents.BaseDynaAgent import BaseDynaAgent
from DataStructures.Node import Node
import Utils as utils
import Config as config

from profilehooks import timecall, profile, coverage

episodes_only_dqn = config.episodes_only_dqn
episodes_only_mcts = config.episodes_only_mcts


# MCTS uses DQN values for nodes initialization
class DQNMCTSAgent_InitialValue(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_InitialValue"

    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        self.episode_counter = -1

    def start(self, observation):
        self.episode_counter += 1
        if self.episode_counter < episodes_only_dqn:
            action = BaseDynaAgent.start(self, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.start(self, observation)
        else:
            if self.episode_counter % 2 == 0:
                action = BaseDynaAgent.start(self, observation)
            else:
                action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter < episodes_only_dqn:
            action = BaseDynaAgent.step(self, reward, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.step(self, reward, observation)
        else:
            if self.episode_counter % 2 == 0:
                action = BaseDynaAgent.step(self, reward, observation)
            else:
                action = MCTSAgent.step(self, reward, observation)
        return action

    def end(self, reward):
        if self.episode_counter < episodes_only_dqn:
            BaseDynaAgent.end(self, reward)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            MCTSAgent.end(self, reward)
        else:
            if self.episode_counter % 2 == 0:
                BaseDynaAgent.end(self, reward)
            else:
                MCTSAgent.end(self, reward)

    def get_initial_value(self, state):
        state_representation = self.getStateRepresentation(state)
        value = self.getStateActionValue(state_representation)
        return value.item()

class DQNMCTSAgent_InitialValue_offline(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_InitialValue"

    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        with open("dqn_vf_4by4.p",'rb') as file:
            self._vf = pickle.load(file)
        self.episode_counter = -1

    def start(self, observation):
        self.episode_counter += 1
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)
        action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        action = MCTSAgent.step(self, reward, observation)
        return action

    def end(self, reward):
        pass

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
        if self.episode_counter < episodes_only_dqn:
            action = BaseDynaAgent.start(self, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.start(self, observation)
        else:
            if self.episode_counter % 2 == 0:
                action = BaseDynaAgent.start(self, observation)
            else:
                action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter < episodes_only_dqn:
            action = BaseDynaAgent.step(self, reward, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.step(self, reward, observation)
        else:
            if self.episode_counter % 2 == 0:
                action = BaseDynaAgent.step(self, reward, observation)
            else:
                action = MCTSAgent.step(self, reward, observation)
        return action

    def end(self, reward):
        if self.episode_counter < episodes_only_dqn:
            BaseDynaAgent.end(self, reward)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            MCTSAgent.end(self, reward)
        else:
            if self.episode_counter % 2 == 0:
                BaseDynaAgent.end(self, reward)
            else:
                MCTSAgent.end(self, reward)

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
        if self.episode_counter % 2 == 0:
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
        while len(selected_node.get_childs()) > 0:
            max_uct_value = -np.inf
            child_values = list(map(lambda n: n.get_avg_value()+n.reward_from_par, selected_node.get_childs()))
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
                                self.C * ((child.parent.num_visits / child.num_visits) ** 0.5)
                if max_uct_value < uct_value:
                    max_uct_value = uct_value
                    selected_node = child
            prev_state_torch = self.getStateRepresentation(selected_node.parent.state)
            prev_action_index = self.getActionIndex(selected_node.action_from_par)
            prev_action_torch = torch.tensor([prev_action_index], device=self.device).view(1, 1)
            reward = torch.tensor([selected_node.reward_from_par], device=self.device)
            state_torch = self.getStateRepresentation(selected_node.state)
            self.updateTransitionBuffer(utils.transition(prev_state_torch,
                                                         prev_action_torch,
                                                         reward,
                                                         state_torch,
                                                         None, selected_node.is_terminal, self.time_step, 0))
            
        return selected_node

    def selection_SARSA(self):
        selected_node = self.subtree_node
        buffer_prev_state = None
        buffer_prev_action = None
        buffer_reward = None
        reward = 0
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
                                self.C * ((child.parent.num_visits / child.num_visits) ** 0.5)
                if max_uct_value < uct_value:
                    max_uct_value = uct_value
                    selected_node = child

            if selected_node.is_terminal:
                buffer_prev_state_t = self.getStateRepresentation(selected_node.parent.state)
                prev_action_index_t = self.getActionIndex(selected_node.action_from_par)
                buffer_prev_action_t = torch.tensor([prev_action_index_t], device=self.device).view(1, 1)
                buffer_reward_t = torch.tensor([selected_node.reward_from_par], device=self.device)
                buffer_state_t = None
                buffer_action_t = None
                is_terminal_t = True
                self.updateTransitionBuffer(utils.transition(buffer_prev_state_t,
                                                         buffer_prev_action_t,
                                                         buffer_reward_t,
                                                         buffer_state_t,
                                                         buffer_action_t, is_terminal_t, self.time_step, 0))

            buffer_state = self.getStateRepresentation(selected_node.parent.state)
            action_index = self.getActionIndex(selected_node.action_from_par)
            buffer_action = torch.tensor([action_index], device=self.device).view(1, 1)

            if buffer_prev_state is not None:
                reward = selected_node.parent.reward_from_par
                if selected_node.parent.is_terminal:
                    raise ValueError("ridididididididididirididiridiririri")
                self.updateTransitionBuffer(utils.transition(buffer_prev_state,
                                                         buffer_prev_action,
                                                         buffer_reward,
                                                         buffer_state,
                                                         buffer_action, selected_node.parent.is_terminal, self.time_step, 0))
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



# DQN uses actions chosen by MCTS iterations
class DQNMCTSAgent_UseSelectedAction(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_UseSelectedAction"
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
            self.mcts_prev_state = self.getStateRepresentation(observation)
            self.mcts_prev_action = action
        return action

    def step(self, reward, observation):
        if self.episode_counter % 2 == 0:
            self.time_step += 1

            self.state = self.getStateRepresentation(observation)
            self.action = self.policy(self.state)
            
            reward = torch.tensor([reward], device=self.device)

            # store the new transition in buffer
            self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     self.state,
                                                     self.action, False, self.time_step, 0))

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
            prev_action_index = self.getActionIndex(self.mcts_prev_action)
            prev_action_torch = torch.tensor([prev_action_index], device=self.device, dtype=int).view(1, 1)
            reward = torch.tensor([reward], device=self.device).float()
            state_torch = self.getStateRepresentation(observation)
            self.updateTransitionBuffer(utils.transition(self.mcts_prev_state,
                                                         prev_action_torch,
                                                         reward,
                                                         state_torch,
                                                         None, False, self.time_step, 0))
            self.mcts_prev_state = state_torch
            self.mcts_prev_action = action

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
        return action

    def end(self, reward):
        if self.episode_counter % 2 == 0:
            reward = torch.tensor([reward], device=self.device)
            self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     None,
                                                     None, True, self.time_step, 0))

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
        else:
            prev_action_index = self.getActionIndex(self.mcts_prev_action)
            prev_action_torch = torch.tensor([prev_action_index], device=self.device, dtype=int).view(1, 1)
            reward = torch.tensor([reward], device=self.device).float()
            self.updateTransitionBuffer(utils.transition(self.mcts_prev_state,
                                                         prev_action_torch,
                                                         reward,
                                                         None,
                                                         None, True, self.time_step, 0))

            if self._vf['q']['training']:
                if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                    transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                    self.updateValueFunction(transition_batch, 'q')
            if self._vf['s']['training']:
                if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                    transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                    self.updateValueFunction(transition_batch, 's')
    
    def rollout(self, node):
        if self.episode_counter > 200:
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
        else:
            return MCTSAgent.rollout(self, node)

    def get_initial_value(self, state):
        if self.episode_counter > 200:
            state_representation = self.getStateRepresentation(state)
            value = self.getStateActionValue(state_representation)
            return value.item()
        else:
            return 0


# DQN uses MCTS as the policy
class DQNMCTSAgent_MCTSSelectedAction(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_MCTSSelectedAction"

    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        self.episode_counter = -1

    def start(self, observation):
        self.episode_counter += 1
        if self.keep_tree and self.root is None:
            self.root = Node(None, observation)
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, observation)
            self.expansion(self.subtree_node)

        action = BaseDynaAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if not self.keep_subtree:
            self.subtree_node = Node(None, observation)
            self.expansion(self.subtree_node)


        self.time_step += 1

        self.state = self.getStateRepresentation(observation)

        reward = torch.tensor([reward], device=self.device)
        self.action = self.policy(self.state)

        # store the new transition in buffer
        if self.episode_counter % 2 == 1:
            self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                         self.prev_action,
                                                         reward,
                                                         self.state,
                                                         self.action, False, self.time_step, 0))
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

        return self.action_list[self.prev_action.item()]


    def end(self, reward):
        reward = torch.tensor([reward], device=self.device)
        if self.episode_counter % 2 == 1:
            self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                         self.prev_action,
                                                         reward,
                                                         None,
                                                         None, True, self.time_step, 0))


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

    def policy(self, state):
        if self.episode_counter % 2 == 1:
            action, sub_tree = None, None
            for i in range(self.num_iterations):
                action, sub_tree = self.MCTS_iteration()
            # self.render_tree()
            self.subtree_node = sub_tree
            action = torch.from_numpy(np.array([self.getActionIndex(action)])).unsqueeze(0).to(self.device)
        else:
            action = BaseDynaAgent.policy(self, state)
        return action

    # def rollout(self, node):
    #     state = node.get_state()
    #     t_state = torch.from_numpy(state).unsqueeze(0).to(self.device)
    #     value = self.getStateActionValue(t_state)
    #     return value


# MCTS uses DQN values for bootstrap nodes initialization
class DQNMCTSAgent_BootstrapInitial(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_BootstrapInitial"

    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        self.episode_counter = -1

    def start(self, observation):
        self.episode_counter += 1
        if self.episode_counter < episodes_only_dqn:
            action = BaseDynaAgent.start(self, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.start(self, observation)
        else:
            if self.episode_counter % 2 == 0:
                action = BaseDynaAgent.start(self, observation)
            else:
                action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter < episodes_only_dqn:
            action = BaseDynaAgent.step(self, reward, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.step(self, reward, observation)
        else:
            if self.episode_counter % 2 == 0:
                action = BaseDynaAgent.step(self, reward, observation)
            else:
                action = MCTSAgent.step(self, reward, observation)
        return action

    def end(self, reward):
        if self.episode_counter < episodes_only_dqn:
            BaseDynaAgent.end(self, reward)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            MCTSAgent.end(self, reward)
        else:
            if self.episode_counter % 2 == 0:
                BaseDynaAgent.end(self, reward)
            else:
                MCTSAgent.end(self, reward)

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

    def get_initial_value(self, state):
        state_representation = self.getStateRepresentation(state)
        value = self.getStateActionValue(state_representation)
        return value.item()


# MCTS uses DQN values for bootstrap nodes initialization
class DQNMCTSAgent_Rollout(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_Rollout"

    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        self.episode_counter = -1

    def start(self, observation):
        self.episode_counter += 1
        if self.episode_counter < episodes_only_dqn:
            action = BaseDynaAgent.start(self, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.start(self, observation)
        else:
            if self.episode_counter % 2 == 0:
                action = BaseDynaAgent.start(self, observation)
            else:
                action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter < episodes_only_dqn:
            action = BaseDynaAgent.step(self, reward, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.step(self, reward, observation)
        else:
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
                a = self.rollout_policy(state)
                next_state, is_terminal, reward = self.true_model(state, a)
                single_return += reward
                depth += 1
                state = next_state
            sum_returns += single_return
        return sum_returns / self.num_rollouts

    def rollout_policy(self, state):
        # random policy
        # action = random.choice(self.action_list)

        # DQNs policy
        state = self.getStateRepresentation(state)


        action_ind = BaseDynaAgent.policy(self, state)
        action = self.action_list[action_ind.item()]


        return action


# DQN uses Tree made by MCTS tree trajectories, but with a priority
class DQNMCTSAgent_UseMCTSwPriority(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_UseMCTSwPriority"

    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        self.episode_counter = -1
        self.td_average = 0
        self.average_rate = 0.1
        self.mcts_count = 0

    def start(self, observation):
        print(self.mcts_count)
        self.mcts_count = 0


        self.episode_counter += 1
        if self.episode_counter % 2 == 0:
            action = BaseDynaAgent.start(self, observation)
        else:
            action = MCTSAgent.start(self, observation)
        return action



    def step(self, reward, observation):
        self.time_step += 1
        if self.episode_counter % 2 == 0:

            self.state = self.getStateRepresentation(observation)
            self.action = self.policy(self.state)
            
            reward = torch.tensor([reward], device=self.device)

            with torch.no_grad():
                real_prev_action = self.action_list[self.prev_action.item()]
                prev_state_value = self.getStateActionValue(self.prev_state, real_prev_action).item()
                state_value = self._vf['q']['network'](self.state).max(1)[1].view(1, 1).item()                
                td_error = reward.item() + self.gamma * state_value - prev_state_value
                self.update_average_td_error(td_error)

            # store the new transition in buffer
            self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     self.state,
                                                     self.action, False, self.time_step, 0))

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
        if self.episode_counter % 2 == 0:

            real_prev_action = self.action_list[self.prev_action.item()]
            prev_state_value = self.getStateActionValue(self.prev_state, real_prev_action).item()
            td_error = reward - prev_state_value
            self.update_average_td_error(td_error)

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
        while len(selected_node.get_childs()) > 0:
            max_uct_value = -np.inf
            child_values = list(map(lambda n: n.get_avg_value()+n.reward_from_par, selected_node.get_childs()))
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
                                self.C * ((child.parent.num_visits / child.num_visits) ** 0.5)
                if max_uct_value < uct_value:
                    max_uct_value = uct_value
                    selected_node = child
            prev_state_torch = self.getStateRepresentation(selected_node.parent.state)
            prev_action_index = self.getActionIndex(selected_node.action_from_par)
            prev_action_torch = torch.tensor([prev_action_index], device=self.device).view(1, 1)
            reward = torch.tensor([selected_node.reward_from_par], device=self.device)
            state_torch = self.getStateRepresentation(selected_node.state)
            with torch.no_grad():
                real_prev_action = self.action_list[prev_action_torch.item()]
                prev_state_value = self.getStateActionValue(prev_state_torch, real_prev_action).item()
                state_value = 0
                if not selected_node.is_terminal:
                    state_value = self._vf['q']['network'](state_torch).max(1)[1].view(1, 1).item()
                    state_torch = state_torch.float()            
                td_error = reward.item() + self.gamma * state_value - prev_state_value
                if (td_error >= self.td_average):
                    self.updateTransitionBuffer(utils.transition(prev_state_torch.float(),
                                                    prev_action_torch,
                                                    reward.float(),
                                                    state_torch,
                                                    None, selected_node.is_terminal, self.time_step, 0))
                    self.mcts_count += 1
                    self.update_average_td_error(td_error)            
        return selected_node



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
            with torch.no_grad():
                real_prev_action = self.action_list[buffer_prev_action.item()]
                prev_state_value = self.getStateActionValue(buffer_prev_state, real_prev_action).item()
                state_value = 0
                if not is_terminal:
                    state_value = self._vf['q']['network'](buffer_state).max(1)[1].view(1, 1).item()  
                    buffer_state = buffer_state.float()          
                td_error = buffer_reward.item() + self.gamma * state_value - prev_state_value
                if (td_error >= self.td_average):
                    self.updateTransitionBuffer(utils.transition(buffer_prev_state.float(),
                                                    buffer_prev_action,
                                                    buffer_reward.float(),
                                                    buffer_state,
                                                    None, is_terminal, self.time_step, 0))
                    self.mcts_count += 1

                    self.update_average_td_error(td_error)     
    
    def update_average_td_error(self, td_error):
        self.td_average = self.td_average * (1 - self.average_rate) + np.abs(td_error) * self.average_rate
        self.td_average = 1000
        # print(self.td_average)


# MCTS uses DQN value function for for reducing breadth of the search
class DQNMCTSAgent_ReduceBreadth(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent_ReduceBreadth"

    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        self.episode_counter = -1
        self.branch_factor = 3

    def start(self, observation):
        self.episode_counter += 1
        if self.episode_counter < episodes_only_dqn:
            action = BaseDynaAgent.start(self, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.start(self, observation)
        else:
            if self.episode_counter % 2 == 0:
                action = BaseDynaAgent.start(self, observation)
            else:
                action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter < episodes_only_dqn:
            action = BaseDynaAgent.step(self, reward, observation)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            action = MCTSAgent.step(self, reward, observation)
        else:
            if self.episode_counter % 2 == 0:
                action = BaseDynaAgent.step(self, reward, observation)
            else:
                action = MCTSAgent.step(self, reward, observation)
        return action

    def end(self, reward):
        if self.episode_counter < episodes_only_dqn:
            BaseDynaAgent.end(self, reward)
        elif self.episode_counter < episodes_only_dqn + episodes_only_mcts:
            MCTSAgent.end(self, reward)
        else:
            if self.episode_counter % 2 == 0:
                BaseDynaAgent.end(self, reward)
            else:
                MCTSAgent.end(self, reward)

    def expansion(self, node):
        children_list = []
        sort_list = []
        for a in self.action_list:
            next_state, is_terminal, reward = self.true_model(node.get_state(),
                                                              a)  # with the assumption of deterministic model
            # if np.array_equal(next_state, node.get_state()):
            #     continue
            value = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                         value=value)
            children_list.append(child)
            sort_value = self.get_state_value(next_state)
            sort_list.append(sort_value)

        children_list = [x for _, x in sorted(zip(sort_list, children_list), key=lambda pair: pair[0], reverse=True)]
        for i in range(self.branch_factor):
            node.add_child(children_list[i])

    def get_state_value(self, state):
        state_representation = self.getStateRepresentation(state)
        value = self.getStateActionValue(state_representation)
        return value.item()