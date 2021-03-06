import numpy as np
import torch
import random
import gc
from torch.utils.tensorboard import SummaryWriter
from ete3 import Tree, TreeStyle, TextFace, add_face_to_node

from Agents.BaseAgent import BaseAgent
from DataStructures.Node_Torch import Node_Torch as Node
from profilehooks import timecall, profile, coverage
from Networks.RepresentationNN.StateRepresentation import StateRepresentation



#Warning: for other environments check the true model
is_gridWorld = False
class MCTSAgent_Torch(BaseAgent):
    name = "MCTSAgent_Torch"

    def __init__(self, params={}):

        self.time_step = 0
        # self.writer = SummaryWriter()

        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon']

        self.device = params['device']

        if is_gridWorld:
            self.transition_dynamics = params['transition_dynamics']
        else:
            self.true_model = params['true_fw_model']

        # MCTS parameters
        self.C = params['c']
        self.num_iterations = params['num_iteration']
        self.num_rollouts = params['num_simulation']
        self.rollout_depth = params['simulation_depth']
        self.keep_subtree = False
        self.keep_tree = False
        self.root = None

        self.is_model_imperfect = False
        self.corrupt_prob = 0.025
        self.corrupt_step = 1

        self._sr = dict(network=None,
                        layers_type=[],
                        layers_features=[],
                        batch_size=None,
                        step_size=None,
                        batch_counter=None,
                        training=False)
    
    def start(self, observation, info=None):
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)
        
        state = self.getStateRepresentation(observation)
        

        if self.keep_tree and self.root is None:
            self.root = Node(None, state)
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree
        return action

    def step(self, reward, observation):

        state = self.getStateRepresentation(observation)
        if not self.keep_subtree:
            self.subtree_node = Node(None, state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree
        return action

    def end(self, reward):
        pass

    def get_initial_value(self, state):
        return 0

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

    def MCTS_iteration(self):
        selected_node = self.selection()
        if selected_node.is_terminal:
            self.backpropagate(selected_node, 0)
        elif selected_node.num_visits == 0:
            rollout_value = self.rollout(selected_node)
            self.backpropagate(selected_node, rollout_value)
        else:
            self.expansion(selected_node)
            rollout_value = self.rollout(selected_node.get_childs()[0])
            self.backpropagate(selected_node.get_childs()[0], rollout_value)

    @timecall(immediate=False)
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
        return selected_node

    # @timecall(immediate=False)
    def expansion(self, node):
        for a in range(self.num_actions):
            action_index = torch.tensor([a]).unsqueeze(0)
            next_state, is_terminal, reward = self.model(node.get_state(),
                                                              action_index)  # with the assumption of deterministic model
            # if np.array_equal(next_state, node.get_state()):
            #     continue
            value = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                         value=value)
            node.add_child(child)

    # @timecall(immediate=False)
    def rollout(self, node):
        sum_returns = 0
        for i in range(self.num_rollouts):
            depth = 0
            single_return = 0
            is_terminal = node.is_terminal
            state = node.get_state()
            while not is_terminal and depth < self.rollout_depth:
                action_index = torch.randint(0, self.num_actions, (1, 1))
                # action_index = torch.randint(0, self.num_actions, (1, 1), device=self.device)
                next_state, is_terminal, reward = self.model(state, action_index)
                single_return += reward
                depth += 1
                state = next_state

            sum_returns += single_return
        return sum_returns / self.num_rollouts

    @timecall(immediate=False)
    def backpropagate(self, node, value):
        while node is not None:
            node.add_to_values(value)
            node.inc_visits()
            value *= self.gamma
            value += node.reward_from_par
            node = node.parent

    def true_model(self, state, action_index):
        print("true model")
        transition = self.transition_dynamics[int(state[0]), int(state[1]), action_index]
        next_state, is_terminal, reward = transition[0:2], transition[2], transition[3]
        if self.is_model_imperfect:
            r = random.random()
            if r < self.corrupt_prob:
                for _ in range(self.corrupt_step):
                    action_index = random.randint(0, self.num_actions - 1)
                    transition = self.transition_dynamics[int(state[0]), int(state[1]), action_index]
                    next_state, is_terminal, reward = transition[0:2], transition[2], transition[3]
                    state = next_state
        return next_state, is_terminal, reward

    def model(self, state, action_index):
        state_np = state.cpu().numpy()
        action_index = action_index.cpu().numpy()
        next_state_np, is_terminal, reward = self.true_model(state_np[0], action_index.item())
        next_state = torch.from_numpy(next_state_np).unsqueeze(0).to(self.device)
        return next_state, is_terminal, reward

    def getActionIndex(self, action):
        if is_gridWorld:
            if action[0] == 0:
                if action[1] == 1:
                    return 2
                else:
                    return 0
            elif action[0] == 1:
                return 3
            else:
                return 1
        for i, a in enumerate(self.action_list):
            if np.array_equal(a, action):
                return i
        raise ValueError("action is not defined")

    def updateStateRepresentation(self):
        if len(self._sr['layers_type']) == 0:
            return None
        if self._sr['batch_counter'] == self._sr['batch_size'] and self._sr['training']:
            self.updateNetworkWeights(self._sr['network'], self._sr['step_size'] / self._sr['batch_size'])
            self._sr['batch_counter'] = 0

    def init_s_representation_network(self, observation):
        '''
        :param observation: numpy array
        :return: None
        '''

        nn_state_shape = observation.shape
        self._sr['network'] = StateRepresentation(nn_state_shape,
                                                  self._sr['layers_type'],
                                                  self._sr['layers_features']).to(self.device)


    def getStateRepresentation(self, observation, gradient=False):
        '''
        :param observation: numpy array -> [obs_shape]
        :param gradient: boolean
        :return: torch including batch -> [1, state_shape]
        '''
        if gradient:
            self._sr['batch_counter'] += 1
        observation = torch.tensor([observation], device=self.device)
        if gradient:
            rep = self._sr['network'](observation)
        else:
            with torch.no_grad():
                rep = self._sr['network'](observation)
        return rep




# import numpy as np
# import torch
# import random
# import gc
# from torch.utils.tensorboard import SummaryWriter
# from ete3 import Tree, TreeStyle, TextFace, add_face_to_node

# from Agents.BaseAgent import BaseAgent
# from DataStructures.Node_Torch import Node_Torch as Node
# from profilehooks import timecall, profile, coverage
# from Networks.RepresentationNN.StateRepresentation import StateRepresentation



# #Warning: for other environments check the true model
# is_gridWorld = True
# class MCTSAgent_Torch(BaseAgent):
#     name = "MCTSAgent_Torch"

#     def __init__(self, params={}):

#         self.time_step = 0
#         # self.writer = SummaryWriter()

#         self.prev_state = None
#         self.state = None

#         self.action_list = params['action_list']
#         self.num_actions = self.action_list.shape[0]

#         self.gamma = params['gamma']
#         self.epsilon = params['epsilon']

#         self.device = params['device']

#         if is_gridWorld:
#             self.transition_dynamics = params['transition_dynamics']
#         else:
#             self.true_model = params['true_fw_model']
#         # MCTS parameters
#         self.C = params['c']
#         self.num_iterations = params['num_iteration']
#         self.num_rollouts = params['num_simulation']
#         self.rollout_depth = params['simulation_depth']
#         self.keep_subtree = False
#         self.keep_tree = False
#         self.root = None

#         self.is_model_imperfect = False
#         self.corrupt_prob = 0.025
#         self.corrupt_step = 1

#         self._sr = dict(network=None,
#                         layers_type=[],
#                         layers_features=[],
#                         batch_size=None,
#                         step_size=None,
#                         batch_counter=None,
#                         training=False)

    
#     def start(self, observation):
        
#         if self._sr['network'] is None:
#             self.init_s_representation_network(observation)
        
#         state = self.getStateRepresentation(observation)

#         if self.keep_tree and self.root is None:
#             self.root = Node(None, state)
#             self.expansion(self.root)

#         if self.keep_tree:
#             self.subtree_node = self.root
#         else:
#             self.subtree_node = Node(None, state)
#             self.expansion(self.subtree_node)

#         for i in range(self.num_iterations):
#             self.MCTS_iteration()
#         action, sub_tree = self.choose_action()
#         self.subtree_node = sub_tree
#         return action

#     def step(self, reward, observation):
#         state = self.getStateRepresentation(observation)
#         if not self.keep_subtree:
#             self.subtree_node = Node(None, state)
#             self.expansion(self.subtree_node)

#         for i in range(self.num_iterations):
#             self.MCTS_iteration()
#         action, sub_tree = self.choose_action()
#         self.subtree_node = sub_tree
#         return action

#     def end(self, reward):
#         pass

#     def get_initial_value(self, state):
#         return 0

#     def choose_action(self):
#         max_visit = -np.inf
#         max_action_list = []
#         max_child_list = []
#         for child in self.subtree_node.get_childs():
#             if child.num_visits > max_visit:
#                 max_visit = child.num_visits
#                 max_action_list = [child.get_action_from_par()]
#                 max_child_list = [child]
#             elif child.num_visits == max_visit:
#                 max_action_list.append(child.get_action_from_par())
#                 max_child_list.append(child)
#         random_ind = random.randint(0, len(max_action_list) - 1)
#         return max_action_list[random_ind], max_child_list[random_ind]

# #     @timecall(immediate=False)
#     def MCTS_iteration(self):
#         # self.render_tree()
#         selected_node = self.selection()
#         # now we decide to expand the leaf or rollout
#         if selected_node.is_terminal:
#             self.backpropagate(selected_node, 0)
#         elif selected_node.num_visits == 0:  # don't expand just roll-out
#             rollout_value = self.rollout(selected_node)
#             self.backpropagate(selected_node, rollout_value)
#         else:  # expand then roll_out
#             self.expansion(selected_node)
#             rollout_value = self.rollout(selected_node.get_childs()[0])
#             self.backpropagate(selected_node.get_childs()[0], rollout_value)


# #     @timecall(immediate=False)
#     def selection(self):
#         selected_node = self.subtree_node
#         while len(selected_node.get_childs()) > 0:
#             max_uct_value = -np.inf
#             child_values = list(map(lambda n: n.get_avg_value()+n.reward_from_par, selected_node.get_childs()))
#             max_child_value = max(child_values)
#             min_child_value = min(child_values)
#             for ind, child in enumerate(selected_node.get_childs()):
#                 if child.num_visits == 0:
#                     selected_node = child
#                     break
#                 else:
#                     child_value = child_values[ind]
#                     if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
#                         child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
#                     uct_value = child_value + \
#                                 self.C * ((child.parent.num_visits / child.num_visits) ** 0.5)
#                 if max_uct_value < uct_value:
#                     max_uct_value = uct_value
#                     selected_node = child
#         return selected_node

# #     @timecall(immediate=False)
#     def expansion(self, node):
#         for a in range(self.num_actions):
#             action_index = torch.tensor([a], device=self.device).unsqueeze(0)
#             next_state, is_terminal, reward = self.model(node.get_state(),
#                                                               action_index)  # with the assumption of deterministic model
#             # if np.array_equal(next_state, node.get_state()):
#             #     continue
#             value = self.get_initial_value(next_state)
#             child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
#                          value=value)
#             node.add_child(child)

# #     @timecall(immediate=False)
#     def rollout(self, node):
#         sum_returns = 0
#         for i in range(self.num_rollouts):
#             depth = 0
#             single_return = 0
#             is_terminal = node.is_terminal
#             state = node.get_state()
#             while not is_terminal and depth < self.rollout_depth:
#                 action_index = torch.randint(0, self.num_actions, (1, 1), device=self.device)
#                 next_state, is_terminal, reward = self.model(state, action_index)
#                 single_return += reward
#                 depth += 1
#                 state = next_state
#             sum_returns += single_return
#         return sum_returns / self.num_rollouts

# #     @timecall(immediate=False)
#     def backpropagate(self, node, value):
#         while node is not None:
#             node.add_to_values(value)
#             node.inc_visits()
#             value *= self.gamma
#             value += node.reward_from_par
#             node = node.parent

#     def true_model(self, state, action_index):
#         transition = self.transition_dynamics[int(state[0]), int(state[1]), action_index]
#         next_state, is_terminal, reward = transition[0:2], transition[2], transition[3]
#         if self.is_model_imperfect:
#             r = random.random()
#             if r < self.corrupt_prob:
#                 for _ in range(self.corrupt_step):
#                     action_index = random.randint(0, self.num_actions - 1)
#                     transition = self.transition_dynamics[int(state[0]), int(state[1]), action_index]
#                     next_state, is_terminal, reward = transition[0:2], transition[2], transition[3]
#                     state = next_state
#         return next_state, is_terminal, reward

#     def model(self, state, action_index):
#         state_np = state.cpu().numpy()
#         action_index = action_index.cpu().numpy()
#         next_state_np, is_terminal, reward = self.true_model(state_np, action_index)
#         next_state = torch.from_numpy(next_state_np).to(self.device)
#         return next_state, is_terminal, reward

#     def show(self):
#         queue = [self.subtree_node, "*"]
#         while queue:
#             node = queue.pop(0)
#             if node == "*":
#                 print("********")
#                 continue
#             node.show()
#             for child in node.get_childs():
#                 queue.append(child)
#             if len(node.get_childs()) > 0:
#                 queue.append("*")

#     def render_tree(self):
#         def my_layout(node):
#             F = TextFace(node.name, tight_text=True)
#             add_face_to_node(F, node, column=0, position="branch-right")

#         t = Tree()
#         ts = TreeStyle()
#         ts.show_leaf_name = False
#         queue = [(self.subtree_node, None)]
#         while queue:
#             node, parent = queue.pop(0)
#             uct_value = 0
#             if node.parent is not None:
#                 child_values = list(map(lambda n: n.get_avg_value() + n.reward_from_par, node.parent.get_childs()))
#                 max_child_value = max(child_values)
#                 min_child_value = min(child_values)
#                 child_value = node.get_avg_value()
#                 if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
#                     child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
#                 if node.num_visits == 0:
#                     uct_value = np.inf
#                 else:
#                     uct_value = child_value + \
#                                 self.C * ((node.parent.num_visits / node.num_visits) ** 0.5)



#             node_face = str(node.get_state()) + "," + str(node.num_visits) + "," + str(node.get_avg_value()) \
#                         + "," + str(node.is_terminal) + "," + str(uct_value)
#             if parent is None:
#                 p = t.add_child(name=node_face)
#             else:
#                 p = parent.add_child(name=node_face)
#             for child in node.get_childs():
#                 queue.append((child, p))

#         ts.layout_fn = my_layout
#         # t.render('t.png', tree_style=ts)
#         # print(t.get_ascii(show_internal=Tree))
#         t.show(tree_style=ts)

#     def getActionIndex(self, action):
#         # print(action)
#         if is_gridWorld:
#             if action[0] == 0:
#                 if action[1] == 1:
#                     return 2
#                 else:
#                     return 0
#             elif action[0] == 1:
#                 return 3
#             else:
#                 return 1
#         for i, a in enumerate(self.action_list):
#             if np.array_equal(a, action):
#                 return i
#         raise ValueError("action is not defined")

#     def updateStateRepresentation(self):

#         if len(self._sr['layers_type']) == 0:
#             return None
#         if self._sr['batch_counter'] == self._sr['batch_size'] and self._sr['training']:
#             self.updateNetworkWeights(self._sr['network'], self._sr['step_size'] / self._sr['batch_size'])
#             self._sr['batch_counter'] = 0

#     def init_s_representation_network(self, observation):
#         '''
#         :param observation: numpy array
#         :return: None
#         '''

#         nn_state_shape = observation.shape
#         self._sr['network'] = StateRepresentation(nn_state_shape,
#                                                   self._sr['layers_type'],
#                                                   self._sr['layers_features']).to(self.device)


#     def getStateRepresentation(self, observation, gradient=False):
#         '''
#         :param observation: numpy array -> [obs_shape]
#         :param gradient: boolean
#         :return: torch including batch -> [1, state_shape]
#         '''
#         if gradient:
#             self._sr['batch_counter'] += 1
#         observation = torch.tensor([observation], device=self.device)
#         if gradient:
#             rep = self._sr['network'](observation)
#         else:
#             with torch.no_grad():
#                 rep = self._sr['network'](observation)
#         return rep