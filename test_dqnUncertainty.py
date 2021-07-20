import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from abc import abstractmethod
import random

import Utils as utils
from Agents.BaseAgent import BaseAgent
from Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN
from Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN_het
from Networks.ValueFunctionNN.StateValueFunction import StateVFNN
from Networks.RepresentationNN.StateRepresentation import StateRepresentation
from Networks.ValueFunctionNN.StateActionValueFunction import rnd_network
import pickle

from Experiments.GridWorldExperiment import GridWorldExperiment
from Environments.GridWorldRooms import GridWorldRooms

import Config as config
#this is an DQN agent.
writer = SummaryWriter()

class BaseDynaAgent(BaseAgent):
    name = 'BaseDynaAgent'
    global_step = 0

    def __init__(self, params={}):

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
        self.transition_buffer_size = 4096

        self.policy_values = 'q'  # 'q' or 's' or 'qs'
        self.policy_values = params['vf']['type']  # 'q' or 's' or 'qs'

        self._vf = {'q': dict(network=None,
                              layers_type=params['vf']['layers_type'],
                              layers_features=params['vf']['layers_features'],
                              action_layer_num=params['vf']['action_layer_num'],
                              # if one more than layer numbers => we will have num of actions output
                              batch_size=32,
                              step_size=params['max_stepsize'],
                              training=True),
                    's': dict(network=None,
                              layers_type=['fc'],
                              layers_features=[32],
                              batch_size=1,
                              step_size=0.01,
                              training=False)}

        self._sr = dict(network=None,
                        layers_type=[],
                        layers_features=[],
                        batch_size=None,
                        step_size=None,
                        batch_counter=None,
                        training=False)

        self._target_vf = dict(network=None,
                               counter=0,
                               layers_num=None,
                               action_layer_num=None,
                               update_rate=500,
                               type=None)

        self.reward_function = params['reward_function']
        self.device = params['device']
        if params['goal'] is not None:
            self.goal = torch.from_numpy(params['goal']).float().to(self.device)

        self.num_steps = 0
        self.num_terminal_steps = 0

        self.is_pretrained = False


    def start(self, observation):
        '''
        :param observation: numpy array -> (observation shape)
        :return: action : numpy array
        '''
        # print(self.num_terminal_steps, ' - ', self.num_steps)
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)

        self.prev_state = self.getStateRepresentation(observation)
        if self._vf['q']['network'] is None and self._vf['q']['training']:
            self.init_q_value_function_network(self.prev_state)  # a general state action VF for all actions
            self.init_rnd_network(self.prev_state)

        if self._vf['s']['network'] is None and self._vf['s']['training']:
            self.init_s_value_function_network(self.prev_state)  # a separate state VF for each action

        self.setTargetValueFunction(self._vf['q'], 'q')
        self.prev_action = self.policy(self.prev_state)
        self.initModel(self.prev_state)
        return self.action_list[self.prev_action.item()]

    def step(self, reward, observation):
        self.time_step += 1


        self.state = self.getStateRepresentation(observation)

        reward = torch.tensor([reward], device=self.device)
        self.action = self.policy(self.state)

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
                self.update_rnd_network(transition_batch)
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
                ind = self._vf['q']['network'](state)[0].max(1)[1].view(1, 1)
                return ind
            else:
                raise ValueError('policy is not defined')

    # ***
    def init_q_value_function_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''

        nn_state_shape = state.shape
        self._vf['q']['network'] = StateActionVFNN_het(nn_state_shape, self.num_actions,
                                                   self._vf['q']['layers_type'],
                                                   self._vf['q']['layers_features'],
                                                   self._vf['q']['action_layer_num']).to(self.device)
        #remove later
        if self.is_pretrained:
            value_function_file = "Results_EmptyRoom/DQNVF_16x8/dqn_vf_7.p"
            print("loading ", value_function_file)
            self.loadValueFunction(value_function_file)
            self._vf['q']['training'] = False
        self.optimizer = optim.Adam(self._vf['q']['network'].parameters(), lr=self._vf['q']['step_size'])

    def init_rnd_network(self, state):
        nn_state_shape = state.shape
        self.rnd_network_learner = rnd_network(nn_state_shape, self.num_actions,
                                            ['fc', 'fc'],
                                            [8, 8],
                                            3, mean=0.0).to(self.device)
        self.rnd_network_target = rnd_network(nn_state_shape, self.num_actions,
                                            ['fc', 'fc'],
                                            [8, 8],
                                            3, mean=0.0).to(self.device).eval()
        self.rnd_optimizer = optim.Adam(self.rnd_network_learner.parameters(), lr=0.01)


    def init_s_value_function_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''

        nn_state_shape = state.shape
        self._vf['s']['network'] = []
        for i in range(self.num_actions):
            self._vf['s']['network'].append(StateVFNN(nn_state_shape,
                                                      self._vf['s']['layers_type'],
                                                      self._vf['s']['layers_features']).to(self.device))

    def init_s_representation_network(self, observation):
        '''
        :param observation: numpy array
        :return: None
        '''

        nn_state_shape = observation.shape
        self._sr['network'] = StateRepresentation(nn_state_shape,
                                                  self._sr['layers_type'],
                                                  self._sr['layers_features']).to(self.device)

    # ***
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

        #BEGIN DQN
        state_action_values = self._vf['q']['network'](prev_state_batch)[0].gather(1, prev_action_batch)
        state_action_var = self._vf['q']['network'](prev_state_batch)[1].gather(1, prev_action_batch)
        next_state_values = torch.zeros(self._vf['q']['batch_size'], device=self.device)
        next_state_values[non_final_mask] = self._target_vf['network'](non_final_next_states)[0].max(1)[0].detach()
        #END DQN

        #BEGIN SARSA
        # non_final_next_actions = torch.cat([a for a in batch.action
        #                                    if a is not None])
        # state_action_values = self._vf['q']['network'](prev_state_batch).gather(1, prev_action_batch)
        # next_state_values = torch.zeros(self._vf['q']['batch_size'], device=self.device)
        # next_state_values[non_final_mask] = self._target_vf['network'](non_final_next_states).gather(1, non_final_next_actions).detach()[:, 0]
        #END SARSA

        expected_state_action_values = ((next_state_values * self.gamma) + reward_batch).unsqueeze(1)

        loss = torch.mean( ((expected_state_action_values - state_action_values) ** 2) / (2 * state_action_var) + 0.5 * torch.log(state_action_var))
        # loss = F.mse_loss(state_action_values,
        #                   expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._target_vf['counter'] += 1

    def getStateActionValue(self, state, action=None, vf_type='q', gradient=False):
        '''
        :param state: torch -> [1, state_shape]
        :param action: numpy array
        :param vf_type: str -> 'q' or 's'
        :param gradient: boolean
        :return: value: int
        '''

        if action is not None:
            action_index = self.getActionIndex(action)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

            if vf_type == 'q':
                if len(self._vf['q']['layers_type']) + 1 == self._vf['q']['action_layer_num']:
                    value = self._vf['q']['network'](state)[0].detach()[:, action_index] if not gradient \
                        else self._vf['q']['network'](state)[0][:, action_index]
                    var = self._vf['q']['network'](state)[1].detach()[:, action_index] if not gradient \
                        else self._vf['q']['network'](state)[1][:, action_index]
                else:
                    value = self._vf['q']['network'](state, action_onehot)[0].detach()[0] if not gradient \
                        else self._vf['q']['network'](state, action_onehot)[0][0]
                    var = self._vf['q']['network'](state, action_onehot)[1].detach()[0] if not gradient \
                        else self._vf['q']['network'](state, action_onehot)[1][0]

            elif vf_type == 's':
                value = self._vf['s']['network'][action_index](state).detach()[0] if not gradient \
                    else self._vf['s']['network'][action_index](state)[0]

            else:
                raise ValueError('state action value type is not defined')
            return value, var
        else:
            # state value (no gradient)
            if gradient:
                raise ValueError("cannot calculate the gradient for state values!")
            sum = 0
            for action in self.action_list:
                action_index = self.getActionIndex(action)
                action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

                if vf_type == 'q':
                    if len(self._vf['q']['layers_type']) + 1 == self._vf['q']['action_layer_num']:
                        value = self._vf['q']['network'](state).detach()[:, action_index]
                    else:
                        value = self._vf['q']['network'](state, action_onehot).detach()[0]

                elif vf_type == 's':
                    value = self._vf['s']['network'][action_index](state).detach()[0]

                else:
                    raise ValueError('state action value type is not defined')

                sum += value

            return sum / len(self.action_list)

    def update_rnd_network(self, transition_batch):
        batch = utils.transition(*zip(*transition_batch))

        prev_state_batch = torch.cat(batch.prev_state)
        prev_action_batch = torch.cat(batch.prev_action)
        reward_batch = torch.cat(batch.reward)

        state_action_rnd_target = self.rnd_network_target(prev_state_batch, prev_action_batch).gather(1, prev_action_batch)
        state_action_rnd_learner = self.rnd_network_learner(prev_state_batch, prev_action_batch).gather(1, prev_action_batch)
        loss = torch.mean((state_action_rnd_learner - state_action_rnd_target) ** 2)
        self.rnd_optimizer.zero_grad()
        loss.backward()
        writer.add_scalar("rnd_loss", loss, self.global_step)
        self.global_step += 1
        self.rnd_optimizer.step()

    def get_rnd_error(self, state, action):
        action_index = self.getActionIndex(action)
        state_action_rnd_target = self.rnd_network_target(state, action)[:, action_index].item()
        state_action_rnd_learner = self.rnd_network_learner(state, action)[:, action_index].item()
        error = (state_action_rnd_learner - state_action_rnd_target) ** 2
        return error

    # ***
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

    def updateStateRepresentation(self):

        if len(self._sr['layers_type']) == 0:
            return None
        if self._sr['batch_counter'] == self._sr['batch_size'] and self._sr['training']:
            self.updateNetworkWeights(self._sr['network'], self._sr['step_size'] / self._sr['batch_size'])
            self._sr['batch_counter'] = 0

    # ***
    def setTargetValueFunction(self, vf, type):

        if self._target_vf['network'] is None:
            nn_state_shape = self.prev_state.shape
            self._target_vf['network'] = StateActionVFNN_het(
                nn_state_shape,
                self.num_actions,
                vf['layers_type'],
                vf['layers_features'],
                vf['action_layer_num']).to(self.device)

        self._target_vf['network'].load_state_dict(vf['network'].state_dict())  # copy weights and stuff
        if type != 's':
            self._target_vf['action_layer_num'] = vf['action_layer_num']
        self._target_vf['layers_num'] = len(vf['layers_type'])
        self._target_vf['counter'] = 0
        self._target_vf['type'] = type

    def getTargetValue(self, state, action=None):
        '''
        :param state: torch -> (1, state_shape)
        :param action: numpy array
        :return value: int
        '''

        with torch.no_grad():
            if action is not None:
                action_index = self.getActionIndex(action)
                action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)
                if self._target_vf['type'] == 'q':
                    if self._target_vf['layers_num'] + 1 == self._target_vf['action_layer_num']:
                        value = self._target_vf['network'](state).detach()[:, action_index]
                    else:
                        value = self._target_vf['network'](state, action_onehot).detach()[0]

                elif self._target_vf['type'] == 's':
                    value = self._target_vf['network'][action_index](state).detach()[0]

                else:
                    raise ValueError('state action value type is not defined')
                return value

            else:
                # state value (no gradient)
                sum = 0
                for action in self.action_list:
                    action_index = self.getActionIndex(action)
                    action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

                    if self._target_vf['type'] == 'q':
                        if self._target_vf['layers_num'] + 1 == self._target_vf['action_layer_num']:
                            value = self._target_vf['network'](state).detach()[:, action_index]
                        else:
                            value = self._target_vf['network'](state, action_onehot).detach()[0]

                    elif self._target_vf['type'] == 's':
                        value = self._target_vf['network'][action_index](state).detach()

                    else:
                        raise ValueError('state action value type is not defined')

                    sum += value
                return sum / len(self.action_list)

    # ***
    def getTransitionFromBuffer(self, n):

        # both model and value function are using this buffer
        if len(self.transition_buffer) < n:
            n = len(self.transition_buffer)
        return random.sample(self.transition_buffer, k=n)

    def updateTransitionBuffer(self, transition):
        self.num_steps += 1
        if transition.is_terminal:
            self.num_terminal_steps += 1
        self.transition_buffer.append(transition)
        if len(self.transition_buffer) > self.transition_buffer_size:
            self.removeFromTransitionBuffer()

    def removeFromTransitionBuffer(self):
        self.num_steps -= 1
        transition = self.transition_buffer.pop(0)
        if transition.is_terminal:
            self.num_terminal_steps -= 1

    # ***
    def getActionIndex(self, action):

        for i, a in enumerate(self.action_list):
            if np.array_equal(a, action):
                return i
        raise ValueError("action is not defined")

    def getActionOnehot(self, action):

        res = np.zeros([len(self.action_list)])
        res[self.getActionIndex(action)] = 1
        return res

    def saveValueFunction(self, name):
        with open(name, "wb") as file:
            pickle.dump(self._vf, file)
    
    def loadValueFunction(self, name):
        with open(name, "rb") as file:
            self._vf = pickle.load(file)

    # ***
    @abstractmethod
    def trainModel(self):
        pass

    @abstractmethod
    def plan(self):
        pass

    @abstractmethod
    def initModel(self, state):
        pass

def print_qvalues(env, agent):
    for s in env.getAllStates():
        for a in env.getAllActions():
            state = torch.from_numpy(s).unsqueeze(0)
            value, var = agent.getStateActionValue(state, a)
            true_value = env.calculate_state_action_value(s, a, 0.99)
            rnd_error = agent.get_rnd_error(state, a)

            print(s, a, (value - true_value)**2, rnd_error, var)

def true_qvalues(env):
    for s in env.getAllStates():
        for a in env.getAllActions():
            value = env.calculate_state_action_value(s, a, 0.99)
            print(s, a, value)

if __name__ == "__main__":
    device = torch.device("cpu")
    num_episodes = 200
    env = GridWorldRooms(params=config.n_room_params)
    agent = BaseDynaAgent({'action_list': np.asarray(env.getAllActions()),
                            'gamma': 0.99, 'epsilon': 0.1,
                            'max_stepsize': 2**-7,
                            'model_stepsize': 0.001,
                            'reward_function': env.rewardFunction,
                            'goal': np.asarray(env.posToState((0, 8), state_type='coord')),
                            'device': device,
                            'model': None,
                            'true_bw_model': env.transitionFunctionBackward,
                            'true_fw_model': env.coordTransitionFunction,
                            'transition_dynamics':env.transition_dynamics,
                            'vf':  {'type': 'q', 'layers_type': ['fc', 'fc'], 'layers_features': [64, 64], 'action_layer_num': 3},})
    exp = GridWorldExperiment(agent, env, device)
    for i in range(num_episodes):
        print(i)
        exp.runEpisode(50)
    print_qvalues(env, agent)

