from ssl import RAND_pseudo_bytes
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from abc import abstractmethod
import random

import Utils as utils
from Agents.BaseAgent import BaseAgent
from Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN, StateActionVFNN_het, rnd_network
from Networks.ValueFunctionNN.StateValueFunction import StateVFNN
from Networks.RepresentationNN.StateRepresentation import StateRepresentation
import pickle

#this is an DQN agent.
class HetDQN(BaseAgent):
    name = 'HetDQN'

    def __init__(self, params={}):

        self.time_step = 0
        # self.writer = SummaryWriter()

        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.epsilon_max = params['epsilon_max']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon = self.epsilon_max

        self.transition_buffer = []
        self.transition_buffer_size = 4096

        self.policy_values = 'q'  # 'q' or 's' or 'qs'
        self.policy_values = params['vf']['type']  # 'q' or 's' or 'qs'

        self._vf = {'q': dict(network=None,
                              num_ensembles=params['vf']['num_ensembles'],
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
        print(self.epsilon)
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)

        self.prev_state = self.getStateRepresentation(observation)
        self.init_rnd_networks(self.prev_state)
        
        if self._vf['q']['network'] is None and self._vf['q']['training']:
            self.init_q_value_function_network(self.prev_state)  # a general state action VF for all actions

        if self._vf['s']['network'] is None and self._vf['s']['training']:
            self.init_s_value_function_network(self.prev_state)  # a separate state VF for each action

        self.setTargetValueFunction(self._vf['q'], 'q')
        self.prev_action = self.policy(self.prev_state)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon_max *=  self.epsilon_decay
        #     self.epsilon = self.epsilon_max

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
                self.trainRndNetworks(transition_batch)

        if self._vf['s']['training']:
            if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 's')


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
        
        self.updateStateRepresentation()

    def policy(self, state):
        '''
        :param state: torch -> (1, state_shape)
        :return: action: index torch
        '''
        self.num_steps += 1
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-1 * self.num_steps / self.epsilon_decay)
        if random.random() < self.epsilon:
            ind = torch.tensor([[random.randrange(self.num_actions)]],
                               device=self.device, dtype=torch.long)
            return ind
        with torch.no_grad():
            v = []
            if self.policy_values == 'q':
                mu, var = self._vf['q']['network'][0](state)
                # var = self._vf['q']['network'](state)[1]
                # sample = torch.normal(mu, var)
                ind = mu.max(1)[1].view(1, 1)
                return ind
            else:
                raise ValueError('policy is not defined')

    # ***
    def init_rnd_networks(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''

        nn_state_shape = state.shape

        self.rnd_target = rnd_network(nn_state_shape, self.num_actions,
                                                   ['fc', 'fc', 'fc'],
                                                   [256, 256, 256],
                                                   4).to(self.device)
        self.rnd_target.eval()
        self.rnd_learner = rnd_network(nn_state_shape, self.num_actions,
                                                   ['fc', 'fc'],
                                                   [256, 256],
                                                   3).to(self.device)


        self.rnd_optimizer = optim.Adam(self.rnd_learner.parameters(), lr=0.001)
    
    def init_q_value_function_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''

        nn_state_shape = state.shape
        self._vf['q']['network'] = []
        self.optimizer = []
        for i in range(self._vf['q']['num_ensembles']):
            self._vf['q']['network'].append(StateActionVFNN_het(nn_state_shape, self.num_actions,
                                                   self._vf['q']['layers_type'],
                                                   self._vf['q']['layers_features'],
                                                   self._vf['q']['action_layer_num']).to(self.device))
            self.optimizer.append(optim.Adam(self._vf['q']['network'][i].parameters(), lr=self._vf['q']['step_size']))


        # self.het_vf = StateActionVFNN_het(nn_state_shape, self.num_actions,
        #                                            self._vf['q']['layers_type'],
        #                                            self._vf['q']['layers_features'],
        #                                            self._vf['q']['action_layer_num']).to(self.device)
        # self.het_optimizer = optim.Adam(self.het_vf.parameters(), lr=self._vf['q']['step_size'])

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
        next_state_values = torch.zeros(self._vf['q']['batch_size'], device=self.device)
        next_state_values[non_final_mask] = self._target_vf['network'](non_final_next_states)[0].max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        for i in range(self._vf['q']['num_ensembles']):
            # state_action_values = self._vf['q']['network'][i](prev_state_batch).gather(1, prev_action_batch)
            state_action_values = self._vf['q']['network'][i](prev_state_batch)[0].gather(1, prev_action_batch)
            state_action_values_var = self._vf['q']['network'][i](prev_state_batch)[1].gather(1, prev_action_batch)
            loss = torch.mean( (state_action_values - expected_state_action_values.unsqueeze(1)) ** 2
                            / (2 * state_action_values_var) 
                            + 0.5 * torch.log(state_action_values_var)  
                        )
            self.optimizer[i].zero_grad()
            loss.backward()
            self.optimizer[i].step()
        #END DQN

        #BEGIN SARSA
        # non_final_next_actions = torch.cat([a for a in batch.action
        #                                    if a is not None])
        # state_action_values = self._vf['q']['network'](prev_state_batch).gather(1, prev_action_batch)
        # next_state_values = torch.zeros(self._vf['q']['batch_size'], device=self.device)
        # next_state_values[non_final_mask] = self._target_vf['network'](non_final_next_states).gather(1, non_final_next_actions).detach()[:, 0]
        #END SARSA

        # state_action_values = self.het_vf(prev_state_batch)[0].gather(1, prev_action_batch)
        # state_action_values_var = self.het_vf(prev_state_batch)[1].gather(1, prev_action_batch)
        # loss = torch.mean( (state_action_values - expected_state_action_values.unsqueeze(1)) ** 2
        #                     / (2 * state_action_values_var) 
        #                     + 0.5 * torch.log(state_action_values_var)  
        #                 )
        # self.het_optimizer.zero_grad()
        # loss.backward()
        # self.het_optimizer.step()
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
                    value = self._vf['q']['network'](state).detach()[:, action_index] if not gradient \
                        else self._vf['q']['network'](state)[:, action_index]
                else:
                    value = self._vf['q']['network'](state, action_onehot).detach()[0] if not gradient \
                        else self._vf['q']['network'](state, action_onehot)[0]

            elif vf_type == 's':
                value = self._vf['s']['network'][action_index](state).detach()[0] if not gradient \
                    else self._vf['s']['network'][action_index](state)[0]

            else:
                raise ValueError('state action value type is not defined')
            return value
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
        random_ensemble = np.random.randint(0, vf['num_ensembles'])
        self._target_vf['network'].load_state_dict(vf['network'][random_ensemble].state_dict())  # copy weights and stuff
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
        # self.num_steps += 1
        # if transition.is_terminal:
        #     self.num_terminal_steps += 1
        self.transition_buffer.append(transition)
        if len(self.transition_buffer) > self.transition_buffer_size:
            self.removeFromTransitionBuffer()

    def removeFromTransitionBuffer(self):
        # self.num_steps -= 1
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

    def checkHetValues(self, transition_batch):
        batch = utils.transition(*zip(*transition_batch))
        prev_state_batch = torch.cat(batch.prev_state)
        prev_action_batch = torch.cat(batch.prev_action)
        with torch.no_grad():
            state_action_values = self._vf['q']['network'](prev_state_batch).gather(1, prev_action_batch)
            state_action_het_values = self.het_vf(prev_state_batch)[0].gather(1, prev_action_batch)
            state_action_het_var = self.het_vf(prev_state_batch)[1].gather(1, prev_action_batch)
            value_diff = (state_action_values - state_action_het_values) ** 2
            het_error = torch.mean((value_diff - state_action_het_var) ** 2)
            print(het_error)

    def getRndError(self, state_batch, action_batch):
        with torch.no_grad():
            rnd_target = self.rnd_target(state_batch).gather(1, action_batch)
            rnd_learner = self.rnd_learner(state_batch).gather(1, action_batch)
            err = torch.sum((rnd_learner - rnd_target) ** 2, dim=1) ** 0.5
        return err
    
    def trainRndNetworks(self, transition_batch):
        batch = utils.transition(*zip(*transition_batch))
        state_batch = torch.cat(batch.prev_state)
        action_batch = torch.cat(batch.prev_action)
        rnd_target = self.rnd_target(state_batch).gather(1, action_batch)
        rnd_learner = self.rnd_learner(state_batch).gather(1, action_batch)
        loss = torch.mean( (rnd_target - rnd_learner) ** 2)
        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()

    def getEnsembleError(self, state_batch, action_batch):
        state_action_values = torch.zeros([self._vf['q']['num_ensembles'], len(state_batch)])
        with torch.no_grad():
            for i in range(self._vf['q']['num_ensembles']):
                value = self._vf['q']['network'][i](state_batch).gather(1, action_batch)     
                state_action_values[i] = value[:, 0]
        return torch.mean(state_action_values, axis=0), torch.std(state_action_values, axis=0)
    
    def getStateValueUncertainty(self, state_batch, type):
        if type == "ensemble":
            values = torch.zeros([self.num_actions, len(state_batch)])
            uncertainties = torch.zeros([self.num_actions, len(state_batch)])
            for i, a in enumerate(self.action_list):
                action_batch = torch.tensor([a]*len(state_batch)).unsqueeze(1)
                value, uncertainty = self.getEnsembleError(state_batch, action_batch)
                values[i] = value
                uncertainties[i] = uncertainty
            value = torch.mean(values, axis=0)
            uncertainty = torch.mean(uncertainties, axis=0) 
            return value, uncertainty
        
        elif type == "rnd":
            values = torch.zeros([self.num_actions, len(state_batch)])
            uncertainties = torch.zeros([self.num_actions, len(state_batch)])
            with torch.no_grad():
                for i, a in enumerate(self.action_list):
                    action_batch = torch.tensor([a]*len(state_batch)).unsqueeze(1)
                    uncertainty = self.getRndError(state_batch, action_batch)
                    value, _ = self.getEnsembleError(state_batch, action_batch)
                    values[i] = value
                    uncertainties[i] = uncertainty
                value = torch.mean(values, axis=0)
                uncertainty = torch.mean(uncertainties, axis=0) 
                return value, uncertainty
        
        elif type == "het":
            values = torch.zeros([self.num_actions, len(state_batch)])
            uncertainties = torch.zeros([self.num_actions, len(state_batch)])
            with torch.no_grad():
                for i, a in enumerate(self.action_list):
                    action_batch = torch.tensor([a]*len(state_batch)).unsqueeze(1)
                    value = self.het_vf(state_batch)[0].gather(1, action_batch)
                    uncertainty = self.het_vf(state_batch)[1].gather(1, action_batch)
                    values[i] = value[:, 0]
                    uncertainties[i] = uncertainty[:, 0]
                value = torch.mean(values, axis=0)
                uncertainty = torch.mean(uncertainties, axis=0) 
                return value, uncertainty
        else:
            raise ValueError("The type of uncertainty is not known")