import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from abc import abstractmethod
import random
from profilehooks import timecall, profile, coverage

import Utils as utils
from Agents.BaseAgent import BaseAgent
from Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN
from Networks.ValueFunctionNN.StateValueFunction import StateVFNN
from Networks.RepresentationNN.StateRepresentation import StateRepresentation
from Networks.ModelNN.StateTransitionModel import StateTransitionModel
from Networks.ModelNN.StateTransitionModel import StateTransitionModelHeter
import pickle


# this is an DQN agent.
class RealBaseDynaAgent(BaseAgent):
    name = 'RealBaseDynaAgent'

    def __init__(self, params={}):
        self.model_loss = []
        self.time_step = 0
        self.writer = SummaryWriter()
        self.writer_iterations = 0
        self.prev_state = None
        self.state = None

        self.dataset = params['dataset']

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
                              layers_type=params['vf']['layers_type'],
                              layers_features=params['vf']['layers_features'],
                              action_layer_num=params['vf']['action_layer_num'],
                              # if one more than layer numbers => we will have num of actions output
                              batch_size=16,
                              step_size=params['max_stepsize'],
                              training=True),
                    's': dict(network=None,
                              layers_type=['fc'],
                              layers_features=[32],
                              batch_size=1,
                              step_size=0.01,
                              training=False)}
        self.model_type = params['model']['type']
        self._model = {self.model_type: dict(network=None,
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

    def start(self, observation, info=None):
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

        if self._vf['s']['network'] is None and self._vf['s']['training']:
            self.init_s_value_function_network(self.prev_state)  # a separate state VF for each action

        self.setTargetValueFunction(self._vf['q'], 'q')
        self.prev_action = self.policy(self.prev_state)

        if self._model[self.model_type]['network'] is None and self._model[self.model_type]['training']:
            self.initModel(self.prev_state, self.prev_action)

        return self.action_list[self.prev_action.item()]

    def step(self, reward, observation, info=None):
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
        if self._vf['s']['training']:
            if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 's')

        # train/plan with model
        if self._model[self.model_type]['training']:
            if len(self.transition_buffer) >= self._model[self.model_type]['batch_size']:
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

        if self._model[self.model_type]['training']:
            if len(self.transition_buffer) >= self._model[self.model_type]['batch_size']:
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
                ind = self._vf['q']['network'](state).max(1)[1].view(1, 1)
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
        self._vf['q']['network'] = StateActionVFNN(nn_state_shape, self.num_actions,
                                                   self._vf['q']['layers_type'],
                                                   self._vf['q']['layers_features'],
                                                   self._vf['q']['action_layer_num']).to(self.device)
        # remove later
        if self.is_pretrained:
            value_function_file = "Results_EmptyRoom/DQNVF_16x8/dqn_vf_7.p"
            print("loading ", value_function_file)
            self.loadValueFunction(value_function_file)
            self._vf['q']['training'] = False
        self.optimizer = optim.Adam(self._vf['q']['network'].parameters(), lr=self._vf['q']['step_size'])

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

        # BEGIN DQN
        state_action_values = self._vf['q']['network'](prev_state_batch).gather(1, prev_action_batch)
        next_state_values = torch.zeros(self._vf['q']['batch_size'], device=self.device)
        next_state_values[non_final_mask] = self._target_vf['network'](non_final_next_states).max(1)[0].detach()
        # END DQN

        # BEGIN SARSA
        # non_final_next_actions = torch.cat([a for a in batch.action
        #                                    if a is not None])
        # state_action_values = self._vf['q']['network'](prev_state_batch).gather(1, prev_action_batch)
        # next_state_values = torch.zeros(self._vf['q']['batch_size'], device=self.device)
        # next_state_values[non_final_mask] = self._target_vf['network'](non_final_next_states).gather(1, non_final_next_actions).detach()[:, 0]
        # END SARSA

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = F.mse_loss(state_action_values,
                          expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._target_vf['counter'] += 1

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
            self._target_vf['network'] = StateActionVFNN(
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

    def getActionOnehotTorch(self, action):
        '''
        action = index torch
        output = onehot torch
        '''
        batch_size = action.shape[0]
        num_actions = len(self.action_list)
        onehot = torch.zeros([batch_size, num_actions], device=self.device)
        # onehot.zero_()
        onehot.scatter_(1, action, 1)
        return onehot

    def saveValueFunction(self, name):
        with open(name, "wb") as file:
            pickle.dump(self._vf, file)

    def loadValueFunction(self, name):
        with open(name, "rb") as file:
            self._vf = pickle.load(file)

    def saveModelFile(self, name):
        with open(name, "wb") as file:
            pickle.dump(self._model, file)

    def loadModelFile(self, name):
        with open(name, "rb") as file:
            self._model = pickle.load(file)

    # ***
    @abstractmethod
    def trainModel(self):
        if self.model_type == 'general':
            transition_batch = self.getTransitionFromBuffer(n=self._model['general']['batch_size'])
            batch = utils.transition(*zip(*transition_batch))

            non_final_next_states_batch = torch.cat([s for s in batch.state
                                                     if s is not None])
            non_final_prev_states_batch = torch.cat([s for s, t in zip(batch.prev_state, transition_batch)
                                                     if t.state is not None])
            non_final_prev_action_batch = torch.cat([a for a, t in zip(batch.prev_action, transition_batch)
                                                     if t.state is not None])
            non_final_prev_action_onehot_batch = self.getActionOnehotTorch(non_final_prev_action_batch)
            predicted_next_state = self._model['general']['network'](non_final_prev_states_batch,
                                                                     non_final_prev_action_onehot_batch)

            loss = F.mse_loss(predicted_next_state.float(),
                              non_final_next_states_batch.float())
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_loss.append(loss)
            self.model_optimizer.step()

        elif self.model_type == 'ensemble':
            for i in range(self.num_ensembles):
                transition_batch = self.getTransitionFromBuffer(n=self._model['ensemble']['batch_size'])
                batch = utils.transition(*zip(*transition_batch))
                non_final_next_states_batch = torch.cat([s for s in batch.state
                                                         if s is not None])
                non_final_prev_states_batch = torch.cat([s for s, t in zip(batch.prev_state, transition_batch)
                                                         if t.state is not None])
                non_final_prev_action_batch = torch.cat([a for a, t in zip(batch.prev_action, transition_batch)
                                                         if t.state is not None])
                non_final_prev_action_onehot_batch = self.getActionOnehotTorch(non_final_prev_action_batch)
                predicted_next_state = self._model['ensemble']['network'][i](non_final_prev_states_batch,
                                                                             non_final_prev_action_onehot_batch)
                loss = F.mse_loss(predicted_next_state.float(),
                                  non_final_next_states_batch.float())
                self.model_optimizer[i].zero_grad()
                loss.backward()
                self.model_loss.append(loss)
                self.model_optimizer[i].step()

        elif self.model_type == 'heter':
            transition_batch = self.getTransitionFromBuffer(n=self._model['heter']['batch_size'])
            batch = utils.transition(*zip(*transition_batch))
            non_final_next_states_batch = torch.cat([s for s in batch.state
                                                     if s is not None])
            non_final_prev_states_batch = torch.cat([s for s, t in zip(batch.prev_state, transition_batch)
                                                     if t.state is not None])
            non_final_prev_action_batch = torch.cat([a for a, t in zip(batch.prev_action, transition_batch)
                                                     if t.state is not None])
            non_final_prev_action_onehot_batch = self.getActionOnehotTorch(non_final_prev_action_batch)

            # predicted_next_state_mu = self._model['heter']['network'][0](non_final_prev_states_batch, non_final_prev_action_onehot_batch)
            # predicted_next_state_var = F.softplus(self._model['heter']['network'][1](non_final_prev_states_batch, non_final_prev_action_onehot_batch)) + 10**-6
            # predicted_next_state_var = torch.diag_embed(predicted_next_state_var)

            predicted_next_state_mu = \
            self._model['heter']['network'](non_final_prev_states_batch, non_final_prev_action_onehot_batch)[0]
            predicted_next_state_var = \
            self._model['heter']['network'](non_final_prev_states_batch, non_final_prev_action_onehot_batch)[1]

            A = (predicted_next_state_mu - non_final_next_states_batch) ** 2
            inv_var = 1 / predicted_next_state_var
            loss_element1 = torch.sum(A * inv_var, dim=1)
            loss_element2 = torch.log(torch.prod(predicted_next_state_var, dim=1))
            loss = torch.mean(loss_element1 + loss_element2)

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_loss.append(loss)
            self.model_optimizer.step()

            true_var = torch.sum((predicted_next_state_mu - non_final_next_states_batch) ** 2, dim=1)
            predicted_next_state_var_trace = torch.sum(predicted_next_state_var, dim=1)
            var_err = torch.mean((true_var - predicted_next_state_var_trace) ** 2)
            mu_err = torch.mean(true_var)
            self.writer.add_scalar('VarLoss/step_size' + str(self._model['heter']['step_size']), var_err,
                                   self.writer_iterations)
            self.writer.add_scalar('MuLoss/step_size' + str(self._model['heter']['step_size']), mu_err,
                                   self.writer_iterations)
            self.writer.add_scalar('HetLoss/step_size' + str(self._model['heter']['step_size']), loss,
                                   self.writer_iterations)
            self.writer_iterations += 1

        else:
            raise NotImplementedError("train model not implemented")

    @abstractmethod
    def plan(self):
        pass
        # with torch.no_grad():
        #     states = torch.from_numpy(self.dataset[:, 0:2])
        #     actions = torch.from_numpy(self.dataset[:, 2:6])
        #     next_states = torch.from_numpy(self.dataset[:, 6:8])

        #     predicted_next_state_var = self._model['heter']['network'](states, actions)[1].float().detach()
        #     predicted_next_state_var_trace = torch.sum(predicted_next_state_var, dim=1)
        #     predicted_next_state = self._model['heter']['network'](states, actions)[0].float().detach()

        #     true_var = torch.sum((predicted_next_state - next_states) ** 2, dim=1)
        #     var_err = torch.mean((true_var - predicted_next_state_var_trace)**2)
        #     mu_err = torch.mean(true_var) 

        #     # true_var_argsort = torch.argsort(true_var)
        #     # pred_var_argsort = torch.argsort(predicted_next_state_var_trace)
        #     # print(np.count_nonzero(pred_var_argsort != true_var_argsort), np.count_nonzero(pred_var_argsort == true_var_argsort))

        #     A = (predicted_next_state - next_states) ** 2
        #     inv_var = 1 / predicted_next_state_var
        #     loss_element1 = torch.sum(A * inv_var, dim=1)
        #     loss_element2 = torch.log(torch.prod(predicted_next_state_var, dim=1))
        #     loss_het = torch.mean(loss_element1 + loss_element2)

        #     self.writer.add_scalar('VarLoss/step_size'+str(self._model['heter']['step_size']), var_err, self.writer_iterations)
        #     self.writer.add_scalar('MuLoss/step_size'+str(self._model['heter']['step_size']), mu_err, self.writer_iterations)
        #     self.writer.add_scalar('HetLoss/step_size'+str(self._model['heter']['step_size']), loss_het, self.writer_iterations)
        #     self.writer_iterations += 1

    @abstractmethod
    def initModel(self, state, action):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''
        nn_state_shape = state.shape
        action_onehot = self.getActionOnehotTorch(action)
        nn_action_onehot_shape = action_onehot.shape
        if self.model_type == 'general':
            self._model['general']['network'] = StateTransitionModel(nn_state_shape, nn_action_onehot_shape,
                                                                     self._model['general']['layers_type'],
                                                                     self._model['general']['layers_features']).to(
                self.device)
            self.model_optimizer = optim.Adam(self._model[self.model_type]['network'].parameters(),
                                              lr=self._model[self.model_type]['step_size'])

        elif self.model_type == 'ensemble':
            self._model['ensemble']['network'] = []
            self.model_optimizer = []
            for i in range(self.num_ensembles):
                self._model['ensemble']['network'].append(StateTransitionModel(nn_state_shape, nn_action_onehot_shape,
                                                                               self._model['ensemble']['layers_type'],
                                                                               self._model['ensemble'][
                                                                                   'layers_features']).to(self.device))
                self.model_optimizer.append(optim.Adam(self._model['ensemble']['network'][i].parameters(),
                                                       lr=self._model['ensemble']['step_size']))

        elif self.model_type == 'heter':
            self._model['heter']['network'] = StateTransitionModelHeter(nn_state_shape, nn_action_onehot_shape,
                                                                        self._model['heter']['layers_type'],
                                                                        self._model['heter']['layers_features']).to(
                self.device)
            self.model_optimizer = optim.Adam(self._model['heter']['network'].parameters(),
                                              lr=self._model['heter']['step_size'])


        else:
            raise NotImplementedError("model not implemented")

    @timecall(immediate=False)
    def modelRollout(self, state, action_index):
        '''
        :param state: torch -> (1, state), 
        :return: next state
        '''
        if self.model_type == "general":
            with torch.no_grad():
                one_hot_action = self.getActionOnehotTorch(action_index)
                predicted_next_state = self._model['general']['network'](state, one_hot_action).detach()
                return predicted_next_state, 0

        elif self.model_type == "ensemble":
            with torch.no_grad():
                one_hot_action = self.getActionOnehotTorch(action_index)
                predicted_next_state_ensembles = torch.zeros_like(self.prev_state)
                next_state_list = torch.tensor([], device=self.device)
                for i in range(self.num_ensembles):
                    predicted_next_state = self._model['ensemble']['network'][i](state, one_hot_action).detach()
                    predicted_next_state_ensembles = torch.add(predicted_next_state_ensembles, predicted_next_state)
                    next_state_list = torch.cat((next_state_list, predicted_next_state))
                std = torch.std(next_state_list, dim=0)
                avg_std = torch.mean(std)
                predicted_next_state_ensembles = torch.div(predicted_next_state_ensembles, self.num_ensembles)
                return predicted_next_state_ensembles, avg_std

        elif self.model_type == 'heter':
            with torch.no_grad():
                one_hot_action = self.getActionOnehotTorch(action_index)
                # predicted_next_state_mu = self._model['heter']['network'](state, one_hot_action)[0].detach()
                # predicted_next_state_var = self._model['heter']['network'](state, one_hot_action)[1].detach().trace()

                predicted_next_state_var = self._model['heter']['network'](state, one_hot_action)[1].float().detach()
                predicted_next_state_var_trace = torch.sum(predicted_next_state_var, dim=1)
                predicted_next_state_mu = self._model['heter']['network'](state, one_hot_action)[0].float().detach()
                return predicted_next_state_mu, predicted_next_state_var_trace

        else:
            raise NotImplementedError("model not implemented")
