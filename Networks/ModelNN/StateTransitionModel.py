
# State Transition Model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter
import torchvision

from Datasets.TransitionDataGrid import data_store
import Utils as utils

class StateTransitionModel(nn.Module):
    def __init__(self, state_shape, num_actions, layers_type, layers_features, action_layer_num):
        super(StateTransitionModel, self).__init__()
        # state : B, state_size(linear)
        # action: A
        self.layers_type = layers_type
        self.layers = []

        linear_input_size = 0
        self.action_layer_num = action_layer_num
        self.num_actions = num_actions
        state_size = state_shape[1]

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                action_shape_size = 0
                if i == self.action_layer_num:
                    # insert action to this layer
                    action_shape_size = num_actions
                if i == 0:
                    linear_input_size = state_size + action_shape_size
                    layer = nn.Linear(linear_input_size, layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)
                else:
                    layer = nn.Linear(layers_features[i - 1] + action_shape_size, layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)
            else:
                print(layer)
                raise ValueError("layer is not defined")

        if self.action_layer_num < len(self.layers_type):
            self.head = nn.Linear(layers_features[-1], state_size)
            self.is_terminal = nn.Linear(layers_features[-1], 1)

        elif self.action_layer_num == len(self.layers_type):
            self.head = nn.Linear(layers_features[-1] + num_actions, state_size)
            self.is_terminal = nn.Linear(layers_features[-1] + num_actions, 1)

        elif self.action_layer_num == len(self.layers_type) + 1:
            self.head = nn.Linear(layers_features[-1], self.num_actions * state_size)
            self.is_terminal = nn.Linear(layers_features[-1], self.num_actions * 1)

        else:
            raise ValueError("action layer number is out of range")



    def forward(self, state, action=None):
        if self.action_layer_num == len(self.layers) + 1 and action is None:
            raise ValueError("action is not given")
        x = 0
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)
                if i == self.action_layer_num:
                    # insert action to this layer
                    a = action.flatten(start_dim=1)
                    x = torch.cat((x.float(), a.float()), dim=1)
                x = self.layers[i](x.float())
                x = torch.relu(x)
            else:
                raise ValueError("layer is not defined")

        if self.action_layer_num == len(self.layers_type):
            a = action.flatten(start_dim=1)
            x = torch.cat((x.float(), a.float()), dim=1)

        head = self.head(x.float())
        is_terminal = torch.sigmoid(5 * self.is_terminal(x.float()))

        if self.action_layer_num == len(self.layers_type) + 1:
            return head.view((-1,) + (self.num_actions,) + state.shape[1:]), \
                   is_terminal.view((-1,) + (self.num_actions,) + (1,))  # -1 is for the batch size
        else:
            return head.view(state.shape), is_terminal #


# PreTrain Forward
def preTrainForward(env, device):

    def getActionIndex(action):
        for i, a in enumerate(env.getAllActions()):
            if list(a) == list(action):
                return i
        raise ValueError("action is not defined")

    def getActionOnehot(action):
        res = np.zeros([len(env.getAllActions())])
        res[getActionIndex(action)] = 1
        return res

    def testAccuracy(state_transition_model, test):
        sum = 0.0
        for data in test:
            state, action, next_state, reward = data
            state = torch.from_numpy(state).unsqueeze(0).float().to(device)
            next_state = torch.from_numpy(next_state).unsqueeze(0).float().to(device)
            action_index = getActionIndex(action)
            action_onehot = torch.from_numpy(getActionOnehot(action)).float().unsqueeze(0).to(device)
            if action_layer_num == len(model_layers_type) + 1:
                pred_state = state_transition_model(state, action_onehot).detach()[:, action_index]
            else:
                pred_state = state_transition_model(state, action_onehot).detach()

            err = torch.mean((next_state - pred_state)**2)
            sum += err
        mse = sum / len(test)
        return mse

    def createVisitCountsDict(env):
        visit_count = {}
        for state in env.getAllStates():
            for action in env.getAllActions():
                pos = env.stateToPos(state)
                visit_count[(pos, tuple(action))] = 0
        return visit_count

    train, test = data_store(env)
    model_layers_type = ['fc','fc']
    model_layers_features = [256, 64]
    action_layer_num = 3
    model_step_size = 1.0
    batch_size = 4

    writer = SummaryWriter()
    xxx = False

    state_transition_model = StateTransitionModel(train[0].state.shape,
                                                   len(env.getAllActions()),
                                                   model_layers_type,
                                                   model_layers_features,
                                                   action_layer_num).to(device)


    num_samples = 0
    max_samples = 10000
    plot_y = []
    plot_x = []
    visit_count = createVisitCountsDict(env)
    print("Forward model is being trained")
    while num_samples < max_samples:
        transition_batch = random.choices(train, k=batch_size)
        num_samples += batch_size
        for transition in transition_batch:
            state, action, next_state, reward = transition

            pos = env.stateToPos(state)
            visit_count[(pos, tuple(action))] += 1

            x_old = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(device)
            x_new = torch.from_numpy(np.asarray(next_state)).float().unsqueeze(0).to(device)
            action_index = getActionIndex(action)
            action_onehot = torch.from_numpy(getActionOnehot(action)).float().unsqueeze(0).to(device)

            if action_layer_num == len(model_layers_type) + 1:
                input = state_transition_model(x_old, action_onehot)[:, action_index]
            else:
                input = state_transition_model(x_old, action_onehot)

            target = x_new
            assert target.shape == input.shape, "target and input should have same shapes"

            grid_target = torchvision.utils.make_grid(utils.reshape_for_grid(target))
            grid_input = torchvision.utils.make_grid(utils.reshape_for_grid(input))
            writer.add_image('true_images', grid_target, global_step=num_samples)
            writer.add_image('pred_images', grid_input, global_step=num_samples)
            if not xxx:
                xxx = True
                writer.add_graph(state_transition_model, input_to_model=[x_old, action_onehot])
            writer.close()

            loss = nn.MSELoss()(input, target)
            loss.backward()
            optimizer = optim.SGD(state_transition_model.parameters(), lr=(1 / batch_size) * model_step_size)
            optimizer.step()
            optimizer.zero_grad()

        mse = testAccuracy(state_transition_model, test)
        plot_y.append(mse)
        plot_x.append(num_samples)

        print(mse)
    print(visit_count)

    print("model training finished")
    return state_transition_model, visit_count, plot_y, plot_x


# PreTrain Backward
def preTrainBackward(env, device):

    def getActionIndex(action):
        for i, a in enumerate(env.getAllActions()):
            if list(a) == list(action):
                return i
        raise ValueError("action is not defined")

    def getActionOnehot(action):
        res = np.zeros([len(env.getAllActions())])
        res[getActionIndex(action)] = 1
        return res

    def testAccuracy(state_transition_model, test):
        sum = 0.0
        for data in test:
            state, action, next_state, reward = data
            state = torch.from_numpy(state).unsqueeze(0).float().to(device)
            next_state = torch.from_numpy(next_state).unsqueeze(0).float().to(device)
            action_index = getActionIndex(action)
            action_onehot = torch.from_numpy(getActionOnehot(action)).float().unsqueeze(0).to(device)
            if action_layer_num == len(model_layers_type) + 1:
                pred_state = state_transition_model(next_state, action_onehot).detach()[:, action_index]
            else:
                pred_state = state_transition_model(next_state, action_onehot).detach()

            err = torch.mean((state - pred_state)**2)
            sum += err
        mse = sum / len(test)
        return mse

    def createVisitCountsDict(env):
        visit_count = {}
        for state in env.getAllStates():
            for action in env.getAllActions():
                pos = env.stateToPos(state)
                visit_count[(pos, tuple(action))] = 0
        return visit_count

    train, test = data_store(env)
    model_layers_type = ['fc', 'fc']
    model_layers_features = [256, 64]
    action_layer_num = 3
    model_step_size = 1.0
    batch_size = 10
    state_transition_model = StateTransitionModel(train[0].state.shape,
                                                  len(env.getAllActions()),
                                                  model_layers_type,
                                                  model_layers_features,
                                                  action_layer_num).to(device)
    num_samples = 0
    max_samples = 10000
    plot_y = []
    plot_x = []
    visit_count = createVisitCountsDict(env)
    print("Backward model is being trained")
    while num_samples < max_samples:
        transition_batch = random.choices(train, k=batch_size)
        num_samples += batch_size
        for transition in transition_batch:
            state, action, next_state, reward = transition

            pos = env.stateToPos(state)
            visit_count[(pos, tuple(action))] += 1

            x_old = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(device)
            x_new = torch.from_numpy(np.asarray(next_state)).float().unsqueeze(0).to(device)
            action_index = getActionIndex(action)
            action_onehot = torch.from_numpy(getActionOnehot(action)).float().unsqueeze(0).to(device)
            if action_layer_num == len(model_layers_type) + 1:
                input = state_transition_model(x_new, action_onehot)[:, action_index]
            else:
                input = state_transition_model(x_new, action_onehot)

            target = x_old
            assert target.shape == input.shape, "target and input should have same shapes"

            loss = nn.MSELoss()(input, target)
            loss.backward()
            optimizer = optim.SGD(state_transition_model.parameters(), lr=(1 / batch_size) * model_step_size)
            optimizer.step()
            optimizer.zero_grad()
        mse = testAccuracy(state_transition_model, test)
        plot_y.append(mse)
        plot_x.append(num_samples)

        print(mse)
    print(visit_count)

    print("model training finished")
    return state_transition_model, visit_count, plot_y, plot_x