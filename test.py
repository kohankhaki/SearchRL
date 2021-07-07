from numpy import mask_indices
from Environments.GridWorldRooms import GridWorldRooms
import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Utils as utils
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


class StateTransitionModelHeter2(nn.Module):
    def __init__(self, state_shape, action_shape, layers_type, layers_features):
        # state : B, state_size(linear)
        # action: A
        super(StateTransitionModelHeter2, self).__init__()
        self.layers_type = layers_type
        self.mu_layers = []
        self.var_layers = []
        state_size = state_shape[1]
        action_size = action_shape[1]

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    linear_input_size = state_size + action_size
                    mu_layer = nn.Linear(linear_input_size, layers_features[i])
                    var_layer = nn.Linear(linear_input_size, layers_features[i])
                else:
                    mu_layer = nn.Linear(layers_features[i - 1] + action_size, layers_features[i])
                    var_layer = nn.Linear(layers_features[i - 1], layers_features[i])
                self.add_module('hidden_layer_' + str(i), mu_layer)
                self.add_module('hidden_varlayer_' + str(i), var_layer)
                self.mu_layers.append(mu_layer)
                self.var_layers.append(var_layer)
            else:
                raise ValueError("layer is not defined")

        self.mu_head = nn.Linear(layers_features[-1], state_size)
        self.var_head = nn.Linear(layers_features[-1], state_size)

    def forward(self, state, action):
        x = None
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)
                a = action.flatten(start_dim=1)
                x = torch.cat((x.float(), a.float()), dim=1)
                x = self.mu_layers[i](x.float())
                x = torch.relu(x)
            else:
                raise ValueError("layer is not defined")
        mu_head = self.mu_head(x.float())

        x = None
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)
                    a = action.flatten(start_dim=1)
                    x = torch.cat((x.float(), a.float()), dim=1)
                x = self.var_layers[i](x.float())
                x = torch.relu(x)
            else:
                raise ValueError("layer is not defined")
        var_head = F.softplus(self.var_head(x.float())) + 1e-6
        return mu_head, var_head

class StateTransitionModelHeter(nn.Module):
    def __init__(self, state_shape, action_shape, layers_type, layers_features):
        # state : B, state_size(linear)
        # action: A
        super(StateTransitionModelHeter, self).__init__()
        self.layers_type = layers_type
        self.mu_layers = []
        state_size = state_shape[1]
        action_size = action_shape[1]

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    linear_input_size = state_size + action_size
                    mu_layer = nn.Linear(linear_input_size, layers_features[i])
                else:
                    mu_layer = nn.Linear(layers_features[i - 1] + action_size, layers_features[i])
                self.add_module('hidden_layer_' + str(i), mu_layer)
                self.mu_layers.append(mu_layer)
            else:
                raise ValueError("layer is not defined")

        self.mu_head = nn.Linear(layers_features[-1], state_size)

        self.var_layer = nn.Linear(layers_features[-1], 200)
        self.var_head = nn.Linear(200, state_size)

    def forward(self, state, action):
        x = None
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)
                a = action.flatten(start_dim=1)
                x = torch.cat((x.float(), a.float()), dim=1)
                x = self.mu_layers[i](x.float())
                x = torch.relu(x)
            else:
                raise ValueError("layer is not defined")
        mu_head = self.mu_head(x.float())
        var_layer = self.var_layer(x.float())
        var_head = F.softplus(self.var_head(var_layer)) + 1e-6
        return mu_head, var_head






writer = SummaryWriter()

def init(device, params):
    model = StateTransitionModelHeter2([1, 2], [1, 4], 
                                      ['fc', 'fc'],
                                      [32, 16]).to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=params['step_size'])
    return model, model_optimizer

def train(model, model_optimizer, state_batch, action_batch, nextstate_batch):                    

    # predicted_next_state_mu = self._model['heter']['network'][0](non_final_prev_states_batch, non_final_prev_action_onehot_batch)
    # predicted_next_state_var = F.softplus(self._model['heter']['network'][1](non_final_prev_states_batch, non_final_prev_action_onehot_batch)) + 10**-6
    # predicted_next_state_var = torch.diag_embed(predicted_next_state_var)

    predicted_next_state_mu = model(state_batch, action_batch)[0]
    predicted_next_state_var = model(state_batch, action_batch)[1]

    
    A = (predicted_next_state_mu - nextstate_batch) ** 2
    inv_var = 1 / predicted_next_state_var
    loss_element1 = torch.sum(A * inv_var, dim=1)
    loss_element2 = torch.log(torch.prod(predicted_next_state_var, dim=1))
    loss = torch.mean(loss_element1 + loss_element2)

    # predicted_next_state_var = torch.diag_embed(predicted_next_state_var)
    # A = (predicted_next_state_mu.float()-nextstate_batch.float()).unsqueeze(2)
    # inv_var = torch.inverse(predicted_next_state_var.float())
    # loss = torch.mean(torch.matmul(torch.matmul(A.permute(0,2,1), inv_var), A).squeeze(2).squeeze(1) + torch.logdet(predicted_next_state_var.float()))

    # self.model_optimizer[0].zero_grad()
    # self.model_optimizer[1].zero_grad()
    model_optimizer.zero_grad()
    loss.backward()
    model_optimizer.step()
    # self.model_optimizer[0].step()
    # self.model_optimizer[1].step()

def one_hot_encoder(action, action_list):
    action_index = None
    for i, a in enumerate(action_list):
            if np.array_equal(a, action):
                action_index = i
                break
    res = np.zeros([len(action_list)])
    res[action_index] = 1
    return res

def validate(model, dataset, iter, params):
    # test ******
    i = 0
    # for m in model.modules():
    #     if isinstance(m, nn.Linear):
    #         print(m.weight.data)    
    #         print("*****************")   
    with torch.no_grad():
        states = torch.from_numpy(dataset[:, 0:2])
        actions = torch.from_numpy(dataset[:, 2:6])
        next_states = torch.from_numpy(dataset[:, 6:8])
        
        predicted_next_state_var = model(states, actions)[1].float().detach()
        predicted_next_state_var_trace = torch.sum(predicted_next_state_var, dim=1)
        predicted_next_state = model(states, actions)[0].float().detach()
        true_var = torch.sum((predicted_next_state - next_states) ** 2, dim=1)
        var_err = torch.mean((true_var - predicted_next_state_var_trace)**2)
        mu_err = torch.mean(true_var) 

        # true_var_argsort = torch.argsort(true_var)
        # pred_var_argsort = torch.argsort(predicted_next_state_var_trace)
        # print(np.count_nonzero(pred_var_argsort != true_var_argsort), np.count_nonzero(pred_var_argsort == true_var_argsort))

        A = (predicted_next_state - next_states) ** 2
        inv_var = 1 / predicted_next_state_var
        loss_element1 = torch.sum(A * inv_var, dim=1)
        loss_element2 = torch.log(torch.prod(predicted_next_state_var, dim=1))
        loss_het = torch.mean(loss_element1 + loss_element2)

        writer.add_scalar('VarLoss/test'+str(params), var_err, iter)
        writer.add_scalar('MuLoss/test'+str(params), mu_err, iter)
        writer.add_scalar('HetLoss/test'+str(params), loss_het, iter)

        print(var_err, mu_err, loss_het)
        return loss_het, var_err, mu_err
    

if __name__ == "__main__":
    fig, ax = plt.subplots()
    # step_size_list = [2**-4, 2**-6, 2**-8, 2**-10, 2**-12]
    step_size_list = [2**-10]

    env = GridWorldRooms(params= Config.n_room_params)
    all_states = env.getAllStates()
    all_actions = env.getAllActions()
    batch_size = 32
    dataset = np.zeros([len(all_states)*len(all_actions), 8])
    i = 0
    num_runs = 3
    epochs = 100
    for s in all_states:
        for a in all_actions:
            next_state, is_terminal, reward = env.fullTransitionFunction(s, a)
            dataset[i] = np.concatenate([s, one_hot_encoder(a, all_actions), next_state])
            i += 1
    
    for count, s in enumerate(step_size_list):
        params = {'step_size':s}
        model, model_optimizer = init(device="cpu", params=params)
        for e in tqdm(range(epochs)):
            mask_index = np.arange(len(dataset))
            np.random.shuffle(mask_index)
            for i in range(len(dataset) // batch_size + 1):
                if (i+1) * batch_size > len(mask_index):
                    index = mask_index[i * batch_size:]
                else:
                    index = mask_index[i * batch_size: (i+1) * batch_size]
                state_batch = dataset[index, 0:2]
                action_batch = dataset[index, 2:6]
                next_state_batch = dataset[index, 6:8]
                train(model, model_optimizer, torch.from_numpy(state_batch), torch.from_numpy(action_batch), torch.from_numpy(next_state_batch))

            if e % 10 == 0:
                loss_het, var_err, mu_err = validate(model, dataset, e, params)


    
    print("input****")
    err = 0
    counter = 0
    for s in all_states:
        for a in all_actions:
            one_hot_action = torch.from_numpy(one_hot_encoder(a, all_actions)).unsqueeze(0)
            state = torch.from_numpy(s).unsqueeze(0)
            predicted_next_state_var = model(state, one_hot_action)[1].float().detach().sum()
            predicted_next_state = model(state, one_hot_action)[0].float().detach()
            next_state, _, _ = env.fullTransitionFunction(s, a)
            true_var = torch.sum((predicted_next_state - next_state) ** 2)
            print(s, a, predicted_next_state, next_state, predicted_next_state_var, true_var, torch.abs(predicted_next_state_var - true_var))
            err += torch.abs(predicted_next_state_var - true_var)
            counter += 1
    print(err, counter, err/counter)
    
    # print("random input****")
    # for i in range(100):
    #     state = torch.FloatTensor(1, 2).uniform_(0, 20)
    #     one_hot_action = torch.tensor([[1, 0, 0, 0]])
    #     predicted_next_state_var = model(state, one_hot_action)[1].detach().trace()
    #     print(state, predicted_next_state_var)