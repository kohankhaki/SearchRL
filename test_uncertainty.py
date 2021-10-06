import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import pickle
import Utils as utils
import random
import matplotlib.pyplot as plt

class UncertaintyNN(nn.Module):
    def __init__(self, state_shape, action_shape, layers_type, layers_features):
        # state : B, state_size(linear)
        # action: A
        super(UncertaintyNN, self).__init__()
        self.layers_type = layers_type
        self.layers = []
        state_size = state_shape[1]
        action_size = action_shape

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    linear_input_size = state_size + action_size
                    layer = nn.Linear(linear_input_size, layers_features[i])
                else:
                    layer = nn.Linear(layers_features[i - 1] + action_size, layers_features[i])
                self.add_module('hidden_layer_' + str(i), layer)
                self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")

        self.head = nn.Linear(layers_features[-1], 1)
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
                x = self.layers[i](x.float())
                x = torch.tanh(x)
            else:
                raise ValueError("layer is not defined")
        head = self.head(x.float())
        return head

def train_uncertainty(corrupt_transition_batch, network, optimizer, max_uncertainty, min_uncertainty):
    batch = utils.corrupt_transition(*zip(*corrupt_transition_batch))
    true_next_states = torch.cat([s for s in batch.true_state]).float()
    corrupt_next_states = torch.cat([s for s in batch.corrupt_state]).float()
    prev_state_batch = torch.cat(batch.prev_state).float()
    prev_action_batch = torch.cat(batch.prev_action).float()
    prev_state_batch = prev_state_batch[:, -6].unsqueeze(1) / 10 #only pos
    # print(prev_state_batch, prev_action_batch)
    predicted_uncertainty = network(prev_state_batch, prev_action_batch)
    true_uncertainty = torch.mean((true_next_states - corrupt_next_states) ** 2, axis=1).unsqueeze(1) / max_uncertainty
    loss = F.mse_loss(predicted_uncertainty,
                        true_uncertainty)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test_uncertainty(corrupt_transition_batch, network):
    with torch.no_grad():
        batch = utils.corrupt_transition(*zip(*corrupt_transition_batch))
        true_next_states = torch.cat([s for s in batch.true_state]).float()
        corrupt_next_states = torch.cat([s for s in batch.corrupt_state]).float()
        prev_state_batch = torch.cat(batch.prev_state).float()
        prev_action_batch = torch.cat(batch.prev_action).float()
        prev_state_batch = prev_state_batch[:, -6].unsqueeze(1) / 10#only pos

        predicted_uncertainty = network(prev_state_batch, prev_action_batch)
        true_uncertainty = torch.mean((true_next_states - corrupt_next_states) ** 2, axis=1).unsqueeze(1)
        loss = torch.mean((predicted_uncertainty - true_uncertainty)**2, dim=0)
    return loss.item()

def draw_data(data):
    data = utils.corrupt_transition(*zip(*data))
    prev_action = torch.cat(data.prev_action)
    action_index = []
    for i in range(len(prev_action)):
        if np.array_equal(prev_action[i], [0, 0, 0, 1]):
            action_index.append(i)

    prev_state = torch.cat(data.prev_state)
    prev_state = prev_state[:, -6]
    true_next_states = torch.cat([s for s in data.true_state]).float()
    corrupt_next_states = torch.cat([s for s in data.corrupt_state]).float()

    prev_state = prev_state[action_index]
    true_next_states = true_next_states[action_index]
    corrupt_next_states = corrupt_next_states[action_index]
    print(prev_state.shape, true_next_states.shape, corrupt_next_states.shape)

    true_uncertainty = torch.mean((true_next_states - corrupt_next_states) ** 2, axis=1)
    plt.scatter(prev_state, true_uncertainty)
    plt.savefig("data.png")
    print(prev_state.shape, true_uncertainty.shape)
with open("MiniAtariResult/Buffer/SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_Buffer15700_t15700.p", 'rb') as file:
    buffer = pickle.load(file)
print(len(buffer))

draw_data(buffer)
exit(0)
network = UncertaintyNN([1, 1], 4, ['fc', 'fc'], [128, 128])
# network = UncertaintyNN(buffer[0].true_state.shape, 4, ['fc', 'fc'], [128, 128])
optimizer = optim.Adam(network.parameters(), lr=0.001)
batch_size = 10
num_epochs = 100
show_freq = 99

true_uncertainty_list = []
for i in range(len(buffer)):
    true_uncertainty = torch.mean(torch.pow(buffer[i].true_state - buffer[i].corrupt_state, 2)).item()
    true_uncertainty_list.append(true_uncertainty)
max_uncertainty = max(true_uncertainty_list)
min_uncertainty = min(true_uncertainty_list)


for i in range(num_epochs):
    batch = random.sample(buffer, k=batch_size)
    train_uncertainty(batch, network, optimizer, max_uncertainty, min_uncertainty)
    loss = test_uncertainty(buffer, network)
    print(loss)
    if i % show_freq == 0 and i != 0:
        with torch.no_grad():
            for x in buffer[0:100]:
                pred_uncertainty = network(x.prev_state[:, -6].unsqueeze(1) / 10, x.prev_action).item()
                true_uncertainty = torch.mean(torch.pow(x.true_state - x.corrupt_state, 2)).item() / max_uncertainty
                
                # if true_uncertainty != 0:
                print(pred_uncertainty, true_uncertainty)

                    
exit(0)

good_states = [0, 1, 7, 8, 9]
bad_states = [2, 3, 4, 5, 6]
batch_size = 10
num_epochs = 100000
show_freq = 100
data_size = 100

network = UncertaintyNN([1, 300], 4, ['fc', 'fc'], [128, 128])
optimizer = optim.Adam(network.parameters(), lr=0.001)

x = torch.tensor([1]).unsqueeze(0)
a = torch.tensor([0]).unsqueeze(0)

test_x = torch.arange(0, 10).unsqueeze(1)
test_a = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

data_x = torch.randint(10, size=(data_size, 300), dtype=int)
data_a = test_a[torch.randint(4, size=(data_size, 1))]
data_y = torch.zeros(size=(data_size, 1))
for i in range(data_size):
    if data_x[i, 0] in bad_states and np.array_equal(data_a[i][0], [0, 0, 0, 1]):
        data_y[i] = 1
data_y = data_y.float()

for i in range(num_epochs):
    index = np.random.randint(data_x.shape[0], size=batch_size)
    batch_x = data_x[index]
    batch_y = data_y[index]
    batch_a = data_a[index]
    
    output = network(batch_x, batch_a)
    loss = F.mse_loss(output, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % show_freq == 0:
        with torch.no_grad():
            for x in data_x:
                for a in test_a:
                    pred = network(x.unsqueeze(0), a.unsqueeze(0))
                    y = 0
                    if x[0] in bad_states and np.array_equal(a, [0, 0, 0, 1]):
                        y = 1
                    print(pred, y)