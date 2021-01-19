
# State VF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateVFNN(nn.Module):
    def __init__(self, state_shape, layers_type, layers_features):
        # state : Batch, W, H, Channels
        # action: Batch, A
        super(StateVFNN, self).__init__()
        self.layers_type = layers_type
        self.layers = []
        linear_input_size = state_shape[1]

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer =='fc':
                if i == 0:
                    layer = nn.Linear(linear_input_size, layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)
                else:
                    layer = nn.Linear(layers_features[i-1], layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")

        self.head = nn.Linear(layers_features[-1], 1)

    def forward(self, state):
        x = 0

        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)
                x = self.layers[i](x.float())
                x = torch.relu(x)
            else:
                raise ValueError("layer is not defined")
        return self.head(x.float())
