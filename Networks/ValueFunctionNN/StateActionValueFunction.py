import torch
import torch.nn as nn


class StateActionVFNN(nn.Module):  # last layer has number of actions' output
    def __init__(self, state_shape, num_actions, layers_type, layers_features, action_layer_num):
        # state : Batch, Linear State
        # action: Batch, A
        super(StateActionVFNN, self).__init__()
        self.layers_type = layers_type
        self.action_layer_num = action_layer_num
        self.layers = []
        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                action_shape_size = 0
                if i == self.action_layer_num:
                    # insert action to this layer
                    action_shape_size = num_actions

                if i == 0:
                    linear_input_size = state_shape[1] + action_shape_size
                    layer = nn.Linear(linear_input_size, layers_features[i])
                    # nn.init.normal_(layer.weight)
                    self.add_module('hidden_layer_' + str(i), layer)
                    self.layers.append(layer)

                else:
                    layer = nn.Linear(layers_features[i - 1] + action_shape_size, layers_features[i])
                    # nn.init.normal_(layer.weight)
                    self.add_module('hidden_layer_' + str(i), layer)
                    self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")

        if len(layers_type) > 0:
            if self.action_layer_num == len(self.layers_type):
                self.head = nn.Linear(layers_features[-1] + num_actions, 1)
                # nn.init.normal_(self.head.weight)

            elif self.action_layer_num == len(self.layers_type) + 1:
                self.head = nn.Linear(layers_features[-1], num_actions)
                # nn.init.normal_(self.head.weight)

            else:
                self.head = nn.Linear(layers_features[-1], 1)
                # nn.init.normal_(self.head.weight)
        else:
            # simple linear regression
            if self.action_layer_num == len(self.layers_type):
                self.head = nn.Linear(state_shape[1] + num_actions, 1, bias=False)
            elif self.action_layer_num == len(self.layers_type) + 1:
                self.head = nn.Linear(state_shape[1], num_actions, bias=False)
                # nn.init.normal_(self.head.weight)

    def forward(self, state, action=None):
        if self.action_layer_num != len(self.layers) + 1 and action is None:
            raise ValueError("action is not given")
        x = state.flatten(start_dim=1)
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim=1)
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

        x = self.head(x.float())
        return x
