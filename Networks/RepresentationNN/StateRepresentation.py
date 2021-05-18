import torch
import torch.nn as nn


class StateRepresentation(nn.Module): # action inserted to a layer
    def __init__(self, state_shape, layers_type, layers_features):
        # state : W, H, Channels
        super(StateRepresentation, self).__init__()
        self.layers_type = layers_type
        self.layers = []
        if len(state_shape) == 3:
            linear_input_size = state_shape[0] * state_shape[1] * state_shape[2]
        elif len(state_shape) == 1:
            linear_input_size = state_shape[0]
        else:
            raise ValueError('representation not defined')

        if len(self.layers_type) == 0:
            return None

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


    def forward(self, state):
        if len(self.layers_type) == 0:
            return state.flatten(start_dim=1)
        x = 0
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim=1)
                x = self.layers[i](x.float())
                x = torch.relu(x)
            else:
                raise ValueError("layer is not defined")


        return x.float() # -1 is for the batch size
