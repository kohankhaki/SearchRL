import numpy as np
import torch


class Node_Torch:
    def __init__(self, parent, state, value=0, is_terminal=False, action_from_par=None, reward_from_par=0, uncertainty=0):
        self.state = state
        self.sum_values = value
        self.num_visits = 0
        self.childs_list = []
        self.parent = parent
        self.is_terminal = is_terminal
        self.action_from_par = action_from_par
        self.reward_from_par = reward_from_par
        self.uncertainty = uncertainty

    def get_action_from_par(self):
        return self.action_from_par

    def add_child(self, child):
        self.childs_list.append(child)

    def get_childs(self):
        return self.childs_list.copy()

    def add_to_values(self, value):
        self.sum_values += value

    def get_avg_value(self):
        # return self.sum_values / (self.num_visits + 1)

        if self.num_visits > 0:
            avg_value = self.sum_values / self.num_visits
        else:
            avg_value = self.sum_values
        return avg_value

    def inc_visits(self):
        self.num_visits += 1

    def get_state(self):
        # return self.state.detach().clone()
        return self.state

    
    def get_uncertainty(self):
        return self.uncertainty

    def show(self):
        try:
            print("state: ", self.state, " value: ", self.sum_values, " num_visits: ", self.num_visits, " parent: ",
                  self.parent.get_state())
        except AttributeError:
            print("state: ", self.state, " value: ", self.sum_values, " num_visits: ", self.num_visits, " parent: ",
                  None)
