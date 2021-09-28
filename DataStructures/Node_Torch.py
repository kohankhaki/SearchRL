import numpy as np
import torch


class Node_Torch:
    def __init__(self, parent, state, env_state=None, value=0, is_terminal=False, action_from_par=None, reward_from_par=0, uncertainty=0.0, cloned_env=None):
        self.state = state
        self.env_state = env_state
        self.sum_values = value
        self.num_visits = 0
        self.childs_list = []
        self.parent = parent
        self.is_terminal = is_terminal
        self.action_from_par = action_from_par
        self.reward_from_par = reward_from_par
        self.uncertainty = uncertainty
        self.cloned_env = cloned_env

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

    def get_weighted_avg_value(self):
        if self.is_terminal:
            return 0.0
        if self.num_visits > 1:
            children_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, self.childs_list)))
            softmax_denominator = np.sum(np.exp(-children_uncertainties))
            weights = np.exp(-children_uncertainties) / softmax_denominator
            children_visit = np.asarray(list(map(lambda n: n.num_visits, self.childs_list)))
            avg_value = self.sum_values / np.dot(weights, children_visit)
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
