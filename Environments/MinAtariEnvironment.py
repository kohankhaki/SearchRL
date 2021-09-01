import numpy as np
from minatar import Environment

class Freeway():
    def __init__(self):
        self.game = Environment("freeway")

    def start(self):
        self.game.reset()
        observation = self.game.game_state()
        return observation

    def step(self, action):
        reward, is_terminal = self.game.act(action)
        observation = self.game.game_state()
        self.game.display_state(50)
        return reward, observation, is_terminal

    def getAllActions(self):
        return self.game.minimal_action_set()

    def getState(self):
        return self.game.game_state()

    def transitionFunction(self, state, action):
        return self.game.transition_function(state, action)
