import numpy as np
from minatar import Environment

class MinAtar():
    def __init__(self, name):
        self.game = Environment(name)

    def start(self):
        self.game.reset()
        observation = self.game.game_state()
        return observation

    def step(self, action):
        reward, is_terminal = self.game.act(action)
        observation = self.game.game_state()
        self.game.display_state(50)
        # self.game.close_display()

        return reward, observation, is_terminal

    def getAllActions(self):
        return self.game.minimal_action_set()

    def getState(self):
        return self.game.game_state()

    def transitionFunction(self, state, action, is_corrupted):
        return self.game.transition_function(state, action, is_corrupted)

if __name__ == "__main__":
    env = MinAtar("space_invaders")
    env.start()
    actions = env.getAllActions()
    print(actions)

    for a in range(10000):
        action = np.random.choice(actions)
        reward, observation1, is_terminal1 = env.step(action)


