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
        # self.game.display_state(50)
        return reward, observation, is_terminal

    def getAllActions(self):
        return self.game.minimal_action_set()

    def getState(self):
        return self.game.game_state()

    def transitionFunction(self, state, action):
        return self.game.transition_function(state, action)

if __name__ == "__main__":
    env = MinAtar("space_invaders") 
    env.start()
    actions = env.getAllActions()
    print(actions)
    action_list = [1, 3]
    observation2 = env.getState()
    for i in observation2:
        pass
    for a in range(10000):
        action = np.random.choice(actions)  
        # action = action_list[a]
        # print(action)  
        reward, observation1, is_terminal1 = env.step(action)
        reward, observation2, is_terminal2 = env.transitionFunction(observation2, action)
        for i, j in zip(observation1, observation2):
            if not np.array_equal(i, j):
                print("False")
                print(observation1, i, "\n", observation2, j, "\n\n****\n")
                exit(0)
        if is_terminal1 and is_terminal2:
            observation2 = env.start()
    # # print(observation2)
    # print("***********")
    # print(observation1)
