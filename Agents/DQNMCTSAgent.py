from Agents.MCTSAgent import MCTSAgent
from Agents.BaseDynaAgent import BaseDynaAgent


class DQNMCTSAgent(MCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent"
    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        MCTSAgent.__init__(self, params)
        self.episode_counter = 0

    def start(self, observation):
        if self.episode_counter % 2 == 0:
            action = BaseDynaAgent.start(self, observation)
        else:
            action = MCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter % 2 == 0:
            action = BaseDynaAgent.step(self, reward, observation)
        else:
            action = MCTSAgent.step(self, reward, observation)

        return action

    def end(self, reward):
        BaseDynaAgent.end(self, reward)
        self.episode_counter += 1

    def get_initial_value(self, state):
        state_representation = self.getStateRepresentation(state)
        value = self.getStateActionValue(state_representation)
        return value.item()