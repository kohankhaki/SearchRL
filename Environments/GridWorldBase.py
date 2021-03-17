import numpy as np
import random
import pygame

class GridWorld():
    def __init__(self, params=None):
        if params is None:  # default grid
            params = {'size': (5, 4), 'init_state': 'random', 'state_mode': 'coord',
                      'obstacles_pos': [(1, 1), (2, 1), (3, 1), (4,1)],
                      'rewards_pos': [(4, 2), (4, 3)], 'rewards_value': [1, 2],
                      'terminals_pos': [(4,3)], 'termination_probs':[1],
                      'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                      'neighbour_distance': 0,
                      'agent_color': [0,1,0], 'ground_color':[0,0,0], 'obstacle_color': [1,1,1],
                      'transition_randomness': 0.0,
                      'window_size' : (255, 255),
                      'aging_reward': -1
                       }

        if self.checkSimilarShapes(params['agent_color'], params['obstacle_color']) and \
            self.checkSimilarShapes(params['agent_color'], params['ground_color']):
            self._num_channel = len(params['agent_color'])
            self._obstacle_color = params['obstacle_color']
            self._ground_color = params['ground_color']
            self._agent_color = params['agent_color']
        else:
            raise ValueError("colors shape mismatch")

        self._grid_shape = params['size']+ (self._num_channel,) # size[0], size[1], num_channel
        self._state_mode = params['state_mode'] # 'coord', 'full_obs', 'nei_obs'
        self._neighbour_distance = params['neighbour_distance'] #an int to decide the neighbours of the agent
        self._transition_randomness = params['transition_randomness'] # float between 0,1
        self._actions_list = np.copy(params['actions']) # each action is a tuple that shows the movement in directions
        self._window_size = params['window_size'] # size of pygame window

        for pos in params['rewards_pos']:
            if not self.checkPosInsideGrid(pos):
                raise ValueError("reward pos is out of range")
        for pos in params['obstacles_pos']:
            if not self.checkPosInsideGrid(pos):
                raise ValueError("obstacle pos is out of range")
        for pos in params['terminals_pos']:
            if not self.checkPosInsideGrid(pos):
                raise ValueError("terminal pos is out of range")

        if len(params['rewards_pos']) != len(params['rewards_value']):
            raise ValueError('rewards number mismatch')
        if len(params['terminals_pos']) != len(params['termination_probs']):
            raise ValueError('terminals mismatch')

        self._grid = self.__createEmptyGrid()

        self._agent_pos = None
        self._obstacles_pos = []
        self._rewards_pos = params['rewards_pos']
        self._rewards_value = params['rewards_value']
        self._terminals_pos = params['terminals_pos']
        self._terminate_probs = params['termination_probs']
        self._aging_reward = params['aging_reward']

        if params['init_state'] != 'random':  # 'random' , tuple(x,y)
            if self.checkPosInsideGrid(params['init_state']):
                self._init_pos = params['init_state']
            else:
                raise ValueError("initial position is out of range")
        self._init_pos = params['init_state']
        self.addObstacles(params['obstacles_pos'], self._grid)
        self.__createOnehotMap()
        self.screen = None # for visualization
        self.__create_transition_dynamics()

    def start(self):
        """
        return environment state
        """
        self._grid = self.__createEmptyGrid()
        self.addObstacles(self._obstacles_pos, self._grid)

        if self._init_pos == 'random':
            pos = np.random.randint(0, self._grid_shape[0]), np.random.randint(0, self._grid_shape[1])
            while pos in self._obstacles_pos or pos in self._terminals_pos:
                pos = np.random.randint(0, self._grid_shape[0]), np.random.randint(0, self._grid_shape[1])
            self._agent_pos = pos
        else:
            self._agent_pos = self._init_pos

        self._initial_agent_pos = self._agent_pos
        self._grid[self._agent_pos] = self._agent_color

        if self._state_mode == 'coord':
            self._state = np.asarray(self._agent_pos)  # a tuple of the agent's position

        elif self._state_mode == 'full_obs':
            self._state = np.copy(self._grid)  # a np.array of the full grid

        elif self._state_mode == 'nei_obs':
            raise NotImplemented("need to be implemented")  # a np.array of partially neighbours grid

        elif self._state_mode == 'one_hot':
            self._state = self.one_hot_map[self._agent_pos]

        else:
            raise ValueError("state type is unknown")

        return self._state

    def step(self, action):
        """
        return reward, next env state, is terminal, info
        """
        if action not in self._actions_list:
            raise ValueError("Incorrect action")

        # update agent.pos and grid
        self._grid[self._agent_pos] = self._ground_color
        next_agent_pos = self.__transitionFunction(self._agent_pos, action)
        # if next_agent_pos not in self._obstacles_pos:
        self._agent_pos = next_agent_pos

        self._grid[self._agent_pos] = self._agent_color

        # calculate the reward and termination of the transition
        reward = self.__rewardFunction(self._agent_pos)
        is_terminal = self.__terminalFunction(self._agent_pos)

        # calculate the state
        if self._state_mode == 'coord':
            self._state = np.asarray(self._agent_pos)  # a tuple of the agent's position

        elif self._state_mode == 'full_obs':
            self._state = np.copy(self._grid)  # a np.array of the full grid

        elif self._state_mode == 'nei_obs':
            raise NotImplemented("need to be implemented")  # a np.array of partially neighbours grid

        elif self._state_mode == 'one_hot':
            self._state = self.one_hot_map[self._agent_pos]

        else:
            raise ValueError("state type is unknown")

        return reward, self._state, is_terminal

    def getAllStates(self, state_type= None):
        agent_pos_list = []
        state_list = []
        if state_type == None:
            state_type = self._state_mode
        for x in range(self._grid_shape[0]):
            for y in range(self._grid_shape[1]):
                if (x,y) not in self._obstacles_pos:
                    agent_pos_list.append((x,y))

        if state_type == 'coord':
            for pos in agent_pos_list:
                state_list.append(np.asarray(pos))
            return state_list

        elif state_type == 'full_obs':
            for pos in agent_pos_list:
                grid = self.__createEmptyGrid()
                self.addObstacles(self._obstacles_pos, grid)
                grid[pos] = self._agent_color
                state_list.append(grid)
            return state_list

        elif state_type == 'nei_obs':
            raise NotImplemented("need to be implemented")

        elif self._state_mode == 'one_hot':
            for pos in agent_pos_list:
                state_list.append(self.one_hot_map[pos])
            return state_list

        else:
            raise ValueError("state type is unknown")

    def getAllActions(self):
        return np.copy(self._actions_list)

    def __createEmptyGrid(self):
        grid = np.zeros([self._grid_shape[0], self._grid_shape[1], self._num_channel])
        for x in range(self._grid_shape[0]):
            for y in range(self._grid_shape[1]):
                grid[x, y] = self._ground_color
        return grid

    def addObstacles(self, obstacles, grid):
        """ change the grid values, add obstacle in their position."""
        for pos in obstacles:
            if self.checkPosInsideGrid(pos):  # if inside the grid
                if list(grid[pos]) == self._ground_color:
                    if pos not in self._obstacles_pos:
                        self._obstacles_pos.append(pos)
                    grid[pos] = self._obstacle_color
                else:
                    raise ValueError("obstacle position already filled")
            else:
                raise ValueError("obstacle position is out of range")

    def checkPosInsideGrid(self, pos):
        if 0 <= pos[0] < self._grid_shape[0] \
                and 0 <= pos[1] < self._grid_shape[1]:  # if inside the grid
            return True
        return False

    def checkSimilarShapes(self, arr1, arr2):
        if len(arr1) == len(arr2):
            return True
        return False

    def __rewardFunction(self, pos):
        reward = 0
        if self.checkPosInsideGrid(pos):
            for i, reward_pos in enumerate(self._rewards_pos):
                if pos == reward_pos:
                    reward += self._rewards_value[i]
        else:
            raise ValueError('position for reward function is out of range')
        return reward + self._aging_reward

    def __terminalFunction(self, pos):
        if self.checkPosInsideGrid(pos):
            for i, terminal_pos in enumerate(self._terminals_pos):
                if np.array_equal(pos, terminal_pos):
                    prob = self._terminate_probs[i]
                    termination = np.random.rand() < prob
                    return termination
        else:
            raise ValueError('position for reward function is out of range')
        return False

    def __transitionFunction(self, pos, action):
        if np.random.rand() < self._transition_randomness :
            # choose a random action
            action = self.getAllActions()[random.randint(0, len(self.getAllActions()) - 1)]

        next_pos = tuple(sum(x) for x in zip(pos, action)) #stochastic backward
        # next_pos = tuple(sum(x) % self._grid_shape[i] for i,x in enumerate(zip(pos, action))) #deterministic backward
        if self.checkPosInsideGrid(next_pos) and next_pos not in self._obstacles_pos:
            return next_pos
        return pos

    def posToState(self, pos, state_type):
        if self._grid is None:
            raise NotImplementedError("grid is not defined")

        if state_type == 'full_obs':
            grid = self._grid.copy()
            grid[self._agent_pos] = self._ground_color
            grid[pos] = self._agent_color
            return grid

        elif state_type == 'nei_obs':
            raise NotImplementedError("neighbouring observation is not implemented")

        elif state_type == 'coord':
            return np.asarray(pos)

        elif state_type == 'one_hot':
            return self.one_hot_map[pos]

        else:
            raise ValueError('state type not defined')

    def stateToPos(self, state, state_type= None):
        if state_type is None:
            state_type = self._state_mode
        if self._grid is None:
            raise NotImplementedError("grid is not defined")

        if state_type == 'full_obs':
            for i in range(state.shape[0]):
                for j in range(state.shape[1]):
                    if np.array_equal(state[i,j], self._agent_color):
                        pos = i,j
                        if pos == None: 
                          print("fuck")
                        return pos
            raise ValueError("agent is not in the grid")

        elif state_type == 'nei_obs':
            raise NotImplementedError("neighbouring observation is not implemented")

        elif state_type == 'coord':
            return tuple(state)

        elif state_type == 'one_hot':
            for i, pos in zip(state, self.getAllStates(state_type='coord')):
                if i == 1:
                    return tuple(pos)

        else:
            raise ValueError('state type not defined')

    def rewardFunction(self, state, state_type= 'full_obs'):
        pos = self.stateToPos(state, state_type=state_type)
        reward = self.__rewardFunction(pos)
        return reward

    def transitionFunction(self, state, action, state_type= 'coord'):
        pos = self.stateToPos(state, state_type)
        pos = self.__transitionFunction(pos, action)
        next_state = self.posToState(pos, state_type)
        return next_state

    def fullTransitionFunction(self, state, action, state_type='coord'):
        pos = self.stateToPos(state, state_type)
        pos = self.__transitionFunction(pos, action)
        is_terminal = self.__terminalFunction(pos)
        reward = self.__rewardFunction(pos)
        next_state = self.posToState(pos, state_type)
        return next_state, is_terminal, reward

    def coordTransitionFunction(self, state, action):
        action_index = self.getActionIndex(action)
        transition = self.transition_dynamics[int(state[0]), int(state[1]), action_index]
        next_state, is_terminal, reward = transition[0:2], transition[2], transition[3]
        return next_state, is_terminal, reward

    def getActionIndex(self, action):
        for i, a in enumerate(self.getAllActions()):
            if np.array_equal(a, action):
                return i
        raise ValueError("action is not defined")

    def __create_transition_dynamics(self):
        all_states = self.getAllStates(state_type="coord")
        all_actions = self.getAllActions()
        num_actions = len(all_actions)
        self.transition_dynamics = np.zeros([self._grid_shape[0], self._grid_shape[1], num_actions, 4]) #x, y, is_terminal, reward
        for state in all_states:
            for action in all_actions:
                next_state, is_terminal, reward = self.fullTransitionFunction(state, action, state_type="coord")
                action_index = self.getActionIndex(action)
                self.transition_dynamics[state[0], state[1], action_index, 0] = next_state[0]
                self.transition_dynamics[state[0], state[1], action_index, 1] = next_state[1]
                self.transition_dynamics[state[0], state[1], action_index, 2] = is_terminal
                self.transition_dynamics[state[0], state[1], action_index, 3] = reward

    def transitionFunctionBackward(self, state, prev_action, state_type='coord', type='sample'):
        if type == 'expectation':
            pos = self.stateToPos(state, state_type)
            possible_prev_states = []

            # come back to the same state with same action
            next_pos = self.__transitionFunction(pos, prev_action)
            if next_pos == pos:
                possible_prev_states.append(self.posToState(pos, state_type))

            # reverse the action
            prev_pos = tuple(np.subtract(pos, prev_action)) #stochastic backward
            
            # prev_pos = np.remainder(prev_pos, [self._grid_shape[0], self._grid_shape[1]]) #deterministic backward
            
            if self.checkPosInsideGrid(prev_pos) and prev_pos not in self._obstacles_pos:
                possible_prev_states.append(self.posToState(prev_pos, state_type))
            expected_prev_state = 0
            for s in possible_prev_states:
                expected_prev_state += s

            if len(possible_prev_states) > 0:
                expected_prev_state /= len(possible_prev_states)
            else:
                expected_prev_state = None
            return expected_prev_state

        elif type == 'sample':
            pos = self.stateToPos(state, state_type)
            possible_prev_states = []

            # come back to the same state with same action
            next_pos = self.__transitionFunction(pos, prev_action)
            if next_pos == pos:
                possible_prev_states.append(self.posToState(pos, state_type))

            # reverse the action
            # prev_pos = tuple(np.subtract(pos, prev_action)) # stochastic backward
            prev_pos = tuple((pos[i] - prev_action[i]) % self._grid_shape[i] for i, x in enumerate(zip(pos, prev_action))) #deterministic backward
            if self.checkPosInsideGrid(prev_pos) and prev_pos not in self._obstacles_pos:
                possible_prev_states.append(self.posToState(prev_pos, state_type))
            expected_prev_state = 0
            for s in possible_prev_states:
                expected_prev_state += s

            if len(possible_prev_states) > 0:
                expected_prev_state = random.choice(possible_prev_states)
            else:
                expected_prev_state = None
            return expected_prev_state

        else:
            raise NotImplementedError("backward transition function other expectation hasn't been implemented")

    def get_obstacles_pos(self):
        return self._obstacles_pos

    def __createOnehotMap(self):
        all_states = self.getAllStates('coord')
        self.one_hot_map = {}
        for i, s in enumerate(all_states):
            res = np.zeros((len(all_states)))
            res[i] = 1
            self.one_hot_map[self.stateToPos(s, 'coord')] = res

    def render(self, grid= None, values= None):
        if grid == None:
            grid = self._grid

        agent_color = [i * 255 for i in self._agent_color]
        ground_color = [i * 255 for i in self._ground_color]
        obstacle_color = [i * 255 for i in self._obstacle_color]
        text_color = (240,240,10)
        info_color = (200, 50, 50)
        # This sets the WIDTH and HEIGHT of each grid location
        WIDTH = int(self._window_size[0] / grid.shape[1])
        HEIGHT = int(self._window_size[1] / grid.shape[0])
        # This sets the margin between each cell
        MARGIN = 1

        reward_text_rect_list = []
        reward_text_list = []
        # Initialize pygame
        pygame.init()

        # Set the HEIGHT and WIDTH of the screen
        WINDOW_SIZE = [self._window_size[0], self._window_size[1]]
        if self.screen == None:
            self.screen = pygame.display.set_mode(WINDOW_SIZE)

        # Set title of screen
        pygame.display.set_caption("Grid_World")

        # Used to manage how fast the screen updates
        clock = pygame.time.Clock()

        font = pygame.font.Font('freesansbold.ttf', 15)
        info_font = pygame.font.Font('freesansbold.ttf', 10)
        for i, pos in enumerate(self._rewards_pos):
            # create a text suface object, on which text is drawn on it.
            text = font.render(str(self._rewards_value[i]), True, text_color)

            # create a rectangular object for the text surface object
            textRect = text.get_rect()

            # set the center of the rectangular object.
            textRect.center = ((pos[1])*WIDTH + WIDTH/2, (pos[0])*HEIGHT+ HEIGHT/2)
            reward_text_rect_list.append(textRect)
            reward_text_list.append(text)

        # adding initial position to the list
        text = font.render("S", True, text_color)
        textRect = text.get_rect()
        textRect.center = ((self._initial_agent_pos[1]) * WIDTH + WIDTH / 2,
                           (self._initial_agent_pos[0]) * HEIGHT + HEIGHT / 2)
        reward_text_rect_list.append(textRect)
        reward_text_list.append(text)

        # -------- Main Program Loop -----------
        # while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                exit(0)

        # Set the screen background
        self.screen.fill((100,100,100))


        # Draw the grid
        for x in range(self._grid_shape[0]):
            for y in range(self._grid_shape[1]):
                color = ground_color
                if list(grid[x][y]) == self._agent_color:
                    color = agent_color
                elif list(grid[x][y]) == self._obstacle_color:
                    color = obstacle_color
                pygame.draw.rect(self.screen,
                                 color,
                                 [(MARGIN + WIDTH) * y + MARGIN,
                                  (MARGIN + HEIGHT) * x + MARGIN,
                                  WIDTH,
                                  HEIGHT])
                if color != obstacle_color and values is not None:
                    # showing values only for 4 basic actions
                    up_left_corner = [(MARGIN + WIDTH) * y + MARGIN,
                                      (MARGIN + HEIGHT) * x + MARGIN]
                    up_right_corner = [(MARGIN + WIDTH) * y + MARGIN + WIDTH,
                                      (MARGIN + HEIGHT) * x + MARGIN]
                    down_left_corner = [(MARGIN + WIDTH) * y + MARGIN,
                                       (MARGIN + HEIGHT) * x + MARGIN + HEIGHT]
                    down_right_corner = [(MARGIN + WIDTH) * y + MARGIN + WIDTH,
                                        (MARGIN + HEIGHT) * x + MARGIN + HEIGHT]
                    center = [(up_right_corner[0] + up_left_corner[0]) // 2,
                              (up_right_corner[1] + down_right_corner[1]) // 2]

                    pygame.draw.polygon(self.screen, info_color,
                                        [up_left_corner, up_right_corner, center],
                                        1)
                    pygame.draw.polygon(self.screen, info_color,
                                        [up_right_corner, down_right_corner, center],
                                        1)
                    pygame.draw.polygon(self.screen, info_color,
                                        [down_right_corner, down_left_corner, center],
                                        1)
                    pygame.draw.polygon(self.screen, info_color,
                                        [down_left_corner, up_left_corner, center],
                                        1)
                    for a in self.getAllActions():
                        if tuple(a) == (0,1):
                            right = info_font.render(str(values[(x,y), tuple(a)]), True, info_color)
                        elif tuple(a) == (1,0):
                            down = info_font.render(str(values[(x,y), tuple(a)]), True, info_color)
                        elif tuple(a) == (0,-1):
                            left = info_font.render(str(values[(x,y), tuple(a)]), True, info_color)
                        elif tuple(a) == (-1,0):
                            up = info_font.render(str(values[(x,y), tuple(a)]), True, info_color)
                        else:
                            raise ValueError("action cannot be rendered")
                    self.screen.blit(left,
                                     (up_left_corner[0] + 0.5 * left.get_rect().width, center[1])) #left
                    self.screen.blit(right,
                                     (up_right_corner[0] - 1.5 * right.get_rect().width, center[1]))  # right
                    self.screen.blit(up,
                                     (center[0] - up.get_rect().width // 2,
                                      center[1] - up.get_rect().width))  # up
                    self.screen.blit(down,
                                     (center[0] - down.get_rect().width // 2,
                                      center[1] + down.get_rect().width - down.get_rect().height))  # down

        # Limit to 60 frames per second
        clock.tick(60)
        for i in range(len(reward_text_list)):
            self.screen.blit(reward_text_list[i], reward_text_rect_list[i])

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
    
    def calculate_state_action_value(self, state, action, gamma):
        next_state, is_terminal, reward = self.fullTransitionFunction(state, action)
        queue = [(next_state, reward)]
        visited_states = []
        while(len(queue) != 0):
            node, distance = queue.pop(0)
            # print(node, '------', distance)
            if self.__terminalFunction(node):
                break
            for a in self.getAllActions():
                child, is_terminal, reward = self.fullTransitionFunction(node, a)
                if not self.is_in(child, visited_states):
                    queue.append((child, distance * gamma + reward))
            visited_states.append(node)
        return distance

    def is_in(self, child, visited_states):
        for visited in visited_states:
            if np.array_equal(child, visited):
                return True
        return False

if __name__ == "__main__":
    env = GridWorld()







