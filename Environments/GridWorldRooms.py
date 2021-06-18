from Environments.GridWorldBase import GridWorld


class GridWorldRooms(GridWorld):

    def __init__(self, params):
        self.house_shape = params['house_shape']
        self.rooms_shape = params['rooms_shape']
        self.grid_size = self.calculateGridShape()
        self.obstacles_pos = self.calculateWallsPos()
        params['size'] = self.grid_size
        params['obstacles_pos'] = self.obstacles_pos

        params['rewards_pos'] = [(0, self.grid_size[1]-1)] # can change later
        params['terminals_pos'] = params['rewards_pos']
        params['init_state'] = (self.grid_size[0]-1, 0) # corner (can change later)
        GridWorld.__init__(self, params)

    def calculateWallsPos(self):
        obstacles_pos = []
        for i in range(1, self.house_shape[0]):
            x = i * self.rooms_shape[0] + (i-1)
            for j in range(self.grid_size[1]):
                if (x,j) not in obstacles_pos:
                    obstacles_pos.append((x,j))

        for j in range(1, self.house_shape[1]):
            y = j * self.rooms_shape[1] + (j-1)
            for i in range(self.grid_size[0]):
                if (i,y) not in obstacles_pos:
                    obstacles_pos.append((i,y))
        # doorways
        for i in range(1, self.house_shape[0]):
            x = i * self.rooms_shape[0] + (i-1)
            left = 0
            for j in range(1, self.house_shape[1] + 1):
                y = j * self.rooms_shape[1] + (j - 1)
                right = y - 1
                door = x, (right + left) // 2
                left = y + 1
                obstacles_pos.remove(door)

        for j in range(1, self.house_shape[1]):
            y = j * self.rooms_shape[1] + (j-1)
            up = 0
            for i in range(1, self.house_shape[0] + 1):
                x = i * self.rooms_shape[0] + (i - 1)
                down = x - 1
                door = (down + up) // 2, y
                up = x + 1
                obstacles_pos.remove(door)

        return obstacles_pos


    def calculateGridShape(self):
        size = self.house_shape[0] * self.rooms_shape[0] + (self.house_shape[0] - 1),\
               self.house_shape[1] * self.rooms_shape[1] + (self.house_shape[1] - 1)
        return size


    
if __name__ == "__main__":
    params = {'init_state': 'random' , 'state_mode': 'coord', #init_state (_n-1, 0)
    'house_shape': (2,2), 'rooms_shape': (4, 4),
    'obstacles_pos': [],
    'rewards_value': [10],
    'termination_probs': [1],
    'actions': [(0, -1), (-1, 0), (0, 1) , (1, 0)], # L, U, R, D
    'neighbour_distance': 0,
    'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
    'transition_randomness': 0.0,
    'window_size': (900, 900),
    'aging_reward': -1,
    }
    env = GridWorldRooms(params)
    error = env.calculate_state_action_value((0, 8), (0, 1), 1)
    print (error)



