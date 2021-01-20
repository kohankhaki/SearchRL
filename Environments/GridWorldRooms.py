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
        params['init_state'] = (self.grid_size[0]-2, 1) # corner (can change later)
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