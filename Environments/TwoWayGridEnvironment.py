from Environments.GridWorldBase import GridWorld



if __name__ == "__main__":
    params = {'size': (3, 8), 'init_state': (1, 0), 'state_mode': 'coord',
              'obstacles_pos': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)],
              'rewards_pos': [(1, 7)], 'rewards_value': [1],
              'terminals_pos': [(1, 7)], 'termination_probs': [1],
              'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
              'neighbour_distance': 0,
              'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
              'transition_randomness': 0.0,
              'window_size': (255, 255),
              'aging_reward': -1
              }
    env = GridWorld(params)
    env.start()
    for i in range(10000):
        env.render()



