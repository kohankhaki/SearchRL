# # experiment
num_runs = 10
num_episode = 200

max_step_each_episode = 100

episodes_only_dqn = 0
episodes_only_mcts = 0

num_thread = 2

# # environment
# empty room parameters
_n = 2
empty_room_params = \
    {'size': (_n, _n), 'init_state':(_n-1, 0), 'state_mode': 'coord', #init_state (_n-1, 0)
    'obstacles_pos': [],
    'rewards_pos': [(0, _n-1)], 'rewards_value': [1],
    'terminals_pos': [(0, _n-1)], 'termination_probs': [1],
    'actions': [(0, -1), (-1, 0), (0, 1) , (1, 0)], # L, U, R, D
    'neighbour_distance': 0,
    'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
    'transition_randomness': 0.0,
    'window_size': (600, 600),
    'aging_reward': -10,
    }

n_room_params = \
    {'init_state': 'random' , 'state_mode': 'coord', #init_state (_n-1, 0)
    'house_shape': (2,2), 'rooms_shape': (4, 4),
    'obstacles_pos': [],
    'rewards_value': [1],
    'termination_probs': [1],
    'actions': [(0, -1), (-1, 0), (0, 1) , (1, 0)], # L, U, R, D
    'neighbour_distance': 0,
    'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
    'transition_randomness': 0.0,
    'window_size': (900, 900),
    'aging_reward': -1,
    }




