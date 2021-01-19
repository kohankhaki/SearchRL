from collections import namedtuple
import csv
from torch.utils.data import Dataset, DataLoader
import random


def data_store(env, train_test_split = 0.7):
    transition = namedtuple('transition', ['state', 'action', 'next_state', 'reward'])

    all_states = env.getAllStates(state_type='coord')
    random.shuffle(all_states)
    # train_states = all_states[0 : int(len(all_states) * train_test_split)]
    train_states = all_states #  ***** change later, now we are training on all states****

    # test_state = all_states[int(len(all_states) * train_test_split) : ]
    test_state = all_states #  ***** change later, now we are testing on all states****
    all_actions = env.getAllActions()
    train_list = []
    test_list = []

    for state in train_states:
        for action in all_actions:
            next_state = env.transitionFunction(state, action, state_type='coord')
            reward = env.rewardFunction(next_state, state_type='coord')
            t = transition(state, action, next_state, reward)
            train_list.append(t)

    for state in test_state:
        for action in all_actions:
            next_state = env.transitionFunction(state, action, state_type='coord')
            reward = env.rewardFunction(next_state, state_type='coord')
            t = transition(state, action, next_state, reward)
            test_list.append(t)
    return train_list, test_list

    # with open('3dTransition_train.csv', 'w') as f:
    #     w = csv.writer(f)
    #     w.writerows([(t.state, t.action, t.next_state, t.reward) for t in transition_list])

def data_loader():
    with open('3dTransition.csv', 'r') as f:
        r = csv.reader(f)
        for row in r:
            state, action, next_state, reward = row
            yield state, action, next_state, reward
