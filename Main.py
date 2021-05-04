'''
train a DQN with MCTS
See if DQN agrees with DQN (saving the best path)
At what step DQN starts to work with MCTS
'''
# use both mcts trajectories and dqn trajectories in the dqn buffer
# use the selection path but with all children in the path
# rollout with dqn policy in mcts
import threading
import time

import Utils as utils, Config as config

from Experiments.ExperimentObject import ExperimentObject
from Experiments.GridWorldExperiment import RunExperiment as GridWorld_RunExperiment
from Environments.GridWorldRooms import GridWorldRooms
from Agents.BaseDynaAgent import BaseDynaAgent
from Agents.MCTSAgent import MCTSAgent
from Agents.DQNMCTSAgent import *



if __name__ == '__main__':

    # agent_class_list = [BaseDynaAgent]
    # agent_class_list = [DQNMCTSAgent_MCTSPolicy]
    # agent_class_list = [DQNMCTSAgent_InitialValue]
    # agent_class_list = [DQNMCTSAgent_BootstrapInitial]
    # agent_class_list = [DQNMCTSAgent_Bootstrap]
    # agent_class_list = [MCTSAgent]
    # agent_class_list = [DQNMCTSAgent_UseTreeExpansion]
    # agent_class_list = [DQNMCTSAgent_UseTreeSelection]
    # agent_class_list = [DQNMCTSAgent_Rollout]
    # agent_class_list = [DQNMCTSAgent_MCTSSelectedAction]
    # agent_class_list = [DQNMCTSAgent_UseSelectedAction]
    # agent_class_list = [DQNMCTSAgent_UseMCTSwPriority]
    # agent_class_list = [DQNMCTSAgent_InitialValue_offline]
    agent_class_list = [DQNMCTSAgent_ReduceBreadth]


    show_pre_trained_error_grid = [False, False],
    show_values_grid = [False, False],
    show_model_error_grid = [False, False]

    # s_vf_list = [2 ** -5, 2 ** -7, 2 ** -9, 2 ** -11]
    s_vf_list = [2 ** -7]

    s_md_list = [2 ** -9]


    c_list = [2 ** -1, 2 ** 0, 2 ** 0.5, 2 ** 1]
    # c_list = [2 ** 1]

    num_iteration_list = [100]#[i for i in range(30, 40, 10)]
    # simulation_depth_list = [200]
    simulation_depth_list = [5, 10, 75]
    # num_simulation_list = [10]
    num_simulation_list = [1]

    # model_list = [{'type':'forward', 'num_networks':1, 'layers_type':['fc'], 'layers_features':[128]},
    #               {'type': 'forward', 'num_networks': 2, 'layers_type': ['fc'], 'layers_features': [64]},
    #               {'type': 'forward', 'num_networks': 4, 'layers_type': ['fc'], 'layers_features': [32]}
    #               ]

    model_list = [{'type': None, 'num_networks': 1, 'layers_type': ['fc'], 'layers_features': [128]}]
    vf_list = [
        # {'type': 'q', 'layers_type': ['fc', 'fc'], 'layers_features': [64, 64], 'action_layer_num': 3},
        # {'type': 'q', 'layers_type': ['fc', 'fc'], 'layers_features': [32, 32], 'action_layer_num': 3},
        {'type': 'q', 'layers_type': ['fc', 'fc'], 'layers_features': [16, 16], 'action_layer_num': 3},
        # {'type': 'q', 'layers_type': ['fc', 'fc'], 'layers_features': [16, 8], 'action_layer_num': 3},
        # {'type': 'q', 'layers_type': ['fc', 'fc'], 'layers_features': [8, 8], 'action_layer_num': 3},
        # {'type': 'q', 'layers_type': ['fc', 'fc'], 'layers_features': [4, 4], 'action_layer_num': 3}
               ]

    experiment = GridWorld_RunExperiment()

    experiment_object_list = []
    for agent_class in agent_class_list:
        for s_vf in s_vf_list:
            for model in model_list:
                for vf in vf_list:
                    for s_md in s_md_list:
                        for c in c_list:
                            for num_iteration in num_iteration_list:
                                for simulation_depth in simulation_depth_list:
                                    for num_simulation in num_simulation_list:
                                        params = {'pre_trained': None,
                                                  'vf_step_size': s_vf,
                                                  'vf': vf,
                                                  'model': model,
                                                  'model_step_size': s_md,
                                                  'c': c,
                                                  'num_iteration': num_iteration,
                                                  'simulation_depth': simulation_depth,
                                                  'num_simulation': num_simulation}
                                        obj = ExperimentObject(agent_class, params)
                                        experiment_object_list.append(obj)
    # x = time.time()
    # detail = "Env = 4room - 4x4; Not keep subtree; max_episode = 100; Pretrained DQN - DQN VF: 16x8 dqn_vf_9.p"
    # detail = "Env = Empty Room; _n = 20; max_episode = 100; Pretrained DQN - DQN VF: 16x8 dqn_vf_5.p"

    # experiment.run_experiment(experiment_object_list, result_file_name="DQNMCTS_InitialValue_PretrainedDQN_AutoImperfect15", detail=detail)
    # experiment.run_experiment(experiment_object_list, result_file_name="DQNMCTS_BootstrapInitial_PretrainedDQN_AutoImperfect15", detail=detail)
    # experiment.run_experiment(experiment_object_list, result_file_name="DQNMCTS_BootstrapInitial_PretrainedDQN_AutoImperfect_prob=0.025_step=1", detail=detail)

    # detail = "Env = 4room - 4x4; Not keep subtree; max_episode = 100"
    # experiment.run_experiment(experiment_object_list, result_file_name="ddd", detail=detail)
    detail = "Env = 4room - 4x4; Not keep subtree; max_episode = 100"
    experiment.run_experiment(experiment_object_list, result_file_name="DQNMCTSAgent_Bootstrap_VF16x16", detail=detail)

    # detail = "Env = Empty Room; _n = 20; max_episode = 100"
    # experiment.run_experiment(experiment_object_list, result_file_name="MCTS_BestParameter", detail=detail)


    # print(time.time() - x)