'''
train a DQN with MCTS
See if DQN agrees with DQN (saving the best path)
At what step DQN starts to work with MCTS
'''
from Experiments.ExperimentObject import ExperimentObject
from Experiments.GridWorldExperiment import RunExperiment as GridWorld_RunExperiment
from Environments.GridWorldRooms import GridWorldRooms
from Agents.BaseDynaAgent import BaseDynaAgent
from Agents.MCTSAgent import MCTSAgent
from Agents.DQNMCTSAgent import *



if __name__ == '__main__':

    # agent_class_list = [BaseDynaAgent]
    # agent_class_list = [DQNMCTSAgent_InitialValue]
    # agent_class_list = [DQNMCTSAgent_Bootstrap]
    # agent_class_list = [MCTSAgent]
    agent_class_list = [DQNMCTSAgent_UseTreeExpansion]
    # agent_class_list = [DQNMCTSAgent_UseTree]



    show_pre_trained_error_grid = [False, False],
    show_values_grid = [False, False],
    show_model_error_grid = [False, False]

    s_vf_list = [2 ** -7]
    s_md_list = [2 ** -9]

    c_list = [2]
    num_iteration_list = [50]
    simulation_depth_list = [25]
    num_simulation_list = [1]


    # model_list = [{'type':'forward', 'num_networks':1, 'layers_type':['fc'], 'layers_features':[128]},
    #               {'type': 'forward', 'num_networks': 2, 'layers_type': ['fc'], 'layers_features': [64]},
    #               {'type': 'forward', 'num_networks': 4, 'layers_type': ['fc'], 'layers_features': [32]}
    #               ]

    model_list = [{'type': None, 'num_networks': 1, 'layers_type': ['fc'], 'layers_features': [128]}]

    experiment = GridWorld_RunExperiment()

    experiment_object_list = []
    for agent_class in agent_class_list:
        for s_vf in s_vf_list:
            for model in model_list:
                for s_md in s_md_list:
                    for c in c_list:
                        for num_iteration in num_iteration_list:
                            for simulation_depth in simulation_depth_list:
                                for num_simulation in num_simulation_list:
                                    params = {'pre_trained': None,
                                              'vf_step_size': s_vf,
                                              'model': model,
                                              'model_step_size': s_md,
                                              'c': c,
                                              'num_iteration': num_iteration,
                                              'simulation_depth': simulation_depth,
                                              'num_simulation': num_simulation}
                                    obj = ExperimentObject(agent_class, params)
                                    experiment_object_list.append(obj)

    experiment.run_experiment(experiment_object_list, result_file_name="DQN")
