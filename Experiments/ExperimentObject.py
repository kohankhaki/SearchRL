class ExperimentObject:
    def __init__(self, agent_class, params={}):
        self.agent_class = agent_class
        self.pre_trained = params['pre_trained']
        self.vf_step_size = params['vf_step_size']
        self.model = params['model']
        self.model_step_size = params['model_step_size']

        #mcts params
        self.c = params['c']
        self.num_iteration = params['num_iteration']
        self.simulation_depth = params['simulation_depth']
        self.num_simulation = params['num_simulation']
