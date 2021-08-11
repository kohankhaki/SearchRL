from Environments.GridWorldBase import GridWorld
import numpy as np
import torch
import os
import Utils as utils, Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter

from Experiments.BaseExperiment import BaseExperiment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
debug = True


class TwoWayGridExperiment(BaseExperiment):
    def __init__(self, agent, env, device, params=None):
        if params is None:
            params = {'render': False}
        super().__init__(agent, env)

        self._render_on = params['render']
        self.num_steps_to_goal_list = []
        self.num_samples = 0
        self.device = device
        self.visited_states = np.array([[0, 0, 0, 0]])

    def start(self):
        self.num_steps = 0
        s = self.environment.start()
        obs = self.observationChannel(s)
        self.last_action = self.agent.start(obs)
        return (obs, self.last_action)

    def step(self):
        (reward, s, term) = self.environment.step(self.last_action)
        self.num_samples += 1
        obs = self.observationChannel(s)
        self.total_reward += reward

        if self._render_on and self.num_episodes >= 0:
            self.environment.render()

        if term:
            self.agent.end(reward)
            roat = (reward, obs, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.step(reward, obs)
            roat = (reward, obs, self.last_action, term)

        self.recordTrajectory(roat[1], roat[2], roat[0], roat[3])
        return roat

    def runEpisode(self, max_steps=0):
        is_terminal = False
        self.start()

        while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
            rl_step_result = self.step()
            is_terminal = rl_step_result[3]

        self.num_episodes += 1
        self.num_steps_to_goal_list.append(self.num_steps)
        if debug:
            print("num steps: ", self.num_steps)
        return is_terminal

    def observationChannel(self, s):
        return np.asarray(s)

    def recordTrajectory(self, s, a, r, t):
        pass


class RunExperiment():
    def __init__(self):
        self.device = torch.device("cpu")
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(self.device)

    def run_experiment(self, experiment_object_list, result_file_name, detail=None):
        num_runs = config.num_runs
        num_episode = config.num_episode
        max_step_each_episode = config.max_step_each_episode
        self.num_steps_run_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        for i, obj in tqdm(enumerate(experiment_object_list)):
            print("---------------------")
            print("This is the case: ", i)

            for r in range(num_runs):
                print("starting runtime ", r + 1)
                random_obstacle_x = 0#np.random.randint(0, 8)
                random_obstacle_y = np.random.choice([0, 2])
                env = GridWorld(params={'size': (3, 8), 'init_state': (1, 0), 'state_mode': 'coord',
                                      'obstacles_pos': [(1, 1),(1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                                                        (random_obstacle_y, random_obstacle_x)],
                                      'rewards_pos': [(1, 7)], 'rewards_value': [1],
                                      'terminals_pos': [(1, 7)], 'termination_probs': [1],
                                      'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                                      'neighbour_distance': 0,
                                      'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
                                      'transition_randomness': 0.0,
                                      'window_size': (255, 255),
                                      'aging_reward': -1
                                      })

                corrupt_env = GridWorld(params={'size': (3, 8), 'init_state': (1, 0), 'state_mode': 'coord',
                                        'obstacles_pos': [(1, 1),(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)],
                                        'rewards_pos': [(1, 7)], 'rewards_value': [1],
                                        'terminals_pos': [(1, 7)], 'termination_probs': [1],
                                        'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                                        'neighbour_distance': 0,
                                        'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0],
                                        'obstacle_color': [1, 1, 1],
                                        'transition_randomness': 0.0,
                                        'window_size': (255, 255),
                                        'aging_reward': -1
                                        })
                # initializing the agent
                agent = obj.agent_class({'action_list': env.getAllActions(),
                                         'gamma': 1.0, 'epsilon_max': 0.9, 'epsilon_min': 0.05, 'epsilon_decay': 200,
                                         'model_corruption': obj.model_corruption,
                                         'max_stepsize': obj.vf_step_size,
                                         'model_stepsize': obj.model_step_size,
                                         'reward_function': None,
                                         'goal': None,
                                         'device': self.device,
                                         'model': obj.model,
                                         'true_bw_model': None,
                                         'true_fw_model': env.fullTransitionFunction,
                                         'corrupted_fw_model':corrupt_env.fullTransitionFunction,
                                         'transition_dynamics': None,
                                         'c': obj.c,
                                         'num_iteration': obj.num_iteration,
                                         'simulation_depth': obj.simulation_depth,
                                         'num_simulation': obj.num_simulation,
                                         'vf': obj.vf_network, 'dataset': None})

                # initialize experiment
                experiment = TwoWayGridExperiment(agent, env, self.device)
                for e in range(num_episode):
                    if debug:
                        print("starting episode ", e + 1)
                    experiment.runEpisode(max_step_each_episode)
                    self.num_steps_run_list[i, r, e] = experiment.num_steps

        with open("TwoWayGridResult/" + result_file_name + '.p', 'wb') as f:
            result = {'num_steps': self.num_steps_run_list,
                      'experiment_objs': experiment_object_list,
                      'detail': detail,}
            pickle.dump(result, f)
        f.close()
        # show_num_steps_plot(self.num_steps_run_list, ["Uncertain_MCTS"])
        # save_num_steps_plot(self.num_steps_run_list, experiment_object_list)

    def show_experiment_result(self, result_file_name):
        with open("TwoWayGridResult/" + result_file_name + '.p', 'rb') as f:
            result = pickle.load(f)
        f.close()
        show_num_steps_plot(result['num_steps'], ["Uncertain_MCTS"])
        save_num_steps_plot(result['num_steps'], result['experiment_objs'])

def show_num_steps_plot(num_steps, agent_names):
    num_steps_avg = np.mean(num_steps, axis=1)
    writer = SummaryWriter()  # (comment="TuneModelwithDQN_LR{2**-4to-16}_BATCH{16}")
    counter = 0

    for agent in range(num_steps_avg.shape[0]):
        name = agent_names[agent]
        for i in range(num_steps_avg.shape[1]):
            writer.add_scalar(name, num_steps_avg[agent, i], counter)
            counter += 1
    writer.close()


def save_num_steps_plot(num_steps, experiment_objs):
    names = experiment_obj_to_name(experiment_objs)
    fig, axs = plt.subplots(1, 1, constrained_layout=False)

    for i, name in enumerate(names):
        num_steps_avg = np.mean(num_steps[i], axis=0)
        num_steps_std = np.mean(num_steps[i], axis=0)
        x = range(len(num_steps_avg))
        if len(num_steps_avg) == 1:
            color = generate_hex_color()
            axs.axhline(num_steps_avg, label=name, color=color)
            # axs.axhspan(num_steps_avg - 0.1 * num_steps_std,
            #             num_steps_avg + 0.1 * num_steps_std,
            #             alpha=0.4, color=color)

        else:
            axs.plot(x, num_steps_avg, label=name)
            axs.fill_between(x,
                             num_steps_avg - 0.1 * num_steps_std,
                             num_steps_avg + 0.1 * num_steps_std,
                             alpha=.4, edgecolor='none')
        axs.legend()
        fig.savefig("test.png")


def experiment_obj_to_name(experiment_objs):
    def all_elements_equal(List):
        result = all(element == List[0] for element in List)
        if (result):
            return True
        else:
            return False

    names = [""] * len(experiment_objs)
    print(names)
    agent_class = [i.agent_class for i in experiment_objs]
    if not all_elements_equal(agent_class):
        for i in range(len(experiment_objs)):
            names[i] += "agent:" + agent_class[i].name

    pre_trained = [i.pre_trained for i in experiment_objs]
    if not all_elements_equal(pre_trained):
        for i in range(len(experiment_objs)):
            names[i] += "pre_trained:" + str(pre_trained[i])

    model = [i.model for i in experiment_objs]
    if not all_elements_equal(model):
        for i in range(len(experiment_objs)):
            names[i] += "model:" + str(model[i])

    model_step_size = [i.model_step_size for i in experiment_objs]
    if not all_elements_equal(model_step_size):
        for i in range(len(experiment_objs)):
            names[i] += "m_stepsize:" + str(model_step_size[i])

    # dqn params
    vf_network = [i.vf_network for i in experiment_objs]
    if not all_elements_equal(vf_network):
        for i in range(len(experiment_objs)):
            names[i] += "vf:" + str(vf_network[i])

    vf_step_size = [i.vf_step_size for i in experiment_objs]
    if not all_elements_equal(vf_step_size):
        for i in range(len(experiment_objs)):
            names[i] += "vf_stepsize:" + str(vf_step_size[i])

    # mcts params
    c = [i.c for i in experiment_objs]
    if not all_elements_equal(c):
        for i in range(len(experiment_objs)):
            names[i] += "c:" + str(c[i])

    num_iteration = [i.num_iteration for i in experiment_objs]
    if not all_elements_equal(num_iteration):
        for i in range(len(experiment_objs)):
            names[i] += "N_I:" + str(num_iteration[i])

    simulation_depth = [i.simulation_depth for i in experiment_objs]
    if not all_elements_equal(simulation_depth):
        for i in range(len(experiment_objs)):
            names[i] += "S_D:" + str(simulation_depth[i])

    num_simulation = [i.num_simulation for i in experiment_objs]
    if not all_elements_equal(num_simulation):
        for i in range(len(experiment_objs)):
            names[i] += "N_S:" + str(num_simulation[i])

    model_corruption = [i.model_corruption for i in experiment_objs]
    if not all_elements_equal(model_corruption):
        for i in range(len(experiment_objs)):
            names[i] += "M_C:" + str(model_corruption[i])
    print(names)
    return names


def generate_hex_color():
    import random
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    c = (r, g, b)
    hex_c = '#%02x%02x%02x' % c
    return hex_c