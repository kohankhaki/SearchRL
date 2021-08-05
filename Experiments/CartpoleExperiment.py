from Environments.Cartpole import CartPoleEnv
import numpy as np
import torch
import os
import Utils as utils, Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter

from Experiments.BaseExperiment import BaseExperiment

os.environ['KMP_DUPLICATE_LIB_OK']='True'

debug = True

class CartpoleExperiment(BaseExperiment):
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
        s = self.environment.reset()
        obs = self.observationChannel(s)
        self.last_action = self.agent.start(obs)
        self.visited_states = np.append(self.visited_states, [s], axis=0)
        return (obs, self.last_action)

    def step(self):
        (s, reward, term, info) = self.environment.step(self.last_action)
        self.visited_states = np.append(self.visited_states, [s], axis=0)
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
        writer = SummaryWriter()
        num_runs = config.num_runs
        num_episode = config.num_episode
        max_step_each_episode = config.max_step_each_episode
        self.num_steps_run_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        # ****
        self.simulation_steps_run_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        self.consistency = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        # ****
        self.model_error_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.float)
        self.agent_model_error_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.float)
        self.model_error_samples = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        # ****
        self.het_mu_error = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.float)
        self.het_var_error = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.float)
        self.het_error = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.float)

        for i, obj in tqdm(enumerate(experiment_object_list)):
            print("---------------------")
            print("This is the case: ", i)

            for r in range(num_runs):
                print("starting runtime ", r+1)
                env = CartPoleEnv()
                # initializing the agent
                agent = obj.agent_class({'action_list': np.arange(env.action_space.n),
                                       'gamma': 0.99, 'epsilon_max': 0.9, 'epsilon_min': 0.05, 'epsilon_decay': 200,
                                       'model_corruption':obj.model_corruption,
                                       'max_stepsize': obj.vf_step_size,
                                       'model_stepsize': obj.model_step_size,
                                       'reward_function': None,
                                       'goal': None,
                                       'device': self.device,
                                       'model': obj.model,
                                       'true_bw_model': None,
                                       'true_fw_model': env.transition_dynamics,
                                       'transition_dynamics':None,
                                       'c': obj.c,
                                       'num_iteration': obj.num_iteration,
                                       'simulation_depth': obj.simulation_depth,
                                       'num_simulation': obj.num_simulation,
                                       'vf': obj.vf_network, 'dataset':None})
                
               

                #initialize experiment
                experiment = CartpoleExperiment(agent, env, self.device)
                for e in range(num_episode):
                    if debug:
                        print("starting episode ", e + 1)
                    experiment.runEpisode(max_step_each_episode)
                    
                    # transition_batch = agent.getTransitionFromBuffer(agent.transition_buffer_size)
                    # batch = utils.transition(*zip(*transition_batch))
                    # prev_state_batch = torch.cat(batch.prev_state)
                    # prev_action_batch = torch.cat(batch.prev_action)
                    # rnd_error = agent.getRndError(prev_state_batch, prev_action_batch)
                    # ensemble_error = agent.getEnsembleError(prev_state_batch, prev_action_batch)
                    # v, u = agent.getStateValueUncertainty(prev_state_batch, type="rnd")
                    # print(torch.mean(rnd_error), torch.mean(ensemble_error), len(prev_state_batch))
                    # agent.checkHetValues(transition_batch)

                    self.num_steps_run_list[i, r, e] = experiment.num_steps
                    # writer.add_scalar("num_steps"+str(obj.vf_step_size), experiment.num_steps, e)
                # agent.saveModelFile("LearnedModel/HeteroscedasticLearnedModel/r"+str(r)+ "_stepsize"+str(obj.model_step_size)+"_network"+"16x4")
                # agent.saveModelFile("LearnedModel/HeteroscedasticLearnedModel/TestCartpole"+"_stepsize"+str(obj.model_step_size)+"_network"+"6")

                #**********************
                # agent.loadModelFile("LearnedModel/HeteroscedasticLearnedModel/r0_stepsize0.0009765625_network64x32")
                # print(agent._model['heter']['network'])
                # import torch.nn as nn
                # import torch.nn.functional as F
                # for m in agent._model['heter']['network'].modules():
                #     if isinstance(m, nn.Linear):
                #         print(m.weight.data)    
                #         print("*****************")   
                # mu_err_sum = 0
                # var_err_sum = 0
                # counter = 0
                # for s in experiment.visited_states[-100:-1]:
                #     for a in range(env.action_space.n):
                #         action_index = torch.tensor([agent.getActionIndex(a)], device=self.device).unsqueeze(0)
                #         one_hot_action = agent.getActionOnehotTorch(action_index)
                #         state = torch.from_numpy(s).unsqueeze(0)
                #         predicted_next_state_var = agent._model['heter']['network'](state, one_hot_action)[1].float().detach()
                #         predicted_next_state = agent._model['heter']['network'](state, one_hot_action)[0].float().detach()
                #         predicted_next_state_var_trace = torch.sum(predicted_next_state_var, dim=1)
                #         true_next_state = torch.tensor(np.asarray(env.transition_dynamics(s, a)[0]))

                #         true_var = torch.sum((predicted_next_state - true_next_state) ** 2, dim=1)
                #         var_err = torch.mean((true_var - predicted_next_state_var_trace)**2)
                #         mu_err = torch.mean(true_var) 
                #         mu_err_sum += mu_err
                #         var_err_sum += var_err
                #         counter += 1
                #         print(s, a, true_next_state, predicted_next_state, true_var.item(), predicted_next_state_var_trace.item())
                # print(mu_err_sum/counter, var_err_sum/counter)
                # exit(0)
                #**********************

        with open("CartpoleResults/" + result_file_name + '.p', 'wb') as f:
            result = {'num_steps': self.num_steps_run_list,
                      'experiment_objs': experiment_object_list,
                      'detail': detail,
                      'het_error':self.het_error, 
                      'mu_error':self.het_mu_error, 
                      'var_error':self.het_var_error}
            pickle.dump(result, f)
        # show_num_steps_plot(self.num_steps_run_list, ["Uncertain_MCTS"])
        save_num_steps_plot(self.num_steps_run_list, experiment_object_list)


def show_num_steps_plot(num_steps, agent_names):
    num_steps_avg = np.mean(num_steps, axis=1)
    writer = SummaryWriter()#(comment="TuneModelwithDQN_LR{2**-4to-16}_BATCH{16}")
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
    
    #dqn params
    vf_network = [i.vf_network for i in experiment_objs]
    if not all_elements_equal(vf_network):
        for i in range(len(experiment_objs)):
            names[i] += "vf:" + str(vf_network[i])

    vf_step_size = [i.vf_step_size for i in experiment_objs]
    if not all_elements_equal(vf_step_size):
        for i in range(len(experiment_objs)):
            names[i] += "vf_stepsize:" + str(vf_step_size[i])

    #mcts params
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