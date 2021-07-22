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
                                       'gamma': 0.99, 'epsilon': 0.01,
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
                    self.num_steps_run_list[i, r, e] = experiment.num_steps

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

        with open("Results/" + result_file_name + '.p', 'wb') as f:
            result = {'num_steps': self.num_steps_run_list,
                      'experiment_objs': experiment_object_list,
                      'detail': detail,
                      'het_error':self.het_error, 
                      'mu_error':self.het_mu_error, 
                      'var_error':self.het_var_error}
            pickle.dump(result, f)
        # show_num_steps_plot(self.num_steps_run_list, ["Uncertain_MCTS"])



def show_num_steps_plot (num_steps, agent_names):
        num_steps_avg = np.mean(num_steps, axis=1)
        writer = SummaryWriter()
        counter = 0

        for agent in range(num_steps_avg.shape[0]):
            name = agent_names[agent] 
            for i in range(num_steps_avg.shape[1]):
                writer.add_scalar(name, num_steps_avg[agent, i], counter)
                counter += 1
        writer.close()












