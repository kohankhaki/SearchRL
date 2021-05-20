import numpy as np
import torch
import os
import Utils as utils, Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from Experiments.BaseExperiment import BaseExperiment
from Environments.GridWorldBase import GridWorld
from Environments.GridWorldRooms import GridWorldRooms
# from Networks.ModelNN.StateTransitionModel import preTrainBackward, preTrainForward
from Datasets.TransitionDataGrid import data_store

os.environ['KMP_DUPLICATE_LIB_OK']='True'

debug = True

class GridWorldExperiment(BaseExperiment):
    def __init__(self, agent, env, device, params=None):
        if params is None:
            params = {'render': False}
        super().__init__(agent, env)

        self._render_on = params['render']
        self.num_steps_to_goal_list = []
        self.visit_counts = self.createVisitCounts(env)
        self.num_samples = 0
        self.device = device

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

        # while ((max_steps == 0) or (self.num_steps < max_steps)):
        #     rl_step_result = self.step()
        #     is_terminal = rl_step_result[3]
        #     if is_terminal:
        #         self.start()

        self.num_episodes += 1
        self.num_steps_to_goal_list.append(self.num_steps)
        if debug:
            print("num steps: ", self.num_steps)
        return is_terminal

    def observationChannel(self, s):
        return np.asarray(s)

    def recordTrajectory(self, s, a, r, t):
        self.updateVisitCounts(s, a)

    def agentPolicyEachState(self):
        all_states = self.environment.getAllStates()

        for state in all_states:
            agent_state = self.agent.getStateRepresentation(state)
            policy = self.agent.policy(agent_state, greedy= True)
            print(policy)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def updateVisitCounts(self, s, a):
        if a is None:
            return 0
        pos = self.environment.stateToPos(s)
        self.visit_counts[(pos), tuple(a)] += 1

    def createVisitCounts(self, env):
        visit_count = {}
        for state in env.getAllStates():
            for action in env.getAllActions():
                pos = env.stateToPos(state)
                visit_count[(pos, tuple(action))] = 0
        return visit_count
        

class RunExperiment():
    def __init__(self):
        gpu_counts = torch.cuda.device_count()
        # self.device = torch.device("cuda:"+str(random.randint(0, gpu_counts-1)) if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        # self.show_pre_trained_error_grid = config.show_pre_trained_error_grid
        # self.show_values_grid = config.show_values_grid
        # self.show_model_error_grid = config.show_model_error_grid

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

        for i, obj in tqdm(enumerate(experiment_object_list)):
            print("---------------------")
            print("This is the case: ", i)
            pre_trained_plot_y_run_list = []
            pre_trained_plot_x_run_list = []
            for r in range(num_runs):
                print("starting runtime ", r+1)
                # env = GridWorld(params=config.empty_room_params)
                env = GridWorldRooms(params=config.n_room_params)

                train, test = data_store(env)
                reward_function = env.rewardFunction
                goal = np.asarray(env.posToState((0, config._n - 1), state_type='coord'))

                # Pre-train the model
                # pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x = \
                #     self.pretrain_model(obj.pre_trained_model, env)
                # pre_trained_plot_y_run_list.append(pre_trained_plot_y)
                # pre_trained_plot_x_run_list.append(pre_trained_plot_x)

                # initializing the agent
                agent = obj.agent_class({'action_list': np.asarray(env.getAllActions()),
                                       'gamma': 0.99, 'epsilon': 0.1,
                                       'max_stepsize': obj.vf_step_size,
                                       'model_stepsize': obj.model_step_size,
                                       'reward_function': reward_function,
                                       'goal': goal,
                                       'device': self.device,
                                       'model': obj.model,
                                       'true_bw_model': env.transitionFunctionBackward,
                                       'true_fw_model': env.coordTransitionFunction,
                                       'transition_dynamics':env.transition_dynamics,
                                       'c': obj.c,
                                       'num_iteration': obj.num_iteration,
                                       'simulation_depth': obj.simulation_depth,
                                       'num_simulation': obj.num_simulation,
                                       'vf': obj.vf_network,})

                #initialize experiment
                experiment = GridWorldExperiment(agent, env, self.device)
                # self.test_model(env, agent,"before.txt")
                # self.pretrain_model2(env, agent)
                # self.test_model(env, agent, "after.txt")
                # exit(0)
                for e in range(num_episode):
                    if debug:
                        print("starting episode ", e + 1)
                    experiment.runEpisode(max_step_each_episode)
                    self.num_steps_run_list[i, r, e] = experiment.num_steps
                    # print(self.model_error(agent, env))
                    # if e % 100 == 0:
                    #     self.test_model(env, agent, "after-episode" + str(e) + ".txt")


                    # if agent.name == 'DQNMCTSAgent':
                    #     self.simulation_steps_run_list[i, r, e] = self.simulate_dqn(agent.policy, agent.true_model,
                    #                                                                 env.start(), env.getAllActions())
                    #     self.consistency[i, r, e] = agent.action_consistency / experiment.num_steps

                # vf_error = self.calculate_dqn_vf_error(agent, env)
                # print('DQN VF ERROR:', vf_error)
                    # if e % 100 == 0:
                    #     mean = np.mean(self.num_steps_run_list[0], axis=0)
                    #     plt.plot(mean[0:e])
                    #     plt.show()

                    # if agent.name != 'BaseDynaAgent' and agent.name != 'BaseMCTSAgent' and agent.name != 'DQNMCTSAgent':
                    #     model_type = list(agent.model.keys())[0]
                    #     # agent_model_error = experiment.calculateModelErrorError(agent.model[model_type],
                    #     #                                     test,
                    #     #                                     type=str(model_type),
                    #     #                                     true_transition_function=env.transitionFunction)[0]
                    #
                    #     model_error = experiment.calculateModelErrorWithData(agent.model[model_type],
                    #                                                      test,
                    #                                                      type=str(model_type),
                    #                                                      true_transition_function=env.transitionFunction)
                    #     self.model_error_list[i, r, e] = model_error
                    #     # self.agent_model_error_list[agent_counter, r, e] = agent_model_error
                    #     self.model_error_samples[i, r, e] = experiment.num_samples

                # agent.saveValueFunction("Results_EmptyRoom/DQNVF_16x8/dqn_vf_" + str(r) + ".p")


                # *********
                # model_type = list(agent.model.keys())[0]
                # utils.draw_grid((config._n, config._n), (900, 900),
                #                 state_action_values=experiment.calculateModelErrorError(agent.model[model_type],
                #                                             test,
                #                                             type=str(model_type),
                #                                             true_transition_function=env.transitionFunction)[1],
                #                 all_actions=env.getAllActions(),
                #                 obstacles_pos=env.get_obstacles_pos())
                # *********

        # self.show_model_error_plot()
        # self.show_agent_model_error_plot()
        # with open('sim_num_steps_run_list.npy', 'wb') as f:
        #     np.save(f, self.simulation_steps_run_list)
        with open("Results/" + result_file_name + '.p', 'wb') as f:
            result = {'num_steps': self.num_steps_run_list,
                      'experiment_objs': experiment_object_list,
                      'detail': detail}
            pickle.dump(result, f)
            # np.save(f, self.num_steps_run_list)
        # with open('model_error_run.npy', 'wb') as f:
        #     np.save(f, self.model_error_list)
        #     np.save(f, self.model_error_samples)
        self.show_num_steps_plot()

    def calculate_dqn_vf_error(self, agent, env):
        states = env.getAllStates()
        actions = env.getAllActions()
        error = 0
        for state in states:
            for action in actions:
                env_vf = env.calculate_state_action_value(state, action, agent.gamma)
                torch_state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                agent_vf = agent.getStateActionValue(torch_state, action)
                error += (env_vf - agent_vf) ** 2
        return error

    def model_error(self, agent, env):
        error = 0
        counter = 0
        for s in env.getAllStates(state_type='coord'):
            state = agent.getStateRepresentation(s)
            for a in env.getAllActions():
                # true_next_state = env.transitionFunctionBackward(s, a)
                action_index = torch.tensor([agent.getActionIndex(a)], device=self.device).unsqueeze(0)
                true_next_state = torch.tensor([env.transitionFunction(s, a)], device=self.device)
                pred_next_state, pred_error = agent.modelRollout(state, action_index)
                error += torch.dist(pred_next_state, true_next_state)
                counter += 1
        return error / counter

    def calculate_model_error(self, agent, env):
        error = {}
        for s in env.getAllStates(state_type='coord'):
            state = agent.getStateRepresentation(s)
            for a in env.getAllActions():
                # true_next_state = env.transitionFunctionBackward(s, a)
                true_next_state = env.transitionFunction(s, a)

                distance_pred = []
                for i in range(agent.model['forward']['num_networks']):
                    distance_pred.append(torch.dist(agent.rolloutWithModel(state, a, agent.model['forward'], net_index=i),
                                                    torch.from_numpy(true_next_state).float()))
                # print(distance_pred)
                pos = env.stateToPos(s, state_type='coord')
                model_index = distance_pred.index(min(distance_pred))
                if ((pos),tuple(a)) not in error:
                    # error[(pos), tuple(a)] = round(distance_pred[0].item(),3), round(distance_pred[1].item(),3), round(distance_pred[2].item(),3)
                    # error[(pos), tuple(a)] = str(model_index), round(distance_pred[model_index].item(), 3)
                    try:
                        error[(pos), tuple(a)] = str(round(distance_pred[model_index].item(), 3)) + "\n" + \
                                                 str(agent.counter[(pos), tuple(a)])

                        # error[(pos), tuple(a)] = str(model_index) + "\n" + \
                        #                          str(agent.counter[(pos), tuple(a)])
                    except:
                        error[(pos), tuple(a)] = str(round(distance_pred[model_index].item(), 3)) + "\n" + \
                                                 str(0)
                        # error[(pos), tuple(a)] = str(model_index) + "\n" + \
                        #                          str((0, 0, 0))

        utils.draw_grid((env.grid_size[0], env.grid_size[1]), (900, 900),
                        state_action_values=error,
                        all_actions=env.getAllActions(),
                        obstacles_pos=env.get_obstacles_pos())

    def pretrain_model(self, model_type, env):
        if model_type == 'forward':
            pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x = preTrainForward(
                env, self.device)
        elif model_type == 'backward':
            pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x = preTrainBackward(
                env, self.device)
        elif model_type is None:
            return None, None, None, None
        else:
            raise ValueError("model type not defined")

        return pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x

    def show_num_steps_plot(self):
        if False:
            for a in range(self.num_steps_run_list.shape[0]):
                agent_name = self.agents[a].name
                for r in range(self.num_steps_run_list.shape[1]):
                    utils.draw_plot(range(len(self.num_steps_run_list[a,r])), self.num_steps_run_list[a,r],
                            xlabel='episode_num', ylabel='num_steps', show=True,
                            label=agent_name, title='run_number '+str(r+1))
        if False:
            for r in range(self.num_steps_run_list.shape[1]):
                for a in range(self.num_steps_run_list.shape[0]):
                    agent_name = self.agents[a].name
                    utils.draw_plot(range(len(self.num_steps_run_list[a,r])), self.num_steps_run_list[a,r],
                            xlabel='episode_num', ylabel='num_steps', show=False,
                            label=agent_name, title='run_number '+str(r+1))
                plt.show()

        if False:
            color=['blue','orange','green']
            for a in range(self.num_steps_run_list.shape[0]):
                agent_name = self.agents[a].name
                average_num_steps_run = np.mean(self.num_steps_run_list[a], axis=0)
                std_err_num_steps_run = np.std(self.num_steps_run_list[a], axis=0)
                AUC = sum(average_num_steps_run)
                print("AUC:", AUC, agent_name)
                utils.draw_plot(range(len(average_num_steps_run)), average_num_steps_run,
                        std_error = std_err_num_steps_run,
                        xlabel='episode_num', ylabel='num_steps', show=False,
                        label=agent_name + str(a), title= 'average over runs',
                        sub_plot_num='4'+'1' + str(a+1), color=color[a])

                utils.draw_plot(range(len(average_num_steps_run)), average_num_steps_run,
                                std_error=std_err_num_steps_run,
                                xlabel='episode_num', ylabel='num_steps', show=False,
                                label=agent_name + str(a), title='average over runs',
                                sub_plot_num=414)

            # plt.savefig('')
            plt.show()

    def show_model_error_plot(self):

        if False:
            for a in range(self.model_error_list.shape[0]):
                agent_name = self.agents[a].name
                for r in range(self.model_error_list.shape[1]):
                    utils.draw_plot(range(len(self.model_error_samples[a,r])), self.model_error_list[a,r],
                            xlabel='num_samples', ylabel='model_error', show=True,
                            label=agent_name, title='run_number '+str(r+1))
        if False:
            for r in range(self.model_error_list.shape[1]):
                for a in range(self.model_error_list.shape[0]):
                    agent_name = self.agents[a].name
                    utils.draw_plot(range(len(self.model_error_list[a,r])), self.model_error_list[a,r],
                            xlabel='num_samples', ylabel='model_error', show=False,
                            label=agent_name, title='run_number '+str(r+1))
                plt.show()

        if False:
            color=['blue','orange','green']
            for a in range(self.model_error_list.shape[0]):
                agent_name = self.agents[a].name
                average_model_error_run = np.mean(self.model_error_list[a], axis=0)
                std_err_model_error_run = np.std(self.model_error_list[a], axis=0)
                AUC = sum(average_model_error_run)
                print("AUC:", AUC, agent_name)
                utils.draw_plot(range(len(average_model_error_run)), average_model_error_run,
                        std_error = std_err_model_error_run,
                        xlabel='num_samples', ylabel='model_error', show=False,
                        label=agent_name + str(a), title= 'average over runs',
                        sub_plot_num='4'+'1' + str(a+1), color=color[a])

                utils.draw_plot(range(len(average_model_error_run)), average_model_error_run,
                                std_error=std_err_model_error_run,
                                xlabel='num_samples', ylabel='model_error', show=False,
                                label=agent_name + str(a), title='average over runs',
                                sub_plot_num=414)

            # plt.savefig('')
            plt.show()

    def show_agent_model_error_plot(self):
        for a in range(self.agent_model_error_list.shape[0]):
            agent_name = self.agents[a].name
            for r in range(self.agent_model_error_list.shape[1]):
                utils.draw_plot(range(len(self.model_error_samples[a, r])), self.agent_model_error_list[a, r],
                                xlabel='num_samples', ylabel='agent_model_error', show=True,
                                label=agent_name, title='run_number ' + str(r + 1))

    def simulate_dqn(self, policy, model, init_state, action_list):
        num_steps = 0
        is_terminal = False
        state = init_state
        while num_steps < config.max_step_each_episode and not is_terminal:
            num_steps += 1
            torch_state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            action_ind = policy(torch_state).item()
            action = action_list[action_ind]
            next_state, is_terminal, reward = model(state, action)
            state = next_state
        return num_steps

    def pretrain_model2(self, env, agent):
        # don't use it on the real agent
        buffer = []
        num_states = len(env.getAllStates())
        num_actions = len(env.getAllActions())
        agent.start(env.getAllStates(state_type='coord')[0])
        for s in env.getAllStates(state_type='coord'):
            state = agent.getStateRepresentation(s)
            for a in env.getAllActions():
                # true_next_state = env.transitionFunctionBackward(s, a)
                action_index = torch.tensor([agent.getActionIndex(a)], device=self.device)
                true_next_state = torch.tensor([env.transitionFunction(s, a)], device=self.device)
                
                agent.updateTransitionBuffer(utils.transition(state, action_index, 0, true_next_state, None, False, 0, 0))
        for i in range(100):
            for i in range((num_states*num_actions) // agent._model['general']['batch_size']):
                agent.trainModel()
            print(self.model_error(agent, env))
    
    def test_model(self, env, agent, file_name):
        # don't use it on the real agent
        agent.start(env.getAllStates(state_type='coord')[0])
        for s in env.getAllStates(state_type='coord'):
            state = agent.getStateRepresentation(s)
            for a in env.getAllActions():
                # true_next_state = env.transitionFunctionBackward(s, a)
                action_index = torch.tensor([agent.getActionIndex(a)], device=self.device).unsqueeze(0)
                true_next_state = torch.tensor([env.transitionFunction(s, a)], device=self.device)
                pred_next_state, model_error = agent.modelRollout(state, action_index)
                print(true_next_state, pred_next_state, model_error, torch.dist(pred_next_state, true_next_state))
                with open(file_name, "a") as file:
                    file.write(str(s)+ str(a) + str( true_next_state.cpu().numpy()) + str(pred_next_state.cpu().numpy()) + "\n")




