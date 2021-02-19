import numpy as np
import random
from matplotlib import pyplot as plt
import pickle

def generate_hex_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    c = (r, g, b)
    hex_c = '#%02x%02x%02x' % c
    return hex_c


def drawPlotUncertainty(x, y, y_err, label, color, axis):
    axis.plot(x, y, color, label=label)
    axis.fill_between(x,
                      y - y_err,
                      y + y_err,
                      facecolor=color, alpha=.4, edgecolor='none')


def plot_simple_agent(steps_run_list, label_name, axs):
    # with open(file_name, 'rb') as f:
    #     steps_run_list = np.load(f)

    print(steps_run_list.shape)
    mean_steps_run_list = np.mean(steps_run_list, axis=1)
    std_steps_run_list = np.std(steps_run_list, axis=1)
    x_steps_run_list = np.arange(steps_run_list.shape[2])

    for stepsize_index in range(mean_steps_run_list.shape[0]):
        drawPlotUncertainty(x_steps_run_list,
                            mean_steps_run_list[stepsize_index],
                            std_steps_run_list[stepsize_index],
                            # label="MCTS(Baseline)" + str(stepsize_index),
                            label=label_name,
                            color=generate_hex_color(),
                            axis=axs)


def plot_simple_agent_single_episode(steps_run_list, label_name, axs):

    mean_steps_run_list = np.mean(steps_run_list, axis=1)
    std_steps_run_list = np.std(steps_run_list, axis=1)
    x_steps_run_list = np.arange(steps_run_list.shape[2])

    mean_steps_run_list = np.mean(mean_steps_run_list, axis=1)
    std_steps_run_list = np.std(std_steps_run_list, axis=1)

    best_par = np.argmin(mean_steps_run_list)
    # print(best_par)
    print("best parameter case: ", mean_steps_run_list[best_par])
    for stepsize_index in range(mean_steps_run_list.shape[0]):
        axs.axhline(mean_steps_run_list[stepsize_index], color=generate_hex_color(), label=label_name+str(stepsize_index))



def plot_alternate_agents(file_name, label_name1, label_name2, axs):
    with open(file_name, 'rb') as f:
        steps_run_list = np.load(f)

    num_step = steps_run_list.shape[2] // 2
    odd_ind = list(range(1, num_step * 2, 2))
    even_ind = list(range(0, num_step * 2, 2))
    agent1_step_list = np.delete(steps_run_list, odd_ind, 2)
    agent2_step_list = np.delete(steps_run_list, even_ind, 2)

    mean_agent1_steps_run_list = np.mean(agent1_step_list, axis=1)
    std_agent1_steps_run_list = np.std(agent1_step_list, axis=1)
    x_agent1_steps_run_list = np.arange(agent1_step_list.shape[2])

    mean_agent2_steps_run_list = np.mean(agent2_step_list, axis=1)
    std_agent2_steps_run_list = np.std(agent2_step_list, axis=1)
    x_agent2_steps_run_list = np.arange(agent2_step_list.shape[2])

    drawPlotUncertainty(x_agent1_steps_run_list,
                        mean_agent1_steps_run_list[0],
                        std_agent1_steps_run_list[0],
                        label=label_name1,
                        color=generate_hex_color(),
                        axis=axs)

    drawPlotUncertainty(x_agent2_steps_run_list,
                        mean_agent2_steps_run_list[0],
                        std_agent2_steps_run_list[0],
                        label=label_name2,
                        color=generate_hex_color(),
                        axis=axs)

if __name__ == "__main__":
    file_name = 'Results/DQN_ParameterStudy.p'
    with open(file_name, "rb") as f:
        res = pickle.load(f)
    # print(res.keys())
    # print(res['num_steps'][0].shape)
    # print(res['experiment_objs'][0].num_iteration)

    fig, axs = plt.subplots(1, 1, constrained_layout=False)

    print(res['num_steps'])
    steps_run_list = res['num_steps']
    label_name = 'DQN'
    # plot_simple_agent_single_episode(steps_run_list, label_name, axs)
    plot_simple_agent(steps_run_list, label_name, axs)
    # file_name = 'Results/DQNMCTS_InitialValue_4by4_num_steps_list.npy'
    # label_name1 = '-'
    # label_name2 = 'Initial Value'
    # plot_alternate_agents(file_name, label_name1, label_name2, axs)
    #
    # file_name = 'Results/DQNMCTS_Bootstrap_4by4_num_steps_list.npy'
    # label_name1 = '-'
    # label_name2 = 'Bootstrap'
    # plot_alternate_agents(file_name, label_name1, label_name2, axs)
    #
    # file_name = 'Results/DQNMCTS_BootstrapInitial_4by4_i30d75_keeptree_num_steps_list.npy'
    # label_name1 = '-'
    # label_name2 = 'Initial Value + Bootstrap'
    # plot_alternate_agents(file_name, label_name1, label_name2, axs)


    # file_name = 'Results/DQNMCTS_InitialValue_4by4_Offline_6464_num_steps_list.npy'
    # label_name = 'Offline MCTS'
    # plot_simple_agent_single_episode(file_name, label_name, axs)
    #
    axs.title.set_text("DQN Parameter Study")
    axs.legend()
    fig.savefig("Results/Plots/DQN_ParameterStudy.png")
    fig.show()
