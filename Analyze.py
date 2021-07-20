import numpy as np
import random
from matplotlib import pyplot as plt
import pickle
import Config as config

generate_random_color = False
color_counter = 0
color_list = ['#FF2929', '#19A01D', '#F4D03F', '#FF7F50', '#8E44AD', '#34495E', '#95A5A6', '#5DADE2', '#A2FF00', '#003AFF', '#FF008F']


def generate_hex_color():
    global color_counter
    if generate_random_color:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        c = (r, g, b)
        hex_c = '#%02x%02x%02x' % c
    else:
        hex_c = color_list[color_counter]
        color_counter = (color_counter + 1) % len(color_list)
    return hex_c


def drawPlotUncertainty(x, y, y_err, color, label, axis):
    axis.plot(x, y, color = color, label=label)
    axis.fill_between(x,
                      y - y_err,
                      y + y_err,
                      facecolor=color, alpha=.4, edgecolor='none')


def plot_simple_agent(steps_run_list, label_name, axs):
    # steps_run_list = steps_run_list[:, :, 0:25]
    mean_steps_run_list = np.mean(steps_run_list, axis=1)
    std_steps_run_list = np.std(steps_run_list, axis=1)
    x_steps_run_list = np.arange(steps_run_list.shape[2])

    for stepsize_index in range(steps_run_list.shape[0]):
        drawPlotUncertainty(x_steps_run_list,
                            mean_steps_run_list[stepsize_index],
                            std_steps_run_list[stepsize_index],
                            # label=label_name + str(stepsize_index),
                            label=label_name,
                            color=generate_hex_color(),
                            axis=axs)

def plot_simple_agent_each_run(steps_run_list, label_name, axs):

    x_steps_run_list = np.arange(steps_run_list.shape[2])
    for run_index in range(steps_run_list.shape[1]):
        fig, axs = plt.subplots(1, 1, constrained_layout=False)
        drawPlotUncertainty(x_steps_run_list,
                            steps_run_list[1][run_index],
                            np.zeros(steps_run_list[1][run_index].shape),
                            label=label_name + "_" + str(run_index),
                            color=generate_hex_color(),
                            axis=axs)

        axs.title.set_text("dqn vf 6464")
        axs.legend()
        fig.savefig("dqn vf 6464_" + str(run_index) + ".png")
        fig.show()


def plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True):

    mean_steps_run_list = np.mean(steps_run_list, axis=1)
    std_steps_run_list = np.std(steps_run_list, axis=1)
    # x_steps_run_list = np.arange(steps_run_list.shape[2])

    mean_steps_run_list = np.mean(mean_steps_run_list, axis=1)
    std_steps_run_list = np.std(std_steps_run_list, axis=1)

    # best_par = np.argmin(mean_steps_run_list)
    
    # for stepsize_index in range(mean_steps_run_list.shape[0]):
    #     axs.axhline(mean_steps_run_list[stepsize_index], color=generate_hex_color(), label=label_name+str(stepsize_index))
    
    line_style = "dashed"
    if is_imperfect:
        line_style = "solid"
        # for stepsize_index in range(mean_steps_run_list.shape[0]):
        for stepsize_index in range(mean_steps_run_list.shape[0]):
            axs.axhline(mean_steps_run_list[stepsize_index], color=generate_hex_color(), label=label_name, linestyle=line_style)
    else:
        for stepsize_index in range(mean_steps_run_list.shape[0]):
            axs.axhline(mean_steps_run_list[stepsize_index], color=generate_hex_color(), label=label_name + str(stepsize_index // 2), linestyle=line_style)

def plot_alternate_agents(steps_run_list, label_name1, label_name2, axs):
    # steps_run_list = steps_run_list[:, :, 0:800]
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

    for p in range(steps_run_list.shape[0]):
        color = generate_hex_color()
        drawPlotUncertainty(x_agent1_steps_run_list,
                            mean_agent1_steps_run_list[p],
                            std_agent1_steps_run_list[p],
                            label=label_name1,
                            color=color,
                            axis=axs)
        color = generate_hex_color()
        drawPlotUncertainty(x_agent2_steps_run_list,
                            mean_agent2_steps_run_list[p],
                            std_agent2_steps_run_list[p],
                            label=label_name2,
                            color=color,
                            axis=axs)

def plot_alternate_agents_single_episode(steps_run_list, label_name1, label_name2, axs, plot_first=True,  is_imperfect = False):



    num_step = config.episodes_only_dqn
    agent1_ind = list(range(0, num_step, 1))
    agent2_ind = list(range(num_step, steps_run_list.shape[2], 1))
    agent1_step_list = np.delete(steps_run_list, agent2_ind, 2)
    agent2_step_list = np.delete(steps_run_list, agent1_ind, 2)

    # num_step = steps_run_list.shape[2] // 2
    # odd_ind = list(range(1, num_step * 2, 2))
    # even_ind = list(range(0, num_step * 2, 2))
    # agent1_step_list = np.delete(steps_run_list, odd_ind, 2)
    # agent2_step_list = np.delete(steps_run_list, even_ind, 2)


    # mean_agent1_steps_run_list = np.mean(agent1_step_list, axis=1)
    # std_agent1_steps_run_list = np.std(agent1_step_list, axis=1)
    # x_agent1_steps_run_list = np.arange(agent1_step_list.shape[2])


    # drawPlotUncertainty(x_agent1_steps_run_list,
    #                     mean_agent1_steps_run_list[0],
    #                     std_agent1_steps_run_list[0],
    #                     label=label_name1,
    #                     color=generate_hex_color(),
    #                     axis=axs)


    # mean_agent2_steps_run_list = np.mean(agent2_step_list, axis=1)
    # std_agent2_steps_run_list = np.std(agent2_step_list, axis=1)
    # x_agent2_steps_run_list = np.arange(agent2_step_list.shape[2])
    
    # drawPlotUncertainty(x_agent2_steps_run_list,
    #                 mean_agent2_steps_run_list[0],
    #                 std_agent2_steps_run_list[0],
    #                 label=label_name2,
    #                 color=generate_hex_color(),
    #                 axis=axs)


    line_style = "dashed"
    if is_imperfect:
        line_style = "solid"

    mean_agent1_steps_run_list = np.mean(agent1_step_list, axis=1)
    mean_agent1_steps_run_list = np.mean(mean_agent1_steps_run_list, axis=1)

    if plot_first:
        for stepsize_index in range(mean_agent1_steps_run_list.shape[0]):
            axs.axhline(mean_agent1_steps_run_list[stepsize_index], color=generate_hex_color(), label=label_name1, linestyle=line_style)

    mean_agent2_steps_run_list = np.mean(agent2_step_list, axis=1)
    mean_agent2_steps_run_list = np.mean(mean_agent2_steps_run_list, axis=1)

    for stepsize_index in range(mean_agent2_steps_run_list.shape[0]):
        axs.axhline(mean_agent2_steps_run_list[stepsize_index], color=generate_hex_color(), label=label_name2, linestyle=line_style)



fig, axs = plt.subplots(1, 1, constrained_layout=False)






file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25.p'
with open(file_name, "rb") as f:
    res = pickle.load(f)
steps_run_list = res['num_steps']

label_name = 'Imperfect MCTS'

plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)




file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionDivLinear2m2_2.p'
with open(file_name, "rb") as f:
    res = pickle.load(f)
steps_run_list = res['num_steps']
label_name = 'Selection Linear 1'
plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)

file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionDivLinear2m3_2.p'
with open(file_name, "rb") as f:
    res = pickle.load(f)
steps_run_list = res['num_steps']
label_name = 'Selection Linear 2'
plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)

file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionDivLinear2m4_2.p'
with open(file_name, "rb") as f:
    res = pickle.load(f)
steps_run_list = res['num_steps']
label_name = 'Selection Linear 3'
plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)

file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionDivLinear2p2_2.p'
with open(file_name, "rb") as f:
    res = pickle.load(f)
steps_run_list = res['num_steps']
label_name = 'Selection Linear 4'
plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)

file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionDivRoot_2.p'
with open(file_name, "rb") as f:
    res = pickle.load(f)
steps_run_list = res['num_steps']
label_name = 'Selection Root 1'
plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)


file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionDivRoot2m2.p'
with open(file_name, "rb") as f:
    res = pickle.load(f)
steps_run_list = res['num_steps']
label_name = 'Selection Root 2'
plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)



file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionDivRoot2m2_2.p'
with open(file_name, "rb") as f:
    res = pickle.load(f)
steps_run_list = res['num_steps']
label_name = 'Selection Root 3'
plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)


# file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionDivRoot2p2_2.p'
# with open(file_name, "rb") as f:
#     res = pickle.load(f)
# steps_run_list = res['num_steps']
# label_name = 'Selection Root 2'
# plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)



# file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionLinear2p0.p'
# with open(file_name, "rb") as f:
#     res = pickle.load(f)
# steps_run_list = res['num_steps']
# label_name = 'Selection_Linear2p0'
# plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)




# file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionLinear2p1.p'
# with open(file_name, "rb") as f:
#     res = pickle.load(f)
# steps_run_list = res['num_steps']
# label_name = 'Selection_Linear2p1'
# plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)



# file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionRoot2p0.p'
# with open(file_name, "rb") as f:
#     res = pickle.load(f)
# steps_run_list = res['num_steps']
# label_name = 'Selection_Root2p0'
# plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)






# file_name = 'Results_GivenModelPerfectUncertainty/ImperfectMCTSAgentIdeas_S7P25_SelectionRoot2p1.p'
# with open(file_name, "rb") as f:
#     res = pickle.load(f)
# steps_run_list = res['num_steps']
# label_name = 'Selection_Root2p1'
# plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)




# file_name = 'Results/ImperfectMCTSAgentIdeas_S3P25_UseUncertaintySelection_Root2p1.p'
# with open(file_name, "rb") as f:
#     res = pickle.load(f)
# steps_run_list = res['num_steps']

# label_name = 'Root 3'

# plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)



# file_name = 'Results/ImperfectMCTSAgentIdeas_S3P25_UseUncertaintySelection_Root2pm1.p'
# with open(file_name, "rb") as f:
#     res = pickle.load(f)
# steps_run_list = res['num_steps']

# label_name = 'Root 1'

# plot_simple_agent_single_episode(steps_run_list, label_name, axs, is_imperfect = True)


# axs.title.set_text("Imperfect MCTS Using Given Uncertainty")
axs.legend()
fig.savefig("Imperfect_MCTS_Ideas_Selection_Div_Coefficient")
fig.show()