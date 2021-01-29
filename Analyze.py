import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

def drawPlotUncertainty(x, y, y_err, label, color, axis):
    axis.plot(x, y, color, label=label)
    axis.fill_between(x,
                      y - y_err,
                      y + y_err,
                      facecolor=color, alpha=.4, edgecolor='none')


with open('Results/DQNnum_steps_run_list.npy', 'rb') as f:
    DQNnum_steps_run_list = np.load(f)

print(DQNnum_steps_run_list.shape)
mean_DQNnum_steps_run_list = np.mean(DQNnum_steps_run_list, axis=1)
std_DQNnum_steps_run_list = np.std(DQNnum_steps_run_list, axis=1)
x_DQNnum_steps_run_list = np.arange(DQNnum_steps_run_list.shape[2])


fig, axs_DQN = plt.subplots(1, 1, constrained_layout=False)
for stepsize_index in range(mean_DQNnum_steps_run_list.shape[0]):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    c = (r, g, b)
    hex_c = '#%02x%02x%02x' % c
    drawPlotUncertainty(x_DQNnum_steps_run_list,
                        mean_DQNnum_steps_run_list[stepsize_index],
                        std_DQNnum_steps_run_list[stepsize_index],
                        label="dqn"+str(stepsize_index),
                        color=hex_c,
                        axis=axs_DQN)

axs_DQN.legend()
fig.show()

with open('Results/MCTS2num_steps_run_list.npy', 'rb') as f:
    MCTSnum_steps_run_list = np.load(f)

print(MCTSnum_steps_run_list.shape)
mean_MCTSnum_steps_run_list = np.mean(MCTSnum_steps_run_list, axis=1)
std_MCTSnum_steps_run_list = np.std(MCTSnum_steps_run_list, axis=1)
x_MCTSnum_steps_run_list = np.arange(MCTSnum_steps_run_list.shape[2])

mean_MCTSnum_steps_run_list = np.mean(mean_MCTSnum_steps_run_list, axis=1)
std_MCTSnum_steps_run_list = np.std(std_MCTSnum_steps_run_list, axis=1)

fig2, axs_MCTS = plt.subplots(1, 1, constrained_layout=False)
best_par = np.argmin(mean_MCTSnum_steps_run_list)
for stepsize_index in range(mean_MCTSnum_steps_run_list.shape[0]):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    c = (r, g, b)
    hex_c = '#%02x%02x%02x' % c
    axs_MCTS.axhline(mean_MCTSnum_steps_run_list[2], color = hex_c, label=str(stepsize_index))

axs_MCTS.legend()
fig2.show()


with open('Results/DQNMCTSAgent_UseTreeSelectionnum_steps_run_list.npy', 'rb') as f:
    DQNMCTSnum_steps_run_list = np.load(f)

num_step = DQNMCTSnum_steps_run_list.shape[2] // 2
# num_step = 4
odd_ind = list(range(1, num_step * 2, 2))
even_ind = list(range(0, num_step * 2, 2))
dqn_step_list = np.delete(DQNMCTSnum_steps_run_list, odd_ind, 2)
mcts_step_list = np.delete(DQNMCTSnum_steps_run_list, even_ind, 2)

mean_dqnnum_steps_run_list = np.mean(dqn_step_list, axis=1)
std_dqnnum_steps_run_list = np.std(dqn_step_list, axis=1)
x_dqnnum_steps_run_list = np.arange(dqn_step_list.shape[2])

mean_mctsnum_steps_run_list = np.mean(mcts_step_list, axis=1)
std_mctsnum_steps_run_list = np.std(mcts_step_list, axis=1)
x_mctsnum_steps_run_list = np.arange(mcts_step_list.shape[2])
fig, axs_DQNMCTS = plt.subplots(1, 1, constrained_layout=False)


drawPlotUncertainty(x_dqnnum_steps_run_list,
                    mean_dqnnum_steps_run_list[0],
                    std_dqnnum_steps_run_list[0],
                    label="dqn",
                    color="blue",
                    axis=axs_DQNMCTS)

drawPlotUncertainty(x_mctsnum_steps_run_list,
                    mean_mctsnum_steps_run_list[0],
                    std_mctsnum_steps_run_list[0],
                    label="mcts",
                    color="red",
                    axis=axs_DQNMCTS)
# totalmean_mcts = np.mean(mean_mctsnum_steps_run_list, axis=1)
# print(totalmean_mcts, mean_MCTSnum_steps_run_list[best_par])
axs_DQNMCTS.axhline(mean_MCTSnum_steps_run_list[best_par], color = "green", label="normal mcts")
# axs_DQNMCTS.axhline(totalmean_mcts, color = "black", label="mean mcts")
axs_DQNMCTS.legend()
axs_DQNMCTS.title.set_text("DQNMCTSAgent_UseTreeSelectionnum")
fig.show()


with open('Results/MCTSAgent_different_iterations.npy', 'rb') as f:
    MCTSnum_steps_run_list = np.load(f)

print(MCTSnum_steps_run_list.shape)
mean_MCTSnum_steps_run_list = np.mean(MCTSnum_steps_run_list, axis=1)
std_MCTSnum_steps_run_list = np.std(MCTSnum_steps_run_list, axis=1)
x_MCTSnum_steps_run_list = np.arange(mean_MCTSnum_steps_run_list.shape[0])
fig, axs_MCTS = plt.subplots(1, 1, constrained_layout=False)
drawPlotUncertainty(x_MCTSnum_steps_run_list,
                    mean_MCTSnum_steps_run_list[:,0],
                    std_MCTSnum_steps_run_list[:,0],
                    label="MCTS",
                    color="blue",
                    axis=axs_MCTS)

axs_MCTS.title.set_text("MCTSAgent")
fig.show()