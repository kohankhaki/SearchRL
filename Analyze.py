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

