from matplotlib import pyplot as plt
import numpy as np

def drawPlotUncertainty(x, y, y_err, label, color, axis):
    axis.plot(x, y, color, label=label)
    axis.fill_between(x,
                      y - y_err,
                      y + y_err,
                      facecolor=color, alpha=.2, edgecolor='none')


with open('Results/DQNnum_steps_run_list.npy', 'rb') as f:
    DQNnum_steps_run_list = np.load(f)

print(DQNnum_steps_run_list.shape)
mean_DQNnum_steps_run_list = np.mean(DQNnum_steps_run_list, axis=1)
std_DQNnum_steps_run_list = np.std(DQNnum_steps_run_list, axis=1)
x_DQNnum_steps_run_list = np.arange(DQNnum_steps_run_list.shape[2])

fig, axs_DQN = plt.subplots(1, 1, constrained_layout=False)
for stepsize_index in range(mean_DQNnum_steps_run_list.shape[0]):
    drawPlotUncertainty(x_DQNnum_steps_run_list,
                        mean_DQNnum_steps_run_list[stepsize_index],
                        std_DQNnum_steps_run_list[stepsize_index],
                        label="dqn"+str(stepsize_index),
                        color="",
                        axis=axs_DQN)
axs_DQN.legend()
fig.show()
