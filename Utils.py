
import matplotlib.pyplot as plt
import torch
import numpy as np
import pygame
import random
from collections import namedtuple

transition = namedtuple('transition', ['prev_state', 'prev_action', 'reward', 'state', 'action', 'is_terminal', 'time_step', 'error'])
# transition = namedtuple('transition', ['prev_state', 'prev_action', 'reward', 'state', 'action'])


def draw_plot(x, y, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None, show=False, label='', std_error=None, sub_plot_num=None, color=None):
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    if xlabel is not None:
        # naming the x axis
        plt.xlabel(xlabel)
    if ylabel is not None:
        # naming the y axis
        plt.ylabel(ylabel)
    if title is not None:
        # giving a title to my graph
        plt.title(title)

    # plotting the points
    if std_error is None:
        plt.plot(x, y, label=label)
    else:
        if color is not None:
            plt.errorbar(x, y, yerr=std_error, label=label, color=color)
        else:
            plt.errorbar(x, y, yerr=std_error, label=label)

    plt.legend()
    if show:
        # # function to show the plot
        plt.show()

def draw_grid(grid_size, window_size, state_action_values=None, all_actions=None, obstacles_pos=[]):
    ground_color = [255, 255, 255]
    # agent_color = [i * 255 for i in self._agent_color]
    # ground_color = [i * 255 for i in self._ground_color]
    # obstacle_color = [i * 255 for i in self._obstacle_color]
    text_color = (240,240,10)
    info_color = (200, 50, 50)
    # This sets the WIDTH and HEIGHT of each grid location
    WIDTH = int(window_size[0] / grid_size[1])
    HEIGHT = int(window_size[1] / grid_size[0])

    # This sets the margin between each cell
    MARGIN = 1


    # Initialize pygame
    pygame.init()

    # Set the HEIGHT and WIDTH of the screen
    WINDOW_SIZE = [window_size[0], window_size[1]]
    screen = pygame.display.set_mode(WINDOW_SIZE)

    # Set title of screen
    pygame.display.set_caption("Grid")

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    font = pygame.font.Font('freesansbold.ttf', 20)
    info_font = pygame.font.Font('freesansbold.ttf', int(60 / ((grid_size[0]+grid_size[1])/2)) )


    done = False
    # -------- Main Program Loop -----------
    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True

        # Set the screen background
        screen.fill((100,100,100))


        # Draw the grid
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                if (x,y) in obstacles_pos:
                    continue
                color = ground_color
                # if list(grid[x][y]) == self._agent_color:
                #     color = agent_color
                # elif list(grid[x][y]) == self._obstacle_color:
                #     color = obstacle_color
                pygame.draw.rect(screen,
                                 color,
                                 [(MARGIN + WIDTH) * y + MARGIN,
                                  (MARGIN + HEIGHT) * x + MARGIN,
                                  WIDTH,
                                  HEIGHT])
                if state_action_values is not None:
                    # showing values only for 4 basic actions
                    up_left_corner = [(MARGIN + WIDTH) * y + MARGIN,
                                      (MARGIN + HEIGHT) * x + MARGIN]
                    up_right_corner = [(MARGIN + WIDTH) * y + MARGIN + WIDTH,
                                      (MARGIN + HEIGHT) * x + MARGIN]
                    down_left_corner = [(MARGIN + WIDTH) * y + MARGIN,
                                       (MARGIN + HEIGHT) * x + MARGIN + HEIGHT]
                    down_right_corner = [(MARGIN + WIDTH) * y + MARGIN + WIDTH,
                                        (MARGIN + HEIGHT) * x + MARGIN + HEIGHT]
                    center = [(up_right_corner[0] + up_left_corner[0]) // 2,
                              (up_right_corner[1] + down_right_corner[1]) // 2]

                    pygame.draw.polygon(screen, info_color,
                                        [up_left_corner, up_right_corner, center],
                                        1)
                    pygame.draw.polygon(screen, info_color,
                                        [up_right_corner, down_right_corner, center],
                                        1)
                    pygame.draw.polygon(screen, info_color,
                                        [down_right_corner, down_left_corner, center],
                                        1)
                    pygame.draw.polygon(screen, info_color,
                                        [down_left_corner, up_left_corner, center],
                                        1)
                    for a in all_actions:
                        if tuple(a) == (0,1):
                            right = info_font.render(str(state_action_values[(x,y), tuple(a)]), True, info_color)
                        elif tuple(a) == (1,0):
                            down = info_font.render(str(state_action_values[(x,y), tuple(a)]), True, info_color)
                        elif tuple(a) == (0,-1):
                            left = info_font.render(str(state_action_values[(x,y), tuple(a)]), True, info_color)
                        elif tuple(a) == (-1,0):
                            up = info_font.render(str(state_action_values[(x,y), tuple(a)]), True, info_color)
                        else:
                            raise ValueError("action cannot be rendered")

                    margin = 1
                    screen.blit(left,
                               (up_left_corner[0] + margin,
                                center[1])) #left
                    screen.blit(right,
                               (up_right_corner[0] - right.get_rect().width,
                                center[1]))  # right
                    screen.blit(up,
                               (center[0] - up.get_rect().width // 2,
                                up_right_corner[1] + margin)) # up
                    screen.blit(down,
                               (center[0] - down.get_rect().width // 2,
                                down_left_corner[1] - down.get_rect().height - margin)) # down

        # Limit to 60 frames per second
        clock.tick(60)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

def reshape_for_grid(img):
    # get a tenser as input with shape B, W, H, C and return a tensor with shape B, C, W, H
    grid_img = torch.cat([img[:, :, :, 0], img[:, :, :, 1], img[:, :, :, 2]]).unsqueeze(0)
    return grid_img

def calculate_true_values(env, gamma):
    states = env.getAllStates()
    actions = env.getAllActions()
    values = {}
    alpha = 0.1
    max_iter = 10000
    td_differ = 0.1
    for s in states:
        for a in actions:
            pos = env.stateToPos(s)
            values[pos, tuple(a)] = 0
    tderror_sum = 1000
    i = 0
    while tderror_sum > td_differ and i < max_iter:
        tderror_sum = 0
        i += 1
        random.shuffle(states)
        for s in states:
            s = env.stateToPos(s)
            for a in actions:
                next_state, is_terminal, reward = env.fullTransitionFunction(s, a, state_type='coord')
                if not is_terminal:
                    next_state_value = 0
                    for aa in actions:
                        next_state_value += values[next_state, tuple(aa)]
                    next_state_value /= len(actions)
                    tderror = reward + gamma * next_state_value - values[s, tuple(a)]
                else:
                    tderror = reward - values[s, tuple(a)]
                tderror_sum += abs(tderror)
                values[s, tuple(a)] += alpha * tderror

    return values
