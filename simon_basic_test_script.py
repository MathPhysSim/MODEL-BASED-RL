import os
import pickle

import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pernaf.pernaf.naf import NAF
from pernaf.pernaf.utils.statistic import Statistic

# from simple_environment import simpleEnv
# from pendulum import PendulumEnv as simpleEnv
# set random seed
random_seed = 123
# set random seed
tf.set_random_seed(random_seed)
np.random.seed(random_seed)


import numpy as np
import tensorflow as tf
import gym
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
rms_threshold = 0.75

element_actor_list = ['rmi://virtual_awake/logical.RCIBH.430029/K',
                      'rmi://virtual_awake/logical.RCIBH.430040/K',
                      'rmi://virtual_awake/logical.RCIBH.430104/K',
                      'rmi://virtual_awake/logical.RCIBH.430130/K',
                      'rmi://virtual_awake/logical.RCIBH.430204/K',
                      'rmi://virtual_awake/logical.RCIBH.430309/K',
                      'rmi://virtual_awake/logical.RCIBH.412344/K',
                      'rmi://virtual_awake/logical.RCIBH.412345/K',
                      'rmi://virtual_awake/logical.RCIBH.412347/K',
                      'rmi://virtual_awake/logical.RCIBH.412349/K',
                      'rmi://virtual_awake/logical.RCIBH.412353/K',
                      'rmi://virtual_awake/logical.RCIBV.430029/K',
                      'rmi://virtual_awake/logical.RCIBV.430040/K',
                      'rmi://virtual_awake/logical.RCIBV.430104/K',
                      'rmi://virtual_awake/logical.RCIBV.430130/K',
                      'rmi://virtual_awake/logical.RCIBV.430204/K',
                      'rmi://virtual_awake/logical.RCIBV.430309/K',
                      'rmi://virtual_awake/logical.RCIBV.412344/K',
                      'rmi://virtual_awake/logical.RCIBV.412345/K',
                      'rmi://virtual_awake/logical.RCIBV.412347/K',
                      'rmi://virtual_awake/logical.RCIBV.412349/K',
                      'rmi://virtual_awake/logical.RCIBV.412353/K']

element_state_list = ['BPM.430028_horizontal',
                      'BPM.430039_horizontal',
                      'BPM.430103_horizontal',
                      'BPM.430129_horizontal',
                      'BPM.430203_horizontal',
                      'BPM.430308_horizontal',
                      'BPM.412343_horizontal',
                      'BPM.412345_horizontal',
                      'BPM.412347_horizontal',
                      'BPM.412349_horizontal',
                      'BPM.412351_horizontal',
                      'BPM.430028_vertical',
                      'BPM.430039_vertical',
                      'BPM.430103_vertical',
                      'BPM.430129_vertical',
                      'BPM.430203_vertical',
                      'BPM.430308_vertical',
                      'BPM.412343_vertical',
                      'BPM.412345_vertical',
                      'BPM.412347_vertical',
                      'BPM.412349_vertical',
                      'BPM.412351_vertical']

simulation = True
element_actor_list_selected = pd.Series(element_actor_list[:11])

element_state_list_selected = pd.Series(element_state_list[:11])
number_bpm_measurements = 30
simulation = True
if simulation:
    from new_simulated_environment import e_trajectory_simENV as awakeEnv
else:
    from awake_environment_machine import awakeEnv

reference_position = np.zeros(len(element_state_list_selected))

env = awakeEnv(action_space=element_actor_list_selected, state_space=element_state_list_selected,
               number_bpm_measurements=number_bpm_measurements, noSet=False, debug=True, scale=1e-4)

env.__name__ = 'AWAKE'

env.seed(random_seed)

for _ in range(10):
    env.reset()

label = 'NAF on: AWAKE'

directory = "checkpoints/test_implementation/"

#TODO: Test the loading

def plot_results(env, label):
    # plotting
    print('now plotting')
    rewards = env.rewards
    initial_states = env.initial_conditions

    iterations = []
    finals = []
    starts = []

    # init_states = pd.read_pickle('/Users/shirlaen/PycharmProjects/DeepLearning/spinningup/Environments/initData')

    for i in range(len(rewards)):
        if (len(rewards[i]) > 0):
            finals.append(rewards[i][len(rewards[i]) - 1])
            starts.append(-np.sqrt(np.mean(np.square(initial_states[i]))))
            iterations.append(len(rewards[i]))

    plot_suffix = f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    ax=axs[0]
    ax.plot(iterations)
    ax.set_title('Iterations' + plot_suffix)
    fig.suptitle(label, fontsize=12)

    ax = axs[1]
    color = 'blue'
    ax.set_ylabel('Final RMS', color=color)  # we already handled the x-label with ax1
    ax.tick_params(axis='y', labelcolor=color)
    ax.plot(finals, color=color)
    ax.axhline(env.threshold, ls=':',c='r')
    ax.set_title('Final reward per episode')  # + plot_suffix)
    ax.set_xlabel('Episodes (1)')

    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.set_ylabel('Initial RMS', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(starts, color=color)

    ax.set_ylim(ax1.get_ylim())
    plt.savefig(label+'.pdf')
    plt.savefig(label + '.png')
    # fig.tight_layout()
    plt.show()


def plot_convergence(agent, label):
    losses, vs = agent.losses, agent.vs
    fig, ax = plt.subplots()
    ax.set_title(label)
    ax.set_xlabel('episodes')

    color = 'tab:blue'
    ax.plot(losses, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel('td_loss', color=color)
    # ax.set_ylim(0, 1)

    ax1 = plt.twinx(ax)
    # ax1.set_ylim(-2, 1)
    color = 'lime'

    ax1.set_ylabel('V', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(vs, color=color)
    plt.savefig(label + 'convergence' + '.pdf')
    plt.show()


if __name__ == '__main__':

    discount = 0.9999
    batch_size = 100
    learning_rate = 1e-3
    max_steps = 200
    update_repeat = 10
    max_episodes = 100
    tau = 1 - 0.999
    is_train = True
    is_continued = False

    nafnet_kwargs = dict(hidden_sizes=[32, 32], activation=tf.nn.tanh
                         , weight_init=tf.random_uniform_initializer(-0.05, 0.05))

    prio_info = dict(alpha=.0, beta=.5)

    # filename = 'Scan_data.obj'
    # filehandler = open(filename, 'rb')
    # scan_data = pickle.load(filehandler)


    with tf.Session() as sess:
        # statistics and running the agent
        stat = Statistic(sess=sess, env_name=env.__name__, model_dir=directory,
                         max_update_per_step=update_repeat, is_continued=is_continued, save_frequency=500)
        # init the agent
        agent = NAF(sess=sess, env=env, stat=stat, discount= discount, batch_size=batch_size,
                    learning_rate=learning_rate, max_steps=max_steps, update_repeat=update_repeat,
                    max_episodes=max_episodes, tau=tau, pretune = None, prio_info=prio_info, **nafnet_kwargs)
        # run the agent
        agent.run(is_train)

    # plot the results
    plot_convergence(agent=agent, label=label)
    plot_results(env, label)

