# For the moment only basic functionality is provided (select a full plane)


# matplotlib.use("Qt5Agg")
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# import PyQt5
from spinup import td3, ddpg, sac, ppo, trpo

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

element_actor_list_selected = pd.Series(element_actor_list[:11])
element_state_list_selected = pd.Series(element_state_list[:11])
number_bpm_measurements = 10
# matplotlib.use("Qt5Agg")
simulation = True

if simulation:
    from new_simulated_environment import e_trajectory_simENV as awakeEnv
else:
    from awake_environment_machine import awakeEnv

reference_position = np.zeros(len(element_state_list_selected))

env = awakeEnv(action_space=element_actor_list_selected, state_space=element_state_list_selected,
               number_bpm_measurements=number_bpm_measurements, noSet=True, debug=True, scale=5e-4)

env_fn = lambda: env
#
# nafnet_kwargs = dict(hidden_sizes=[100, 100], activation=tf.nn.tanh
#                      , weight_init=tf.random_uniform_initializer(-0.05, 0.05))
act_noise = .1
# output_dir = 'machine_logging/awake/NAF/'
# logger_kwargs = dict(output_dir=output_dir, exp_name='transport_awake')
#
# agent = naf(env_fn=env_fn, epochs=20, steps_per_epoch=250, logger_kwargs=logger_kwargs,
#             nafnet_kwargs=nafnet_kwargs, act_noise=act_noise, gamma=0.999, start_steps=2500,
#             batch_size=10, q_lr=1e-3, update_repeat=7, polyak=0.995, seed=123)

#
# output_dir = 'logging/awake/PPO/'
# logger_kwargs = dict(output_dir=output_dir, exp_name='transport_awake')
# agent = ppo(env_fn=env_fn, epochs=100, steps_per_epoch=5000, logger_kwargs=logger_kwargs, seed=123, save_freq=100)

# output_dir = 'logging/awake/SAC/'
# logger_kwargs = dict(output_dir=output_dir, exp_name='transport_awake')
# agent = sac(env_fn=env_fn, epochs=25, steps_per_epoch=1000, logger_kwargs=logger_kwargs, start_steps=1000)


output_dir = 'logging/awake/TRPO/'
logger_kwargs = dict(output_dir=output_dir, exp_name='transport_awake')
# agent = td3(env_fn=env_fn, epochs=25, steps_per_epoch=1000,
#             logger_kwargs=logger_kwargs, start_steps=500)

agent = trpo(env_fn=env_fn, epochs=50, steps_per_epoch=1000,
            logger_kwargs=logger_kwargs, delta=0.5)

plot_name = 'Stats'
name = plot_name
data = pd.read_csv(output_dir + '/progress.txt', sep="\t")

data.index = data['TotalEnvInteracts']
data_plot = data[['EpLen', 'MinEpRet', 'AverageEpRet']]
data_plot.plot(secondary_y=['MinEpRet', 'AverageEpRet'])

plt.title(name)
# plt.savefig(name + '.pdf')
plt.show()

# # plotting
# print('now plotting')
# rewards = env.rewards
#
# iterations = []
# finals = []
#
# # init_states = pd.read_pickle('/Users/shirlaen/PycharmProjects/DeepLearning/spinningup/Environments/initData')
#
# for i in range(len(rewards)):
#     if (len(rewards[i]) > 0):
#         iterations.append(len(rewards[i]))
#         finals.append(rewards[i][len(rewards[i]) - 1])


# plot_suffix = ', number of iterations: ' + str(env.TOTAL_COUNTER)
#
# plt.figure(1)
# plt.subplot(211)
# plt.ylim()
# plt.plot(iterations)
# plt.title('Iterations' + plot_suffix)
#
# plt.subplot(212)
# plt.plot(finals, 'r--')
# plt.title('Reward' + plot_suffix)
#
# plt.savefig('progress1')
#
# plt.show()

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
        starts.append(-np.sqrt(np.mean(np.power(initial_states[i], 2))))
        iterations.append(len(rewards[i]))

plot_suffix = f', number of iterations: {env.TOTAL_COUNTER}, awake time: {env.TOTAL_COUNTER / 600:.1f} h'

plt.figure(1)
plt.subplot(211)
plt.ylim()
plt.plot(iterations)
plt.title('Iterations' + plot_suffix)

plt.subplot(212)
plt.plot(finals, 'r--')
plt.plot(starts, c='lime')
plt.title('Final reward per episode')  # + plot_suffix)

plt.savefig('progress1')

plt.show()

plt.figure()
plt.scatter(starts, finals, c="g", alpha=0.5, marker=r'$\clubsuit$',
            label="Luck")

plt.show()