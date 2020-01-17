import os

import numpy as np
import tensorflow as tf
import gym
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os

sns.set(style="white")
# Number of networks, number of starting points pure policy if no randomness after init
data_folder = 'data_25_25_pure_policy_50_acquisition_long/'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# TODO: Exploration noise decay
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
    from simulated_environment_final import e_trajectory_simENV as awakeEnv
# else:
#     from awake_environment_machine import awakeEnv

reference_position = np.zeros(len(element_state_list_selected))
scaling = 1e-4
env = awakeEnv(action_space=element_actor_list_selected, state_space=element_state_list_selected,
               number_bpm_measurements=number_bpm_measurements, noSet=False, debug=True, scale=scaling)

rms_threshold = env.threshold
print('rms threshold:', rms_threshold)


def plot_results(env, label, **kwargs):
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

    plot_suffix = f', number of iterations: {env.TOTAL_COUNTER}, AWAKE time: {env.TOTAL_COUNTER / 600:.1f} h'

    fig, axs = plt.subplots(2, 1)  # , constrained_layout=True)

    ax = axs[0]
    ax.plot(iterations)
    ax.set_title('Iterations' + plot_suffix)

    fig.suptitle(label, fontsize=12)

    ax = axs[1]
    ax.set_title('Final reward per episode')  # + plot_suffix)
    ax.set_xlabel('Episodes (1)')

    color = 'lime'
    ax.axhline(env.threshold, ls=':', c='r')
    ax.set_ylabel('Initial RMS', color=color)  # we already handled the x-label with ax1
    ax.tick_params(axis='y', labelcolor=color)
    ax.plot(starts, color=color)

    ax1 = plt.twinx(ax)

    color = 'blue'
    ax1.set_ylabel('Final RMS', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(finals, color=color)

    # ax1.set_ylim(-1,0)
    # ax.set_ylim(-1, 0)
    if 'save_name' in kwargs:
        plt.savefig(data_folder + kwargs.get('save_name')+'.pdf')
    # plt.savefig(label + '.pdf')
    # plt.savefig(label + '.png')
    # fig.tight_layout()
    plt.show()


def make_env():
    # env.seed(123)
    # env.reset()
    return env


def mlp(x, hidden_layers, output_layer, activation=tf.tanh, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
    return tf.layers.dense(x, units=output_layer, activation=last_activation)


def softmax_entropy(logits):
    '''
    Softmax Entropy
    '''
    return -tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.nn.log_softmax(logits, axis=-1), axis=-1)


def gaussian_log_likelihood(ac, mean, log_std):
    '''
    Gaussian Log Likelihood
    '''
    log_p = ((ac - mean) ** 2 / (tf.exp(log_std) ** 2 + 1e-9) + 2 * log_std) + np.log(2 * np.pi)
    return -0.5 * tf.reduce_sum(log_p, axis=-1)


def conjugate_gradient(A, b, x=None, iters=10):
    '''
    Conjugate gradient method: approximate the solution of Ax=b
    It solve Ax=b without forming the full matrix, just compute the matrix-vector product (The Fisher-vector product)

    NB: A is not the full matrix but is a useful matrix-vector product between the averaged Fisher information matrix and arbitrary vectors
    Descibed in Appendix C.1 of the TRPO paper
    '''
    if x is None:
        x = np.zeros_like(b)

    r = A(x) - b
    p = -r
    for _ in range(iters):
        a = np.dot(r, r) / (np.dot(p, A(p)) + 1e-8)
        x += a * p
        r_n = r + a * A(p)
        b = np.dot(r_n, r_n) / (np.dot(r, r) + 1e-8)
        p = -r_n + b * p
        r = r_n
    return x


def gaussian_DKL(mu_q, log_std_q, mu_p, log_std_p):
    '''
    Gaussian KL divergence in case of a diagonal covariance matrix
    '''
    return tf.reduce_mean(tf.reduce_sum(
        0.5 * (log_std_p - log_std_q + tf.exp(log_std_q - log_std_p) + (mu_q - mu_p) ** 2 / tf.exp(log_std_p) - 1),
        axis=1))


def backtracking_line_search(Dkl, delta, old_loss, p=0.8):
    '''
    Backtracking line searc. It look for a coefficient s.t. the constraint on the DKL is satisfied
    It has both to
     - improve the non-linear objective
     - satisfy the constraint

    '''
    ## Explained in Appendix C of the TRPO paper
    a = 1
    it = 0

    new_dkl, new_loss = Dkl(a)
    while (new_dkl > delta) or (new_loss > old_loss):
        a *= p
        it += 1
        new_dkl, new_loss = Dkl(a)

    return a


def GAE(rews, v, v_last, gamma=0.99, lam=0.95):
    '''
    Generalized Advantage Estimation
    '''
    assert len(rews) == len(v)
    vs = np.append(v, v_last)
    d = np.array(rews) + gamma * vs[1:] - vs[:-1]
    gae_advantage = discounted_rewards(d, 0, gamma * lam)
    return gae_advantage


def discounted_rewards(rews, last_sv, gamma):
    '''
    Discounted reward to go

    Parameters:
    ----------
    rews: list of rewards
    last_sv: value of the last state
    gamma: discount value
    '''
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1] + gamma * last_sv
    for i in reversed(range(len(rews) - 1)):
        rtg[i] = rews[i] + gamma * rtg[i + 1]
    return rtg


def flatten_list(tensor_list):
    '''
    Flatten a list of tensors
    '''
    return tf.concat([flatten(t) for t in tensor_list], axis=0)


def flatten(tensor):
    '''
    Flatten a tensor
    '''
    return tf.reshape(tensor, shape=(-1,))


def test_agent(env_test, agent_op, num_games=10, model_buffer=False, **kwargs):
    '''
    Test an agent 'agent_op', 'num_games' times
    Return mean and std
    '''
    games_r = []
    lengths = []
    successes = []
    for _ in range(num_games):
        d = False
        game_r = 0
        length = 0
        o = env_test.reset()

        while not d:
            o0 = o.copy()
            a_s, _ = agent_op([o])
            o, r, d, _ = env_test.step(a_s)
            length += 1
            game_r += r  # TODO: modify (game_r = r)

            if length > 100 and model_buffer:
                break
            elif length > 100:
                break
            if model_buffer:
                # add the new transition to the temporary buffer
                # print(o0, a_s, r, o, d)
                model_buffer.store(o0, a_s[0], r, o.copy(), d)
        # game_r = r # final reward only
        successes.append(env_test.success)
        games_r.append(game_r)
        lengths.append(length)
    return games_r, lengths, successes


def to_pickle(data, data_name):
    pickling_on = open(data_folder + data_name, "wb")
    pickle.dump(data, pickling_on)
    pickling_on.close()


class Buffer():
    '''
    Class to store the experience from a unique policy
    '''

    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.adv = []
        self.ob = []
        self.ac = []
        self.rtg = []

    def store(self, temp_traj, last_sv):
        '''
        Add temp_traj values to the buffers and compute the advantage and reward to go

        Parameters:
        -----------
        temp_traj: list where each element is a list that contains: observation, reward, action, state-value
        last_sv: value of the last state (Used to Bootstrap)
        '''
        # store only if there are temporary trajectories
        if len(temp_traj) > 0:
            self.ob.extend(temp_traj[:, 0])
            rtg = discounted_rewards(temp_traj[:, 1], last_sv, self.gamma)
            self.adv.extend(GAE(temp_traj[:, 1], temp_traj[:, 3], last_sv, self.gamma, self.lam))
            self.rtg.extend(rtg)
            self.ac.extend(temp_traj[:, 2])

    def get_batch(self):
        # standardize the advantage values
        norm_adv = (self.adv - np.mean(self.adv)) / (np.std(self.adv) + 1e-10)
        return np.array(self.ob), np.array(np.expand_dims(self.ac, -1)), np.array(norm_adv), np.array(self.rtg)

    def __len__(self):
        assert (len(self.adv) == len(self.ob) == len(self.ac) == len(self.rtg))
        return len(self.ob)


class FullBuffer():
    def __init__(self):
        self.rew = []
        self.obs = []
        self.act = []
        self.nxt_obs = []
        self.done = []

        self.train_idx = []
        self.valid_idx = []
        self.idx = 0

    def store(self, obs, act, rew, nxt_obs, done):
        self.rew.append(rew)
        self.obs.append(obs)
        self.act.append(act)
        self.nxt_obs.append(nxt_obs)
        self.done.append(done)

        self.idx += 1

    def generate_random_dataset(self):
        rnd = np.arange(len(self.obs))
        np.random.shuffle(rnd)
        self.valid_idx = rnd[: int(len(self.obs) / 3)]
        self.train_idx = rnd[int(len(self.obs) / 3):]
        print('Train set:', len(self.train_idx), 'Valid set:', len(self.valid_idx))

    def get_training_batch(self):
        return np.array(self.obs)[self.train_idx], np.array(np.expand_dims(self.act, -1))[self.train_idx], \
               np.array(self.rew)[self.train_idx], np.array(self.nxt_obs)[self.train_idx], np.array(self.done)[
                   self.train_idx]

    def get_valid_batch(self):
        return np.array(self.obs)[self.valid_idx], np.array(np.expand_dims(self.act, -1))[self.valid_idx], \
               np.array(self.rew)[self.valid_idx], np.array(self.nxt_obs)[self.valid_idx], np.array(self.done)[
                   self.valid_idx]

    def __len__(self):
        assert (len(self.rew) == len(self.obs) == len(self.act) == len(self.nxt_obs) == len(self.done))
        return len(self.obs)


def simulate_environment(env, policy, simulated_steps):
    buffer = Buffer(0.999, 0.95)
    # lists to store rewards and length of the trajectories completed
    steps = 0
    number_episodes = 0

    while steps < simulated_steps:
        temp_buf = []
        obs = env.reset()
        number_episodes += 1
        done = False

        while not done:
            act, val = policy([obs])

            obs2, rew, done, _ = env.step([act])

            temp_buf.append([obs.copy(), rew, np.squeeze(act), np.squeeze(val)])

            obs = obs2.copy()
            steps += 1

            if done:
                buffer.store(np.array(temp_buf), 0)
                temp_buf = []

            if steps == simulated_steps:
                break

        buffer.store(np.array(temp_buf), np.squeeze(policy([obs])[1]))

    print('Sim ep:', number_episodes, end=' \n')

    return buffer.get_batch()


class NetworkEnv(gym.Wrapper):
    def __init__(self, env, model_func, reward_func, done_func, number_models):
        gym.Wrapper.__init__(self, env)

        self.model_func = model_func
        self.reward_func = reward_func
        self.done_func = done_func
        self.number_models = number_models
        self.len_episode = 0
        self.action_0 = None

        self.success = None

    def cut_trajectory(self, ob):
        rms = (np.sqrt(np.mean(np.square(ob))))
        done = -1 * rms > rms_threshold
        ob_ret = ob.copy()
        # if any(abs(ob) > abs(10 * rms_threshold)):
        #     ob_ret[np.argmax(abs(ob) > abs(10 * rms_threshold)):] = 10 * rms_threshold
        #     # done = True
        # rms = (np.sqrt(np.mean(np.square(ob_ret))))
        return ob_ret, -1 * rms, done

    def reset(self, **kwargs):
        self.len_episode = 0
        self.success = 0
        # kwargs['simulation'] = True
        # self.action_0 = np.random.uniform(low=-.5, high=.5, size=env.action_space.shape[0])  # self.env.reset(**kwargs)
        # self.current_action = self.action_0.copy()
        # self.obs = self.model_func(np.zeros(env.observation_space.shape[0]), [np.squeeze(self.current_action)],
        #                            np.random.randint(0, self.number_models))
        self.obs = self.env.reset()
        return self.obs

    def step(self, action):
        self.current_action = np.squeeze(action)
        # predict the next state on a random model np.mean(np.squeeze([self.model_func(self.obs, [np.squeeze(action)], i) for i in range(self.number_models)]))
        obs = self.model_func(self.obs, [np.squeeze(action)], np.random.randint(0, self.number_models))
        # Take the mean of all models
        # obs = np.mean(np.array([self.model_func(self.obs, [np.squeeze(action)], i) for i in range(self.number_models)]),axis=0)
        # print('obs: ', obs)
        # print('obs1: ', np.mean(np.array([self.model_func(self.obs, [np.squeeze(action)], i) for i in range(self.number_models)]),axis=0))
        # rew = self.reward_func(self.obs, self.len_episode+1)
        # done = self.done_func(obs)
        obs, rew, done = self.cut_trajectory(obs)
        self.len_episode += 1

        # if self.len_episode >= 100:
        #     done = True

        self.obs = obs
        # print('rew:', rew, done)
        rms = (np.sqrt(np.mean(np.square(obs))))
        # done = -1 * rms > rms_threshold
        if -1 * rms > rms_threshold:
            done = True
            self.success = 1
            # rew += .1
        # rew *= self.len_episode
        return self.obs, rew, done, ""


class StructEnv(gym.Wrapper):
    '''
    Gym Wrapper to store information like number of steps and total reward of the last espisode.
    '''

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.n_obs = self.env.reset()
        self.total_rew = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        self.n_obs = self.env.reset(**kwargs)
        self.total_rew = 0
        self.len_episode = 0
        return self.n_obs.copy()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.total_rew += reward
        self.len_episode += 1
        return ob, reward, done, info

    def get_episode_reward(self):
        return self.total_rew / self.len_episode

    def get_episode_length(self):
        return np.mean(self.len_episode)


def episode_done(ob):
    # return np.abs(np.arcsin(np.squeeze(ob[3]))) > .2
    return -1 * (np.sqrt(np.mean(np.square(ob)))) > rms_threshold


def reward_function(ob, len):
    return -1 * np.sqrt(np.mean(np.square(ob)))


def restore_model(old_model_variables, m_variables):
    # variable used as index for restoring the actor's parameters
    it_v2 = tf.Variable(0, trainable=False)
    restore_m_params = []

    for m_v in m_variables:
        upd_m_rsh = tf.reshape(old_model_variables[it_v2: it_v2 + tf.reduce_prod(m_v.shape)], shape=m_v.shape)
        restore_m_params.append(m_v.assign(upd_m_rsh))
        it_v2 += tf.reduce_prod(m_v.shape)

    return tf.group(*restore_m_params)

def number_to_text(it):
    if it < 10:
            text = '00' + str(it)
    elif it < 100:
            text = '0' + str(it)
    return text

def METRPO(env_name, hidden_sizes=[32], cr_lr=5e-3, num_epochs=50, gamma=0.99, lam=0.95, number_envs=1,
           critic_iter=10, steps_per_env=100, delta=0.05, algorithm='TRPO', conj_iters=10, minibatch_size=1000,
           mb_lr=0.0001, model_batch_size=512, simulated_steps=300, num_ensemble_models=2, model_iter=15):
    '''
    Model Ensemble Trust Region Policy Optimization

    Parameters:
    -----------
    env_name: Name of the environment
    hidden_sizes: list of the number of hidden units for each layer
    cr_lr: critic learning rate
    num_epochs: number of training epochs
    gamma: discount factor
    lam: lambda parameter for computing the GAE
    number_envs: number of "parallel" synchronous environments
        # NB: it isn't distributed across multiple CPUs
    critic_iter: Number of SGD iterations on the critic per epoch
    steps_per_env: number of steps per environment
            # NB: the total number of steps per epoch will be: steps_per_env*number_envs
    delta: Maximum KL divergence between two policies. Scalar value
    algorithm: type of algorithm. Either 'TRPO' or 'NPO'
    conj_iters: number of conjugate gradient iterations
    minibatch_size: Batch size used to train the critic
    mb_lr: learning rate of the environment model
    model_batch_size: batch size of the environment model
    simulated_steps: number of simulated steps for each policy update
    num_ensemble_models: number of models
    model_iter: number of iterations without improvement before stopping training the model
    '''
    # TODO: add ME-TRPO hyperparameters

    tf.reset_default_graph()

    # Create a few environments to collect the trajectories

    # envs = [StructEnv(gym.make(env_name)) for _ in range(number_envs)]
    envs = [StructEnv(make_env()) for _ in range(number_envs)]
    # env_test = gym.make(env_name)
    env_test = make_env()
    # env_test = gym.wrappers.Monitor(env_test, "VIDEOS/", force=True, video_callable=lambda x: x%10 == 0)

    low_action_space = envs[0].action_space.low
    high_action_space = envs[0].action_space.high

    obs_dim = envs[0].observation_space.shape
    act_dim = envs[0].action_space.shape[0]

    print(envs[0].action_space, envs[0].observation_space)

    # Placeholders
    act_ph = tf.placeholder(shape=(None, act_dim), dtype=tf.float32, name='act')
    obs_ph = tf.placeholder(shape=(None, obs_dim[0]), dtype=tf.float32, name='obs')
    # NEW
    nobs_ph = tf.placeholder(shape=(None, obs_dim[0]), dtype=tf.float32, name='nobs')
    ret_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='ret')
    adv_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='adv')
    old_p_log_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='old_p_log')
    old_mu_ph = tf.placeholder(shape=(None, act_dim), dtype=tf.float32, name='old_mu')
    old_log_std_ph = tf.placeholder(shape=(act_dim), dtype=tf.float32, name='old_log_std')
    p_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='p_ph')

    # result of the conjugate gradient algorithm
    cg_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='cg')

    #########################################################
    ######################## POLICY #########################
    #########################################################

    old_model_variables = tf.placeholder(shape=(None,), dtype=tf.float32, name='old_model_variables')

    # Neural network that represent the policy
    with tf.variable_scope('actor_nn'):
        p_means = mlp(obs_ph, hidden_sizes, act_dim, tf.tanh, last_activation=tf.tanh)
        p_means = tf.clip_by_value(p_means, low_action_space, high_action_space)
        log_std = tf.get_variable(name='log_std', initializer=np.ones(act_dim, dtype=np.float32))

    # Neural network that represent the value function
    with tf.variable_scope('critic_nn'):
        s_values = mlp(obs_ph, hidden_sizes, 1, tf.tanh, last_activation=None)
        s_values = tf.squeeze(s_values)

    # Add "noise" to the predicted mean following the Gaussian distribution with standard deviation e^(log_std)
    p_noisy = p_means + tf.random_normal(tf.shape(p_means), 0, 1) * tf.exp(log_std)
    # Clip the noisy actions
    a_sampl = tf.clip_by_value(p_noisy, low_action_space, high_action_space)
    # Compute the gaussian log likelihood
    p_log = gaussian_log_likelihood(act_ph, p_means, log_std)

    # Measure the divergence
    diverg = tf.reduce_mean(tf.exp(old_p_log_ph - p_log))

    # ratio
    ratio_new_old = tf.exp(p_log - old_p_log_ph)
    # TRPO surrogate loss function
    p_loss = - tf.reduce_mean(ratio_new_old * adv_ph)

    # MSE loss function
    v_loss = tf.reduce_mean((ret_ph - s_values) ** 2)
    # Critic optimization
    v_opt = tf.train.AdamOptimizer(cr_lr).minimize(v_loss)

    def variables_in_scope(scope):
        # get all trainable variables in 'scope'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    # Gather and flatten the actor parameters
    p_variables = variables_in_scope('actor_nn')
    p_var_flatten = flatten_list(p_variables)

    # Gradient of the policy loss with respect to the actor parameters
    p_grads = tf.gradients(p_loss, p_variables)
    p_grads_flatten = flatten_list(p_grads)

    ########### RESTORE ACTOR PARAMETERS ###########
    p_old_variables = tf.placeholder(shape=(None,), dtype=tf.float32, name='p_old_variables')
    # variable used as index for restoring the actor's parameters
    it_v1 = tf.Variable(0, trainable=False)
    restore_params = []

    for p_v in p_variables:
        upd_rsh = tf.reshape(p_old_variables[it_v1: it_v1 + tf.reduce_prod(p_v.shape)], shape=p_v.shape)
        restore_params.append(p_v.assign(upd_rsh))
        it_v1 += tf.reduce_prod(p_v.shape)

    restore_params = tf.group(*restore_params)

    # gaussian KL divergence of the two policies
    dkl_diverg = gaussian_DKL(old_mu_ph, old_log_std_ph, p_means, log_std)

    # Jacobian of the KL divergence (Needed for the Fisher matrix-vector product)
    dkl_diverg_grad = tf.gradients(dkl_diverg, p_variables)

    dkl_matrix_product = tf.reduce_sum(flatten_list(dkl_diverg_grad) * p_ph)
    print('dkl_matrix_product', dkl_matrix_product.shape)
    # Fisher vector product
    # The Fisher-vector product is a way to compute the A matrix without the need of the full A
    Fx = flatten_list(tf.gradients(dkl_matrix_product, p_variables))

    ## Step length
    beta_ph = tf.placeholder(shape=(), dtype=tf.float32, name='beta')
    # NPG update
    npg_update = beta_ph * cg_ph

    ## alpha is found through line search
    alpha = tf.Variable(1., trainable=False)
    # TRPO update
    trpo_update = alpha * npg_update

    ####################   POLICY UPDATE  ###################
    # variable used as an index
    it_v = tf.Variable(0, trainable=False)
    p_opt = []
    # Apply the updates to the policy
    for p_v in p_variables:
        upd_rsh = tf.reshape(trpo_update[it_v: it_v + tf.reduce_prod(p_v.shape)], shape=p_v.shape)
        p_opt.append(p_v.assign_sub(upd_rsh))
        it_v += tf.reduce_prod(p_v.shape)

    p_opt = tf.group(*p_opt)

    #########################################################
    ######################### MODEL #########################
    #########################################################

    m_opts = []
    m_losses = []

    nobs_pred_m = []
    act_obs = tf.concat([obs_ph, act_ph], 1)
    # computational graph of N models
    for i in range(num_ensemble_models):
        with tf.variable_scope('model_' + str(i) + '_nn'):
            nobs_pred = mlp(act_obs, [100, 100], obs_dim[0], tf.nn.relu, last_activation=None)
            nobs_pred_m.append(nobs_pred)

        m_loss = tf.reduce_mean((nobs_ph - nobs_pred) ** 2)
        m_losses.append(m_loss)

        m_opts.append(tf.train.AdamOptimizer(mb_lr).minimize(m_loss))

    ##################### RESTORE MODEL ######################
    initialize_models = []
    models_variables = []
    for i in range(num_ensemble_models):
        m_variables = variables_in_scope('model_' + str(i) + '_nn')
        initialize_models.append(restore_model(old_model_variables, m_variables))

        models_variables.append(flatten_list(m_variables))

    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)

    # Set scalars and hisograms for TensorBoard
    tf.summary.scalar('p_loss', p_loss, collections=['train'])
    tf.summary.scalar('v_loss', v_loss, collections=['train'])
    tf.summary.scalar('p_divergence', diverg, collections=['train'])
    tf.summary.scalar('ratio_new_old', tf.reduce_mean(ratio_new_old), collections=['train'])
    tf.summary.scalar('dkl_diverg', dkl_diverg, collections=['train'])
    tf.summary.scalar('alpha', alpha, collections=['train'])
    tf.summary.scalar('beta', beta_ph, collections=['train'])
    tf.summary.scalar('p_std_mn', tf.reduce_mean(tf.exp(log_std)), collections=['train'])
    tf.summary.scalar('s_values_mn', tf.reduce_mean(s_values), collections=['train'])
    tf.summary.histogram('p_log', p_log, collections=['train'])
    tf.summary.histogram('p_means', p_means, collections=['train'])
    tf.summary.histogram('s_values', s_values, collections=['train'])
    tf.summary.histogram('adv_ph', adv_ph, collections=['train'])
    tf.summary.histogram('log_std', log_std, collections=['train'])
    scalar_summary = tf.summary.merge_all('train')

    tf.summary.scalar('old_v_loss', v_loss, collections=['pre_train'])
    tf.summary.scalar('old_p_loss', p_loss, collections=['pre_train'])
    pre_scalar_summary = tf.summary.merge_all('pre_train')

    hyp_str = '-spe_' + str(steps_per_env) + '-envs_' + str(number_envs) + '-cr_lr' + str(cr_lr) + '-crit_it_' + str(
        critic_iter) + '-delta_' + str(delta) + '-conj_iters_' + str(conj_iters)
    file_writer = tf.summary.FileWriter('log_dir/' + env_name + '/' + algorithm + '_' + clock_time + '_' + hyp_str,
                                        tf.get_default_graph())

    # create a session
    sess = tf.Session()
    # initialize the variables
    sess.run(tf.global_variables_initializer())

    def action_op(o):
        return sess.run([p_means, s_values], feed_dict={obs_ph: o})

    def action_op_noise(o):
        return sess.run([a_sampl, s_values], feed_dict={obs_ph: o})

    def model_op(o, a, md_idx):
        # TODO: Modified code by Simon
        mo = sess.run(nobs_pred_m[md_idx], feed_dict={obs_ph: [o], act_ph: [a[0]]})
        return np.squeeze(mo)

    def run_model_loss(model_idx, r_obs, r_act, r_nxt_obs):
        # print({'obs_ph': r_obs.shape, 'act_ph': r_act.shape, 'nobs_ph': r_nxt_obs.shape})
        # TODO: Modified code by Simon
        r_act = np.squeeze(r_act, axis=2)
        return sess.run(m_losses[model_idx], feed_dict={obs_ph: r_obs, act_ph: r_act, nobs_ph: r_nxt_obs})

    def run_model_opt_loss(model_idx, r_obs, r_act, r_nxt_obs):
        # TODO: Modified code by Simon
        r_act = np.squeeze(r_act, axis=2)
        return sess.run([m_opts[model_idx], m_losses[model_idx]],
                        feed_dict={obs_ph: r_obs, act_ph: r_act, nobs_ph: r_nxt_obs})

    def model_assign(i, model_variables_to_assign):
        '''
        Update the i-th model's parameters
        '''
        return sess.run(initialize_models[i], feed_dict={old_model_variables: model_variables_to_assign})

    def policy_update(obs_batch, act_batch, adv_batch, rtg_batch):
        # log probabilities, logits and log std of the "old" policy
        # "old" policy refer to the policy to optimize and that has been used to sample from the environment
        # TODO: Modified code by Simon
        act_batch = np.squeeze(act_batch, axis=2)
        old_p_log, old_p_means, old_log_std = sess.run([p_log, p_means, log_std],
                                                       feed_dict={obs_ph: obs_batch, act_ph: act_batch,
                                                                  adv_ph: adv_batch, ret_ph: rtg_batch})
        # get also the "old" parameters
        old_actor_params = sess.run(p_var_flatten)

        # old_p_loss is later used in the line search
        # run pre_scalar_summary for a summary before the optimization
        old_p_loss, summary = sess.run([p_loss, pre_scalar_summary],
                                       feed_dict={obs_ph: obs_batch, act_ph: act_batch, adv_ph: adv_batch,
                                                  ret_ph: rtg_batch, old_p_log_ph: old_p_log})
        file_writer.add_summary(summary, step_count)

        def H_f(p):
            '''
            Run the Fisher-Vector product on 'p' to approximate the Hessian of the DKL
            '''
            return sess.run(Fx,
                            feed_dict={old_mu_ph: old_p_means, old_log_std_ph: old_log_std, p_ph: p, obs_ph: obs_batch,
                                       act_ph: act_batch, adv_ph: adv_batch, ret_ph: rtg_batch})

        g_f = sess.run(p_grads_flatten,
                       feed_dict={old_mu_ph: old_p_means, obs_ph: obs_batch, act_ph: act_batch, adv_ph: adv_batch,
                                  ret_ph: rtg_batch, old_p_log_ph: old_p_log})
        ## Compute the Conjugate Gradient so to obtain an approximation of H^(-1)*g
        # Where H in reality isn't the true Hessian of the KL divergence but an approximation of it computed via Fisher-Vector Product (F)
        conj_grad = conjugate_gradient(H_f, g_f, iters=conj_iters)

        # Compute the step length
        beta_np = np.sqrt(2 * delta / (1e-10 + np.sum(conj_grad * H_f(conj_grad))))

        def DKL(alpha_v):
            '''
            Compute the KL divergence.
            It optimize the function to compute the DKL. Afterwards it restore the old parameters.
            '''
            sess.run(p_opt, feed_dict={beta_ph: beta_np, alpha: alpha_v, cg_ph: conj_grad, obs_ph: obs_batch,
                                       act_ph: act_batch, adv_ph: adv_batch, old_p_log_ph: old_p_log})
            a_res = sess.run([dkl_diverg, p_loss],
                             feed_dict={old_mu_ph: old_p_means, old_log_std_ph: old_log_std, obs_ph: obs_batch,
                                        act_ph: act_batch, adv_ph: adv_batch, ret_ph: rtg_batch,
                                        old_p_log_ph: old_p_log})
            sess.run(restore_params, feed_dict={p_old_variables: old_actor_params})
            return a_res

        # Actor optimization step
        # Different for TRPO or NPG
        # Backtracing line search to find the maximum alpha coefficient s.t. the constraint is valid
        best_alpha = backtracking_line_search(DKL, delta, old_p_loss, p=0.8)
        sess.run(p_opt,
                 feed_dict={beta_ph: beta_np, alpha: best_alpha, cg_ph: conj_grad, obs_ph: obs_batch, act_ph: act_batch,
                            adv_ph: adv_batch, old_p_log_ph: old_p_log})

        lb = len(obs_batch)
        shuffled_batch = np.arange(lb)
        np.random.shuffle(shuffled_batch)

        # Value function optimization steps
        for _ in range(critic_iter):
            # shuffle the batch on every iteration
            np.random.shuffle(shuffled_batch)
            for idx in range(0, lb, minibatch_size):
                minib = shuffled_batch[idx:min(idx + minibatch_size, lb)]
                sess.run(v_opt, feed_dict={obs_ph: obs_batch[minib], ret_ph: rtg_batch[minib]})

    def train_model(tr_obs, tr_act, tr_nxt_obs, v_obs, v_act, v_nxt_obs, step_count, model_idx):

        # Get validation loss on the old model
        mb_valid_loss1 = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs)

        # Restore the random weights to have a new, clean neural network
        model_assign(model_idx, initial_variables_models[model_idx])

        mb_valid_loss = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs)

        acc_m_losses = []
        last_m_losses = []
        md_params = sess.run(models_variables[model_idx])
        best_mb = {'iter': 0, 'loss': mb_valid_loss, 'params': md_params}
        it = 0

        lb = len(tr_obs)
        shuffled_batch = np.arange(lb)
        np.random.shuffle(shuffled_batch)

        while best_mb['iter'] > it - model_iter:

            # update the model on each mini-batch
            last_m_losses = []
            for idx in range(0, lb, model_batch_size):
                minib = shuffled_batch[idx:min(idx + minibatch_size, lb)]

                if len(minib) != minibatch_size:
                    _, ml = run_model_opt_loss(model_idx, tr_obs[minib], tr_act[minib], tr_nxt_obs[minib])
                    acc_m_losses.append(ml)
                    last_m_losses.append(ml)
                else:
                    pass
                    # print('Warning!')

            # Check if the loss on the validation set has improved
            mb_valid_loss = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs)
            if mb_valid_loss < best_mb['loss']:
                best_mb['loss'] = mb_valid_loss
                best_mb['iter'] = it
                best_mb['params'] = sess.run(models_variables[model_idx])

            it += 1
            # print('iteration: ', it)

        # Restore the model with the lower validation loss
        model_assign(model_idx, best_mb['params'])

        print('Model:{}, iter:{} -- Old Val loss:{:.6f}  New Val loss:{:.6f} -- New Train loss:{:.6f}'.format(model_idx,
                                                                                                              it,
                                                                                                              mb_valid_loss1,
                                                                                                              best_mb[
                                                                                                                  'loss'],
                                                                                                              np.mean(
                                                                                                                  last_m_losses)))
        summary = tf.Summary()
        summary.value.add(tag='supplementary/m_loss', simple_value=np.mean(acc_m_losses))
        summary.value.add(tag='supplementary/iterations', simple_value=it)
        file_writer.add_summary(summary, step_count)
        file_writer.flush()

    # variable to store the total number of steps
    step_count = 0
    model_buffer = FullBuffer()
    print('Env batch size:', steps_per_env, ' Batch size:', steps_per_env * number_envs)

    # Create a simulated environment
    sim_env = NetworkEnv(make_env(), model_op, reward_function, episode_done, num_ensemble_models)

    # Get the initial parameters of each model
    # These are used in later epochs when we aim to re-train the models anew with the new dataset
    initial_variables_models = []
    for model_var in models_variables:
        initial_variables_models.append(sess.run(model_var))
    converged = False
    ep = -1
    history_data = []
    ep_data = []

    taken_steps = 0

    while not (converged) and ep < num_epochs:
        ep += 1
        # lists to store rewards and length of the trajectories completed
        batch_rew = []
        batch_len = []
        print('============================', ep, '============================')
        # Execute in serial the environment, storing temporarily the trajectories.
        for env in envs:
            init_log_std = np.ones(act_dim) * np.log(np.random.rand() * 1)
            env.reset()

            if ep==0:
                env_new = awakeEnv(action_space=element_actor_list_selected, state_space=element_state_list_selected,
                                   number_bpm_measurements=number_bpm_measurements, noSet=False, debug=True,
                                   scale=scaling)
                mn_test, length, success_rate = test_agent(env_new, action_op, num_games=100,
                                                           model_buffer=False)  #
                df = pd.concat([pd.DataFrame(mn_test), pd.DataFrame(length),
                                pd.DataFrame(success_rate)], axis=1)
                df.columns = ['Ep. rews', 'Ep. lens', 'Ep. success']
                to_pickle(df, 'Awake_test_init.pkl')

                print(' \nContinuous Test score on awake: ', np.round(np.mean(mn_test), 2),
                      np.round(np.std(mn_test), 2),
                      np.round(np.mean(length), 2), np.round(np.mean(success_rate), 2), '\n')

                plot_results(env_new,
                             'ME-TRPO on AWAKE ep.: , {0[0]}, {0[1]}, {0[2]}, {0[3]}'.format(
                                 (np.round(np.mean(mn_test), 2),
                                  np.round(np.std(mn_test), 2),
                                  np.round(np.mean(length), 2),
                                  np.round(
                                      np.mean(success_rate),
                                      2))))

            # iterate over a fixed number of steps
            # TODO: Changed to rest avoid if not converged empty set

            if taken_steps > 0:
                rest_steps_out = 0
            else:
                rest_steps_out = steps_per_env
             # if ep < 1 else 0
            # max(0, (ep + 1) * steps_per_env - len(model_buffer))
            print('rest steps', rest_steps_out)
            # rest_steps = steps_per_env
            for _ in range(rest_steps_out):
                # run the policy

                if ep == 0:
                    # Sample random action during the first epoch
                    # act = np.random.randn(env.action_space.shape[0])
                    act = env.action_space.sample()
                    # print(np.mean(act))
                else:
                    # TODO: change feedback to new version
                    act = sess.run(a_sampl, feed_dict={obs_ph: [env.n_obs], log_std: init_log_std})

                act = np.squeeze(act)

                # take a step in the real environment
                obs2, rew, done, _ = env.step(np.array(act))

                # add the new transition to the temporary buffer
                model_buffer.store(env.n_obs.copy(), act, rew, obs2.copy(), done)

                env.n_obs = obs2.copy()
                step_count += 1

                if done:
                    batch_rew.append(env.get_episode_reward())
                    # print(env.get_episode_reward())
                    batch_len.append(env.get_episode_length())
                    env.reset()
                    init_log_std = np.ones(act_dim) * np.log(np.random.rand() * 1)

        print('Ep:%d Rew:%.2f -- Len:%.2f -- Step:%d' % (ep, np.mean(batch_rew), np.mean(batch_len), step_count))

        ############################################################
        ###################### MODEL LEARNING ######################
        ############################################################

        # Initialize randomly a training and validation set
        model_buffer.generate_random_dataset()

        # get both datasets
        train_obs, train_act, _, train_nxt_obs, _ = model_buffer.get_training_batch()
        valid_obs, valid_act, _, valid_nxt_obs, _ = model_buffer.get_valid_batch()

        print('Log Std policy:', sess.run(log_std))
        for i in range(num_ensemble_models):
            # train the dynamic model on the datasets just sampled
            train_model(train_obs, train_act, train_nxt_obs, valid_obs, valid_act, valid_nxt_obs, step_count, i)

        ############################################################
        ###################### POLICY LEARNING ######################
        ############################################################

        # depends on the threshold
        # TODO: Modified code by Simon
        best_sim_test = -1e6 * np.ones(num_ensemble_models)
        dynamic_threshold = rms_threshold  # max(1, 1 - ep) * rms_threshold
        dynamic_done = lambda ob: -np.sqrt(np.mean(np.square(ob))) > dynamic_threshold
        # print('Non-Dynamic reward', dynamic_threshold)
        current_step_size = len(model_buffer)
        quality = False
        count = -1
        for it in range(100):
            ep_data.append([ep, it, current_step_size])
            to_pickle(ep_data, 'ep_data.pkl')
            # Create a dynamic simulated environment
            sim_env = NetworkEnv(make_env(), model_op, reward_function, dynamic_done, num_ensemble_models)
            # print('length is:', len(model_buffer))
            print('\n Policy it', it, end='.. \n')
            simulated_steps = 2000

            for _ in range(10):
                ##################### MODEL SIMLUATION #####################
                obs_batch, act_batch, adv_batch, rtg_batch = simulate_environment(sim_env, action_op_noise,
                                                                                  simulated_steps)

                ################# TRPO UPDATE ################
                policy_update(obs_batch, act_batch, adv_batch, rtg_batch)
                ################# TRPO UPDATE ################

            env_new = awakeEnv(action_space=element_actor_list_selected, state_space=element_state_list_selected,
                               number_bpm_measurements=number_bpm_measurements, noSet=False, debug=True, scale=scaling)
            mn_test, length, success_rate = test_agent(env_new, action_op, num_games=100,
                                                       model_buffer=False)  #
            df = pd.concat([pd.DataFrame(mn_test), pd.DataFrame(length),
                            pd.DataFrame(success_rate)], axis=1)
            df.columns = ['Ep. rews', 'Ep. lens', 'Ep. success']

            to_pickle(df, 'Awake_test_' + number_to_text(ep) + '_' + number_to_text(it) + '.pkl')

            print(' \nContinuous Test score on awake: ', np.round(np.mean(mn_test), 2),
                  np.round(np.std(mn_test), 2),
                  np.round(np.mean(length), 2), np.round(np.mean(success_rate), 2), '\n')

            plot_results(env_new,
                         'ME-TRPO on AWAKE ep.: , {0[0]}, {0[1]}, {0[2]}, {0[3]}'.format((np.round(np.mean(mn_test), 2),
                                                                                          np.round(np.std(mn_test), 2),
                                                                                          np.round(np.mean(length), 2),
                                                                                          np.round(
                                                                                              np.mean(success_rate),
                                                                                              2))))

            # Test the policy on simulated environment.
            if (it + 1) % 1 == 0:
                print('\nSimulated test:', end=' -- \n')
                sim_rewards = []
                sim_lengths = []

                true_rewards = []
                true_lengths = []
                model_data = []
                for i in range(num_ensemble_models):
                    sim_m_env = NetworkEnv(make_env(), model_op, reward_function, episode_done, i + 1)
                    mn_sim_rew, mn_sim_len, success_rate = test_agent(sim_m_env, action_op, num_games=100,
                                                                      model_buffer=False)

                    sim_rewards.append(np.mean(mn_sim_rew))
                    sim_lengths.append(np.mean(mn_sim_len))
                    print(np.mean(mn_sim_rew), ' ', np.mean(mn_sim_len), end=' -- \n')

                    df = pd.concat([pd.DataFrame(mn_sim_rew), pd.DataFrame(mn_sim_len),
                                    pd.DataFrame(success_rate)], axis=1)
                    df.columns = ['Ep. rews', 'Ep. lens', 'Ep. success']

                    model_data.append(df)
                model_all = pd.concat(model_data, keys=range(len(model_data)))
                # print(model_all)
                to_pickle(model_all, 'Awake_model_' + number_to_text(ep) + '_' + number_to_text(it) + '.pkl')

                if np.mean(sim_lengths) < 1.25:
                    quality = True

                try:
                    df = pd.concat([pd.DataFrame(sim_lengths), pd.DataFrame(sim_rewards)], axis=1)
                    df.columns = ['Ep. lens', 'Ep. rews']
                    history_data.append(df)
                    g = sns.PairGrid(df, diag_sharey=False)
                    g.map_lower(sns.kdeplot)
                    g.map_upper(sns.scatterplot)
                    g.map_diag(sns.kdeplot, lw=3)
                    # plt.xlim(1, 3)
                    plt.show()
                except:
                    print('')
                print("")
                sim_rewards = np.array(sim_rewards)
                # print(best_sim_test, sim_rewards)
                # stop training if the policy hasn't improved
                if (np.sum(best_sim_test >= sim_rewards) > int(num_ensemble_models * 0.7)):  # \
                    # or (len(mn_sim_len[mn_sim_len <= 1]) > int(num_ensemble_models * 2 / 3)):
                    print('Break')
                    mn_test, length, success_rate = test_agent(env_test, action_op, num_games=10,
                                                               model_buffer=model_buffer)  #
                    print(' \nTest score on awake: ', np.round(np.mean(mn_test), 2),
                          np.round(np.std(mn_test), 2),
                          np.round(np.mean(length), 2), np.round(np.mean(success_rate), 2), '\n')

                    taken_steps = len(model_buffer) - current_step_size

                    break
                else:
                    best_sim_test = sim_rewards
            if (it + 1) % 10 == 0 and quality:
                # Testing the policy on a real environment
                mn_test, length, success_rate = test_agent(env_test, action_op, num_games=10, model_buffer=model_buffer)
                print(' \nTest score on awake: ', np.round(np.mean(mn_test), 2),
                      np.round(np.std(mn_test), 2),
                      np.round(np.mean(length), 2), np.round(np.mean(success_rate), 2), '\n')

                summary = tf.Summary()
                summary.value.add(tag='test/performance', simple_value=np.mean(mn_test))
                file_writer.add_summary(summary, step_count)
                file_writer.flush()

            # if mn_test > env_test.threshold:
            #     break
            taken_steps = len(model_buffer) - current_step_size
            print('add steps: ', taken_steps)
            if taken_steps >= 100:
                print('\n break max steps')
                break
            rest_steps = len(model_buffer) - current_step_size
    # Testing the policy on a real environment
    mn_test, length, success_rate = test_agent(env_test, action_op, num_games=150)  # , model_buffer=model_buffer)
    print(' Final score on awake: ', np.round(np.mean(mn_test), 2),
          np.round(np.std(mn_test), 2),
          np.round(np.mean(length), 2), np.round(np.mean(success_rate), 2), '\n')
    # to_pickle(history_data, 'all_settings.pkl')

    # closing environments..
    for env in envs:
        env.close()
    file_writer.close()


if __name__ == '__main__':
    # set random seed
    random_seed = 222
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    METRPO('', hidden_sizes=[100, 100], cr_lr=1e-3, gamma=0.999, lam=0.95, num_epochs=10, steps_per_env=50,
           number_envs=1, critic_iter=10, delta=.1, algorithm='TRPO', conj_iters=10, minibatch_size=100,
           mb_lr=1e-3, model_batch_size=100, simulated_steps=50, num_ensemble_models=25, model_iter=5)

    # plot the results
    plot_results(env, 'ME-TRPO on AWAKE', save_name = 'On_the_machine')
