import logging.config
# import environments.twissReader as twissReader
import math
import random
from enum import Enum
import scipy.optimize as opt
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 3rd party modules
from gym import spaces
from bayes_opt import BayesianOptimization

from Application import twissReader


class e_trajectory_simENV(gym.Env):
    """
    Define a simple AWAKE environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, **kwargs):
        self.current_action = None
        self.initial_conditions = []
        self.__version__ = "0.0.1"
        logging.info("e_trajectory_simENV - Version {}".format(self.__version__))

        # General variables defining the environment
        self.MAX_TIME = 25
        self.is_finalized = False
        self.current_episode = -1

        # For internal stats...
        self.action_episode_memory = []
        self.rewards = []
        self.current_steps = 0
        self.TOTAL_COUNTER = 0

        self.seed()
        self.twissH, self.twissV = twissReader.readAWAKEelectronTwiss()

        self.bpmsH = self.twissH.getElements("BP")
        self.bpmsV = self.twissV.getElements("BP")

        self.correctorsH = self.twissH.getElements("MCA")
        self.correctorsV = self.twissV.getElements("MCA")

        self.responseH = self._calculate_response(self.bpmsH, self.correctorsH)
        self.responseV = self._calculate_response(self.bpmsV, self.correctorsV)

        self.positionsH = np.zeros(len(self.bpmsH.elements))
        self.settingsH = np.zeros(len(self.correctorsH.elements))
        self.positionsV = np.zeros(len(self.bpmsV.elements))
        self.settingsV = np.zeros(len(self.correctorsV.elements))
        # golden_data = pd.read_hdf('golden.h5')
        # self.goldenH = 1e-3*golden_data.describe().loc['mean'].values
        # print(self.goldenH)
        self.goldenH = np.zeros(len(self.bpmsV.elements))
        # self.goldenH =0.005*np.ones(len(self.bpmsH.elements))
        self.goldenV = np.zeros(len(self.bpmsV.elements))

        self.plane = Plane.horizontal

        high = 3 * np.ones(len(self.correctorsH.elements))
        low = (-1) * high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.act_lim = self.action_space.high[0]

        high = np.ones(len(self.bpmsH.elements))
        low = (-1) * high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # print('dtype ', self.observation_space.shape)

        if 'scale' in kwargs:
            self.action_scale = kwargs.get('scale')
        else:
            self.action_scale = 5e-4
        # print('selected scale at: ', self.action_scale)

        self.state_scale = 100  # Meters to millimeters as given from BPMs in the measurement later on
        self.reward_scale = 100  # Important
        self.threshold = -0.001 * self.reward_scale
        # self.TOTAL_COUNTER = -1
        self.stored_action = None

    def step(self, action, reference_position=None):

        state, reward = self._take_action(action)

        # For the statistics
        self.action_episode_memory[self.current_episode].append(action)

        # Check if episode time is over
        self.current_steps += 1
        if self.current_steps > self.MAX_TIME:
            self.is_finalized = True
        # To finish the episode if reward is sufficient
        # Rescale to fit millimeter reward
        return_reward = reward * self.reward_scale

        self.rewards[self.current_episode].append(return_reward)

        # state = state - self.goldenH
        return_state = np.array(state * self.state_scale)
        if (return_reward > self.threshold or return_reward < 10*self.threshold):
            self.is_finalized = True
            # if return_reward < -10:
            #     reward = -99
            # print('Finished at reward of:', reward, ' total episode nr.: ', self.current_episode)
            # print(action, return_state, return_reward)
        # print('Total interaction :', self.TOTAL_COUNTER)

        return return_state, return_reward, self.is_finalized, {}

    def setGolden(self, goldenH, goldenV):
        self.goldenH = goldenH
        self.goldenV = goldenV

    def setPlane(self, plane):
        if (plane == Plane.vertical or plane == Plane.horizontal):
            self.plane = plane
        else:
            raise Exception("You need to set plane enum")

    def seed(self, seed):
        np.random.seed(seed)

    def _take_action(self, action):
        # if self.stored_action is None:
        #     self.stored_action = action
        #     set_action = action
        # else:
        #     if np.random.uniform()<.0:
        #         set_action = self.stored_action
        #         print('stuck')
        #     else:
        #         set_action = action
        #         self.stored_action = action
        # The action is scaled here for the communication with the hardware
        if self.current_action is None:
            kicks = action * self.action_scale - self.k0
            self.current_action = action
        else:
            kicks = (action-self.current_action) * self.action_scale
            self.current_action = action

        # kicks = set_action * self.action_scale

        # kicks += 0.1*np.random.randn(self.action_space.shape[0]) * self.action_scale
        # Apply the kicks...
        state, reward = self._get_state_and_reward(kicks, self.plane)
        # state += 0.1*np.random.randn(self.observation_space.shape[0])
        return state, reward

    def _get_reward(self, golden, trajectory):
        rms = np.sqrt(np.mean(np.square(trajectory)))
        return (rms * (-1.))

    def _get_state_and_reward(self, kicks, plane):
        self.TOTAL_COUNTER += 1
        if (plane == Plane.horizontal):
            init_positions = self.positionsH
            rmatrix = self.responseH
            golden = self.goldenH

        if (plane == Plane.vertical):
            init_positions = self.positionsV
            rmatrix = self.responseV
            golden = self.goldenV

        state = self._calculate_trajectory(init_positions, rmatrix, kicks-self.k0)
        state -= self.goldenH
        reward = self._get_reward(golden, state)
        return state, reward

    def _calculate_response(self, bpmsTwiss, correctorsTwiss):
        bpms = bpmsTwiss.elements
        correctors = correctorsTwiss.elements
        rmatrix = np.zeros((len(bpms), len(correctors)))

        for i, bpm in enumerate(bpms):
            for j, corrector in enumerate(correctors):
                if (bpm.mu > corrector.mu):
                    rmatrix[i][j] = math.sqrt(bpm.beta * corrector.beta) * math.sin(
                        (bpm.mu - corrector.mu) * 2. * math.pi)
                else:
                    rmatrix[i][j] = 0.0
        return rmatrix

    def _calculate_trajectory(self, initial_pos, rmatrix, delta_settings ):
        # add_noise = np.random.rand
        initial_pos = 0  # TODO:Remove
        delta_settings = np.squeeze(delta_settings)
        return initial_pos + rmatrix.dot(delta_settings)

    def reset(self, **kwargs):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        simulation = False
        self.is_finalized = False

        if (self.plane == Plane.horizontal):
            self.settingsH = np.clip(np.random.randn(len(self.settingsH)), -self.act_lim, self.act_lim)
            # self.settingsH = (np.random.uniform(-2, 2, self.action_space.shape[0]))
            kicks = self.settingsH * self.action_scale
        if (self.plane == Plane.vertical):
            self.settingsV = np.clip(0.5 * np.random.randn(len(self.settingsV)), -self.act_lim, self.act_lim)
            kicks = self.settingsV * self.action_scale

        if (self.plane == Plane.horizontal):
            init_positions = np.zeros(len(self.positionsH))  # self.positionsH
            rmatrix = self.responseH

        if (self.plane == Plane.vertical):
            init_positions = np.zeros(len(self.positionsV))  # self.positionsV
            rmatrix = self.responseV

        if 'simulation' in kwargs:
            simulation = kwargs.get('simulation')

        if simulation:
            # print('init simulation...')
            return_value =  kicks
        else:
            # print('init')
            self.current_episode += 1
            self.current_steps = 0
            self.action_episode_memory.append([])
            self.rewards.append([])
            self.k0 = kicks
            state = self._calculate_trajectory(init_positions, rmatrix, kicks)

            if (self.plane == Plane.horizontal):
                self.positionsH = state
            if (self.plane == Plane.vertical):
                self.positionsV = state
            # Rescale for agent
            # state = state
            return_initial_state = np.array(state * self.state_scale)

            self.initial_conditions.append([return_initial_state])
            return_value = return_initial_state

        return return_value

    def seed(self, seed=None):
        random.seed(seed)


class Plane(Enum):
    horizontal = 0
    vertical = 1


if __name__ == '__main__':

    environment_instance = e_trajectory_simENV()

    rews = []
    actions = []


    def objective(action):
        actions.append(action.copy())
        _, r, _, _ = environment_instance.step(action=action)
        rews.append(r*1e0)
        return -r


    # print(environment_instance.reset())
    if True:

        def constr(action):
            if any(action > environment_instance.action_space.high[0]):
                return -1
            elif any(action < environment_instance.action_space.low[0]):
                return -1
            else:
                return 1


        print('init: ', environment_instance.reset())
        start_vector = np.zeros(environment_instance.action_space.shape[0])
        rhobeg = 1 * environment_instance.action_space.high[0]
        print('rhobeg: ', rhobeg)
        res = opt.fmin_cobyla(objective, start_vector, [constr], rhobeg=rhobeg, rhoend=.1)
        print(res)

    if False:
        # Bounded region of parameter space
        pbounds = dict([('x' + str(i), (environment_instance.action_space.low[0],
                                        environment_instance.action_space.high[0])) for i in range(1, 12)])


        def black_box_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
            func_val = -1 * objective(np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, ]))
            return func_val


        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=3, )

        optimizer.maximize(
            init_points=25,
            n_iter=100,
            acq="ucb"
        )
        objective(np.array([optimizer.max['params'][x] for x in optimizer.max['params']]))

    fig, axs = plt.subplots(2, sharex=True)
    axs[1].plot(rews)

    pd.DataFrame(actions).plot(ax=axs[0])
    plt.show()
