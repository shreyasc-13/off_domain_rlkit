from rlkit.samplers.rollout_functions import rollout
<<<<<<< HEAD
import os
import sys
# import sys
# print(sys.path)
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# grandparentdir=os.path.dirname(parentdir)
# sys.path.append(grandparentdir)
import torch.nn as nn
import torch


import numpy as np
import rlkit.torch.pytorch_util as ptu

import gym
import collections
import time
import tqdm
import gym.spaces
import networkx as nx
import sys

=======
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger

filename = str(uuid.uuid4())

<<<<<<< HEAD
WALLS = {
        'Small':
                np.array([[0, 0, 0, 0, 0, 0, 0 ],
                          [0, 0, 0, 0, 0, 0, 0 ],
                          [0, 0, 0, 0, 0, 0, 0 ],
                          [0, 0, 0, 0, 0, 0, 0 ],
                          [0, 0, 0, 0, 0, 0, 0 ],
                          [0, 0, 0, 0, 0, 0, 0 ],
                          [0, 0, 0, 0, 0, 0, 0 ]]),
        'Cross':
                np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]]),

        'Maze7x7':
                np.array([[0, 0, 0, 1, 0, 0, 0],
                          [1, 0, 0, 1, 0, 1, 0],
                          [1, 1, 0, 1, 0, 1, 1],
                          [1, 0, 0, 1, 0, 0, 1],
                          [1, 0 ,0, 1, 1, 0, 1],
                          [1, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1, 1, 1]]),
}


class PointEnv(gym.Env):

    def __init__(self, walls=None, resize_factor=1,
                             action_noise=1.0):
        if resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]
        (height, width) = self._walls.shape
        self.init_states=[]
        self._height = height
        self._width = width
        self._action_noise = action_noise
        self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0]),
                high=np.array([1.0, 1.0]),
                dtype=np.float32)
        self.observation_space = gym.spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([self._height, self._width]),
                dtype=np.float32)
        self._step_count=0
        self.reset()

    def _sample_empty_state(self):
        candidate_states = np.where(self._walls == 0)
        num_candidate_states = len(candidate_states[0])
        state_index = np.random.choice(num_candidate_states)
        state = np.array([candidate_states[0][state_index],
                                            candidate_states[1][state_index]],
                                         dtype=np.float)
        state += np.random.uniform(size=2)
        assert not self._is_blocked(state)
        return state
        
    def reset(self):
        self.state = self._sample_empty_state()
        self._step_count = 0
        self.init_states.append(self.state)
        return self.state.copy()
    
    def _get_distance(self, obs, goal):
        """Compute the shortest path distance.
        
        Note: This distance is *not* used for training."""
        (i1, j1) = self._discretize_state(obs)
        (i2, j2) = self._discretize_state(goal)
        return self._apsp[i1, j1, i2, j2]

    def _discretize_state(self, state, resolution=1.0):
        (i, j) = np.floor(resolution * state).astype(np.int)
        # Round down to the nearest cell if at the boundary.
        if i == self._height:
            i -= 1
        if j == self._width:
            j -= 1
        return (i, j)
    
    def _is_blocked(self, state):
        if not self.observation_space.contains(state):
            return True
        (i, j) = self._discretize_state(state)
        return (self._walls[i, j] == 1)

    def step(self, action):
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise, size=2) 
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)
        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                new_state[axis] += dt * action[axis]
                if not self._is_blocked(new_state):
                    self.state = new_state 
        done = False
        # self._step_count += 1
        # if self._step_count >= self._duration or ts.is_last():
        #   done=False
        rew = -1.0 * np.linalg.norm(self.state)
        return self.state.copy(), rew, done, {}

    @property
    def walls(self):
        return self._walls

    def get_env_shape(self):
        return  self._height, self._width, self.action_space, self.observation_space, self._walls

def plot_env(env, env_name):
    plt.title(env_name)
    height, width, action_space, observation_space, wall= env.get_env_shape()
    for (i, j) in zip(*np.where(wall)):
        x = np.array([i, i+1]) / float(width)
        y0 = np.array([j, j]) / float(height)
        y1 = np.array([j+1, j+1]) / float(height)
        plt.fill_between(x, y0, y1, color='grey')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(wspace=0.1, hspace=0.2)



def simulate_policy(args):
    print(args.file)
=======

def simulate_policy(args):
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
