# from gym.envs.mujoco impo/rt HalfCheetahEnv
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
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from trainer import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import gym
import collections
import time
import tqdm
import gym.spaces
import networkx as nx
import sys


# sys.path.insert(0,'../')
# print(s)



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

def experiment(variant):
    # import pdb
    # pdb.set_trace()
    sim_env_name='Small'
    real_env_name = 'Maze7x7' 
    expl_env = PointEnv(walls=sim_env_name)
    eval_env = PointEnv(walls=real_env_name)
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    real_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )
    sim_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    sim_replay_buffer= EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    real_replay_buffer= EnvReplayBuffer(
        variant['replay_buffer_size'],
        eval_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        batch_size=variant['algorithm_kwargs']['batch_size'],
        max_path_length=variant['algorithm_kwargs']['max_path_length'],
        num_epochs=variant['algorithm_kwargs']['num_epochs'],
        num_eval_steps_per_epoch=variant['algorithm_kwargs']['num_eval_steps_per_epoch'],
        num_trains_per_train_loop=variant['algorithm_kwargs']['num_trains_per_train_loop'],

        evaluation_data_collector=eval_path_collector,
        sim_data_collector= sim_path_collector,
        real_data_collector=real_path_collector,
        sim_replay_buffer=sim_replay_buffer,
        real_replay_buffer= real_replay_buffer,
        num_real_steps_at_init=1000,
        num_sim_steps_at_init=0,
        num_real_steps_per_epoch=100,
        num_sim_steps_per_epoch=0,
        num_rl_train_steps_per_iter=1,

        rl_on_real=True,
        modify_reward=False,
        num_classifier_train_steps_per_iter=1

        
    )
    algorithm.to(ptu.device)
    algorithm.train()



if __name__ == "__main__":
    # noinspection PyTypeChecker

    # classifier= Network(input_size = 6, output_size = 2, unit_count = SAS_unit_count)
    # state_dict=torch.load('/home/swapnil/DRL/offDyna/odrl/data/SASNetwork.pt',map_location='cpu')
    # classifier.load_state_dict(state_dict)


    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)

