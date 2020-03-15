import os
import sys
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
from pointenv import * 

def experiment(variant):
    # import pdb
    # pdb.set_trace()
    sim_env_name='Small'
    real_env_name = 'Maze7x7' 
    tolerance=variant["tolerance"]
    sparse=variant["sparse"]
    sim_expl_env  = PointEnv(sim_env_name,  sparse, tolerance)
    sim_eval_env  = PointEnv(sim_env_name,  sparse, tolerance)
    real_expl_env = PointEnv(real_env_name, sparse, tolerance)
    real_eval_env = PointEnv(real_env_name, sparse, tolerance)
    obs_dim = sim_expl_env.observation_space.low.size
    action_dim = sim_expl_env.action_space.low.size
    max_episode_steps=variant['algorithm_kwargs']['max_episode_length']
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
    # eval_policy = MakeDeterministic(policy)
    eval_real_path_collector = MdpPathCollector(
        real_eval_env,
        policy,
        max_episode_steps
    )
    eval_sim_path_collector = MdpPathCollector(
        sim_eval_env,
        policy,
        max_episode_steps
    )
    real_path_collector = MdpPathCollector(
        real_expl_env,
        policy,
        max_episode_steps
    )
    sim_path_collector = MdpPathCollector(
        sim_expl_env,
        policy,
        max_episode_steps
    )
    sim_replay_buffer= EnvReplayBuffer(
        variant['replay_buffer_size'],
        sim_expl_env,
    )
    real_replay_buffer= EnvReplayBuffer(
        variant['replay_buffer_size'],
        real_expl_env,
    )
    trainer_env= real_expl_env if variant['rl_on_real'] ==True else sim_expl_env
    trainer = SACTrainer(
        env=trainer_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        # hardcode_classifier=variant['hardcode_classifier'],
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        sim_exploration_env=sim_expl_env,
        real_exploration_env=real_expl_env,
        evaluation_sim_env=sim_eval_env,
        evaluation_real_env=real_eval_env,
        batch_size=variant['algorithm_kwargs']['batch_size'],
        max_path_length=variant['algorithm_kwargs']['max_path_length'],
        max_episode_steps=variant['algorithm_kwargs']['max_episode_length'],
        num_epochs=variant['algorithm_kwargs']['num_epochs'],
        num_eval_steps_per_epoch=variant['algorithm_kwargs']['num_eval_steps_per_epoch'],
        num_trains_per_train_loop=variant['algorithm_kwargs']['num_trains_per_train_loop'],
        evaluation_real_data_collector=eval_real_path_collector,
        evaluation_sim_data_collector=eval_sim_path_collector,
        sim_data_collector= sim_path_collector,
        real_data_collector=real_path_collector,
        sim_replay_buffer=sim_replay_buffer,
        real_replay_buffer= real_replay_buffer,

        num_real_steps_at_init=     50    if variant['rl_on_real'] else 5000,
        num_sim_steps_at_init=      0       if variant['rl_on_real'] else 5000,
        num_real_steps_per_epoch=   0     if variant['rl_on_real'] else 100 if variant['num_classifier_train_steps_per_iter'] else 0,
        num_sim_steps_per_epoch=    0       if variant['rl_on_real'] else 5000 if variant['num_classifier_train_steps_per_iter'] else 5000,
        num_rl_train_steps_per_iter=1,

        rl_on_real=variant['rl_on_real'],
        modify_reward=False if variant['rl_on_real']==True else True,
        num_classifier_train_steps_per_iter=variant['num_classifier_train_steps_per_iter'],
        num_train_loops_per_epoch=1,
        num_classifier_init_epoch=variant['num_classifier_init_epoch'],
        classifier_batch_size=512,
        tolerance=tolerance,
        plot_episodes_period=1 if variant['algorithm_kwargs']['num_epochs']<5 else int(variant['algorithm_kwargs']['num_epochs']/5),
        hardcode_classifier=variant['hardcode_classifier'],
        init_paths_random=variant['init_paths_random'],
        constant_start_state_init=variant['constant_start_state_init'],
        constant_start_state_while_training=variant['constant_start_state_while_training']
    )
    algorithm.to(ptu.device)
    algorithm.train()



if __name__ == "__main__":

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=22,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            # num_expl_steps_per_train_loop=1000,
            # min_num_steps_before_training=1000,
            max_path_length=1000,
            max_episode_length=50,
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
        rl_on_real=True,
        hardcode_classifier=False, 
        num_classifier_train_steps_per_iter=0,
        num_classifier_init_epoch=0,
        sparse=True,
        tolerance=1,#needed for rewards if sparse, and also for calculating accuracy\
        init_paths_random=True,
        constant_start_state_init=False, 
        constant_start_state_while_training=False
        

    )
    setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant) 

