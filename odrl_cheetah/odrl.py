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

# from half_cheetah import HalfCheetahEnv
from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
from plot_scripts import plotting_evalreturns

def experiment(variant):

    sim_expl_env = NormalizedBoxEnv(HalfCheetahBulletEnv(is_real=False))
    sim_eval_env = NormalizedBoxEnv(HalfCheetahBulletEnv(is_real=False))
    real_expl_env = NormalizedBoxEnv(HalfCheetahBulletEnv(is_real=True))
    real_eval_env = NormalizedBoxEnv(HalfCheetahBulletEnv(is_real=True))

    obs_dim = real_expl_env.observation_space.low.size
    action_dim = real_eval_env.action_space.low.size

    #CHECK: WHY IS THIS NOT DEFINED?
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
    eval_path_collector = MdpPathCollector(
        real_eval_env,
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
    trainer_env= real_expl_env if variant['rl_on_real']==True else sim_expl_env
    trainer = SACTrainer(
        env=trainer_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        sim_exploration_env=sim_expl_env,
        real_exploration_env=real_expl_env,
        evaluation_env=real_eval_env,
        batch_size=variant['algorithm_kwargs']['batch_size'],
        max_path_length=variant['algorithm_kwargs']['max_path_length'],
        # max_episode_steps=variant['algorithm_kwargs']['max_episode_length'],
        num_epochs=variant['algorithm_kwargs']['num_epochs'],
        num_eval_steps_per_epoch=variant['algorithm_kwargs']['num_eval_steps_per_epoch'],
        num_trains_per_train_loop=variant['algorithm_kwargs']['num_trains_per_train_loop'],
        evaluation_data_collector=eval_path_collector,
        sim_data_collector= sim_path_collector,
        real_data_collector=real_path_collector,
        sim_replay_buffer=sim_replay_buffer,
        real_replay_buffer= real_replay_buffer,
        num_real_steps_at_init=     1000    if variant['rl_on_real'] else 10000,
        num_sim_steps_at_init=      0       if variant['rl_on_real'] else 10000,
        num_real_steps_per_epoch=   500     if variant['rl_on_real'] else 100 if variant['num_classifier_train_steps_per_iter'] else 0,
        num_sim_steps_per_epoch=    0       if variant['rl_on_real'] else 500,
        num_rl_train_steps_per_iter=1,

        rl_on_real=variant['rl_on_real'],
        modify_reward=False if variant['rl_on_real']==True else True,
        num_classifier_train_steps_per_iter=variant['num_classifier_train_steps_per_iter'],
        num_train_loops_per_epoch=1,
        num_classifier_init_epoch=variant['num_classifier_init_epoch'],
        classifier_batch_size=512,

        hardcode_classifier=variant['hardcode_classifier']
        # **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=200,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            # num_expl_steps_per_train_loop=1000,
            # min_num_steps_before_training=1000,
            max_path_length=1000,
            max_episode_length=500, #Shreyas Note: Needs to be tweaked
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
            real_reward_scaleup = 10. #Shreyas edit
        ),
        rl_on_real=False,
        num_classifier_train_steps_per_iter=0,
        num_classifier_init_epoch=50,

        hardcode_classifier=False
    )
    log_dir = setup_logger('name-of-experiment', variant=variant) #Returns absolute path
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    # print(log_dir)
    experiment(variant)

    #Add plot(s) to the log folders
    plotting_evalreturns(log_dir, rl_on_real)
