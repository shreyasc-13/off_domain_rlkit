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
from pointenv import * 


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
        num_real_steps_at_init=10000,
        num_sim_steps_at_init=10000,
        num_real_steps_per_epoch=100,
        num_sim_steps_per_epoch=100,
        num_rl_train_steps_per_iter=1,

        rl_on_real=False,
        modify_reward=True,
        num_classifier_train_steps_per_iter=1,
        num_train_loops_per_epoch=1

        
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

