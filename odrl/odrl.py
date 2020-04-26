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
import argparse
import  lunar_lander2 , lunar_lander2_sim, lunar_lander2_real
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import  ReduceLROnPlateau
from collections import OrderedDict
import math
from torch import cuda, device

def experiment(variant):
    sim_env_name='Small'
    real_env_name = 'Maze7x7' 
    tolerance=variant["tolerance"]
    sparse=variant["sparse"]
    resize_factor=variant['resize_factor']
    # max_env_steps=variant['']
    # sim_expl_env  = PointEnv(sim_env_name,  sparse, tolerance, resize_factor)
    # sim_eval_env  = PointEnv(sim_env_name,  sparse, tolerance, resize_factor)
    # real_expl_env = PointEnv(real_env_name, sparse, tolerance, resize_factor)
    # real_eval_env = PointEnv(real_env_name, sparse, tolerance, resize_factor)
    sim_expl_env=lunar_lander2_sim.LunarLanderContinuous ( action_mean=variant["sim_action_mean"], action_std_dev=variant["sim_action_std_dev"])
    sim_eval_env=lunar_lander2_sim.LunarLanderContinuous ( action_mean=variant["sim_action_mean"], action_std_dev=variant["sim_action_std_dev"])
    real_expl_env=lunar_lander2_real.LunarLanderContinuous( action_mean=0, action_std_dev=0.3)#=variant["sim_action_std_dev"])
    real_eval_env=lunar_lander2_real.LunarLanderContinuous( action_mean=0, action_std_dev=0.3)#=variant["sim_action_std_dev"])
    # import pdb;pdb.set_trace()
    obs_dim = sim_expl_env.observation_space.low.size
    action_dim = sim_expl_env.action_space.low.size
    max_episode_steps=variant['algorithm_kwargs']['max_episode_length']
    M = variant['layer_size']
    num_trains_per_train_loop=variant['algorithm_kwargs']['num_trains_per_train_loop']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M, M],
    )
    # eval_policy = MakeDeterministic(policy)
    eval_real_path_collector = MdpPathCollector(
        real_eval_env,
        policy,
        max_episode_steps    )
    eval_sim_path_collector = MdpPathCollector(
        sim_eval_env,
        policy,
        max_episode_steps    )
    real_path_collector = MdpPathCollector(
        real_expl_env,
        policy,
        max_episode_steps    )
    sim_path_collector = MdpPathCollector(
        sim_expl_env,
        policy,
        max_episode_steps    )
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
        seed= variant['seed'], 
        SA=bool(variant['num_SA']), 
        max_classifier_reward=variant['max_classifier_reward'], 
        # train_on_sim_with_modified_rewards= variant['train_on_sim_with_modified_rewards'], 
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        sim_exploration_env=sim_expl_env,
        real_exploration_env=real_expl_env,
        evaluation_sim_env=sim_eval_env,
        evaluation_real_env=real_eval_env,
        batch_size=variant['algorithm_kwargs']['batch_size'],
        max_path_length=variant['algorithm_kwargs']['max_episode_length'],
        # max_episode_steps=variant['algorithm_kwargs']['max_episode_length'],
        num_epochs=variant['algorithm_kwargs']['num_epochs'],
        num_eval_steps_per_epoch=variant['algorithm_kwargs']['num_eval_steps_per_epoch'],
        num_trains_per_train_loop=variant['algorithm_kwargs']['num_trains_per_train_loop'],
        evaluation_real_data_collector=eval_real_path_collector,
        evaluation_sim_data_collector=eval_sim_path_collector,
        sim_data_collector= sim_path_collector,
        real_data_collector=real_path_collector,
        sim_replay_buffer=sim_replay_buffer,
        real_replay_buffer= real_replay_buffer,

        num_real_steps_at_init=     max(5120,variant['init_episode']*max_episode_steps), #if variant['rl_on_real'] else int(max_episode_steps* num_trains_per_train_loop/1000) # if not variant['hardcode_classifier'] else 0, #if variant['rl_on_real']  alse 10*max_episode_steps,
        num_sim_steps_at_init=      0  if variant['rl_on_real'] else   max(5120, int(variant['init_episode']* max_episode_steps)),#* num_trains_per_train_loop/1000)), #variant['init_episode']*max_episode_steps, #int(max_episode_steps* num_trains_per_train_loop/1000)# max(variant['init_episode'], 5)*max_episode_steps ,
        num_real_steps_per_epoch=   variant['real_episodes_per_epoch']*max_episode_steps,
        num_sim_steps_per_epoch=    0  if variant['rl_on_real'] else  variant['sim_episodes_per_epoch']*max_episode_steps,

        rl_on_real=variant['rl_on_real'],
        # modify_reward=False if variant['rl_on_real']==True else True,
        num_classifier_train_steps_per_iter=variant['num_classifier_train_steps_per_iter'],
        num_train_loops_per_epoch=1,
        num_classifier_init_epoch=variant['num_classifier_init_epoch'],
        classifier_batch_size=512,#min(variant['init_episode']*max_episode_steps, 5120)/10,
        tolerance=tolerance,
        plot_episodes_period=1 if variant['algorithm_kwargs']['num_epochs']<5 else int(variant['algorithm_kwargs']['num_epochs']/5),
        hardcode_classifier=variant['hardcode_classifier'],
        init_paths_random=variant['init_paths_random'],
        constant_start_state_init=variant['constant_start_state_init'],
        constant_start_state_while_training=variant['constant_start_state_while_training'],
        should_plot=variant['should_plot'], 
        seed= variant['seed'], 
        num_SA=variant['num_SA'] ,
        num_SAS=variant['num_SAS'],
        iid_at_init=variant['iid_at_init'], 
        render=variant['render'], 
        lamda=variant['lamda'], 
        fixed_lamda=variant['lamda']


    )
    algorithm.to(ptu.device)
    algorithm.train()



if __name__ == "__main__":


    parser= argparse.ArgumentParser()
    parser.add_argument("-s","--seed", type=int, default=1 )
    parser.add_argument("-r","--resize_factor", type=int, default=1 )
    parser.add_argument("-n", "--name", type=str, default="unnamed")
    parser.add_argument("-i", "--init_episodes", type=int, default=1)
    parser.add_argument("--sim_episodes_per_epoch", type= int, default=5)
    parser.add_argument("--real_episodes_per_epoch", type=int, default=1)
    parser.add_argument("-c", "--nctspi", type=int, default=5, help="num_of_classifier_train_steps_per_iter")
    parser.add_argument("-m", "--mean", type=float, default=.6)
    parser.add_argument("-d", "--std", type=float, default=1)
    parser.add_argument("-t", "--num_trains_per_train_loop", type=int, default=2000)
    parser.add_argument("-l", "--rl_on_real", type=int, default=0)
    # parser.add_argument("-u", "--unmodified_reward", type=int, default=0)
    parser.add_argument("-a", "--lamda", type=float, default=1)
    parser.add_argument("-f", "--fixed_lamda", type=bool, default=True)
    # parser.add_argument("-a", "--unmodified_reward", type=int, default=0)

    args=parser.parse_args()
    # print(args.seed)
    # print(args.resize_factor)
    # args = sys.argv[1:]
    # print(args)
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=64,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=100000,
            num_eval_steps_per_epoch=10000*args.resize_factor,
            num_trains_per_train_loop=args.num_trains_per_train_loop,
            # num_expl_steps_per_train_loop=1000,
            # min_num   _steps_before_training=1000,
            # max_path_length=1000,
            max_episode_length=1000*args.resize_factor,
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
        rl_on_real=args.rl_on_real,
        batch_rl=False,
        hardcode_classifier=False , 
        init_episode=args.init_episodes, 
        num_classifier_init_epoch=2, 
        num_classifier_train_steps_per_iter=args.nctspi,
        sparse=True,
        tolerance=1*args.resize_factor,#needed for rewards if sparse, and also for calculating accuracy
        init_paths_random=True,
        constant_start_state_init=False, 
        constant_start_state_while_training=False, 
        should_plot=False,
        resize_factor=args.resize_factor,
        seed=args.seed, 
        num_SAS=1,
        num_SA=0, 
        iid_at_init=False, 
        render=False, 
        max_classifier_reward=100.0, 
        sim_action_mean=args.mean,
        sim_action_std_dev= args.std, 
        # train_on_sim_with_modified_rewards=1-args.unmodified_reward, 
        sim_episodes_per_epoch=args.sim_episodes_per_epoch,
        real_episodes_per_epoch=args.real_episodes_per_epoch,
        fixed_lamda= args.fixed_lamda, 
        lamda=args.lamda


    )
    setup_logger('name-of-experiment', variant=variant, args=args)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant) 
#TODO: auto initialise variables from env name and experiment type 


# layer_size 32 64 128 356

# tanh relu
# 

#expt types:
# RL on real train with new data, 
# Rl on real only train with init data 
# train RL on sim without modification only from init data, test rl on real 
# train RL on sim without modification with new data collected every epoch, test rl on real 
# train RL on sim with modification only from init data, test rl on real 
# train RL on sim with modification with new data, test rl on real 

