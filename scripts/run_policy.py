from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger

#Shreyas edit: (to get the modified env)
from odrl_cheetah.pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv #To run from a director above
from odrl_cheetah import pybullet_envs
from rlkit.envs.wrappers import NormalizedBoxEnv
import gym
from gym import wrappers
import pickle

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file, map_location=torch.device('cpu'))
    policy = data['real_evaluation/policy']
    # env = data['evaluation/env'] #Commented out to load the right modifed pybullet env
    #Shreyas edit: ^ and v: To change for a different env
    # env = NormalizedBoxEnv(HalfCheetahBulletEnv(is_real=True))
    # env = NormalizedBoxEnv(gym.make('HalfCheetahHurdleBulletEnv-v0'))
    # env = gym.make('ReacherObstacleBulletEnv-v0')
    env = gym.make('AntBulletEnv-v0')
    # env = gym.make('PusherBulletEnv-v0')
    # env = wrappers.Monitor(env, '~/Destop/', force=True) #TODO: add render saving
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
        # if hasattr(env, "log_diagnostics"):
        #     env.log_diagnostics([path])
        # logger.dump_tabular()
    # paths = []
    # for _ in range(10):
    #     path = rollout(
    #         env,
    #         policy,
    #         max_path_length=args.H,
    #         render=False,
    #     )
    #     paths.append(path)
    #
    # with open('../temp_odrl.pkl', 'wb') as f:
    #     pickle.dump(paths, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
