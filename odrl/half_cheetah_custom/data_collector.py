from half_cheetah import HalfCheetahEnv
from half_cheetah_ensemble import HalfCheetahEnvEnsemble
import numpy as np
import pickle
from collections import defaultdict

SEED = 13

def env_init(is_real = True):
    if is_real:
        env = HalfCheetahEnv(is_real)
        env.seed(SEED)
    else:
        num_envs = 100
        env = HalfCheetahEnvEnsemble(num_envs)
        # env.seed(SEED)
    return env

def collect_data(episode_length, env_sim, env_real):
    data_sim = defaultdict(list)
    data_real = defaultdict(list)

    for episode_step in range(episode_length):
        #Simulated env
        if episode_step%1000 == 0:
            state_sim = env_sim.reset_model()
        action_sim = env_sim.action_space.sample()
        next_state_sim, reward, done, _ = env_sim.step(action_sim)
        data_sim['observations'].append(state_sim)
        data_sim['actions'].append(action_sim)
        data_sim['next_observations'].append(next_state_sim)
        state_sim = next_state_sim.copy()
        # env_sim.render()

        #Real env
        if episode_step%1000 == 0:
            state_real = env_real.reset_model()
        action_real = env_real.action_space.sample()
        next_state_real, reward, done, _ = env_real.step(action_real)
        data_real['observations'].append(state_real)
        data_real['actions'].append(action_real)
        data_real['next_observations'].append(next_state_real)
        state_real = next_state_real.copy()
        # env_real.render()

    return data_sim, data_real

if __name__ == '__main__':
    episode_length = 50000
    save_data = True
    env_sim = env_init(is_real=False)
    env_real = env_init(is_real=True)
    data_sim, data_real = collect_data(episode_length, env_sim, env_real)

    if save_data:
        with open(f'data/data_sim_{episode_length}.pkl', 'wb') as f:
            pickle.dump(data_sim, f)

        with open(f'data/data_real_{episode_length}.pkl', 'wb') as f:
            pickle.dump(data_real, f)

    ###NOTE: done is ALWAYS False for this env - may have useless data for long episodes

    print('Env details----------')
    print(f'State dim = {data_sim["observations"][0].shape}')
    print(f'Action dim = {data_sim["actions"][0].shape}')

    print(f'Epispde length = {episode_length}')

    print(f'Data sim shape = {len(data_sim["observations"])}')
    print(f'Data real shape = {len(data_real["observations"])}')


    # env_sim.reset_model()
    # while True:
    #     # env_sim.viewer_setup()
    #     env_sim.step(env_real.action_space.sample())
    #     env_sim.render()

    # env_sim.reset_model()
    # print(env_sim.env.action_space)
