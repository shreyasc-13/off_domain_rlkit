from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv

env = HalfCheetahBulletEnv(is_real=False)
print(env.reset())
