import re
import numpy as np
from gym import utils
# from gym.envs.mujoco import mujoco_env
import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, is_real): #params):
        #params is a dict of parameters that need to change
        # if params == {}:
        #     print("Object with default params")
        # done = _edit_xml(params)
        # if done == False:
        #     raise("Error while editing xml")

        self.env_list = []
        if is_real:
            f = open('assets/half_cheetah.xml', 'r')
            model_xml = f.read()
            mujoco_env.MujocoEnv.__init__(self, model_xml, 5)
        else:
            # Value of dist hardcoded based on default value of .4 .1 .1 (set as the real value)
            # np.random.seed(13)
            friction_value = np.random.normal([2, 0.5, 0.5], [2, 0.5, 0.5])
            # TODO: std dev <1 will have low variance

            model_xml = self._edit_xml(friction_value)
            mujoco_env.MujocoEnv.__init__(self, model_xml, 5)

        utils.EzPickle.__init__(self)

    def _edit_xml(self, friction_value):

        replaced_str = 'friction="%1.1f %1.1f %1.1f"'%(friction_value[0], friction_value[1], friction_value[2])

        try:
            f = open('assets/half_cheetah.xml', 'r')
            data = f.read()
            model_xml = data.replace('friction=".4 .1 .1"', replaced_str)
            return model_xml
        except:
            raise('Error with editing xml')


    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class HalfCheetahEnvEnsemble(mujoco_env.MujocoEnv):
    def __init__(self, num_envs):
        self.env_list = []
        for _ in range(num_envs):
            self.env_list.append(HalfCheetahEnv(is_real = False))

        self.action_space = self.env_list[0].action_space



    def reset_model(self):
        self.env = np.random.choice(self.env_list)

        #For simulation
        self.frame_skip = self.env.frame_skip
        self.model = self.env.model
        self.sim = self.env.sim
        self.data = self.env.data
        self.viewer = self.env.viewer
        self._viewers = self.env._viewers

        self.metadata = self.env.metadata

        return self.env.reset_model()

    def step(self,action):
        return self.env.step(action)

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
