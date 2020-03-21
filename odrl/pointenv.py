import numpy as np
import gym
import collections
import time
import tqdm
import gym.spaces
import networkx as nx
import matplotlib.pyplot as plt

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
                np.array([[0, 0, 1, 1, 1, 0, 0],
                          [0, 0, 1, 1, 1, 0, 0],
                          [0, 0, 1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0 ,0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]]),
}


def resize_walls(walls, factor):
    (height, width) = walls.shape
    row_indices = np.array([i for i in range(height) for _ in range(factor)])
    col_indices = np.array([i for i in range(width) for _ in range(factor)])
    walls = walls[row_indices]
    walls = walls[:, col_indices]
    assert walls.shape == (factor * height, factor * width)
    return walls


class PointEnv(gym.Env):

    def __init__(self, walls=None, sparse=False, tolerance=1, resize_factor=1,
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
        self.sparse=sparse
        self.tolerance=tolerance
        self.reset()

    def _sample_empty_state(self,constant_start ):
        if constant_start==True:
            state=np.array((self.observation_space.low[0]+1,self.observation_space.high[1]-1))
        else:
            candidate_states = np.where(self._walls == 0)
            num_candidate_states = len(candidate_states[0])
            state_index = np.random.choice(num_candidate_states)
            state = np.array([candidate_states[0][state_index],
                                                candidate_states[1][state_index]],
                                             dtype=np.float)
        state += np.random.uniform(size=2)
        assert not self._is_blocked(state)
        return state
        
    def reset(self, constant_start=True):
        self.state = self._sample_empty_state( constant_start= constant_start)
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

    def _discretize_states_parellel(self, states, resolution=1.0):
        indexes = np.floor(resolution * states).astype(np.int)
        indexes[indexes[:,0]==self._height]-=1
        indexes[indexes[:,1]==self._width]-=1
        return indexes

    def _is_blocked_parellel(self, states):
        if not states[-1].shape==self.observation_space.shape and all(states >= self.low[None]) and all(states <= self.high[None]):
            return np.ones((1, len(states)))
        indexes=  self._discretize_states_parellel(states)
        return np.array(self._walls[indexes[:,0],indexes[:,1] ] == 1)

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
        if self.sparse:
            rew = 20 if np.linalg.norm(self.state)<self.tolerance else 0
        else:
            rew=-1.0 * np.linalg.norm(self.state)
        return self.state.copy(), rew, done, {}

    @property
    def walls(self):
        return self._walls

    def get_env_shape(self):
        return  self._height, self._width, self.action_space, self.observation_space, self._walls

    def plot_env( env_name=None):
        if env_name:
            plt.title(env_name)
        height, width, action_space, observation_space, wall= env.get_env_shape()
        for (i, j) in zip(*np.where(wall)):
            x = np.array([i, i+1]) #/ float(width)
            y0 = np.array([j, j]) #/ float(height)
            y1 = np.array([j+1, j+1])# / float(height)
            plt.fill_between(x, y0, y1, color='grey')
        plt.xlim([0, width])
        plt.ylim([0, height])
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplots_adjust(wspace=0.1, hspace=0.2)
