import collections

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack: int, stacking_key: str = "pixels",frame_skip=1):
        super().__init__(env)
        self._num_stack = num_stack
        self._stacking_key = stacking_key
        
        # assert stacking_key in self.observation_space.keys()
        pixel_obs_spaces = self.observation_space[stacking_key]

        self._env_dim = pixel_obs_spaces.shape[-1]

        low = np.repeat(pixel_obs_spaces.low[..., np.newaxis], num_stack, axis=-1)
        high = np.repeat(pixel_obs_spaces.high[..., np.newaxis], num_stack, axis=-1)
        new_pixel_obs_spaces = Box(low=low, high=high, dtype=pixel_obs_spaces.dtype)
        obs_dict=dict(self.observation_space)
        obs_dict.update({stacking_key:new_pixel_obs_spaces})
        self.observation_space = gym.spaces.Dict(obs_dict)
        self.frame=0
        self.skip=frame_skip
        # new_obs_space=self.observation_space

        # self.observation_space[stacking_key]({stacking_key:new_pixel_obs_spaces})

        # self.observation_space=new_pixel_obs_spaces
        # print(self.observation_space["obs"].sample().shape)
        self._frames = collections.deque(maxlen=num_stack)

    def reset(self,**kwargs):
        obs,info= self.env.reset(**kwargs)
        # breakpoint()
        self.frame=0
        # breakpoint()
        for i in range(self._num_stack):
            self._frames.append(obs[self._stacking_key])
        obs[self._stacking_key] = self.frames
        return obs,info

    @property
    def frames(self):
        return np.stack(self._frames, axis=-1)

    def step(self, action):
        obs, reward, done,terminate, info = self.env.step(action)
        if self.frame%self.skip==0:
            self._frames.append(obs[self._stacking_key])
        self.frame+=1
        obs[self._stacking_key] = self.frames
        return obs,reward,done,terminate,info
