import collections
import copy
from typing import Iterable, Optional

import gymnasium
import jax
import numpy as np
from flax.core import frozen_dict
from gymnasium.spaces import Box

from jaxrl2.data.dataset import DatasetDict, _sample
from jaxrl2.data.replay_buffer import ReplayBuffer
from collections import defaultdict


           
class HindsightReplayBuffer(ReplayBuffer):
    def __init__(
        self, 
        observation_space: gymnasium.Space, 
        action_space: gymnasium.Space, 
        capacity: int,
        relabel_obs_fn=None
    ):
        super().__init__(observation_space, action_space, capacity)
        self._episodes = defaultdict(list)  # Removed lambda as it's not needed
        self._episode_idx = 0
        self.n_sampled_goals = 3
        self.relabel_obs_fn = relabel_obs_fn

    def insert(self, data_dict: DatasetDict):
        if data_dict["dones"]:
            self._episode_idx += 1
            
            # Sample additional goals from current episode
            if len(self._episodes[self._episode_idx]) > 0:  # Check if episode has data
                virtual_indices = np.random.choice(
                    self._episodes[self._episode_idx], 
                    size=min(2, len(self._episodes[self._episode_idx])), 
                    replace=False
                )
                
                for v_idx in virtual_indices:
                    virtual_dict = {}
                    for k, v in self.dataset_dict.items():
                        if isinstance(v, dict):
                            virtual_dict[k] = {}
                            for sub_k, sub_v in v.items():
                                virtual_dict[k][sub_k] = sub_v[v_idx].copy()
                        else:
                            virtual_dict[k] = v[v_idx].copy()
                    
                    if callable(self.relabel_obs_fn):
                        max_len = max(i for i in self._episodes[self._episode_idx])
                        virtual_dict = self.relabel_obs_fn(virtual_dict, max_len=max_len)
                        super().insert(virtual_dict)
            
            # Reset episode if buffer is full
            if (self._insert_index + 1) % self._capacity == 0:
                self._episode_idx = 0
                self._episodes.clear()  # Clear all episodes
                self._episodes[self._episode_idx] = []

        # Insert the original data
        super().insert(data_dict)
        self._episodes[self._episode_idx].append(self._insert_index)