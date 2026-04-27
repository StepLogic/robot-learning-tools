
from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING, Any, SupportsFloat

import gymnasium as gym
from gymnasium import logger
from gymnasium.core import ActType, ObsType

class RecordEpisodeStatistics(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.
    A vector version of the wrapper exists, :class:`gymnasium.wrappers.vector.RecordEpisodeStatistics`.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
     * time_queue: The time length of the last ``deque_size``-many episodes
     * return_queue: The cumulative rewards of the last ``deque_size``-many episodes
     * length_queue: The lengths of the last ``deque_size``-many episodes

    Change logs:
     * v0.15.4 - Initially added
     * v1.0.0 - Removed vector environment support (see :class:`gymnasium.wrappers.vector.RecordEpisodeStatistics`) and add attribute ``time_queue``
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        buffer_length: int = 100,
        stats_key: str = "episode",
    ):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            buffer_length: The size of the buffers :attr:`return_queue`, :attr:`length_queue` and :attr:`time_queue`
            stats_key: The info key for the episode statistics
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        self._stats_key = stats_key

        self.episode_count = 0
        self.episode_start_time: float = -1
        self.episode_returns: float = 0.0
        self.episode_lengths: int = 0

        self.time_queue: deque[float] = deque(maxlen=buffer_length)
        self.return_queue: deque[float] = deque(maxlen=buffer_length)
        self.length_queue: deque[int] = deque(maxlen=buffer_length)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, recording the episode statistics."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        self.episode_returns += reward
        self.episode_lengths += 1

        if terminated or truncated:
            # assert self._stats_key not in info

            episode_time_length = round(
                time.perf_counter() - self.episode_start_time, 6
            )
            # print(info) 
            info[self._stats_key] = {
                "r": self.episode_returns,
                "l": self.episode_lengths,
                "t": episode_time_length,
            }
            # print(info)
            self.time_queue.append(episode_time_length)
            self.return_queue.append(self.episode_returns)
            self.length_queue.append(self.episode_lengths)

            self.episode_count += 1
            self.episode_start_time = time.perf_counter()

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment using seed and options and resets the episode rewards and lengths."""
        obs, info = super().reset(seed=seed, options=options)
        episode_time_length=round(
                time.perf_counter() - self.episode_start_time, 6
            )
        info[self._stats_key] = {
            "r": self.episode_returns,
            "l": self.episode_lengths,
            "t":episode_time_length ,
        }

        self.episode_start_time = time.perf_counter()
        self.episode_returns = 0.0
        self.episode_lengths = 0
        return obs, info
