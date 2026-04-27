import collections
import random
from typing import Optional, Union, Iterable, Callable

import gymnasium as gym
import gymnasium.spaces
import jax
import numpy as np

from jaxrl2.data.dataset import Dataset, DatasetDict, _sample
from flax.core import frozen_dict


def _init_replay_dict(
        obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _init_replay_dict_with_list(
        obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return []
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict_with_list(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
        dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()
    
def _overwrite_recursively(
        dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _overwrite_recursively(dataset_dict[k], data_dict[k], insert_index)
    elif isinstance(dataset_dict, list):
        dataset_dict=np.array(dataset_dict) 
        dataset_dict[insert_index]=dataset_dict
        # return dataset_dict
    else:
        breakpoint()
        raise TypeError()


def _add_recursively(
        dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, list):
        dataset_dict.append(data_dict) 
        return dataset_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in data_dict.keys():
            dataset_dict[k] = _add_recursively(dataset_dict[k], data_dict[k],insert_index)
        return dataset_dict
    
    else:
        print(dataset_dict)
        raise TypeError()

def _convert_to_np_array_recursively(
        dataset_dict: DatasetDict
):
    if isinstance(dataset_dict, list):
        return np.array(dataset_dict)
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            dataset_dict[k] = _convert_to_np_array_recursively(dataset_dict[k])
        return dataset_dict
    else:
        return dataset_dict



# def _sample_list(
#     dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray
# ) -> DatasetDict:
#     if isinstance(dataset_dict,list):
#         return np.array(dataset_dict)[indx]
#     elif isinstance(dataset_dict, dict):
#         batch = {}
#         for k, v in dataset_dict.items():
#             batch[k] = _sample_list(v, indx)
#     else:
#         raise TypeError("Unsupported type.")
#     return batch

class ReplayBuffer(Dataset):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            capacity: int,
            next_observation_space: Optional[gym.Space] = None,
            relabel_fn: Optional[Callable[[DatasetDict], DatasetDict]] = None,
            slack:int=int(3e4)
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = self._size
        self._start_size=None
        self._slack=slack

        self._relabel_fn = relabel_fn

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)
        if self._start_size==None:
            self._insert_index = (self._insert_index + 1) % self._capacity
        else:
            self._insert_index = np.clip((self._insert_index+1)% self._capacity,self._start_size,self._capacity)
        self._size = min(self._size + 1, self._capacity)
    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
    # def get_expert_iterator(self, queue_size: int = 2, sample_args: dict = {}):
    #     # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
    #     # queue_size = 2 should be ok for one GPU.

    #     queue = collections.deque()
    #     indx = self.np_random.randint(len(self), size=sample_args.get("batch_size", 1))
    #     def enqueue(n):
    #         for _ in range(n):
    #             data = self.sample(**sample_args)
    #             queue.append(jax.device_put(data))

    #     enqueue(queue_size)
    #     while queue:
    #         yield queue.popleft()
    #         enqueue(1)
    
    def get_sequential_iterator(self, queue_size: int = 2, sample_args: dict = None):
        if sample_args is None:
            sample_args = {}
        
        # Shuffle all indices once at the beginning
        all_indices = list(range(len(self)))
        random.shuffle(all_indices)

        batch_size = sample_args.get("batch_size", 1)
        m = 0
        queue = collections.deque()
        def enqueue(n,m):
            for _ in range(n):
                start = m * batch_size
                end = min((m + 1) * batch_size, len(self))
                batch_indices = all_indices[start:end]
                # Sample data using the shuffled indices
                data = self.sequential_sample(batch_size=batch_size, indx=np.array(batch_indices))
                queue.append(jax.device_put(data))
                m += 1
            return m
        m = enqueue(queue_size,m)
        while m * batch_size < len(all_indices):
            # print(m*batch_size,batch_size)
            yield queue.popleft()
            m=enqueue(1,m)

    def freeze(self):
        # if self._start_size==None:
        if self._size==self._capacity:
            self._start_size=self._size-int(1e5)
        else:
            self._start_size=self._size


    def sequential_sample(self,
                        batch_size: int,
                        keys: Optional[Iterable[str]] = None,
                        indx: Optional[np.ndarray] = None,
                        k=0,
                        ) -> frozen_dict.FrozenDict:
        if indx is None:
            # If no indices are provided, generate a random batch of indices
            buffer_size = len(self)
            indx = np.random.choice(buffer_size, size=batch_size, replace=False)
        
        # Sample the data using the provided indices
        samples = super().sample(batch_size, keys, indx)
        return samples
    def sample_future_observation(self, indices: np.ndarray, sample_futures: str = "uniform"):
        if sample_futures == 'uniform':
            ep_begin = indices - _sample(self.dataset_dict['observations']['index'], indices)
            ep_end = ep_begin + _sample(self.dataset_dict['observations']['ep_len'], indices)
            future_indices = np.random.randint(ep_begin, ep_end, indices.shape)
        elif sample_futures == 'exponential':
            ep_len = _sample(self.dataset_dict['observations']['ep_len'], indices)
            indices_in_ep = _sample(self.dataset_dict['observations']['index'], indices)
            ep_begin = indices - indices_in_ep
            ep_end = ep_begin + ep_len
            future_offsets = np.random.exponential(100.0, indices.shape).astype(np.int32) + 1
            offsets_from_ep_begin = (future_offsets + indices - ep_begin) % ep_len
            future_indices = (ep_begin + offsets_from_ep_begin) % self._size
        elif sample_futures == 'exponential_no_wrap':
            future_offsets = np.random.exponential(100.0, indices.shape).astype(np.int32)
            future_indices = (indices + future_offsets + 1) % self._size
        else:
            raise ValueError(f'Unknown sample_futures: {sample_futures}')
        return _sample(self.dataset_dict['observations'], future_indices)

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None,
               sample_futures=None,
               sample_futures_key=None,
               relabel: bool = False) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, 'integers'):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)
        samples = super().sample(batch_size, keys, indx)
        if sample_futures and not sample_futures_key is None:
            samples = frozen_dict.unfreeze(samples)
            samples['future_observations'] = self.sample_future_observation(indx, sample_futures_key)
            samples = frozen_dict.freeze(samples)

        if relabel and self._relabel_fn is not None:
            samples = frozen_dict.unfreeze(samples)
            samples = self._relabel_fn(samples)
            samples = frozen_dict.freeze(samples)
        return samples


class VariableCapacityBuffer(Dataset):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            capacity: int=0,
            next_observation_space: Optional[gym.Space] = None,
            relabel_fn: Optional[Callable[[DatasetDict], DatasetDict]] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space
        
        observation_data = _init_replay_dict_with_list(observation_space, capacity)
        next_observation_data = _init_replay_dict_with_list(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=[],
            rewards=[],
            masks=[],
            dones=[],
        )
        # print(dataset_dict)
        # breakpoint()
        super().__init__(dataset_dict)
        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._relabel_fn = relabel_fn

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        # breakpoint()
        self.dataset_dict=_add_recursively(self.dataset_dict, data_dict, self._insert_index)
        self._size += 1
        # breakpoint()
    def overwrite(self, data_dict: DatasetDict):
        if self._capacity < self._size:
            self._capacity = max(10,int(self._size/3))
            print(self._capacity)
            # breakpoint()
        # breakpoint()
        _overwrite_recursively(self.dataset_dict, data_dict, self._insert_index)
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        

    def get_sequential_iterator(self, queue_size: int = 2, sample_args: dict = None):
        if sample_args is None:
            sample_args = {}
        m = 0
        batch_size = sample_args.get("batch_size", 1)
        while m * batch_size < len(self):
            data = self.sequential_sample(**{**sample_args, "k": m})  
            yield jax.device_put(data)
            m += 1 
    
              
    def sequential_sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None,
               k=0,
               ) -> frozen_dict.FrozenDict:
        buffer_size=len(self)
        if indx is None:
            start=min(k*batch_size,buffer_size)
            end=min(buffer_size,(k+1)*batch_size)
            indx=np.array(list(range(start,end)))
        samples = super().sample(batch_size, keys, indx)
        return samples


    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.
        queue = collections.deque()
        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
    # def get_iterator(self, queue_size: int = 2, sample_args: dict = None):
    #     if sample_args is None:
    #         sample_args = {}
    #     m = 0
    #     batch_size = sample_args.get("batch_size", 1)
    #     while m * batch_size < len(self):
    #         data = self.sample(**sample_args)  
    #         yield jax.device_put(data)
    #         m = (m+1) % len(self)
    def optimize(self):
       self.dataset_dict= _convert_to_np_array_recursively(self.dataset_dict)
    #    print("Done Optimizing")

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None,
               sample_futures=None,
               sample_futures_key=None,
               relabel: bool = False) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, 'integers'):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)
        # print(indx)
        # breakpoint()
        # samples = super().sample(batch_size, keys, indx)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] =self.dataset_dict[k][indx]
        return frozen_dict.freeze(batch)

