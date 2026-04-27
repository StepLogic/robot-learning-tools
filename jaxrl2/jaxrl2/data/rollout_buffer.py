import collections
import math
from typing import Optional, Tuple, Union, Iterable, Callable

import gymnasium as gym
import gymnasium.spaces
import jax
import numpy as np
import jax.numpy as jnp
from jaxrl2.data.dataset import Dataset, DatasetDict, _sample
from flax.core import frozen_dict
import scipy

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    # breakpoint()
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)

# import numpy as np
# from sklearn.metrics import explained_variance_score

# def explained_variance(y_true, y_pred):
#     """
#     Compute the explained variance between true values (y_true) and predicted values (y_pred).

#     Args:
#         y_true (np.ndarray): Array of true values (e.g., actual returns).
#         y_pred (np.ndarray): Array of predicted values (e.g., value function predictions).

#     Returns:
#         float: Explained variance.

#     Interpretation:
#             ev=0  =>  might as well have predicted zero
#             ev=1  =>  perfect prediction
#             ev<0  =>  worse than just predicting zero

#     """


#     # Ensure inputs are numpy arrays
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)

#     # Compute the variance of the residuals
#     residual_variance = np.var(y_true - y_pred)

#     # Compute the variance of the true values
#     true_variance = np.var(y_true)

#     # Handle the case where true_variance is zero (to avoid division by zero)
#     if true_variance == 0:
#         return 0.0

#     # Compute explained variance
#     return 1 - (residual_variance / true_variance)


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


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

def _insert_recursively(
        dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        for k in data_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()

class RolloutBuffer(Dataset):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            capacity: int,
            relabel_fn: Optional[Callable[[DatasetDict], DatasetDict]] = None,
    ):
        self.observation_data = _init_replay_dict(observation_space, capacity)
        dataset_dict = dict(
            observations=self.observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
            values=np.empty((capacity,), dtype=np.float32),
            logps=np.empty((capacity,), dtype=np.float32),
            advantages=np.empty((capacity,), dtype=np.float32),
            episode_starts=np.empty((capacity,), dtype=np.float32),
            returns=np.empty((capacity,), dtype=np.float32),
        )
        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._path_start_idx = 0
        self._relabel_fn = relabel_fn
        self.explained_variance=0.0

    def flush(self):
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        # print(self._insert_index,self._capacity)
        assert self._insert_index<self._capacity
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)
        self._insert_index = self._insert_index + 1
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = None):
        if sample_args is None:
            sample_args = {}
        m = 0
        batch_size = sample_args.get("batch_size", 1)
        while m * batch_size < len(self):
            data = self.sample(**{**sample_args, "k": m})  
            yield jax.device_put(data)
            m += 1 
    
            
                
            

    def sample(self,
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
    # def can_sample(self):
    #     return se>0
    
    def compute_advantage(
            self,
            last_value=0.0,
            done=False,
            discount: float = 0.99,
            gae_lambda: float = 0.95
    ):
        values = self.dataset_dict["values"]
        rewards = self.dataset_dict["rewards"]
        advantages = np.zeros_like(rewards)
        dones=self.dataset_dict["dones"]
        # returns = np.zeros_like(rewards)

        # if len(rewards) > 1:
        #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        last_gae_lam = 0
        for step in reversed(range(len(self))):
            if step == len(self) - 1:
                next_non_terminal = 1.0 - float(done)
                next_values = last_value
            else:
                next_non_terminal = 1.0 - float(dones[step+1])
                next_values = values[step + 1]
            delta = rewards[step] + discount * next_values * next_non_terminal - values[step]
            last_gae_lam = delta + discount * gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
            # print(delta,last_gae_lam,advantages[step],rewards[step])
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        returns = advantages + values
        # if len(returns) > 1:
        #     returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.dataset_dict["advantages"] = advantages
        self.dataset_dict["returns"] = returns
        self.explained_variance = explained_variance(values,returns)
        self._insert_index=0

    def compute_advantage_v2(
            self,
            last_value=0.0,
            discount: float = 0.99,
            gae_lambda: float = 0.95
    ):
        # path_slice = slice(self._path_start_idx, self._insert_index)
        # rews = np.concatenate([self.dataset_dict["rewards"][path_slice], [last_value]])
        # vals = np.concatenate([self.dataset_dict["values"][path_slice], [last_value]])
        # # mask = np.concatenate([self.dataset_dict["masks"][path_slice], [1.0]])
        
        # deltas = rews[:-1] + discount * vals[1:] - vals[:-1]
        
        # advantages = np.zeros_like(deltas)
        # lastgae = 0
        # for t in reversed(range(len(deltas))):
        #     lastgae = deltas[t] + discount * gae_lambda * lastgae 
        #     advantages[t] = lastgae
            
        # self.dataset_dict["advantages"][:self._insert_index] = advantages
        # if np.any(np.isnan(vals[:-1])):
        #     print("Values containe Nan")
        # if np.any(np.isnan(advantages)):
        #     print("advantages containe Nan")
        # returns= advantages+vals[:-1]
        # self.dataset_dict["returns"][:self._insert_index] =np.add(advantages,vals[:-1])
        # if np.any(np.isnan(self.dataset_dict["returns"])):
        #     print("Returns containe Nan")
        # # print(self.dataset_dict["returns"],advantages)
        # # breakpoint()
        if self._path_start_idx>self._insert_index:
            self._path_start_idx=0
        path_slice = slice(self._path_start_idx, self._insert_index)
        rews = np.append(self.dataset_dict["rewards"][path_slice], last_value)
        vals = np.append(self.dataset_dict["values"][path_slice], last_value)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + discount * vals[1:] - vals[:-1]
        advantages =  discount_cumsum(deltas, discount * gae_lambda)
        # the next line computes rewards-to-go, to be targets for the value function
        returns =  discount_cumsum(rews, discount)[:-1]
        # self.path_start_idx = self.ptr
        if np.any(np.isnan(returns)):
            print("Returns containe Nan")
        if np.any(np.isnan(advantages)):
            print("advantages containe Nan")
        if np.any(np.isnan(vals[:-1])):
            print("Values containe Nan")
        self.dataset_dict["advantages"][path_slice] = advantages
        self.dataset_dict["returns"][path_slice] = returns
        self._path_start_idx=self._insert_index
        self.flush()
        
