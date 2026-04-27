from typing import Optional, Sequence
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

class StateDependentNoiseDistribution(nn.Module):
    """
    JAX/Flax implementation of generalized State Dependent Exploration (gSDE).
    Paper: https://arxiv.org/abs/2005.05719
    
    :param action_dim: Dimension of the action space
    :param full_std: Whether to use (n_features x n_actions) parameters for std
    :param use_expln: Use expln() instead of exp() for positive std
    :param squash_output: Whether to squash output using tanh
    :param learn_features: Whether to learn features for gSDE
    :param epsilon: Small value to avoid numerical instability
    """
    action_dim: int
    full_std: bool = True
    use_expln: bool = True
    squash_output: bool = True
    learn_features: bool = False
    epsilon: float = 1e-6
    latent_sde_dim: Optional[int] = None

    def setup(self):
        self.bijector = None
        if self.squash_output:
            self.bijector = distrax.Tanh()

    def get_std(self, log_std: jnp.ndarray) -> jnp.ndarray:
        """Get standard deviation from learned parameter."""
        if self.use_expln:
            # From gSDE paper: keep variance above zero and prevent fast growth
            below_threshold = jnp.exp(log_std) * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (jnp.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            std = jnp.exp(log_std)

        if self.full_std:
            return std
        
        # Reduce parameters by broadcasting
        return jnp.ones((self.latent_sde_dim, self.action_dim)) * std

    def sample_weights(self, key: random.PRNGKey, log_std: jnp.ndarray, batch_size: int = 1) -> tuple:
        """Sample weights for noise exploration matrix."""
        std = self.get_std(log_std)
        dist = distrax.Normal(jnp.zeros_like(std), std)
        
        # Single exploration matrix
        key1, key2 = random.split(key)
        exploration_mat = dist.sample(seed=key1)
        
        # Batch of exploration matrices
        exploration_matrices = dist.sample(seed=key2, sample_shape=(batch_size,))
        
        return exploration_mat, exploration_matrices

    def create_networks(self, latent_dim: int, log_std_init: float = -2.0, 
                       latent_sde_dim: Optional[int] = None) -> tuple:
        """Create networks for mean actions and exploration."""
        mean_actions_net = nn.Dense(self.action_dim)
        
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        
        # Initialize log_std parameter
        if self.full_std:
            log_std_shape = (self.latent_sde_dim, self.action_dim)
        else:
            log_std_shape = (self.latent_sde_dim, 1)
            
        log_std = jnp.ones(log_std_shape) * log_std_init
        
        return mean_actions_net, log_std

    def get_distribution(self, key: random.PRNGKey, mean_actions: jnp.ndarray, 
                        log_std: jnp.ndarray, latent_sde: jnp.ndarray) -> distrax.Distribution:
        """Create distribution given parameters."""
        latent_sde = latent_sde if self.learn_features else jax.lax.stop_gradient(latent_sde)
        
        std = self.get_std(log_std)
        variance = jnp.matmul(latent_sde**2, std**2)
        
        dist = distrax.Normal(mean_actions, jnp.sqrt(variance + self.epsilon))
        
        if self.bijector is not None:
            dist = distrax.Transformed(dist, self.bijector)
            
        return dist

    def get_noise(self, key: random.PRNGKey, latent_sde: jnp.ndarray, 
                 exploration_mat: jnp.ndarray, exploration_matrices: jnp.ndarray) -> jnp.ndarray:
        """Generate exploration noise."""
        latent_sde = latent_sde if self.learn_features else jax.lax.stop_gradient(latent_sde)
        
        # Single exploration matrix case
        if len(latent_sde) == 1 or len(latent_sde) != len(exploration_matrices):
            return jnp.matmul(latent_sde, exploration_mat)
            
        # Batch matrix multiplication
        latent_sde = jnp.expand_dims(latent_sde, axis=1)
        noise = jnp.matmul(latent_sde, exploration_matrices)
        return jnp.squeeze(noise, axis=1)

class GSDEPolicy(nn.Module):
    """Policy network with gSDE noise."""
    hidden_dims: Sequence[int]
    action_dim: int
    latent_sde_dim: Optional[int] = 128
    log_std_init: float = -2.0
    full_std: bool = True
    use_expln: bool = True
    squash_output: bool = True
    learn_features: bool = True
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, key: random.PRNGKey, 
                 training: bool = False) -> tuple:
        # Feature extractor
        x = nn.Sequential([
            nn.Dense(dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))
            for dim in self.hidden_dims
        ])(observations)
        
        # gSDE components
        gsde = StateDependentNoiseDistribution(
            action_dim=self.action_dim,
            full_std=self.full_std,
            use_expln=self.use_expln,
            squash_output=self.squash_output,
            learn_features=self.learn_features
        )
        
        mean_net, log_std = gsde.create_networks(
            self.hidden_dims[-1], 
            self.log_std_init,
            self.latent_sde_dim
        )
        
        mean_actions = mean_net(x)
        # exploration_mat, exploration_matrices = gsde.sample_weights(key, log_std)
        return gsde.get_distribution(key, mean_actions, log_std, x)