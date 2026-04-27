import os
import time
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import gymnasium
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from ml_collections import ConfigDict

from jaxrl2.agents import DrQLearner,IQLLearner,PixelIQLLearner,PixelBCLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers.frame_stack import FrameStack
from vision_rl.rllib_integration.carla_env import CarlaEnv


class RLTrainer:
    def __init__(
        self,
        config: ConfigDict,
        env:Any,
        seed: int = 42,
        save_dir: str = "./logs",
        eval_episodes: int = 10,
        log_interval: int = 1000,
        eval_interval: int = 5000,
        max_steps: int = int(5e5),
        start_training: int = int(1e3),
        save_buffer: bool = False,
        save_video: bool = False,
    ):
        """Initialize the RL Trainer.

        Args:
            config: Configuration dictionary
            env_name: Name of the environment
            seed: Random seed
            save_dir: Directory to save logs and models
            eval_episodes: Number of episodes for evaluation
            log_interval: Interval for logging training metrics
            eval_interval: Interval for evaluation
            max_steps: Maximum number of training steps
            start_training: Steps before starting training
            save_buffer: Whether to save replay buffer
            save_video: Whether to save evaluation videos
        """
        self.config = config
        self.env_name = env_name
        self.seed = seed
        self.save_dir = save_dir
        self.eval_episodes = eval_episodes
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.max_steps = max_steps
        self.start_training = start_training
        self.save_buffer = save_buffer
        self.save_video = save_video

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize environment
        self._setup_environment()
        
        # Initialize agent and replay buffer
        self._setup_agent()
        
        # Initialize logger
        self.logger = self._setup_logger()

    def _setup_environment(self) -> None:
        """Setup and wrap the environment."""
        self.env = CarlaEnv(self.config.env_config)
        self.env = FrameStack(env=self.env, num_stack=1)
        self.env = gymnasium.wrappers.RecordEpisodeStatistics(self.env)

    def _setup_agent(self) -> None:
        """Initialize the agent and replay buffer."""
        self.agent = DrQLearner(
            self.seed,
            self.env.observation_space.sample(),
            self.env.action_space.sample(),
            **self.config
        )
        
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space,
            self.env.action_space,
            self.config.buffer_size
        )
        self.replay_buffer.seed(self.seed)
        self.replay_buffer_iterator = self.replay_buffer.get_iterator(
            sample_args={"batch_size": self.config.train_batch_size}
        )

    def _setup_logger(self) -> 'Logger':
        """Initialize the logger."""
        return Logger(self.save_dir)

    def save_checkpoint(self, step: int) -> None:
        """Save agent checkpoint and replay buffer."""
        # Save agent
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.agent, f)

        # Save replay buffer if requested
        if self.save_buffer:
            buffer_dir = os.path.join(self.save_dir, 'buffers')
            os.makedirs(buffer_dir, exist_ok=True)
            buffer_path = os.path.join(buffer_dir, f'buffer_{step}.pkl')
            with open(buffer_path, 'wb') as f:
                pickle.dump(self.replay_buffer, f)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load agent checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            self.agent = pickle.load(f)

    def train_step(
        self, 
        observation: np.ndarray, 
        training: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute single training step."""
        if training and self.total_steps < self.start_training:
            action = self.env.action_space.sample()
            update_info = {}
        else:
            action = self.agent.sample_actions(observation)
            if training:
                batch = next(self.replay_buffer_iterator)
                update_info = self.agent.update(batch)
            else:
                update_info = {}
                
        return action, update_info

    def train(self) -> None:
        """Main training loop."""
        self.total_steps = 0
        observation, info, done = *self.env.reset(), False
        training_start_time = time.time()

        for i in tqdm.tqdm(
            range(1, self.max_steps + 1),
            smoothing=0.1,
            desc="Training"
        ):
            self.total_steps = i
            
            # Execute training step
            action, update_info = self.train_step(observation, training=True)
            next_observation, reward, done, truncated, info = self.env.step(action)

            # Handle episode termination
            mask = 1.0 if not done or not truncated or "TimeLimit.truncated" in info else 0.0

            # Store transition
            self.replay_buffer.insert(
                dict(
                    observations=observation,
                    actions=action,
                    rewards=reward,
                    masks=mask,
                    dones=done,
                    next_observations=next_observation,
                )
            )

            observation = next_observation

            # Handle episode completion
            if done or truncated:
                observation, info, done = *self.env.reset(), False
                if "episode" in info:
                    episode_info = {
                        "return": info["episode"]["r"],
                        "length": info["episode"]["l"],
                        "time": info["episode"]["t"]
                    }
                    self.logger.log_episode(episode_info, i)

            # Log training metrics
            if i >= self.start_training and i % self.log_interval == 0:
                self.logger.log_training(update_info, i)
                self.logger.print_status(i, self.max_steps)

            # Periodic evaluation and saving
            if i % self.eval_interval == 0:
                self.save_checkpoint(i)
                eval_info = evaluate(
                    self.agent, 
                    self.env, 
                    num_episodes=self.eval_episodes
                )
                self.logger.log_eval(eval_info, i)
                self.logger.print_status(i, self.max_steps)

        # Print final training statistics
        training_duration = time.time() - training_start_time
        print(f"\nTraining completed in {training_duration/3600:.2f} hours")
        print(f"Logs saved to: {self.logger.log_dir}")

    def deploy(self, checkpoint_path: str, num_episodes: int = 10) -> Dict[str, float]:
        """Deploy trained agent for evaluation."""
        self.load_checkpoint(checkpoint_path)
        eval_info = evaluate(self.agent, self.env, num_episodes=num_episodes)
        return eval_info


class Logger:
    """Logger class for tracking metrics during training."""
    
    def __init__(self, log_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.train_metrics = {}
        self.eval_metrics = {}
        self.episode_metrics = {}
        
        print(f"\nLogging to: {self.log_dir}\n")
    
    def log_training(self, metrics: Dict[str, Any], step: int) -> None:
        """Log training metrics."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"training/{k}", np.array(v), step)
            self.train_metrics[k] = np.array(v)
    
    def log_eval(self, metrics: Dict[str, Any], step: int) -> None:
        """Log evaluation metrics."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"evaluation/{k}", np.array(v), step)
            self.eval_metrics[k] = np.array(v)
    
    def log_episode(self, metrics: Dict[str, Any], step: int) -> None:
        """Log episode metrics."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"episode/{k}", np.array(v), step)
            self.episode_metrics[k] = np.array(v)
    
    def print_status(self, step: int, total_steps: int) -> None:
        """Print current training status."""
        print("\n" + "="*80)
        print(f"Step: {step}/{total_steps} ({step/total_steps*100:.1f}%)")
        
        if self.train_metrics:
            print("\nTraining Metrics:")
            for k, v in self.train_metrics.items():
                print(f"  {k:<20} {np.array(v):>10.4f}")
        
        if self.episode_metrics:
            print("\nLatest Episode:")
            for k, v in self.episode_metrics.items():
                print(f"  {k:<20} {np.array(v):>10.4f}")
        
        if self.eval_metrics:
            print("\nLatest Evaluation:")
            for k, v in self.eval_metrics.items():
                print(f"  {k:<20} {np.array(v):>10.4f}")
        
        print("="*80 + "\n")