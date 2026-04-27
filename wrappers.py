#!/usr/bin/env python
"""
DrQ Training with JAX, MobileNetV3 encoder, and Temporal Modeling
Adapted for Donkey Car simulator with Real Robot Environment Architecture
- MobileNetV3 visual feature extraction (PyTorch)
- Frame stacking with feature extraction
- Action+IMU history tracking
- Enhanced reward structure for forward motion and collision avoidance
"""
import random
from collections import deque
import os
from datetime import datetime
import time
import torchvision
import tqdm
from typing import Dict, Any
import uuid
import pickle

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random as jax_random
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium import spaces

# PyTorch imports for MobileNetV3
import torch
import torch.nn as torch_nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import cv2
import pickle

# ============================================================================
# IMU Data Structure
# ============================================================================

# IMU fields used throughout this codebase:
#   accel  : np.ndarray, shape (3,) — linear acceleration [ax, ay, az] in m/s²
#   gyro   : np.ndarray, shape (3,) — angular velocity   [gx, gy, gz] in rad/s
#
# Packed into a single IMU vector of shape (IMU_DIM,) = (6,):
#   imu_vec = [ax, ay, az, gx, gy, gz]
#
# StackingWrapper stacks `num_stack` such vectors into shape (num_stack * IMU_DIM,)
# and exposes them under the 'imu' key of the observation dict.
#
# ACTION / IMU STACKING NOTE:
#   Both actions and IMU are stacked with a history that is 2x the pixel
#   frame stack depth.  This gives the policy a longer proprioceptive
#   memory than the visual context window — useful because recovery from
#   a collision takes more steps than a single visual transition.

IMU_DIM = 6  # 3-axis accelerometer + 3-axis gyroscope

def extract_imu_vector(info: dict) -> np.ndarray:
    """
    Extract a flat IMU vector from an environment info dict.

    Expected keys (all optional, zero-filled if absent):
        info['accel'] : array-like, shape (3,)  — [ax, ay, az] in m/s²
        info['gyro']  : array-like, shape (3,)  — [gx, gy, gz] in rad/s

    Returns:
        np.ndarray, shape (IMU_DIM,) = (6,), dtype float32
    """
    accel = np.asarray(info.get("accel", [0.0, 0.0, 0.0]), dtype=np.float32)
    gyro  = np.asarray(info.get("gyro",  [0.0, 0.0, 0.0]), dtype=np.float32)

    # Guard against unexpected shapes from the simulator
    if accel.shape != (3,):
        accel = np.zeros(3, dtype=np.float32)
    if gyro.shape != (3,):
        gyro = np.zeros(3, dtype=np.float32)
    # breakpoint()
    return np.concatenate([[accel[0],accel[2],accel[1]], [gyro[0],gyro[2],gyro[1]]])  # (6,)


# ============================================================================
# MobileNetV3 Feature Extractor
# ============================================================================

import torchvision.transforms as T

import torch
import torchvision.transforms as T
import random

class RandomRedLightFilter:
    def __init__(self, p=0.5, intensity_range=(0.1, 0.3)):
        """
        Args:
            p: probability of applying the filter
            intensity_range: (min, max) intensity of red tint
        """
        self.p = p
        self.intensity_range = intensity_range
    
    def __call__(self, img):
        if random.random() < self.p:
            import numpy as np
            img_array = np.array(img).astype(np.float32)
            
            intensity = random.uniform(*self.intensity_range)
            
            # Add red tint (increase red channel, slightly reduce others)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + intensity), 0, 255)
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1 - intensity * 0.3), 0, 255)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - intensity * 0.3), 0, 255)
            
            from PIL import Image
            img = Image.fromarray(img_array.astype(np.uint8))
        return img

sim2real_a = lambda input_size : T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(
        size=(input_size, input_size),
        scale=(0.8, 1.0),
        ratio=(3/4, 4/3),
    ),
    T.GaussianBlur(5, 1.0),
    RandomRedLightFilter(p=0.3, intensity_range=(0.15, 0.35)),
    T.ColorJitter(0.1, 0.1, 0.1, 0.02),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================================
# Logger and Checkpoint functions
# ============================================================================

class Logger:
    """TensorBoard logger with comprehensive metrics tracking."""
    
    def __init__(self, log_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.train_metrics = {}
        self.eval_metrics = {}
        self.episode_metrics = {}
        
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        print(f"[Logger] Logging to {self.log_dir}")

    def log_training(self, metrics: Dict[str, Any], step: int, prefix=""):
        for k, v in metrics.items():
            self.writer.add_scalar(f"training{prefix}/{k}", np.array(v), step)
            self.train_metrics[f"{k}{prefix}"] = np.array(v)

    def log_eval(self, metrics: Dict[str, Any], step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(f"evaluation/{k}", np.array(v), step)
            self.eval_metrics[k] = np.array(v)

    def log_episode(self, metrics: Dict[str, Any], step: int):
        for k, v in metrics.items():
            try:
                self.writer.add_scalar(f"episode/{k}", np.array(v), step)
                self.episode_metrics[k] = np.array(v)
            except:
                pass
            
        if "return" in metrics:
            self.episode_returns.append(metrics["return"])
            self.writer.add_scalar(
                "episode/return_mean_100", 
                np.mean(self.episode_returns), 
                step
            )
            
        if "length" in metrics:
            self.episode_lengths.append(metrics["length"])
            self.writer.add_scalar(
                "episode/length_mean_100",
                np.mean(self.episode_lengths),
                step
            )

        if "success" in metrics:
            if not hasattr(self, 'episode_successes'):
                self.episode_successes = deque(maxlen=100)
            self.episode_successes.append(metrics["success"])
            self.writer.add_scalar(
                "episode/success_rate_100",
                np.mean(self.episode_successes),
                step
            )

        if "distance" in metrics:
            if not hasattr(self, 'episode_distances'):
                self.episode_distances = deque(maxlen=100)
            self.episode_distances.append(metrics["distance"])
            self.writer.add_scalar(
                "episode/distance_mean_100",
                np.mean(self.episode_distances),
                step
            )

    def print_status(self, step: int, total_steps: int, extra_stats: dict = None):
        print(f"\n{'='*60}")
        print(f"Step {step:,}/{total_steps:,} ({100*step/total_steps:.1f}%)")

        if self.train_metrics:
            print("\n[Training Metrics]")
            for k, v in sorted(self.train_metrics.items()):
                print(f"  {k:30s}: {v:>10.4f}")

        if self.episode_metrics:
            print("\n[Episode Metrics]")
            for k, v in sorted(self.episode_metrics.items()):
                print(f"  {k:30s}: {v:>10.4f}")

        if self.episode_returns:
            print(f"\n[Running Stats (last 100)]")
            print(f"  Mean Return:  {np.mean(self.episode_returns):>10.2f}")
            print(f"  Mean Length:  {np.mean(self.episode_lengths):>10.1f}")
            if hasattr(self, 'episode_successes') and self.episode_successes:
                print(f"  Success Rate: {np.mean(self.episode_successes):>10.1%}")
            if hasattr(self, 'episode_distances') and self.episode_distances:
                print(f"  Mean Distance:{np.mean(self.episode_distances):>10.2f}m")

        if extra_stats:
            print(f"\n[Additional Stats]")
            for k, v in extra_stats.items():
                print(f"  {k:30s}: {v}")

        print(f"{'='*60}\n")
    
    def close(self):
        self.writer.close()


class MobileNetV3Encoder(torch_nn.Module):
    """MobileNetV3-Small encoder for image features."""

    def __init__(self, device: str = "cuda", num_blocks: int = 4, input_size: int = 84):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder = torch_nn.Sequential(*list(model.features[:num_blocks]))
        self.encoder.to(self.device)
        self.encoder.eval()

        self.transform = sim2real_a(input_size)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size).to(self.device)
            output = self.encoder(dummy)
            self.output_dim = output.numel()

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode image to latent features."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.encoder(img_tensor)
        
        return features.squeeze(0).cpu().numpy().flatten()

    def encode_batch(self, images: np.ndarray) -> np.ndarray:
        if len(images.shape) == 3:
            return self.encode(images)
        
        batch_size = images.shape[0]
        features_list = []
        
        for i in range(batch_size):
            features = self.encode(images[i])
            features_list.append(features)
        
        return np.stack(features_list, axis=0)


# ============================================================================
# MobileNetV3 Feature Wrapper
# ============================================================================

class MobileNetFeatureWrapper(gym.Wrapper):
    """
    Replaces raw stacked-pixel observations with MobileNetV3 features.

    Input observation dict keys  : 'pixels', 'actions', 'imu'
    Output observation dict keys : 'pixels' (features), 'actions', 'imu'

    The 'imu' key is passed through unchanged so downstream policy networks
    can fuse visual features with proprioceptive IMU data.
    """

    def __init__(
        self, 
        env: gym.Env,
        device: str = "cuda",
        encoder=None,
        num_blocks: int = 4,
        input_size: int = 84
    ):
        super().__init__(env)

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = MobileNetV3Encoder(
                device=device,
                num_blocks=num_blocks,
                input_size=input_size
            )
        self.device = device
        
        self.feature_dim = self.encoder.output_dim
        
        original_pixels_shape = env.observation_space['pixels'].shape  # (H, W, C*num_stack)
        self.height = original_pixels_shape[0]
        self.width = original_pixels_shape[1]
        self.num_channels_stacked = original_pixels_shape[2]
        self.num_frames = self.num_channels_stacked // 3
        
        feature_stack_shape = (self.num_frames * self.feature_dim,)
        
        # Build new observation space, preserving 'imu' if it exists
        obs_spaces = {
            'pixels': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=feature_stack_shape,
                dtype=np.float32
            ),
            'actions': env.observation_space['actions'],
        }
        if 'imu' in env.observation_space.spaces:
            obs_spaces['imu'] = env.observation_space['imu']

        self.observation_space = spaces.Dict(obs_spaces)
    
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        stacked_pixels = observation['pixels']  # (H, W, C*num_frames)
        
        # Split and encode each stacked frame
        features_list = []
        for i in range(self.num_frames):
            frame = stacked_pixels[:, :, i * 3 : (i + 1) * 3]
            features_list.append(self.encoder.encode(frame))
        
        stacked_features = np.concatenate(features_list).astype(np.float32)
        
        result = {
            'pixels':  stacked_features,
            'actions': observation['actions'],
        }
        # Pass IMU through unchanged
        if 'imu' in observation:
            result['imu'] = observation['imu'].astype(np.float32)
        
        return result
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# ============================================================================
# Goal Image Wrapper
# ============================================================================

class GoalImageWrapper(gym.Wrapper):
    """
    Encodes a goal image using a Siamese MobileNetV3 encoder and adds
    'goal_features' to the observation dict.

    The encoder is shared with MobileNetFeatureWrapper (same weights) so
    that current-observation features and goal features live in the same
    latent space.

    Input observation dict keys  : 'pixels', 'actions', 'imu'
    Output observation dict keys : 'pixels', 'actions', 'imu', 'goal_features'

    The goal image is sourced from info['goal_image'] (provided by
    HabitatNavEnv).  If the key is missing, a zero-vector is used.
    """

    def __init__(self, env, encoder=None, num_blocks=4, input_size=84,
                 device="cuda"):
        super().__init__(env)
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = MobileNetV3Encoder(
                num_blocks=num_blocks, input_size=input_size, device=device
            )
        self.device = device
        self.goal_feature_dim = self.encoder.output_dim

        # Extend observation space with goal_features
        existing = env.observation_space.spaces
        new_spaces = dict(existing)
        new_spaces["goal_features"] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.goal_feature_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(new_spaces)

    # ── Private ────────────────────────────────────────────────────────────────

    def _encode_goal(self, info: dict) -> np.ndarray:
        """Encode goal image from info dict, or return zeros if absent."""
        goal_image = info.get("goal_image", None)
        if goal_image is not None and goal_image.size > 0:
            # Goal image is RGB (from HabitatNavEnv)
            return self.encoder.encode(goal_image).astype(np.float32)
        return np.zeros(self.goal_feature_dim, dtype=np.float32)

    # ── Gym API ────────────────────────────────────────────────────────────────

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        goal_features = self._encode_goal(info)
        obs = dict(obs)
        obs["goal_features"] = goal_features
        obs["imu"] = obs["imu"]
        if obs["imu"][5] < 0.0:
                obs["goal_features"] = np.zeros_like(obs["goal_features"])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        goal_features = self._encode_goal(info)
        obs = dict(obs)
        obs["goal_features"] = goal_features
        return obs, info


class DoomStackingWrapper(gym.Wrapper):
    """
    Stacks RGB frames, actions, and IMU readings into fixed-length histories.

    The action and IMU histories are kept at 2x the pixel frame stack depth.
    This gives the policy a longer proprioceptive memory than its visual
    context window — important for recovery from collisions which unfold over
    more steps than a single frame transition can capture.

    Observation dict shapes (with num_stack=3 as an example):
        'pixels'  : (120, 160, 3 * num_stack)            e.g. (120,160,9)
        'actions' : (2 * num_stack * action_dim,)         e.g. (12,)  for 2-dim actions
        'imu'     : (2 * num_stack * IMU_DIM,)            e.g. (36,)
    """

    def __init__(self, env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack      = num_stack
        self.num_stack_prop = num_stack * 2   # 2x deeper history for actions + IMU

        self.action_history = deque(maxlen=self.num_stack_prop)
        self.rgb_history    = deque(maxlen=self.num_stack)
        self.imu_history    = deque(maxlen=self.num_stack_prop)

        self.action_dim = env.action_space.shape[0]

        action_stack_shape = (self.num_stack_prop * self.action_dim,)
        imu_stack_shape    = (self.num_stack_prop * IMU_DIM,)

        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(
                low=0, high=255,
                shape=(120, 160, 3 * num_stack),
                dtype=np.uint8
            ),
            "actions": spaces.Box(
                low=np.tile(env.action_space.low,  self.num_stack_prop),
                high=np.tile(env.action_space.high, self.num_stack_prop),
                shape=action_stack_shape,
                dtype=np.float32
            ),
            "imu": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=imu_stack_shape,
                dtype=np.float32
            ),
        })

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_stacked_actions(self) -> np.ndarray:
        actions_list = list(self.action_history)
        while len(actions_list) < self.num_stack_prop:
            actions_list.insert(0, np.zeros(self.action_dim, dtype=np.float32))
        return np.concatenate(actions_list).astype(np.float32)

    def _get_stacked_rgb(self) -> np.ndarray:
        return np.concatenate(list(self.rgb_history), axis=-1).astype(np.uint8)

    def _get_stacked_imu(self) -> np.ndarray:
        """Concatenate IMU history into a flat vector (2 * num_stack * IMU_DIM,)."""
        imu_list = list(self.imu_history)
        while len(imu_list) < self.num_stack_prop:
            imu_list.insert(0, np.zeros(IMU_DIM, dtype=np.float32))
        return np.concatenate(imu_list).astype(np.float32)

    def _build_obs(self) -> Dict[str, np.ndarray]:
        return {
            'pixels':  self._get_stacked_rgb(),
            'actions': self._get_stacked_actions(),
            'imu':     self._get_stacked_imu(),
        }

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Handle dict obs (ViZDoomEnv) vs raw image (DonkeyCar)
        if isinstance(obs, dict):
            obs = obs.get("image", obs.get("pixels", obs))
        obs = cv2.cvtColor(obs.transpose(1,2,0), cv2.COLOR_BGR2RGB)

        self.action_history.append(action.astype(np.float32))
        self.rgb_history.append(obs)
        self.imu_history.append(extract_imu_vector(info))

        return self._build_obs(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict):
            pix = obs.get("image", obs.get("pixels", obs))
        # breakpoint()
        pix = cv2.cvtColor(pix.transpose(1,2,0), cv2.COLOR_BGR2RGB)

        self.action_history.clear()
        self.rgb_history.clear()
        self.imu_history.clear()

        zero_imu = np.zeros(IMU_DIM, dtype=np.float32)
        # Pixel history: num_stack frames
        for _ in range(self.num_stack):
            self.rgb_history.append(pix)

        for _ in range(self.num_stack_prop):
            self.action_history.append(np.zeros(self.action_dim, dtype=np.float32))
            self.imu_history.append(zero_imu)

        return self._build_obs(), info


# ============================================================================
# Stacking Wrapper (RGB frames + actions + IMU)
# ============================================================================

class StackingWrapper(gym.Wrapper):
    """
    Stacks RGB frames, actions, and IMU readings into fixed-length histories.

    The action and IMU histories are kept at 2x the pixel frame stack depth.
    This gives the policy a longer proprioceptive memory than its visual
    context window — important for recovery from collisions which unfold over
    more steps than a single frame transition can capture.

    Observation dict shapes (with num_stack=3 as an example):
        'pixels'  : (120, 160, 3 * num_stack)            e.g. (120,160,9)
        'actions' : (2 * num_stack * action_dim,)         e.g. (12,)  for 2-dim actions
        'imu'     : (2 * num_stack * IMU_DIM,)            e.g. (36,)
    """

    def __init__(self, env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack      = num_stack
        self.num_stack_prop = num_stack * 2   # 2x deeper history for actions + IMU

        self.action_history = deque(maxlen=self.num_stack_prop)
        self.rgb_history    = deque(maxlen=self.num_stack)
        self.imu_history    = deque(maxlen=self.num_stack_prop)

        self.action_dim = env.action_space.shape[0]

        action_stack_shape = (self.num_stack_prop * self.action_dim,)
        imu_stack_shape    = (self.num_stack_prop * IMU_DIM,)

        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(
                low=0, high=255,
                shape=(120, 160, 3 * num_stack),
                dtype=np.uint8
            ),
            "actions": spaces.Box(
                low=np.tile(env.action_space.low,  self.num_stack_prop),
                high=np.tile(env.action_space.high, self.num_stack_prop),
                shape=action_stack_shape,
                dtype=np.float32
            ),
            "imu": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=imu_stack_shape,
                dtype=np.float32
            ),
        })

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_stacked_actions(self) -> np.ndarray:
        actions_list = list(self.action_history)
        while len(actions_list) < self.num_stack_prop:
            actions_list.insert(0, np.zeros(self.action_dim, dtype=np.float32))
        return np.concatenate(actions_list).astype(np.float32)

    def _get_stacked_rgb(self) -> np.ndarray:
        return np.concatenate(list(self.rgb_history), axis=-1).astype(np.uint8)

    def _get_stacked_imu(self) -> np.ndarray:
        """Concatenate IMU history into a flat vector (2 * num_stack * IMU_DIM,)."""
        imu_list = list(self.imu_history)
        while len(imu_list) < self.num_stack_prop:
            imu_list.insert(0, np.zeros(IMU_DIM, dtype=np.float32))
        return np.concatenate(imu_list).astype(np.float32)

    def _build_obs(self) -> Dict[str, np.ndarray]:
        return {
            'pixels':  self._get_stacked_rgb(),
            'actions': self._get_stacked_actions(),
            'imu':     self._get_stacked_imu(),
        }

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Handle dict obs (ViZDoomEnv) vs raw image (DonkeyCar)
        if isinstance(obs, dict):
            obs = obs.get("image", obs.get("pixels", obs))
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

        self.action_history.append(action.astype(np.float32))
        self.rgb_history.append(obs)
        self.imu_history.append(extract_imu_vector(info))

        return self._build_obs(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict):
            obs = obs.get("image", obs.get("pixels", obs))
        # breakpoint()
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

        self.action_history.clear()
        self.rgb_history.clear()
        self.imu_history.clear()

        zero_imu = np.zeros(IMU_DIM, dtype=np.float32)
        # Pixel history: num_stack frames
        for _ in range(self.num_stack):
            self.rgb_history.append(obs)
        # Action + IMU history: 2x num_stack steps
        for _ in range(self.num_stack_prop):
            self.action_history.append(np.zeros(self.action_dim, dtype=np.float32))
            self.imu_history.append(zero_imu)

        return self._build_obs(), info


# ============================================================================
# Environment Wrappers
# ============================================================================

class EnvCompatibility(gym.Env):
    """Wrapper to make old gym.Env environments compatible with gymnasium.Env API."""
    
    def __init__(self, env):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._reward_range = None
        self._metadata = getattr(env, "metadata", {"render_modes": []})

    def step(self, action):
        # print(action)
        obs, reward, done, info = self._env.step(action)
        truncated  = info.get("TimeLimit.truncated", False)
        terminated = done and not truncated
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            try:
                self._env.seed(seed)
            except AttributeError:
                pass
        res = self._env.reset()
        # Handle both old-gym (returns obs only) and gymnasium (returns (obs, info))
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs, info = res, {}
        return obs, info


class RewardWrapper(gym.Wrapper):
    """
    Reward wrapper for forward motion and collision avoidance.

    Forward velocity is estimated by integrating IMU accelerometer readings,
    with dt measured as real elapsed time between step() calls.

    - velocity > 0.01 → reward = +0.1, increment move_count, reset stop_count
    - velocity <= 0.01 → reward = -1.0, increment stop_count
    - stop_count > 20  → truncated
    - hit detected     → terminated, reward -= 10
    - move_count > 50  → terminated, reward += 10
    """

    def __init__(self, env):
        super().__init__(env)
        self.velocity   = 0.0
        self.stop_count = 0
        self.move_count = 0
        self._last_time = None  # wall-clock time of previous step() call

    def step(self, action):
        now = time.monotonic()

        self._last_time = now

        obs, _, _,_, info = self.env.step(action)
        terminated = False
        truncated = False
        reward = 0.0

        vel= info.get("forward_vel",0.0)
        if vel<0.01:
            self.stop_count += 1

        if self.stop_count > 20:
            truncated = True
        # print("Self.strop",self.stop_count)
        # reward += 0.0
        reward += 10*info.get("forward_vel",0.0)
        reward = -1.0
        

        if info.get("hit", "none") != "none":
            terminated = True
            reward -= (100.0+10*info.get("forward_vel",0.0))

        # if self.stop_count > 20:
        #     truncated = True

        # if self.move_count > 50:
        #     reward += 10.0
        #     terminated = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info       = self.env.reset(**kwargs)
        self.velocity   = 0.0
        self.stop_count = 0
        self.move_count = 0
        self._last_time = None  # reset so first step doesn't use stale time
        return obs, info
    


# GOAL_PICKLE_PATH = "/home/kojogyaase/Projects/Research/recovery-from-failure/goal_loc_images.pkl"

# class GoalRelObservationWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
        
#         with open(GOAL_PICKLE_PATH, "rb") as f:
#             self.goal_data = pickle.load(f)

#         if not isinstance(self.goal_data, (list, tuple)) or len(self.goal_data) == 0:
#             raise ValueError("goal_loc_images.pkl must contain a non-empty list of goals")

#         first = self.goal_data[0]
#         if "position" not in first:
#             raise KeyError("Each goal must contain a 'position' field")
#         self.stop_count =0
#         pos = np.asarray(first["position"], dtype=np.float32)
#         low = np.full_like(pos, -np.inf, dtype=np.float32)
#         high = np.full_like(pos, np.inf, dtype=np.float32)
#         goal_rel_space = spaces.Box(low=low, high=high, dtype=np.float32)
#         self.distance_to_goal=None
#         if isinstance(self.observation_space, spaces.Dict):
#             new_spaces = dict(self.observation_space.spaces)
#             new_spaces["goal_rel"] = goal_rel_space
#             self.observation_space = spaces.Dict(new_spaces)
#         else:
#             self.observation_space = spaces.Dict(
#                 {
#                     "obs": self.observation_space,
#                     "goal_rel": goal_rel_space,
#                 }
#             )

#         self.current_goal = None

#     def _sample_goal(self):
#         idx = np.random.randint(len(self.goal_data))
#         # print(len(self.goal_data))
#         g = self.goal_data[idx]
#         return np.asarray(g["position"], dtype=np.float32)

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.current_goal = self._sample_goal()
#         self.stop_count =0
#         curr_pos = np.asarray(info.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
#         goal_rel = self.current_goal - curr_pos
#         self.distance_to_goal = np.linalg.norm(curr_pos)
#         if isinstance(obs, dict):
#             obs = dict(obs)
#             obs["goal_rel"] = goal_rel
#         else:
#             obs = {
#                 "obs": obs,
#                 "goal_rel": goal_rel,
#             }
#         # print(obs.keys())
#         return obs, info

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         terminated, truncated = False, False
#         if self.current_goal is None:
#             self.current_goal = self._sample_goal()
    
#         curr_pos = np.asarray(info.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
#         goal_rel = self.current_goal - curr_pos
#         reward   = -1.0


#         if np.linalg.norm(goal_rel) < 1:
#             terminated = True
#             reward += 100
#         vel= info.get("forward_vel",0.0)
#         reward += vel
#         if vel<0.01:
#             self.stop_count += 1
#         if self.stop_count > 20:
#             truncated = True
        
#         if info.get("hit", "none") != "none":
#             truncated = True
#             reward -= 1.0

#         if isinstance(obs, dict):
#             obs = dict(obs)
#             obs["goal_rel"] = goal_rel
#         else:
#             obs = {
#                 "obs": obs,
#                 "goal_rel": goal_rel,
#             }
#         # if np.linalg.norm(goal_rel) < self.distance_to_goal:
#         self.distance_to_goal = np.linalg.norm(curr_pos)
#         if terminated or truncated:
#             print("Distance To Goal", np.linalg.norm(goal_rel))
#         return obs, reward, terminated, truncated, info

class FrameSkipWrapper(gym.Wrapper):
    """Wrapper to perform action repetition (frame skipping)."""
    
    def __init__(self, env, skip=2):
        super().__init__(env)
        self.num_steps_per_agent_step = skip + 1

    def step(self, action):
        total_reward = 0.0
        terminated   = False
        truncated    = False
        info         = {}
        
        for i in range(self.num_steps_per_agent_step):
            obs, reward, terminated_step, truncated_step, info = self.env.step(action)
            
            total_reward += reward
            terminated    = terminated or terminated_step
            truncated     = truncated  or truncated_step

            if terminated or truncated:
                break
        
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ============================================================================
# Checkpoint utilities
# ============================================================================

def load_checkpoint(agent, checkpoint_path):
    """Load agent checkpoint."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"[Checkpoint] Loading from {checkpoint_path}")
    
    state_dict = {
        "actor":         agent._actor,
        "critic":        agent._critic,
        "target_critic": agent._target_critic_params,
        "temp":          agent._temp,
        "rng":           agent._rng,
    }
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=state_dict,
    )
    
    if restored is None:
        print("[ERROR] Checkpoint loading failed!")
        return agent
    
    agent._actor               = restored['actor']
    agent._critic              = restored['critic']
    agent._target_critic_params = restored['target_critic']
    agent._temp                = restored['temp']
    agent._rng                 = restored['rng']
    
    return agent


def save_checkpoint(agent, replay_buffer, path, step):
    """Save agent checkpoint and replay buffer."""
    try:
        os.makedirs(path, exist_ok=True)
        
        state_dict = {
            "actor":         agent._actor,
            "critic":        agent._critic,
            "target_critic": agent._target_critic_params,
            "temp":          agent._temp,
            "rng":           agent._rng,
        }
        checkpoints.save_checkpoint(
            ckpt_dir=os.path.abspath(path),
            target=state_dict,
            step=step,
            overwrite=True,
            keep=3,
        )
        
        print(f"[Checkpoint] Saved at step {step:,} to {path}")
        print(f"[Checkpoint] Replay buffer saved with {replay_buffer._size} transitions")
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")