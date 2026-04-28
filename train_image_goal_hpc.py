"""
Diagnostic test for Habitat headless HPC operation.

Tests both raw habitat_sim (like test_headless.py) and habitat-lab Env
(like train_habitat_her.py) to narrow down segfault source.

Usage on HPC:
    unset DISPLAY
    export QT_QPA_PLATFORM=offscreen
    python test_hpc_headless.py
"""
import os
import sys

from habitat_wrappers import VideoRecorder


# Apply headless settings first, before any habitat imports
os.environ.pop("DISPLAY", None)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

print("=== Environment ===")
print(f"DISPLAY={os.environ.get('DISPLAY', '<unset>')}")
print(f"QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', '<unset>')}")
print(f"EGL_VISIBLE_DEVICES={os.environ.get('EGL_VISIBLE_DEVICES', '<unset>')}")
print(f"MAGNUM_GPU_VALIDATION={os.environ.get('MAGNUM_GPU_VALIDATION', '<unset>')}")
from habitat_env import HabitatNavEnv
from configs.habitat_config import HabitatNavConfig
from jaxrl2.agents import DrQLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

from racer_imu_env import StackingWrapper
from wrappers import (
    Logger,
    MobileNetFeatureWrapper,
    MobileNetV3Encoder,
    GoalImageWrapper,
    load_checkpoint,
    save_checkpoint,
)
from habitat_wrappers import HabitatRewardWrapper
device = "cuda"
cfg = HabitatNavConfig(headless=True)
env = HabitatNavEnv(cfg, render_mode="rgb_array")
env = StackingWrapper(env, num_stack=3, image_format="rgb")

# # Shared MobileNetV3 encoder for current obs and goal
shared_encoder = MobileNetV3Encoder(
    device=device,
    num_blocks=13,
    input_size=84,
)
env = MobileNetFeatureWrapper(env, encoder=shared_encoder)
env = GoalImageWrapper(env, encoder=shared_encoder)
goal_threshold = 2.0
env = HabitatRewardWrapper(env, goal_threshold=goal_threshold)
env =VideoRecorder(env, video_dir="test_videos", record_episodes=True)
# env = reward_wrapper
replay_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        int(1e6),
    )

print("  HabitatNavEnv created: OK")

obs, info = env.reset()
print(f"  Reset OK, obs keys: {list(obs.keys())}")

obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.1]))
print(f"  Step OK, reward: {reward:.4f}")

env.close()
print("  Test 2 PASSED")
