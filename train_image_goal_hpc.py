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

# ── Test 1: Raw habitat_sim (should work — test_headless.py pattern) ──
print("\n=== Test 1: Raw habitat_sim.Simulator ===")
try:
    import habitat_sim
    import numpy as np

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/gibson/Cantwell.glb"
    backend_cfg.enable_physics = False

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [120, 160]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    print("  Raw Simulator created: OK")

    obs = sim.get_sensor_observations()
    print(f"  Observation shape: {obs['color_sensor'].shape}: OK")
    sim.close()
    print("  Test 1 PASSED")
except Exception as e:
    print(f"  Test 1 FAILED: {e}")
    sys.exit(1)






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
class TrainConfig:
    scene_path = "data/gibson/Cantwell.glb"
    scene_dataset_path = "data/gibson"
    control_frequency = 10
    device = "cuda"
    frame_skip = 3
    max_linear_velocity = 0.5
    max_angular_velocity = 1.5
    imu_noise_std = 0.0
    gpu_device_id = 0
    seed = 42
    debug_render = False
    headless = True
    goal_distance_scale = 3.0
    goal_max_distance = 10.0
    randomize_scenes = False
    replay_buffer_size = int(1e6)

device = "cuda"
habitat_cfg = HabitatNavConfig(
        scene_path=TrainConfig.scene_path,
        scene_dataset_path=TrainConfig.scene_dataset_path,
        control_frequency=TrainConfig.control_frequency,
        frame_skip=TrainConfig.frame_skip,
        max_linear_velocity=TrainConfig.max_linear_velocity,
        max_angular_velocity=TrainConfig.max_angular_velocity,
        imu_noise_std=TrainConfig.imu_noise_std,
        gpu_device_id=TrainConfig.gpu_device_id,
        seed=TrainConfig.seed,
        debug_render=TrainConfig.debug_render,
        headless=not TrainConfig.debug_render,
        goal_distance_scale=TrainConfig.goal_distance_scale,
        goal_max_distance=TrainConfig.goal_max_distance,
        randomize_scenes=TrainConfig.randomize_scenes,
    )
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
        TrainConfig.replay_buffer_size,
    )