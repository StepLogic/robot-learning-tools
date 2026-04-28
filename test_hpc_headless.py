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

# ── Test 2: habitat-lab HabitatEnv (mimics train_habitat_her.py path) ──
print("\n=== Test 2: habitat-lab HabitatEnv (HabitatNavEnv) ===")
try:
    from habitat_env import HabitatNavEnv
    from configs.habitat_config import HabitatNavConfig
    # from train_habitat_her import HabitatRewardWrapper

    from racer_imu_env import StackingWrapper
    from wrappers import (
        Logger,
        MobileNetFeatureWrapper,
        MobileNetV3Encoder,
        GoalImageWrapper,
        load_checkpoint,
        save_checkpoint,
    )
    # device = "cuda"
    cfg = HabitatNavConfig(headless=True)
    env = HabitatNavEnv(cfg, render_mode="rgb_array")
    # env = StackingWrapper(env, num_stack=3, image_format="rgb")

    # # Shared MobileNetV3 encoder for current obs and goal
    # shared_encoder = MobileNetV3Encoder(
    #     device=device,
    #     num_blocks=13,
    #     input_size=84,
    # )
    # env = MobileNetFeatureWrapper(env, encoder=shared_encoder)
    # env = GoalImageWrapper(env, encoder=shared_encoder)
    # goal_threshold = 2.0
    # reward_wrapper = HabitatRewardWrapper(env, goal_threshold=goal_threshold)
    # env = reward_wrapper

    print("  HabitatNavEnv created: OK")

    obs, info = env.reset()
    print(f"  Reset OK, obs keys: {list(obs.keys())}, image shape: {obs['image'].shape}")

    obs, reward, terminated, truncated, info = env.step([0.0, 0.1])
    print(f"  Step OK, reward: {reward:.4f}")

    env.close()
    print("  Test 2 PASSED")
except Exception as e:
    print(f"  Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== All tests passed ===")
