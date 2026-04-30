"""
Diagnostic test for Habitat headless HPC operation.

Tests both raw habitat_sim (like test_headless.py) and habitat-lab Env
(like train_habitat_her.py) to narrow down segfault source.

Usage on HPC:
    unset DISPLAY
    export QT_QPA_PLATFORM=offscreen
    python test_hpc_headless.py
"""
from collections import deque
import datetime
import os
import sys

import numpy as np
import tqdm
from configs import drq_default

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

POLICY_FOLDER = "robot_policy"
def _find_latest_checkpoint(folder):
    """Find the latest checkpoint in a folder, return (path, step) or (None, 0)."""
    if not os.path.isdir(folder):
        return None, 0
    ckpts = []
    for d in os.listdir(folder):
        if d.startswith("checkpoint_"):
            try:
                step = int(d.split("_")[-1])
                ckpts.append((os.path.abspath(os.path.join(folder, d)), step))
            except ValueError:
                continue
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda x: x[1])
    return ckpts[-1]


class TrainConfig:
    scene_path = "data/gibson/Cantwell.glb"
    scene_dataset_path = "data/gibson"
    control_frequency = 10
    device = "cuda"
    frame_skip = 10
    max_linear_velocity = 1.0
    max_angular_velocity = 2.0
    imu_noise_std = 0.0
    gpu_device_id = 0
    seed = 42
    debug_render = False
    headless = True
    goal_distance_scale = 3.0
    goal_max_distance = 10.0
    randomize_scenes = True
    held_out_scenes = ["Airport", "Skokloster-castle", "van-gogh-room"]
    replay_buffer_size = int(1e6)
    max_episode_steps = 3000
    start_training = 1000
    pretrained_checkpoint = None
    video_interval = 1000
    log_interval = 1000
    batch_size = 128
    checkpoint_interval = 1000
    save_dir = "./logs/"
    tqdm = True
    max_steps = int(5e6)


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
        headless=TrainConfig.headless,
        goal_distance_scale=TrainConfig.goal_distance_scale,
        goal_max_distance=TrainConfig.goal_max_distance,
        randomize_scenes=TrainConfig.randomize_scenes,
        held_out_scenes=TrainConfig.held_out_scenes,
    )

# Log scene info
scene_paths = habitat_cfg.get_scene_paths()
print(f"[Scenes] Training on {len(scene_paths)} scenes"
      f"{' (held out: ' + ', '.join(TrainConfig.held_out_scenes) + ')' if TrainConfig.held_out_scenes else ''}")

env = HabitatNavEnv(habitat_cfg, render_mode="rgb_array")
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
env = HabitatRewardWrapper(env, goal_threshold=goal_threshold,curiosity_memory_size=int(3e3))
env = RecordEpisodeStatistics(env)
env = TimeLimit(env, max_episode_steps=TrainConfig.max_episode_steps)
kwargs = drq_default.get_config()

agent = DrQLearner(
        TrainConfig.seed,
        env.observation_space.sample(),
        env.action_space.sample(),
        **kwargs,
    )

# Resume checkpoint if available
os.makedirs(POLICY_FOLDER, exist_ok=True)
resume_path, resume_step = _find_latest_checkpoint(POLICY_FOLDER)
if resume_path is not None:
    agent = load_checkpoint(agent, resume_path)
    print(f"[Checkpoint] Resumed from {resume_path} (step {resume_step:,})")
elif TrainConfig.pretrained_checkpoint:
    agent = load_checkpoint(agent, TrainConfig.pretrained_checkpoint)
    print(f"[Checkpoint] Loaded pre-trained from {TrainConfig.pretrained_checkpoint}")
    resume_step = 0
else:
    resume_step = 0
    print("[Checkpoint] Training from scratch")

start_step = resume_step + 1

# ── Replay buffer ─────────────────────────────────────────────────────────
replay_buffer = ReplayBuffer(
    env.observation_space,
    env.action_space,
    TrainConfig.replay_buffer_size,
)


# ── Noise ─────────────────────────────────────────────────────────────────
ou_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(env.action_space.shape[0]),
    theta=0.15,
    sigma=0.2,
)

# ── Logging ───────────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(TrainConfig.save_dir, f"habitat_nav_{timestamp}")
logger = Logger(log_dir)

# ── Video recording (progress monitoring) ───────────────────────────────
video_rec = None
if TrainConfig.video_interval > 0:
    video_dir = os.path.join(log_dir, "videos")
    video_rec = VideoRecorder(env, video_dir=video_dir)
    env = video_rec  # wrap so step/reset capture frames
    print(f"[Video] Recording full episodes every {TrainConfig.video_interval} steps → {video_dir}")

# ── Training loop ────────────────────────────────────────────────────────
obs, info = env.reset()
episode_reward = 0.0
episode_length = 0
episode_distance = 0.0
episode_collisions = 0
episode_curiosity = 0.0
episode_start_pos = info.get("pos", np.zeros(3)).copy()
episode_prev_pos = episode_start_pos.copy()

# Episode buffer for HER
episode_transitions = []

# Running episode stats
episode_successes = deque(maxlen=100)
episode_distances = deque(maxlen=100)
recent_collisions = deque(maxlen=100)  # running collision rate over last 100 steps
recent_curiosity = deque(maxlen=100)   # running mean curiosity over last 100 steps

pbar = tqdm.tqdm(range(start_step, TrainConfig.max_steps + 1),
                    disable=not TrainConfig.tqdm, desc="Training")

for step in pbar:
    # ── Action selection ──────────────────────────────────────────────────
    if step < TrainConfig.start_training:
        action = env.action_space.sample()
    else:
        action = agent.sample_actions(obs)
        noise = ou_noise()
        action = np.clip(action + noise, env.action_space.low,
                            env.action_space.high)
        

    # ── Environment step ─────────────────────────────────────────────────
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    hit = next_info.get("hit", False)
    recent_collisions.append(float(hit))
    episode_collisions += int(hit)
    curiosity_val = next_info.get("curiosity", 0.0)
    episode_curiosity += curiosity_val
    recent_curiosity.append(curiosity_val)

    # ── Store transition ─────────────────────────────────────────────────
    transition = dict(
        observations=obs,
        actions=action,
        rewards=reward,
        next_observations=next_obs,
        masks=np.float32(1.0 - terminated),
        dones=bool(terminated),
    )
    replay_buffer.insert(transition)
    episode_reward += reward
    episode_length += 1

    # ── Episode end ──────────────────────────────────────────────────────
    if terminated or truncated:
        success = next_info.get("distance_to_goal", float("inf")) < goal_threshold if terminated else False

        episode_transitions = []
        hab_success = next_info.get("habitat_success", 0.0)
        hab_spl = next_info.get("habitat_spl", 0.0)
        episode_successes.append(float(success))
        episode_distances.append(episode_distance)
        collision_rate = episode_collisions / max(episode_length, 1)
        logger.log_episode({
            "return": episode_reward,
            "length": episode_length,
            "success": float(success),
            "distance": episode_distance,
            "collisions": episode_collisions,
            "collision_rate": collision_rate,
            "curiosity": episode_curiosity,
            "habitat_success": float(hab_success),
            "habitat_spl": float(hab_spl),
        }, step)
        episode_reward = 0.0
        episode_length = 0
        episode_distance = 0.0
        episode_collisions = 0
        episode_curiosity = 0.0
        obs, info = env.reset()
        episode_start_pos = info.get("pos", np.zeros(3)).copy()
        episode_prev_pos = episode_start_pos.copy()
    else:
        obs = next_obs
        info = next_info

    # ── Gradient update ──────────────────────────────────────────────────
    if step >= TrainConfig.start_training and len(replay_buffer) >= TrainConfig.batch_size:
        batch = replay_buffer.sample(TrainConfig.batch_size)

        update_info = agent.update(batch)

        if step % TrainConfig.log_interval == 0:
            logger.log_training(update_info, step)

    # ── Checkpoint ───────────────────────────────────────────────────────
    if step % TrainConfig.checkpoint_interval == 0 and step > TrainConfig.start_training:
        ckpt_dir = os.path.join(POLICY_FOLDER, f"checkpoint_{step}")
        save_checkpoint(agent, replay_buffer, ckpt_dir, step)
        print(f"[Checkpoint] Saved at step {step:,}")

    # ── Video recording ──────────────────────────────────────────────────
    if video_rec is not None and step > 0 and step % TrainConfig.video_interval == 0:
        video_rec.record_next_episode()

    # ── Progress ──────────────────────────────────────────────────────────
    if step % TrainConfig.log_interval == 0:
        pbar.set_postfix({
            "step": step,
            "buffer": len(replay_buffer),
            "ep_rew": f"{np.mean(logger.episode_returns):.1f}" if logger.episode_returns else "0.0",
            "sr": f"{np.mean(episode_successes):.0%}" if episode_successes else "0%",
            "coll": f"{np.mean(recent_collisions):.0%}" if recent_collisions else "0%",
            "curio": f"{np.mean(recent_curiosity):.2f}" if recent_curiosity else "0.00",
            "dist": f"{np.mean(episode_distances):.2f}m" if episode_distances else "0m",
        })
        logger.print_status(step, TrainConfig.max_steps, extra_stats={
            "Buffer size": len(replay_buffer),
            "Goal threshold": goal_threshold,
        })

# ── Final save ───────────────────────────────────────────────────────────
final_dir = os.path.join(POLICY_FOLDER, "final")
save_checkpoint(agent, replay_buffer, final_dir, TrainConfig.max_steps)
print(f"\nTraining complete. Final checkpoint saved to {final_dir}")

env.close()

