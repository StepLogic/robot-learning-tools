#!/usr/bin/env python
"""
DrQ Training — Habitat Image-Goal Navigation
=============================================
Uses Habitat-Sim with continuous VelocityControl for navigation.
Goal images are encoded via a Siamese MobileNetV3 encoder (shared
with the current-observation encoder).

Wrapper stack:
  HabitatNavEnv → StackingWrapper → MobileNetFeatureWrapper
  → GoalImageWrapper → HabitatRewardWrapper → RecordEpisodeStatistics → TimeLimit
"""
import os
import pickle
import random
from collections import deque
from datetime import datetime

import cv2
import flax
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from ml_collections import config_flags

import torch
import tqdm

import gymnasium as gym
from gymnasium import spaces

from jaxrl2.agents import DrQLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

from habitat_env import HabitatNavEnv, GymnasiumHabitatNav, HAS_HABITAT_LAB
from configs.habitat_config import HabitatNavConfig
from racer_imu_env import StackingWrapper
from wrappers import (
    Logger,
    MobileNetFeatureWrapper,
    MobileNetV3Encoder,
    GoalImageWrapper,
    load_checkpoint,
    save_checkpoint,
)

flax.config.update("flax_use_orbax_checkpointing", True)
FLAGS = flags.FLAGS

# ── Environment flags ────────────────────────────────────────────────────────
flags.DEFINE_string("env_name", "HabitatImageNav-v0", "Environment name.")
flags.DEFINE_string("scene_path", "data/gibson/Cantwell.glb",
                    "Path to Gibson .glb scene file.")
flags.DEFINE_string("scene_dataset_path", "",
                    "Path to Gibson scene dataset directory (empty for standalone GLB).")
flags.DEFINE_boolean("randomize_scenes", False,
                    "Randomly select a different Gibson scene each episode.")
flags.DEFINE_integer("control_frequency", 5, "Habitat control frequency (Hz).")
flags.DEFINE_integer("frame_skip", 6, "Physics integration steps per action.")
flags.DEFINE_float("max_linear_velocity", 0.5, "Max forward velocity (m/s).")
flags.DEFINE_float("max_angular_velocity", 1.5, "Max turning rate (rad/s).")
flags.DEFINE_float("imu_noise_std", 0.0, "Gaussian noise std for synthesized IMU.")
flags.DEFINE_integer("gpu_device_id", 0, "Habitat-Sim GPU device ID.")
flags.DEFINE_boolean("debug_render", True, "Show cv2 debug window.")

# ── Training flags ───────────────────────────────────────────────────────────
flags.DEFINE_string("save_dir", "./logs/", "Tensorboard log dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 50000, "Eval interval (not used yet).")
flags.DEFINE_integer("checkpoint_interval", 5000, "Checkpoint interval.")
flags.DEFINE_integer("video_interval", 50000, "Save a video every N steps (0 = disabled).")
flags.DEFINE_integer("video_length", 500, "Number of steps per video clip.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Total training steps.")
flags.DEFINE_integer("start_training", int(5e3), "Steps before updates begin.")
flags.DEFINE_integer("replay_buffer_size", int(1e5), "Replay buffer capacity.")
flags.DEFINE_boolean("tqdm", True, "Show tqdm progress bar.")
flags.DEFINE_integer("frame_stack", 3, "Frame stack depth.")
flags.DEFINE_integer("mobilenet_blocks", 13, "MobileNetV3 blocks (full network).")
flags.DEFINE_integer("mobilenet_input_size", 84, "MobileNetV3 input size.")

flags.DEFINE_float("goal_distance_scale", 3.0,
                    "Exponential rate for goal distance (meters, lower = closer goals).")
flags.DEFINE_float("goal_max_distance", 10.0,
                    "Cap on sampled goal distance (meters).")
flags.DEFINE_integer("max_episode_steps", 1000, "Max steps per episode.")

# ── Expert / teleop flags ────────────────────────────────────────────────────
flags.DEFINE_string("pretrained_checkpoint", "",
                    "Path to pre-trained checkpoint (empty = no preload).")
flags.DEFINE_float("expert_sample_ratio", 0.0,
                   "Fraction of batch from expert buffer (0 = disabled).")
flags.DEFINE_string("expert_buffer_path", "",
                    "Path to expert replay buffer .pkl (empty = none).")

# ── HER flags ────────────────────────────────────────────────────────────────
flags.DEFINE_float("her_ratio", 0.0,
                   "Fraction of episode transitions to relabel with HER (0 = disabled).")
flags.DEFINE_float("her_goal_threshold", None,
                   "Feature-distance threshold for HER goal reward (None = auto from goal_threshold).")

config_flags.DEFINE_config_file(
    "config", "./configs/drq_default.py",
    "Training hyperparameter config.", lock_config=False,
)

POLICY_FOLDER = "robot_policy"


# ═══════════════════════════════════════════════════════════════════════════════
# Reward Wrapper for Habitat
# ═══════════════════════════════════════════════════════════════════════════════
class HabitatRewardWrapper(gym.Wrapper):
    """
    Reward shaping for Habitat image-goal navigation.

    Active reward terms:
      + Distance improvement (delta_dist) when throttle is moderate (0.1-0.5)
      + Large bonus (k_goal) for reaching within goal_threshold of goal
      - Collision penalty (k_collision)

    Commented-out terms (steering penalty, stall detection, raw movement bonus)
    can be re-enabled by uncommenting the corresponding lines in step().
    """

    def __init__(self, env, goal_threshold=0.5,
                 k_dist=1.0, k_delta_x=1.0, k_throttle=1.0,
                 k_goal=10.0, k_collision=0.001, k_steering=0.1):
        super().__init__(env)
        self.goal_threshold = goal_threshold
        self.k_dist = k_dist
        self.k_delta_x = k_delta_x
        self.k_throttle = k_throttle
        self.k_goal = k_goal
        self.k_collision = k_collision
        self.k_steering = k_steering
        self.deltas = deque(maxlen=5)  
        self.steering_hist = deque(maxlen=10)
        
        self.throttle_hist = deque(maxlen=10)  
        self._prev_distance = float("inf")
        self.distance_covered = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.distance_covered = 0
        self._prev_distance = info.get("distance_to_goal", float("inf"))
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        reward = -1.0

        # # Distance improvement toward goal
        # curr_distance = info["distance_to_goal"]
        # delta_dist = self._prev_distance - curr_distance

        # # Movement bonus: reward driving across the environment
        habitat_distance_to_goal_reward = info["habitat_distance_to_goal_reward"]
        delta_x = info["delta_x"]
        # self.distance_covered += delta_x
        # # print(("delta_x"),delta_x)
        # # self.deltas.append(delta_x[0])
        # # Throttle bonus: encourage forward driving
        # mean_throttle = float(np.mean(self.unwrapped._rl_env._throttle_history))
        # if  mean_throttle > 0.1:
        #     reward += 0.1

        # reward += float(np.linalg.norm(habitat_distance_to_goal_reward))
        # reward += float(np.linalg.norm(delta_x))
        # reward +=  mean_throttle *
        reward -= abs(action[0])
        
        self.steering_hist.append(action[0])
        self.throttle_hist.append(action[1])
        # reward -= float(np.std(self.deltas))  
        # reward -= float(np.std(self.throttle_hist))
        # reward -= float(np.std(self.steering_hist))*0.1
        # reward -= float(np.mean(self.steering_hist))
        # Goal reached
        if info["habitat_success"] > 0.0:
            reward += self.k_goal
            print("Goal Reached")
            terminated = True

        # # # Collision penalty
        # if info.get("hit", False):
        #     # reward -= self.k_collision
        #     terminated = True

        # Steering penalty (encourages straighter paths)
        # reward -= self.k_steering * abs(action[0])

        # # Stall detection based on position change (more robust than velocity)
        # curr_position = info.get("position", None)
        # if curr_position is not None and self._prev_position is not None:
        #     position_delta = ((curr_position[0] - self._prev_position[0])**2 + 
        #                     (curr_position[2] - self._prev_position[2])**2)**0.5
            
        #     if position_delta < self.stall_threshold:
        #         self._stall_count += 1
        #     else:
        #         self._stall_count = 0
                
        #     self._prev_position = curr_position
            
        # if self._stall_count > self.stall_limit:
        #     truncated = True

        # self._prev_distance = curr_distance
        return obs, reward, terminated, truncated, info
# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers (reuse from wrappers.py)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Video recording
# ═══════════════════════════════════════════════════════════════════════════════

class VideoRecorder(gym.Wrapper):
    """Records episode frames and writes MP4 videos to disk."""

    def __init__(self, env, video_dir: str, fps: int = 30):
        super().__init__(env)
        self.video_dir = video_dir
        self.fps = fps
        os.makedirs(video_dir, exist_ok=True)
        self._frames = []

    def start_recording(self):
        self._frames = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._frames is not None and len(self._frames) < 2000:
            frame = self._get_display_frame(obs, info)
            if frame is not None:
                self._frames.append(frame)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._frames is not None and len(self._frames) < 2000:
            frame = self._get_display_frame(obs, info)
            if frame is not None:
                self._frames.append(frame)
        return obs, info

    def _get_display_frame(self, obs, info):
        raw = self.env.unwrapped.render()
        if raw is not None:
            if raw.ndim == 3 and raw.shape[2] == 4:
                raw = raw[:, :, :3]
            # raw is RGB from HabitatNavEnv
            return raw
        return None

    def save(self, filename: str):
        if not self._frames:
            return
        path = os.path.join(self.video_dir, filename)
        h, w = self._frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        for f in self._frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[Video] Saved {len(self._frames)} frames to {path}")

    def stop_and_save(self, filename: str):
        self.save(filename)
        self._frames = None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(_):
    print("\n" + "=" * 70)
    print("DrQ Habitat Image-Goal Nav | Siamese MobileNetV3")
    print("=" * 70 + "\n")

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Environment ───────────────────────────────────────────────────────────
    print("Building Habitat environment…")
    habitat_cfg = HabitatNavConfig(
        scene_path=FLAGS.scene_path,
        scene_dataset_path=FLAGS.scene_dataset_path,
        control_frequency=FLAGS.control_frequency,
        frame_skip=FLAGS.frame_skip,
        max_linear_velocity=FLAGS.max_linear_velocity,
        max_angular_velocity=FLAGS.max_angular_velocity,
        imu_noise_std=FLAGS.imu_noise_std,
        gpu_device_id=FLAGS.gpu_device_id,
        seed=FLAGS.seed,
        debug_render=FLAGS.debug_render,
        goal_distance_scale=FLAGS.goal_distance_scale,
        goal_max_distance=FLAGS.goal_max_distance,
        randomize_scenes=FLAGS.randomize_scenes,
    )
    if FLAGS.randomize_scenes:
        print(f"Scene randomization: {len(habitat_cfg.get_scene_paths())} scenes available")

    EnvClass = GymnasiumHabitatNav
    render_mode = "human" if FLAGS.debug_render else "rgb_array"
    env = EnvClass(config=habitat_cfg, render_mode=render_mode)
    # env = EnvCompatibility(env)
    env = StackingWrapper(env, num_stack=FLAGS.frame_stack, image_format="rgb")

    # Shared MobileNetV3 encoder for current obs and goal
    shared_encoder = MobileNetV3Encoder(
        device=device,
        num_blocks=FLAGS.mobilenet_blocks,
        input_size=FLAGS.mobilenet_input_size,
    )
    env = MobileNetFeatureWrapper(env, encoder=shared_encoder)
    env = GoalImageWrapper(env, encoder=shared_encoder)
    goal_threshold = 2.0
    reward_wrapper = HabitatRewardWrapper(env, goal_threshold=goal_threshold)
    env = reward_wrapper

    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=FLAGS.max_episode_steps)

    # HER setup
    her_ratio = FLAGS.her_ratio
    feature_dim = env.observation_space.spaces['goal_features'].shape[0]
    # HER threshold is in feature-space distance (L2 norm of MobileNet features),
    # not geodesic meters.  When not explicitly set, use a reasonable default
    # based on the feature dimensionality.
    her_goal_threshold = FLAGS.her_goal_threshold if FLAGS.her_goal_threshold else 1.0
    k_goal = reward_wrapper.k_goal

    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    print(f"HER ratio         : {her_ratio} (threshold: {her_goal_threshold})\n")

    # ── Agent ─────────────────────────────────────────────────────────────────
    kwargs = dict(FLAGS.config) if FLAGS.config else {}
    agent = DrQLearner(
        FLAGS.seed,
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
    elif FLAGS.pretrained_checkpoint:
        agent = load_checkpoint(agent, FLAGS.pretrained_checkpoint)
        print(f"[Checkpoint] Loaded pre-trained from {FLAGS.pretrained_checkpoint}")
        resume_step = 0
    else:
        resume_step = 0
        print("[Checkpoint] Training from scratch")

    start_step = resume_step + 1

    # ── Replay buffer ─────────────────────────────────────────────────────────
    replay_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        FLAGS.replay_buffer_size,
    )

    # Optional expert buffer
    expert_buf = None
    if FLAGS.expert_buffer_path and os.path.exists(FLAGS.expert_buffer_path):
        with open(FLAGS.expert_buffer_path, "rb") as f:
            expert_buf = pickle.load(f)
        print(f"[Expert] Loaded expert buffer: {expert_buf._size} transitions")

    # ── Noise ─────────────────────────────────────────────────────────────────
    ou_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(env.action_space.shape[0]),
        theta=0.15,
        sigma=0.2,
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(FLAGS.save_dir, f"habitat_nav_{timestamp}")
    logger = Logger(log_dir)

    # ── Video recording ──────────────────────────────────────────────────────
    video_rec = None
    if FLAGS.video_interval > 0:
        video_dir = os.path.join(log_dir, "videos")
        video_rec = VideoRecorder(env, video_dir=video_dir)
        env = video_rec  # wrap so env.step/reset captures frames

    # ── Training loop ────────────────────────────────────────────────────────
    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_distance = 0.0
    episode_start_pos = info.get("pos", np.zeros(3)).copy()
    episode_prev_pos = episode_start_pos.copy()
    video_recording = False
    video_step_count = 0

    # Episode buffer for HER
    episode_transitions = []

    # Running episode stats
    episode_successes = deque(maxlen=100)
    episode_distances = deque(maxlen=100)

    pbar = tqdm.tqdm(range(start_step, FLAGS.max_steps + 1),
                     disable=not FLAGS.tqdm, desc="Training")

    for step in pbar:
        # ── Action selection ──────────────────────────────────────────────────
        if step < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(obs)
            noise = ou_noise()
            action = np.clip(action + noise, env.action_space.low,
                             env.action_space.high)
            

        # ── Environment step ─────────────────────────────────────────────────
        next_obs, reward, terminated, truncated, next_info = env.step(action)

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

        # ── HER episode buffer ──────────────────────────────────────────────────
        # Extract current-frame features from stacked pixels for goal relabeling
        current_features = next_obs["pixels"][-feature_dim:]
        episode_transitions.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "terminated": bool(terminated),
            "goal_features": next_obs.get("goal_features", np.zeros(feature_dim)),
            "current_features": current_features,
        })

        # ── Distance covered tracking ──────────────────────────────────────────
        curr_pos = next_info.get("pos", None)
        if curr_pos is not None:
            episode_distance += np.linalg.norm(curr_pos - episode_prev_pos)
            episode_prev_pos = curr_pos.copy()

        # ── Episode end ──────────────────────────────────────────────────────
        if terminated or truncated:
            success = next_info.get("distance_to_goal", float("inf")) < goal_threshold if terminated else False

            # ── HER: relabel transitions with future achieved goals ────────────
            # Standard HER "future" strategy: for each transition, sample a
            # future state from the same episode and use its visual features
            # as the new goal. Reward is based on feature-space distance.
            if her_ratio > 0 and len(episode_transitions) > 1:
                ep_len = len(episode_transitions)

                for idx in range(ep_len - 1):
                    if np.random.random() > her_ratio:
                        continue

                    t = episode_transitions[idx]
                    # Sample a future state as the virtual goal
                    future_idx = np.random.randint(idx + 1, ep_len)
                    future_t = episode_transitions[future_idx]
                    new_goal_features = future_t["current_features"]

                    # Relabel goal_features in obs
                    her_obs = dict(t["obs"])
                    her_next_obs = dict(t["next_obs"])
                    her_obs["goal_features"] = new_goal_features.copy()
                    her_next_obs["goal_features"] = new_goal_features.copy()

                    # Reward: sparse bonus when the agent's state matches the
                    # virtual goal (i.e. agent reached the future state)
                    her_reward = -1.0
                    agent_features = t["current_features"]
                    feat_dist = np.linalg.norm(agent_features - new_goal_features)
                    goal_reached = feat_dist < her_goal_threshold
                    if goal_reached:
                        her_reward += k_goal

                    her_transition = dict(
                        observations=her_obs,
                        actions=t["action"],
                        rewards=her_reward,
                        next_observations=her_next_obs,
                        masks=np.float32(0.0 if goal_reached else 1.0),
                        dones=bool(goal_reached),
                    )
                    replay_buffer.insert(her_transition)

            episode_transitions = []
            hab_success = next_info.get("habitat_success", 0.0)
            hab_spl = next_info.get("habitat_spl", 0.0)
            episode_successes.append(float(success))
            episode_distances.append(episode_distance)
            logger.log_episode({
                "return": episode_reward,
                "length": episode_length,
                "success": float(success),
                "distance": episode_distance,
                "habitat_success": float(hab_success),
                "habitat_spl": float(hab_spl),
            }, step)
            episode_reward = 0.0
            episode_length = 0
            episode_distance = 0.0
            obs, info = env.reset()
            episode_start_pos = info.get("pos", np.zeros(3)).copy()
            episode_prev_pos = episode_start_pos.copy()
        else:
            obs = next_obs
            info = next_info

        # ── Gradient update ──────────────────────────────────────────────────
        if step >= FLAGS.start_training and replay_buffer._size >= FLAGS.batch_size:
            batch = replay_buffer.sample(FLAGS.batch_size)

            # Mixed batch with expert data
            if expert_buf is not None and FLAGS.expert_sample_ratio > 0:
                n_expert = int(FLAGS.batch_size * FLAGS.expert_sample_ratio)
                n_online = FLAGS.batch_size - n_expert
                online_batch = replay_buffer.sample(n_online)
                expert_batch = expert_buf.sample(n_expert)
                # Merge: unfreeze FrozenDicts, concatenate, refreeze
                from flax.core import frozen_dict
                online_unfrozen = frozen_dict.unfreeze(online_batch)
                expert_unfrozen = frozen_dict.unfreeze(expert_batch)
                merged = {}
                for key in online_unfrozen:
                    if isinstance(online_unfrozen[key], dict):
                        merged[key] = {
                            k: np.concatenate(
                                [online_unfrozen[key][k], expert_unfrozen[key][k]]
                            )
                            for k in online_unfrozen[key]
                        }
                    else:
                        merged[key] = np.concatenate(
                            [online_unfrozen[key], expert_unfrozen[key]], axis=0
                        )
                batch = frozen_dict.freeze(merged)

            update_info = agent.update(batch)

            if step % FLAGS.log_interval == 0:
                logger.log_training(update_info, step)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if step % FLAGS.checkpoint_interval == 0 and step > FLAGS.start_training:
            ckpt_dir = os.path.join(POLICY_FOLDER, f"checkpoint_{step}")
            save_checkpoint(agent, replay_buffer, ckpt_dir, step)
            print(f"[Checkpoint] Saved at step {step:,}")

        # ── Video recording ──────────────────────────────────────────────────
        if video_rec is not None:
            if not video_recording and step % FLAGS.video_interval == 0:
                video_rec.start_recording()
                video_recording = True
                video_step_count = 0
            if video_recording:
                video_step_count += 1
                if video_step_count >= FLAGS.video_length:
                    video_rec.stop_and_save(f"step_{step:07d}.mp4")
                    video_recording = False

        # ── Progress ──────────────────────────────────────────────────────────
        if step % FLAGS.log_interval == 0:
            pbar.set_postfix({
                "step": step,
                "buffer": replay_buffer._size,
                "ep_rew": f"{np.mean(logger.episode_returns):.1f}" if logger.episode_returns else "0.0",
                "sr": f"{np.mean(episode_successes):.0%}" if episode_successes else "0%",
                "dist": f"{np.mean(episode_distances):.2f}m" if episode_distances else "0m",
            })
            logger.print_status(step, FLAGS.max_steps, extra_stats={
                "Buffer size": replay_buffer._size,
                "Goal threshold": goal_threshold,
            })

    # ── Final save ───────────────────────────────────────────────────────────
    final_dir = os.path.join(POLICY_FOLDER, "final")
    save_checkpoint(agent, replay_buffer, final_dir, FLAGS.max_steps)
    print(f"\nTraining complete. Final checkpoint saved to {final_dir}")

    env.close()


if __name__ == "__main__":
    app.run(main)