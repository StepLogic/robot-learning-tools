#!/usr/bin/env python
"""
DrQ Training — Real Robot Image-Goal Navigation (HitL + Background Thread + HER)
===================================================================================
Uses RacerEnv (real Donkey Car robot) with image-goal navigation.

Key design:
  1. Goal images are sampled from a persistent pool of past observations.
  2. During data collection the goal is masked (zeroed) — the agent explores
     without a known target.
  3. At episode end transitions sit in an *episode buffer* first.
     70 % are relabeled via HER (future achieved visual goals).
     30 % keep the sampled episode goal.
  4. Gradient updates run in a background thread so robot data collection
     is never blocked by network training.

Wrapper stack:
  RacerEnv → StackingWrapper → RealRobotGoalWrapper → MobileNetFeatureWrapper
  → GoalImageWrapper → RealRobotImageGoalRewardWrapper → RecordEpisodeStatistics → TimeLimit
"""
import os
import pickle
import random
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import flax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

import torch
import tqdm

import gymnasium as gym

from jaxrl2.agents import DrQLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

import pygame

from racer_imu_env import RacerEnv, StackingWrapper
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
flags.DEFINE_string("env_name", "donkey-warehouse-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./logs/", "Tensorboard log dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 50000, "Eval interval (not used yet).")
flags.DEFINE_integer("checkpoint_interval", 5000, "Checkpoint interval.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Total training steps.")
flags.DEFINE_integer("start_training", int(5e3), "Steps before updates begin.")
flags.DEFINE_integer("replay_buffer_size", int(1e5), "Replay buffer capacity.")
flags.DEFINE_boolean("tqdm", True, "Show tqdm progress bar.")
flags.DEFINE_integer("frame_stack", 3, "Frame stack depth.")
flags.DEFINE_integer("mobilenet_blocks", 13, "MobileNetV3 blocks.")
flags.DEFINE_integer("mobilenet_input_size", 84, "MobileNetV3 input size.")

flags.DEFINE_float("goal_feature_threshold", 1.0,
                   "Feature-distance threshold for considering goal reached.")
flags.DEFINE_integer("max_episode_steps", 1000, "Max steps per episode.")

# ── Expert / teleop flags ────────────────────────────────────────────────────
flags.DEFINE_string("pretrained_checkpoint", "",
                    "Path to pre-trained checkpoint (empty = no preload).")
flags.DEFINE_float("expert_sample_ratio", 0.0,
                   "Fraction of batch from expert buffer (0 = disabled).")
flags.DEFINE_string("expert_buffer_path", "",
                    "Path to expert replay buffer .pkl (empty = none).")

# ── HER flags ────────────────────────────────────────────────────────────────
flags.DEFINE_float("her_ratio", 0.7,
                   "Fraction of episode transitions to relabel with HER.")
flags.DEFINE_float("her_goal_threshold", None,
                   "Feature-distance threshold for HER goal reward.")

# ── HitL flags ─────────────────────────────────────────────────────────────────
flags.DEFINE_boolean("enable_hitl", True, "Enable Human-in-the-Loop keyboard override.")
flags.DEFINE_float("steer_step", 0.05, "Steering increment per keypress.")
flags.DEFINE_float("throttle_step", 0.02, "Throttle increment per keypress.")
flags.DEFINE_float("throttle_default", 0.130, "Default throttle when human takes over.")

# ── Checkpoint flags ─────────────────────────────────────────────────────────
flags.DEFINE_string("checkpoint_path", "",
                    "Explicit checkpoint path to load (overrides auto-resume).")

config_flags.DEFINE_config_file(
    "config", "./configs/drq_default.py",
    "Training hyperparameter config.", lock_config=False,
)

POLICY_FOLDER = "robot_policy"
GOAL_POOL_PATH = os.path.join(POLICY_FOLDER, "goal_image_pool.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# Goal Image Pool (persistent storage of candidate goals)
# ═══════════════════════════════════════════════════════════════════════════════
class GoalImagePool:
    """
    Rolling buffer of RGB images that can serve as future navigation goals.
    Saved to disk so the pool survives across training runs.
    """

    def __init__(self, capacity=5000, save_path=GOAL_POOL_PATH):
        self.capacity = capacity
        self.save_path = save_path
        self.pool = deque(maxlen=capacity)
        self._load()

    def add(self, image: np.ndarray):
        """Add a single RGB image (H, W, 3) uint8 to the pool."""
        self.pool.append(image.copy())

    def sample(self) -> np.ndarray | None:
        """Sample a random image from the pool, or None if empty."""
        if len(self.pool) == 0:
            return None
        return random.choice(self.pool)

    def save(self):
        if self.save_path is None:
            return
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        tmp = self.save_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(list(self.pool), f)
        os.replace(tmp, self.save_path)

    def _load(self):
        if self.save_path and os.path.exists(self.save_path):
            try:
                with open(self.save_path, "rb") as f:
                    data = pickle.load(f)
                self.pool = deque(data, maxlen=self.capacity)
                print(f"[GoalPool] Loaded {len(self.pool):,} candidate goals from {self.save_path}")
            except Exception as e:
                print(f"[GoalPool] Load failed ({e}) — starting fresh.")


# ═══════════════════════════════════════════════════════════════════════════════
# Goal Image Wrapper for Real Robot
# ═══════════════════════════════════════════════════════════════════════════════
class RealRobotGoalWrapper(gym.Wrapper):
    """
    Manages goal images for real robot image-goal navigation.

    Uses a persistent GoalImagePool:
      • At episode reset a goal image is sampled from the pool.
      • Every step the current RGB frame is added to the pool.
      • The selected goal is injected into info['goal_image'] for
        GoalImageWrapper downstream.
    """

    def __init__(self, env, goal_pool: GoalImagePool):
        super().__init__(env)
        self.goal_pool = goal_pool
        self.current_goal_image = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        goal_img = self.goal_pool.sample()
        if goal_img is None:
            # Pool empty — bootstrap from the last frame of the current stack
            if isinstance(obs, dict) and "pixels" in obs:
                pixels = obs["pixels"]  # (H, W, 3*num_stack) RGB
                goal_img = pixels[:, :, -3:]
            else:
                goal_img = np.zeros((120, 160, 3), dtype=np.uint8)

        self.current_goal_image = goal_img
        info["goal_image"] = goal_img
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add the current (last) RGB frame to the pool for future episodes
        if isinstance(obs, dict) and "pixels" in obs:
            pixels = obs["pixels"]
            if pixels.shape[-1] >= 3:
                current_img = pixels[:, :, -3:]
                self.goal_pool.add(current_img)

        info["goal_image"] = self.current_goal_image
        return obs, reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════════
# Reward Wrapper for Real Robot Image-Goal Navigation
# ═══════════════════════════════════════════════════════════════════════════════
class RealRobotImageGoalRewardWrapper(gym.Wrapper):
    """
    Reward shaping for real robot image-goal navigation.
    """

    def __init__(self, env, goal_threshold=1.0,
                 k_goal=10.0, k_collision=10.0, k_velocity=0.5,
                 stall_limit=100):
        super().__init__(env)
        self.goal_threshold = goal_threshold
        self.k_goal = k_goal
        self.k_collision = k_collision
        self.k_velocity = k_velocity
        self.stall_limit = stall_limit

        self._prev_feature_dist = float("inf")
        self.stop_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_feature_dist = float("inf")
        self.stop_count = 0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        reward = -1.0

        # ── Feature distance to goal ──────────────────────────────────────────
        feature_dist = float("inf")
        if "goal_features" in obs and "pixels" in obs:
            goal_features = obs["goal_features"]
            feature_dim = goal_features.shape[0]
            current_features = obs["pixels"][-feature_dim:]
            feature_dist = np.linalg.norm(current_features - goal_features)
            info["feature_distance"] = feature_dist

        if feature_dist != float("inf"):
            delta_dist = self._prev_feature_dist - feature_dist
            reward += delta_dist * 0.1

        if feature_dist < self.goal_threshold:
            reward += self.k_goal
            terminated = True
            print(f"[Goal] Reached! Feature dist: {feature_dist:.3f}")

        # ── Velocity bonus ────────────────────────────────────────────────────
        vel_ms = info.get("velocity", {}).get("ms", 0.0)
        if vel_ms > 0.05:
            reward += self.k_velocity * vel_ms

        # ── Collision / blocked penalties ─────────────────────────────────────
        coll_info = info.get("collision", {})
        if coll_info.get("detected", False):
            reward -= self.k_collision
            terminated = True

        if info.get("blocked", False):
            reward -= 2.0

        # ── Stall / low-velocity truncation ───────────────────────────────────
        if vel_ms < 0.01:
            self.stop_count += 1
        else:
            self.stop_count = 0

        if self.stop_count > self.stall_limit:
            truncated = True

        self._prev_feature_dist = feature_dist
        return obs, reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
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


def _load_checkpoint(agent, checkpoint_path):
    """Load a checkpoint and return the agent + step."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return agent, 0
    agent = load_checkpoint(agent, checkpoint_path)
    m = re.search(r"(?:checkpoint_|step_)(\d+)", checkpoint_path)
    step = int(m.group(1)) if m else 0
    print(f"[Checkpoint] Loaded from {checkpoint_path} (step {step:,})")
    return agent, step


# ═══════════════════════════════════════════════════════════════════════════════
# Human-in-the-Loop Controller
# ═══════════════════════════════════════════════════════════════════════════════
class HumanController:
    """Pygame keyboard controller for Human-in-the-Loop override."""

    def __init__(self, action_low, action_high):
        self.low = action_low
        self.high = action_high
        self.steering = 0.0
        self.throttle = FLAGS.throttle_default
        self.paused = False

    def process_events(self):
        quit_req = reset_req = tog_pause = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_req = True
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    quit_req = True
                if event.key == pygame.K_r:
                    reset_req = True
                if event.key == pygame.K_t:
                    tog_pause = True
                if event.key == pygame.K_SPACE:
                    self.throttle = FLAGS.throttle_default
        return quit_req, reset_req, tog_pause

    def read(self):
        keys = pygame.key.get_pressed()
        steer_key = throttle_key = False
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.steering = max(self.low[0], self.steering - FLAGS.steer_step)
            steer_key = True
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.steering = min(self.high[0], self.steering + FLAGS.steer_step)
            steer_key = True
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.throttle = min(self.high[1], self.throttle + FLAGS.throttle_step)
            throttle_key = True
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.throttle = max(self.low[1], self.throttle - FLAGS.throttle_step)
            throttle_key = True
        human_active = steer_key or throttle_key or self.paused
        return np.array([self.steering, self.throttle], dtype=np.float32), human_active

    def reset_controls(self):
        self.steering = 0.0
        self.throttle = FLAGS.throttle_default


def _draw_hud(screen, steering, throttle, step, ep, ep_steps,
              ep_reward, human_active, paused, buffer_size, buf_cap):
    if screen is None:
        return
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.SysFont("monospace", 15, bold=True)

    def txt(msg, x, y, color=(255, 255, 0)):
        screen.blit(font.render(msg, True, color), (x, y))

    bar = pygame.Surface((screen.get_width(), 90), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 170))
    screen.blit(bar, (0, screen.get_height() - 90))
    y0 = screen.get_height() - 88

    if paused:
        lbl, col = "HUMAN [PAUSED]", (255, 80, 80)
    elif human_active:
        lbl, col = "HUMAN OVERRIDE", (80, 255, 80)
    else:
        lbl, col = "AGENT", (80, 180, 255)

    txt(f"[ {lbl} ]", screen.get_width() - 220, y0, color=col)
    txt(f"Steer {steering:+.3f}   Throttle {throttle:+.3f}", 10, y0)
    txt(f"Step {step:>7d}   Ep {ep:>4d}   Ep-step {ep_steps:>4d}", 10, y0 + 18)
    txt(f"Ep reward {ep_reward:+.2f}   Buf {buffer_size:>6d}/{buf_cap}", 10, y0 + 36)
    txt("W/S=throttle  A/D=steer  SPC=coast  T=pause  R=reset  Q=quit",
        10, y0 + 54, color=(190, 190, 190))

    # Steering bar
    bw, bh = 180, 8
    bx, by = screen.get_width() - bw - 10, y0 + 20
    cx = bx + bw // 2
    pygame.draw.rect(screen, (60, 60, 60), (bx, by, bw, bh))
    fw = int(abs(steering) * (bw // 2))
    col = (0, 200, 100) if steering >= 0 else (220, 80, 0)
    pygame.draw.rect(screen, col,
                     (cx if steering >= 0 else cx - fw, by, fw, bh))
    pygame.draw.line(screen, (255, 255, 255), (cx, by), (cx, by + bh), 1)
    pygame.display.flip()


# ═══════════════════════════════════════════════════════════════════════════════
# Episode buffer helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _mask_goal_in_obs(obs, feature_dim):
    """Return a copy of obs with goal_features zeroed out."""
    obs = dict(obs)
    if "goal_features" in obs:
        obs["goal_features"] = np.zeros(feature_dim, dtype=np.float32)
    return obs


def _set_goal_in_obs(obs, goal_features):
    """Return a copy of obs with goal_features replaced."""
    obs = dict(obs)
    obs["goal_features"] = goal_features.copy()
    return obs


# ═══════════════════════════════════════════════════════════════════════════════
# Background training thread
# ═══════════════════════════════════════════════════════════════════════════════
def _train_thread(
    agent,
    batch_queue,
    episode_event,
    stop_event,
    update_info_box,
    n_updates,
    utd_ratio,
):
    """Run gradient updates in a background thread."""
    while not stop_event.is_set():
        triggered = episode_event.wait(timeout=0.5)
        if stop_event.is_set():
            break
        if not triggered:
            continue
        episode_event.clear()

        info = {}
        for _ in range(n_updates):
            if batch_queue:
                batch = batch_queue.popleft()
                info = agent.update(batch, utd_ratio=utd_ratio)
        if info:
            update_info_box[0] = info


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(_):
    print("\n" + "=" * 70)
    print("DrQ Real Robot Image-Goal Nav | HitL | Background Thread | HER")
    print("=" * 70 + "\n")

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Environment ───────────────────────────────────────────────────────────
    print("Building real robot environment…")
    env = RacerEnv(render_mode="human")
    env = StackingWrapper(env, num_stack=FLAGS.frame_stack, image_format="bgr")

    shared_encoder = MobileNetV3Encoder(
        device=device,
        num_blocks=FLAGS.mobilenet_blocks,
        input_size=FLAGS.mobilenet_input_size,
    )

    # Goal image pool (persistent across runs)
    goal_pool = GoalImagePool(capacity=5000, save_path=GOAL_POOL_PATH)

    env = RealRobotGoalWrapper(env, goal_pool=goal_pool)
    env = MobileNetFeatureWrapper(env, encoder=shared_encoder)
    env = GoalImageWrapper(env, encoder=shared_encoder)

    goal_threshold = FLAGS.goal_feature_threshold
    env = RealRobotImageGoalRewardWrapper(env, goal_threshold=goal_threshold)
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=FLAGS.max_episode_steps)

    # HER setup
    her_ratio = FLAGS.her_ratio
    feature_dim = env.observation_space.spaces["goal_features"].shape[0]
    her_goal_threshold = (
        FLAGS.her_goal_threshold if FLAGS.her_goal_threshold else goal_threshold
    )
    k_goal = 10.0

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

    # ── Checkpoint loading ────────────────────────────────────────────────────
    os.makedirs(POLICY_FOLDER, exist_ok=True)
    resume_step = 0

    if FLAGS.checkpoint_path:
        agent, resume_step = _load_checkpoint(agent, FLAGS.checkpoint_path)
    else:
        resume_path, resume_step = _find_latest_checkpoint(POLICY_FOLDER)
        if resume_path is not None:
            agent, resume_step = _load_checkpoint(agent, resume_path)
        elif FLAGS.pretrained_checkpoint:
            agent, resume_step = _load_checkpoint(agent, FLAGS.pretrained_checkpoint)
        else:
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
    log_dir = os.path.join(FLAGS.save_dir, f"real_robot_img_goal_{timestamp}")
    logger = Logger(log_dir)

    # ── Pygame / HitL ─────────────────────────────────────────────────────────
    if FLAGS.enable_hitl:
        pygame.init()
        pygame.font.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((640, 480))
            pygame.display.set_caption("DrQ Image-Goal Nav | HitL")
        clock = pygame.time.Clock()
        human = HumanController(env.action_space.low, env.action_space.high)
    else:
        clock = None
        human = None

    # ── Background training thread ────────────────────────────────────────────
    batch_queue = deque(maxlen=600)
    episode_event = threading.Event()
    stop_event = threading.Event()
    update_info_box = [{}]

    n_expert = int(FLAGS.batch_size * FLAGS.expert_sample_ratio)
    n_online = FLAGS.batch_size - n_expert
    online_iter = replay_buffer.get_iterator(
        sample_args={"batch_size": n_online if n_online > 0 else FLAGS.batch_size}
    )

    # Simple batch generator: if expert buffer exists, mix it in
    def _batch_gen():
        while True:
            batch = next(online_iter)
            if expert_buf is not None and n_expert > 0 and expert_buf._size >= n_expert:
                expert_batch = expert_buf.sample(n_expert)
                import jax
                batch = jax.tree_util.tree_map(
                    lambda o, e: np.concatenate([np.asarray(o), np.asarray(e)], axis=0),
                    batch, expert_batch,
                )
            yield batch

    batch_gen = _batch_gen()

    train_thread = threading.Thread(
        target=_train_thread,
        args=(agent, batch_queue, episode_event, stop_event, update_info_box, 500, 4),
        daemon=True,
        name="train",
    )
    train_thread.start()
    print("[Thread] Gradient-update thread started.\n")

    saver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="saver")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("Training  —  hold A/D/W/S to override the agent at any time")
    if resume_step > 0:
        print(f"Resuming from step {resume_step:,} → target {FLAGS.max_steps:,}")
    print("=" * 70 + "\n")

    obs, info = env.reset()
    done = False
    episode_count = 0
    ep_steps = 0
    ep_reward = 0.0
    best_return = -float("inf")
    human_steps = 0
    step = resume_step

    # Episode buffer: stores transitions *without* inserting to replay_buffer yet
    episode_buffer = []

    # Running stats
    episode_successes = deque(maxlen=100)

    try:
        for i in tqdm.tqdm(range(resume_step, FLAGS.max_steps), smoothing=0.1, disable=not FLAGS.tqdm):
            step = i
            if clock is not None:
                clock.tick(30)

            # ── HitL input handling ───────────────────────────────────────────
            if FLAGS.enable_hitl and human is not None:
                quit_req, reset_req, tog_pause = human.process_events()
                if quit_req:
                    print("\n[HitL] Quit requested.")
                    break
                if tog_pause:
                    human.paused = not human.paused
                    print(f"[HitL] Human pause → {'ON' if human.paused else 'OFF'}")

                human_action, human_active = human.read()
                use_human = human_active or human.paused

                if use_human:
                    action = human_action
                    human_steps += 1
                else:
                    action = agent.sample_actions(obs)
                    noise = ou_noise()
                    action = np.clip(action + noise, env.action_space.low, env.action_space.high)
            else:
                reset_req = False
                human_active = False
                if step < FLAGS.start_training:
                    action = env.action_space.sample()
                else:
                    action = agent.sample_actions(obs)
                    noise = ou_noise()
                    action = np.clip(action + noise, env.action_space.low, env.action_space.high)

            # ── Environment step ────────────────────────────────────────────────
            if reset_req:
                next_obs, reward, terminated, truncated, next_info = obs, -100.0, True, False, info
                env.step(np.array([0.0, 0.0]))
            else:
                next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            ep_steps += 1
            ep_reward += float(reward)

            # ── Episode buffer (masked goal during collection) ────────────────
            # We mask the goal because during collection the agent is exploring
            # without a known target.  The true episode goal is stored separately
            # for the 30 % non-HER samples.
            masked_obs = _mask_goal_in_obs(obs, feature_dim)
            masked_next_obs = _mask_goal_in_obs(next_obs, feature_dim)

            episode_buffer.append({
                "obs": masked_obs,
                "action": action.copy(),
                "reward": float(reward),
                "next_obs": masked_next_obs,
                "terminated": bool(terminated),
                # Original (unmasked) goal features — used for the 30 % non-HER
                "episode_goal_features": next_obs.get("goal_features", np.zeros(feature_dim)).copy(),
                # Current-frame features for HER relabeling
                "current_features": next_obs["pixels"][-feature_dim:].copy(),
            })

            obs = next_obs
            info = next_info

            # ── Feed batches to background thread ───────────────────────────────
            if step >= FLAGS.start_training and len(batch_queue) < batch_queue.maxlen:
                batch_queue.append(next(batch_gen))

            # ── Episode end ─────────────────────────────────────────────────────
            if done:
                replay_buffer.end_episode()

                # ── Commit episode buffer to replay buffer ────────────────────
                ep_len = len(episode_buffer)
                if ep_len > 0:
                    her_count = int(ep_len * her_ratio)
                    # Randomly choose which indices get HER
                    her_indices = set(random.sample(range(ep_len), min(her_count, ep_len)))

                    for idx, transition in enumerate(episode_buffer):
                        if idx in her_indices and ep_len > 1:
                            # ── HER relabeling ──────────────────────────────
                            # Sample a future state as the virtual goal
                            future_idx = random.randint(idx + 1, ep_len - 1)
                            future_t = episode_buffer[future_idx]
                            new_goal_features = future_t["current_features"]

                            her_obs = _set_goal_in_obs(transition["obs"], new_goal_features)
                            her_next_obs = _set_goal_in_obs(transition["next_obs"], new_goal_features)

                            # Sparse HER reward
                            her_reward = -1.0
                            feat_dist = np.linalg.norm(
                                transition["current_features"] - new_goal_features
                            )
                            goal_reached = feat_dist < her_goal_threshold
                            if goal_reached:
                                her_reward += k_goal

                            replay_buffer.insert(dict(
                                observations=her_obs,
                                actions=transition["action"],
                                rewards=her_reward,
                                next_observations=her_next_obs,
                                masks=np.float32(0.0 if goal_reached else 1.0),
                                dones=bool(goal_reached),
                            ))
                        else:
                            # ── Non-HER: use the sampled episode goal ──────
                            ep_goal = transition["episode_goal_features"]
                            final_obs = _set_goal_in_obs(transition["obs"], ep_goal)
                            final_next_obs = _set_goal_in_obs(transition["next_obs"], ep_goal)

                            replay_buffer.insert(dict(
                                observations=final_obs,
                                actions=transition["action"],
                                rewards=transition["reward"],
                                next_observations=final_next_obs,
                                masks=np.float32(1.0 - transition["terminated"]),
                                dones=transition["terminated"],
                            ))

                episode_buffer = []

                # Logging
                success = info.get("feature_distance", float("inf")) < goal_threshold if terminated else False
                episode_successes.append(float(success))

                if "episode" in info:
                    ep_r = info["episode"]["r"]
                    logger.log_episode({
                        "return": ep_r,
                        "length": info["episode"]["l"],
                        "distance": info.get("distance", 0.0),
                        "human_steps": human_steps,
                    }, step)
                    if ep_r > best_return:
                        best_return = ep_r

                episode_count += 1
                obs, info = env.reset()
                ou_noise.reset()
                if human is not None:
                    human.reset_controls()

                ep_steps = 0
                ep_reward = 0.0
                done = False

                # Signal background thread to run updates
                if step >= FLAGS.start_training:
                    episode_event.set()

                if update_info_box[0]:
                    logger.log_training(update_info_box[0], step)
                    logger.print_status(step, FLAGS.max_steps)
                    print(f"  human={human_steps}  buf={replay_buffer._size:,}")

            # ── HUD ───────────────────────────────────────────────────────────
            if FLAGS.enable_hitl:
                _draw_hud(
                    pygame.display.get_surface(),
                    float(action[0]), float(action[1]),
                    step, episode_count, ep_steps, ep_reward,
                    human_active, human.paused if human else False,
                    replay_buffer._size, FLAGS.replay_buffer_size,
                )

            # ── Checkpoint ──────────────────────────────────────────────────────
            if step % FLAGS.checkpoint_interval == 0 and step >= FLAGS.start_training:
                ck_path = os.path.join(POLICY_FOLDER, f"checkpoint_{step}")
                saver.submit(save_checkpoint, agent, replay_buffer, ck_path, step)
                saver.submit(goal_pool.save)
                print(f"[Checkpoint] Saved at step {step:,}")

    except KeyboardInterrupt:
        print("\n[Train] Interrupted.")

    finally:
        stop_event.set()
        episode_event.set()
        train_thread.join(timeout=10)
        saver.shutdown(wait=True)

        # Final saves
        final_dir = os.path.join(POLICY_FOLDER, "final")
        save_checkpoint(agent, replay_buffer, final_dir, step)
        goal_pool.save()
        print(f"\n[Checkpoint] Final save → {final_dir}")

        print("\n" + "=" * 70)
        print("Training complete")
        print(f"  Steps       : {step:,}")
        print(f"  Episodes    : {episode_count}")
        print(f"  Human steps : {human_steps:,}  (this session)")
        print(f"  Buffer size : {replay_buffer._size:,}")
        print(f"  Best return : {best_return:.2f}")
        print(f"  Checkpoints : {POLICY_FOLDER}/")
        print("=" * 70 + "\n")

        logger.close()
        env.close()


if __name__ == "__main__":
    app.run(main)
