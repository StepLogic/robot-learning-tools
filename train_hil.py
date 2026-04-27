#!/usr/bin/env python
"""
DrQ Training — Real Robot Only
================================
Combines:
  • MobileNetV3 visual feature extraction (PyTorch)
  • Frame stacking + action/IMU history (StackingWrapper)
  • Expert teleop buffer (teleop_buffer.pkl) — mixed into every batch
  • Frozen temperature (log_α never updated)
  • Human-in-the-Loop (HitL) override — hold A/D/W/S to take control;
    human transitions go to BOTH online buffer AND teleop_buffer.pkl
  • Gradient updates run in a dedicated background thread
  • K key — award agent a completion bonus for the current transition

Controls (real env window):
  W / Up     – throttle up       A / Left  – steer left
  S / Down   – throttle down     D / Right – steer right
  SPACE      – zero throttle     T         – toggle full human-pause
  K          – award completion bonus to current transition
  R          – reset episode     Q / Esc   – quit & save
"""

import random
import os
import re
import pickle
import queue
import threading
from datetime import datetime
import tqdm
import uuid

import flax
import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

import torch
import pygame

import gymnasium as gym
from jaxrl2.agents import DrQLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

from racer_imu_env import RacerEnv, StackingWrapper, RewardWrapper
from wrappers import (
    EnvCompatibility,
    Logger,
    MobileNetFeatureWrapper,
    load_checkpoint,
    save_checkpoint,
)

flax.config.update("flax_use_orbax_checkpointing", True)

FLAGS = flags.FLAGS

# ── Flags ─────────────────────────────────────────────────────────────────────
flags.DEFINE_string("env_name",        "donkey-warehouse-v0", "Environment name.")
flags.DEFINE_string("sim",             "sim_path",            "Path to unity simulator.")
flags.DEFINE_integer("port",           9091,                  "Port for TCP.")
flags.DEFINE_string("save_dir",        "./logs/",             "Tensorboard log dir.")
flags.DEFINE_integer("seed",           42,                    "Random seed.")
flags.DEFINE_integer("log_interval",   1000,                  "Logging interval.")
flags.DEFINE_integer("eval_interval",  int(50000),            "Eval interval.")
flags.DEFINE_integer("checkpoint_interval", 1000,             "Checkpoint interval.")
flags.DEFINE_integer("batch_size",     64,                    "Batch size.")
flags.DEFINE_integer("max_steps",      int(1e6),              "Total training steps.")
flags.DEFINE_integer("start_training", int(1e3),              "Steps before updates begin.")
flags.DEFINE_integer("replay_buffer_size", int(1e4),          "Online buffer capacity.")
flags.DEFINE_boolean("tqdm",           True,                  "Show tqdm bar.")
flags.DEFINE_integer("frame_stack",    3,                     "Frame stack depth.")
flags.DEFINE_integer("mobilenet_blocks",     4,               "MobileNetV3 blocks.")
flags.DEFINE_integer("mobilenet_input_size", 84,              "MobileNetV3 input size.")

# Pre-trained checkpoint (used only when no resume checkpoint is found)
flags.DEFINE_string(
    "pretrained_checkpoint",
    "/home/kojogyaase/Projects/Research/recovery-from-failure/pretrained_policy/step_520000",
    "Path to pre-trained checkpoint (used only when robot_policy has no prior runs).",
)

# Expert / teleop buffer
flags.DEFINE_string(
    "teleop_buffer_path",
    "/home/kojogyaase/Projects/Research/recovery-from-failure/teleop_buffer.pkl",
    "Path to save/resume teleop (human) transitions.",
)
flags.DEFINE_float(
    "expert_sample_ratio", 0.25,
    "Fraction of each mixed batch drawn from the expert teleop buffer.",
)

# HitL controls
flags.DEFINE_float("steer_step",          0.05, "Steering increment per keypress.")
flags.DEFINE_float("throttle_step",       0.02, "Throttle increment per keypress.")
flags.DEFINE_integer("teleop_save_every", 500,  "Auto-save teleop buffer every N human steps.")

# Human completion reward
flags.DEFINE_float(
    "completion_reward", 100.0,
    "Bonus reward added to the current transition when K is pressed.",
)

# Update worker queue
flags.DEFINE_integer(
    "update_queue_maxsize", 128,
    "Max items in the gradient-update queue before the main loop blocks.",
)

config_flags.DEFINE_config_file(
    "config",
    "./configs/drq_robot.py",
    "Training hyperparameter config.",
    lock_config=False,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Resume helpers
# ═══════════════════════════════════════════════════════════════════════════════

POLICY_FOLDER = "robot_policy"


def _find_latest_checkpoint(folder: str) -> tuple[str | None, int]:
    if not os.path.isdir(folder):
        return None, 0

    best_path, best_step = None, 0
    for name in os.listdir(folder):
        m = re.fullmatch(r"step_(\d+)", name)
        if m:
            n = int(m.group(1))
            if n > best_step:
                best_step = n
                best_path = os.path.join(folder, name)

    return os.path.abspath(best_path), best_step


# ═══════════════════════════════════════════════════════════════════════════════
# Teleop buffer helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_teleop_buffer(path: str, obs_space, act_space,
                        seed: int, capacity: int) -> tuple[ReplayBuffer, int]:
    buf = ReplayBuffer(obs_space, act_space, capacity)
    buf.seed(seed)

    if not os.path.exists(path):
        print(f"[Teleop] No existing file at {path} — starting fresh.")
        return buf, 0

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, ReplayBuffer):
            buf.dataset_dict  = data.dataset_dict
            buf._insert_index = data._insert_index
            buf._size         = data._size
            prior = data._size
        elif isinstance(data, dict) and "data" in data:
            buf.dataset_dict  = data["data"]
            buf._insert_index = data["insert_index"]
            buf._size         = data["size"]
            prior = data["size"]
        else:
            transitions = data if isinstance(data, list) else [data]
            for t in transitions:
                try:
                    buf.insert(t)
                except Exception:
                    pass
            prior = buf._size

        print(f"[Teleop] ✓ Loaded {prior:,} transitions from {path}")
        return buf, prior

    except Exception as e:
        print(f"[Teleop] ✗ Load failed ({e}) — starting fresh.")
        return buf, 0


def _save_teleop_buffer(buf: ReplayBuffer, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "data":         buf.dataset_dict,
            "insert_index": buf._insert_index,
            "size":         buf._size,
        }, f)
    print(f"[Teleop] Saved {buf._size:,} transitions → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Mixed-batch generator  (online + expert)
# ═══════════════════════════════════════════════════════════════════════════════

def _mixed_batch_gen(online_iter, teleop_buf: ReplayBuffer | None,
                     batch_size: int, expert_ratio: float):
    n_expert = int(batch_size * expert_ratio) if teleop_buf is not None else 0
    n_online = batch_size - n_expert

    while True:
        online_batch = next(online_iter) if n_online > 0 else None
        expert_batch = (
            teleop_buf.sample(n_expert)
            if n_expert > 0
               and teleop_buf is not None
               and teleop_buf._size >= n_expert
            else None
        )

        if online_batch is None:
            yield expert_batch
        elif expert_batch is None:
            yield online_batch
        else:
            yield jax.tree_util.tree_map(
                lambda o, e: np.concatenate(
                    [np.asarray(o), np.asarray(e)], axis=0),
                online_batch, expert_batch,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Freeze temperature
# ═══════════════════════════════════════════════════════════════════════════════

def _freeze_temperature(agent):
    try:
        agent._frozen_temp_state = agent._temp
        _orig = agent.update

        def _update_frozen(batch, **kw):
            info = _orig(batch, **kw)
            agent._temp = agent._frozen_temp_state
            return info

        def _update_with_temp(batch, **kw):
            info = _orig(batch, **kw)
            agent._frozen_temp_state = agent._temp
            return info

        agent.update           = _update_frozen
        agent.update_with_temp = _update_with_temp

        print("[Temperature] Frozen for online updates  (agent.update).")
        print("[Temperature] Free   for expert updates  (agent.update_with_temp).")
    except AttributeError as e:
        print(f"[Temperature] ⚠ Could not patch: {e}")
        agent.update_with_temp = agent.update
    return agent


# ═══════════════════════════════════════════════════════════════════════════════
# Background gradient-update worker
# ═══════════════════════════════════════════════════════════════════════════════

# Sentinel that tells the worker thread to exit cleanly.
_STOP_SENTINEL = object()


def _update_worker(agent, update_queue: queue.Queue, info_box: list, lock: threading.Lock):
    """
    Runs in a daemon thread.  Pulls (kind, batch, utd_ratio) tuples from
    *update_queue* and calls the appropriate agent.update* method.

    kind == "expert"  → agent.update        (frozen temperature)
    kind == "online"  → agent.update_with_temp

    The most recent update_info dict is written to info_box[0] under *lock*.
    """
    while True:
        item = update_queue.get()
        if item is _STOP_SENTINEL:
            update_queue.task_done()
            break

        kind, batch, utd_ratio = item
        try:
            if kind == "expert":
                info = agent.update(batch, utd_ratio=utd_ratio)
            else:  # "online"
                info = agent.update_with_temp(batch, utd_ratio=utd_ratio)
            with lock:
                info_box[0] = info
        except Exception as exc:
            print(f"[UpdateWorker] ⚠ {exc}")
        finally:
            update_queue.task_done()


# ═══════════════════════════════════════════════════════════════════════════════
# Human controller (HitL)
# ═══════════════════════════════════════════════════════════════════════════════

class HumanController:
    """Reads pygame key state every step; returns (action, human_active)."""

    def __init__(self, action_low: np.ndarray, action_high: np.ndarray):
        self.low       = action_low
        self.high      = action_high
        self.steering  = 0.0
        self.throttle  = 0.130
        self.paused    = False

        # Completion-reward flag — set by process_events, consumed by main loop
        self.completion_reward_requested = False

    def process_events(self) -> tuple[bool, bool, bool]:
        """Drain pygame event queue. Returns (quit, reset, toggle_pause)."""
        quit_req = reset_req = tog_pause = False
        self.completion_reward_requested = False
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
                    self.throttle = 0.130
                if event.key == pygame.K_k:
                    self.completion_reward_requested = True
        return quit_req, reset_req, tog_pause

    def read(self) -> tuple[np.ndarray, bool]:
        keys = pygame.key.get_pressed()
        steer_key = throttle_key = False

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.steering = max(self.low[0],  self.steering - FLAGS.steer_step)
            steer_key = True
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.steering = min(self.high[0], self.steering + FLAGS.steer_step)
            steer_key = True

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.throttle = min(self.high[1], self.throttle + FLAGS.throttle_step)
            throttle_key = True
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.throttle = max(self.low[1],  self.throttle - FLAGS.throttle_step)
            throttle_key = True

        human_active = steer_key or throttle_key or self.paused
        return np.array([self.steering, self.throttle], dtype=np.float32), human_active

    def reset_controls(self):
        self.steering = 0.0
        self.throttle = 0.130


# ═══════════════════════════════════════════════════════════════════════════════
# HUD overlay
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_hud(screen, steering, throttle, step, ep, ep_steps,
              ep_reward, human_active, paused, teleop_size, buf_cap,
              reward_flash: bool = False):
    if screen is None:
        return
    if not pygame.font.get_init():
        pygame.font.init()

    font = pygame.font.SysFont("monospace", 15, bold=True)

    def txt(msg, x, y, color=(255, 255, 0)):
        screen.blit(font.render(msg, True, color), (x, y))

    bar = pygame.Surface((screen.get_width(), 108), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 170))
    screen.blit(bar, (0, screen.get_height() - 108))
    y0 = screen.get_height() - 106

    if paused:
        lbl, col = "HUMAN [PAUSED]", (255,  80,  80)
    elif human_active:
        lbl, col = "HUMAN OVERRIDE", ( 80, 255,  80)
    else:
        lbl, col = "AGENT",          ( 80, 180, 255)

    txt(f"[ {lbl} ]", screen.get_width() - 220, y0, color=col)
    txt(f"Steer {steering:+.3f}   Throttle {throttle:+.3f}", 10, y0)
    txt(f"Step {step:>7d}   Ep {ep:>4d}   Ep-step {ep_steps:>4d}", 10, y0 + 18)
    txt(f"Ep reward {ep_reward:+.2f}   Teleop {teleop_size:>6d}/{buf_cap}", 10, y0 + 36)
    txt("W/S=throttle  A/D=steer  SPC=coast  T=pause  R=reset  Q=quit",
        10, y0 + 54, color=(190, 190, 190))
    txt("K=completion bonus",
        10, y0 + 72, color=(190, 190, 190))

    # Completion-reward flash
    if reward_flash:
        txt(f"[+REWARD  +{FLAGS.completion_reward:.1f}]",
            screen.get_width() // 2 - 80, y0, color=(255, 220, 0))

    # Steering bar
    bw, bh = 180, 8
    bx, by = screen.get_width() - bw - 10, y0 + 20
    cx = bx + bw // 2
    pygame.draw.rect(screen, (60, 60, 60), (bx, by, bw, bh))
    fw  = int(abs(steering) * (bw // 2))
    col = (0, 200, 100) if steering >= 0 else (220, 80, 0)
    pygame.draw.rect(screen, col,
                     (cx if steering >= 0 else cx - fw, by, fw, bh))
    pygame.draw.line(screen, (255, 255, 255), (cx, by), (cx, by + bh), 1)

    pygame.display.flip()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(_):
    print("\n" + "=" * 70)
    print("DrQ  Real Robot  |  Expert Buffer  |  Frozen Temp  |  HitL")
    print("     Threaded gradient updates  |  K = completion bonus")
    print("=" * 70 + "\n")

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Environment ───────────────────────────────────────────────────────────
    print("Building environment…")
    env = RacerEnv(render_mode="human")
    env = StackingWrapper(env, num_stack=FLAGS.frame_stack)
    env = RewardWrapper(env)
    env = MobileNetFeatureWrapper(
        env,
        device=device,
        num_blocks=FLAGS.mobilenet_blocks,
        input_size=FLAGS.mobilenet_input_size,
    )
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=3000)

    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}\n")

    # ── Agent ─────────────────────────────────────────────────────────────────
    kwargs = dict(FLAGS.config) if FLAGS.config else {}
    agent  = DrQLearner(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample(),
        **kwargs,
    )

    # ── Resume or load pre-trained ────────────────────────────────────────────
    os.makedirs(POLICY_FOLDER, exist_ok=True)
    resume_path, resume_step = _find_latest_checkpoint(POLICY_FOLDER)
    if resume_path is not None:
        agent = load_checkpoint(agent, resume_path)
        print(f"[Checkpoint] ✓ Resumed from {resume_path}  (step {resume_step:,})")
    else:
        agent = load_checkpoint(agent, FLAGS.pretrained_checkpoint)
        print(f"[Checkpoint] ✓ Loaded pre-trained weights from {FLAGS.pretrained_checkpoint}")
        resume_step = 0

    agent = _freeze_temperature(agent)

    # ── Online replay buffer ──────────────────────────────────────────────────
    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.replay_buffer_size)
    replay_buffer.seed(FLAGS.seed)

    # ── Teleop / expert buffer ────────────────────────────────────────────────
    teleop_buf, prior_teleop = _load_teleop_buffer(
        FLAGS.teleop_buffer_path,
        env.observation_space,
        env.action_space,
        FLAGS.seed,
        FLAGS.replay_buffer_size,
    )
    print(f"[Expert]  {teleop_buf._size:,} transitions  "
          f"(ratio={FLAGS.expert_sample_ratio:.0%}/batch)\n")

    # ── Iterators ─────────────────────────────────────────────────────────────
    # n_expert = int(FLAGS.batch_size * FLAGS.expert_sample_ratio)
    # n_online = FLAGS.batch_size - n_expert
    online_iter = replay_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size}
    )
    expert_iter = (
        teleop_buf.get_iterator(sample_args={"batch_size": FLAGS.batch_size})
        if teleop_buf._size >= FLAGS.batch_size
        else None
    )
    batch_gen = _mixed_batch_gen(
        online_iter, teleop_buf, FLAGS.batch_size, FLAGS.expert_sample_ratio)

    # ── Background gradient-update thread ─────────────────────────────────────
    update_queue = queue.Queue(maxsize=FLAGS.update_queue_maxsize)
    update_info_box  = [{}]           # shared; worker writes latest info here
    update_info_lock = threading.Lock()

    worker_thread = threading.Thread(
        target=_update_worker,
        args=(agent, update_queue, update_info_box, update_info_lock),
        daemon=True,
        name="GradientWorker",
    )
    worker_thread.start()
    print("[UpdateWorker] Background gradient thread started.\n")

    # Convenience: enqueue a gradient update without blocking the env loop.
    def _enqueue_update(kind: str, batch, utd_ratio: int = 4):
        try:
            update_queue.put_nowait((kind, batch, utd_ratio))
        except queue.Full:
            pass   # drop silently when the queue is saturated

    # ── Noise / logging ───────────────────────────────────────────────────────
    noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(env.action_space.shape[0]),
        sigma=0.2 * np.ones(env.action_space.shape[0]),
    )
    logger = Logger(log_dir=FLAGS.save_dir)

    # ── Pygame / HitL ─────────────────────────────────────────────────────────
    pygame.init()
    pygame.font.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((640, 480))
        pygame.display.set_caption("DrQ HitL Training")
    clock       = pygame.time.Clock()
    human       = HumanController(env.action_space.low, env.action_space.high)
    human_steps = 0

    # HUD flash timer (steps to show [+REWARD] banner)
    reward_flash_ttl = 0

    # ── Main training loop ────────────────────────────────────────────────────
    print("=" * 70)
    print("Training  —  hold A/D/W/S to override | K = completion bonus")
    if resume_step > 0:
        print(f"Resuming from step {resume_step:,} → target {FLAGS.max_steps:,}")
    print("=" * 70 + "\n")

    observation, info = env.reset()
    done          = False
    episode_count = 0
    ep_steps      = 0
    ep_reward     = 0.0
    best_return   = -float("inf")
    step          = resume_step
    # Holds the last transition dict so K can modify its reward before insert
    pending_transition: dict | None = None

    try:
        for i in tqdm.tqdm(range(resume_step, FLAGS.max_steps, 1),
                           smoothing=0.1, disable=not FLAGS.tqdm):
            step = i
            clock.tick(30)

            # ── HitL events ───────────────────────────────────────────────────
            quit_req, reset_req, tog_pause = human.process_events()
            if quit_req:
                print("\n[HitL] Quit requested.")
                break
            if tog_pause:
                human.paused = not human.paused
                print(f"[HitL] Human pause → {'ON' if human.paused else 'OFF'}")

            # ── K: completion reward on the *previous* transition ─────────────
            if human.completion_reward_requested and pending_transition is not None:
                bonus = float(FLAGS.completion_reward)
                pending_transition["rewards"] += bonus
                ep_reward                     += bonus
                reward_flash_ttl               = 15   # show HUD banner for 15 steps
                print(f"[HitL] Completion bonus +{bonus:.1f} applied at step {step:,}")

            # ── Insert pending transition (now that K may have modified it) ───
            if pending_transition is not None:
                replay_buffer.insert(pending_transition)
                if use_human_prev:                      # only human steps → teleop
                    teleop_buf.insert(pending_transition)
                    human_steps += 1
                    if human_steps % FLAGS.teleop_save_every == 0:
                        _save_teleop_buffer(teleop_buf, FLAGS.teleop_buffer_path)
            pending_transition = None

            human_action, human_active = human.read()
            use_human = human_active or human.paused

            # ── Action ────────────────────────────────────────────────────────
            if use_human:
                action = human_action
            else:
                action = agent.sample_actions(observation)
                action = np.clip(action, env.action_space.low, env.action_space.high)

            # ── Step environment ──────────────────────────────────────────────
            if reset_req:
                next_observation, reward, terminated, truncated, info = \
                    observation, -100, True, False, {}
                env.step(np.array([0.0, 0.0]))
            else:
                next_observation, reward, terminated, truncated, info = \
                    env.step(action)
            done = terminated

            # Build transition but do NOT insert yet — K may augment reward
            pending_transition = dict(
                observations=observation,
                actions=action,
                rewards=float(reward),
                masks=0.0 if done else 1.0,
                dones=bool(done),
                next_observations=next_observation,
            )
            use_human_prev = use_human   # remember for insertion logic above

            observation  = next_observation
            ep_steps    += 1
            ep_reward   += float(reward)
            if expert_iter is not None and teleop_buf._size >= FLAGS.batch_size:
                _enqueue_update("expert", next(expert_iter), utd_ratio=4)
            if replay_buffer._size >= FLAGS.batch_size:
                _enqueue_update("online", next(batch_gen), utd_ratio=4)

            # ── Episode end ───────────────────────────────────────────────────
            if done or truncated:
                episode_count += 1
                observation, info = env.reset()
                noise.reset()
                human.reset_controls()
                ep_steps  = 0
                ep_reward = 0.0
                done      = False

                if "episode" in info:
                    ep_r = info["episode"]["r"]
                    logger.log_episode({
                        "return": ep_r,
                        "length": info["episode"]["l"],
                        "distance": info.get("distance", 0),
                        "human_steps": human_steps,
                    }, step)
                    if ep_r > best_return:
                        best_return = ep_r



                # Log the most recent info (written by background thread)
                with update_info_lock:
                    latest_info = update_info_box[0]
                if latest_info:
                    logger.log_training(latest_info, step)
                logger.print_status(step, FLAGS.max_steps)
                print(f"  human={human_steps}  "
                      f"teleop={teleop_buf._size:,}  "
                      f"online={replay_buffer._size:,}  "
                      f"q={update_queue.qsize()}")

            # ── HUD ───────────────────────────────────────────────────────────
            if reward_flash_ttl > 0:
                reward_flash_ttl -= 1
            _draw_hud(
                pygame.display.get_surface(),
                float(action[0]), float(action[1]),
                step, episode_count, ep_steps, ep_reward,
                human_active, human.paused,
                teleop_buf._size, FLAGS.replay_buffer_size,
                reward_flash=(reward_flash_ttl > 0),
            )

            # ── Checkpoint ────────────────────────────────────────────────────
            if step % FLAGS.checkpoint_interval == 0 and step >= FLAGS.start_training:
                save_checkpoint(
                    agent, replay_buffer,
                    os.path.join(POLICY_FOLDER, f"step_{step}"),
                    step,
                )
                _save_teleop_buffer(teleop_buf, "online_teleop.pkl")

    except KeyboardInterrupt:
        print("\n[Train] Interrupted.")

    finally:
        # Signal background worker to stop and wait for it to drain
        update_queue.put(_STOP_SENTINEL)
        worker_thread.join(timeout=30)
        print("[UpdateWorker] Background thread stopped.")

        save_checkpoint(agent, replay_buffer,
                        os.path.join(POLICY_FOLDER, "final"), step)
        _save_teleop_buffer(teleop_buf, FLAGS.teleop_buffer_path)

        print("\n" + "=" * 70)
        print("Training complete")
        print(f"  Steps       : {step:,}")
        print(f"  Episodes    : {episode_count}")
        print(f"  Human steps : {human_steps:,}  (this session)")
        print(f"  Teleop buf  : {teleop_buf._size:,}")
        print(f"  Best return : {best_return:.2f}")
        print(f"  Checkpoints : {POLICY_FOLDER}/")
        print("=" * 70 + "\n")

        logger.close()
        env.close()


if __name__ == "__main__":
    app.run(main)