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

Threading
---------
  Main thread  — env stepping, HitL input, HUD, episode bookkeeping,
                 batch pre-sampling
  Train thread — gradient updates ONLY (runs concurrently with env)

  At each episode end main sets an Event; the train thread wakes,
  drains the pre-filled batch_queue, runs all gradient updates, then
  sleeps again.  Checkpoint/teleop saves are also offloaded via a
  ThreadPoolExecutor so disk I/O never stalls the main loop.

Controls (real env window):
  W / Up     – throttle up       A / Left  – steer left
  S / Down   – throttle down     D / Right – steer right
  SPACE      – zero throttle     T         – toggle full human-pause
  R          – reset episode     Q / Esc   – quit & save
"""

import random
import os
import re
import pickle
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

flags.DEFINE_string(
    "pretrained_checkpoint",
    "/home/kojogyaase/Projects/Research/recovery-from-failure/pretrained_policy/step_520000",
    "Path to pre-trained checkpoint (used only when robot_policy has no prior runs).",
)
flags.DEFINE_string(
    "teleop_buffer_path",
    "/home/kojogyaase/Projects/Research/recovery-from-failure/teleop_buffer.pkl",
    "Path to save/resume teleop (human) transitions.",
)
flags.DEFINE_float(
    "expert_sample_ratio", 0.25,
    "Fraction of each mixed batch drawn from the expert teleop buffer.",
)
flags.DEFINE_float("steer_step",          0.05, "Steering increment per keypress.")
flags.DEFINE_float("throttle_step",       0.02, "Throttle increment per keypress.")
flags.DEFINE_integer("teleop_save_every", 500,  "Auto-save teleop buffer every N human steps.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/drq_robot.py",
    "Training hyperparameter config.",
    lock_config=False,
)

POLICY_FOLDER = "robot_policy"


# ═══════════════════════════════════════════════════════════════════════════════
# Resume helpers
# ═══════════════════════════════════════════════════════════════════════════════

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
    return (os.path.abspath(best_path), best_step) if best_path else (None, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Teleop buffer helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_teleop_buffer(path, obs_space, act_space, seed, capacity):
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
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({
            "data":         buf.dataset_dict,
            "insert_index": buf._insert_index,
            "size":         buf._size,
        }, f)
    os.replace(tmp, path)   # atomic — never corrupts on crash
    print(f"[Teleop] Saved {buf._size:,} transitions → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Mixed-batch generator
# ═══════════════════════════════════════════════════════════════════════════════

def _mixed_batch_gen(online_iter, teleop_buf, batch_size, expert_ratio):
    n_expert = int(batch_size * expert_ratio) if teleop_buf is not None else 0
    n_online = batch_size - n_expert
    while True:
        online_batch = next(online_iter) if n_online > 0 else None
        expert_batch = (
            teleop_buf.sample(n_expert)
            if n_expert > 0 and teleop_buf is not None and teleop_buf._size >= n_expert
            else None
        )
        if online_batch is None:
            yield expert_batch
        elif expert_batch is None:
            yield online_batch
        else:
            yield jax.tree_util.tree_map(
                lambda o, e: np.concatenate([np.asarray(o), np.asarray(e)], axis=0),
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
# Human controller (HitL)
# ═══════════════════════════════════════════════════════════════════════════════

class HumanController:
    def __init__(self, action_low, action_high):
        self.low      = action_low
        self.high     = action_high
        self.steering  = 0.0
        self.throttle  = 0.130
        self.paused    = False

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
                    self.throttle = 0.130
        return quit_req, reset_req, tog_pause

    def read(self):
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
              ep_reward, human_active, paused, teleop_size, buf_cap):
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

    bw, bh = 180, 8
    bx, by = screen.get_width() - bw - 10, y0 + 20
    cx = bx + bw // 2
    pygame.draw.rect(screen, (60, 60, 60), (bx, by, bw, bh))
    fw  = int(abs(steering) * (bw // 2))
    col = (0, 200, 100) if steering >= 0 else (220, 80, 0)
    pygame.draw.rect(screen, col, (cx if steering >= 0 else cx - fw, by, fw, bh))
    pygame.draw.line(screen, (255, 255, 255), (cx, by), (cx, by + bh), 1)
    pygame.display.flip()


# ═══════════════════════════════════════════════════════════════════════════════
# Gradient-update thread  (the ONLY thing that runs in a separate thread)
# ═══════════════════════════════════════════════════════════════════════════════

def _train_thread(
    agent,
    batch_queue:     deque,
    expert_iter,
    episode_event:   threading.Event,
    stop_event:      threading.Event,
    update_info_box: list,       # [{}]  — single-slot out-param written by train
    n_updates:       int,
    utd_ratio:       int,
    teleop_buf:      ReplayBuffer,
):
    """
    Sleeps until main signals episode end, then runs all gradient updates.

    Protocol
    --------
    • Main appends pre-sampled batches to batch_queue every step.
    • At episode end main calls episode_event.set().
    • This thread wakes, runs n_updates gradient steps, writes the last
      update_info into update_info_box[0], then sleeps again.
    • stop_event tells the thread to exit cleanly.
    """
    while not stop_event.is_set():
        triggered = episode_event.wait(timeout=0.5)
        if stop_event.is_set():
            break
        if not triggered:
            continue
        episode_event.clear()

        info = {}
        for _ in range(n_updates):
            # Expert update — temperature FROZEN
            if expert_iter is not None and teleop_buf._size >= FLAGS.batch_size:
                agent.update(next(expert_iter), utd_ratio=utd_ratio)

            # Mixed online+expert update — temperature FREE
            if batch_queue:
                info = agent.update_with_temp(batch_queue.popleft(), utd_ratio=utd_ratio)

        if info:
            update_info_box[0] = info


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(_):
    print("\n" + "=" * 70)
    print("DrQ  Real Robot  |  Expert Buffer  |  Frozen Temp  |  HitL")
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

    # ── Replay buffers ────────────────────────────────────────────────────────
    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.replay_buffer_size)
    replay_buffer.seed(FLAGS.seed)

    teleop_buf, prior_teleop = _load_teleop_buffer(
        FLAGS.teleop_buffer_path,
        env.observation_space, env.action_space,
        FLAGS.seed, FLAGS.replay_buffer_size,
    )
    print(f"[Expert]  {teleop_buf._size:,} transitions  "
          f"(ratio={FLAGS.expert_sample_ratio:.0%}/batch)\n")

    # ── Iterators ─────────────────────────────────────────────────────────────
    n_expert = int(FLAGS.batch_size * FLAGS.expert_sample_ratio)
    n_online = FLAGS.batch_size - n_expert
    online_iter = replay_buffer.get_iterator(
        sample_args={"batch_size": n_online if n_online > 0 else FLAGS.batch_size}
    )
    expert_iter = (
        teleop_buf.get_iterator(sample_args={"batch_size": FLAGS.batch_size})
        if teleop_buf._size >= FLAGS.batch_size
        else None
    )
    batch_gen = _mixed_batch_gen(
        online_iter, teleop_buf, FLAGS.batch_size, FLAGS.expert_sample_ratio)

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
    clock = pygame.time.Clock()
    human = HumanController(env.action_space.low, env.action_space.high)

    # ── Train thread ──────────────────────────────────────────────────────────
    # deque is thread-safe for append (main) / popleft (train)
    batch_queue      = deque(maxlen=600)   # 500 updates + headroom
    episode_event    = threading.Event()
    stop_event       = threading.Event()
    update_info_box  = [{}]                # out-param: train writes, main reads

    train_thread = threading.Thread(
        target=_train_thread,
        args=(
            agent,
            batch_queue,
            expert_iter,
            episode_event,
            stop_event,
            update_info_box,
            500,    # n_updates_per_episode
            4,      # utd_ratio
            teleop_buf,
        ),
        daemon=True,
        name="train",
    )
    train_thread.start()
    print("[Thread] Gradient-update thread started.\n")

    # Async saver so checkpoint/teleop disk writes never stall main
    saver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="saver")

    # ── Main loop ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Training  —  hold A/D/W/S to override the agent at any time")
    if resume_step > 0:
        print(f"Resuming from step {resume_step:,} → target {FLAGS.max_steps:,}")
    print("=" * 70 + "\n")

    observation, info = env.reset()
    done          = False
    episode_count = 0
    ep_steps      = 0
    ep_reward     = 0.0
    best_return   = -float("inf")
    human_steps   = 0
    step          = resume_step

    try:
        for i in tqdm.tqdm(range(resume_step, FLAGS.max_steps),
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
            done = terminated or truncated

            transition = dict(
                observations=observation,    actions=action,
                rewards=float(reward),       masks=0.0 if done else 1.0,
                dones=bool(done),            next_observations=next_observation,
            )
            replay_buffer.insert(transition)
            observation = next_observation
            ep_steps   += 1
            ep_reward  += float(reward)

            # ── Pre-sample batch for train thread ─────────────────────────────
            # Done every step so the queue is full by episode end.
            if step >= FLAGS.start_training and len(batch_queue) < batch_queue.maxlen:
                batch_queue.append(next(batch_gen))

            # ── Episode end ───────────────────────────────────────────────────
            if done:
                episode_count += 1
                observation, info = env.reset()
                noise.reset()
                human.reset_controls()

                if "episode" in info:
                    ep_r = info["episode"]["r"]
                    logger.log_episode({
                        "return":      ep_r,
                        "length":      info["episode"]["l"],
                        "distance":    info.get("distance", 0),
                        "human_steps": human_steps,
                    }, step)
                    if ep_r > best_return:
                        best_return = ep_r

                ep_steps  = 0
                ep_reward = 0.0
                done      = False

                # Wake the train thread
                if step >= FLAGS.start_training:
                    episode_event.set()

                # Log whatever the train thread finished last episode
                if update_info_box[0]:
                    logger.log_training(update_info_box[0], step)
                    logger.print_status(step, FLAGS.max_steps)
                    print(f"  human={human_steps}  "
                          f"teleop={teleop_buf._size:,}  "
                          f"online={replay_buffer._size:,}")

            # ── HUD ───────────────────────────────────────────────────────────
            _draw_hud(
                pygame.display.get_surface(),
                float(action[0]), float(action[1]),
                step, episode_count, ep_steps, ep_reward,
                human_active, human.paused,
                teleop_buf._size, FLAGS.replay_buffer_size,
            )

            # ── Checkpoint (async so disk I/O doesn't stall main) ─────────────
            if step % FLAGS.checkpoint_interval == 0 and step >= FLAGS.start_training:
                ck_path = os.path.join(POLICY_FOLDER, f"step_{step}")
                saver.submit(save_checkpoint, agent, replay_buffer, ck_path, step)
                saver.submit(_save_teleop_buffer, teleop_buf, "online_teleop.pkl")

    except KeyboardInterrupt:
        print("\n[Train] Interrupted.")

    finally:
        stop_event.set()
        episode_event.set()       # unblock train thread if waiting
        train_thread.join(timeout=10)
        saver.shutdown(wait=True)

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