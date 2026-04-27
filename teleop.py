#!/usr/bin/env python3
"""
Teleoperation data collector for Donkey Car.
Drives the robot with keyboard/joystick, stores transitions in a replay
buffer that is binary-compatible with the one used in train.py.

Controls (keyboard):
  W / Up     – increase throttle
  S / Down   – decrease throttle / brake
  A / Left   – steer left
  D / Right  – steer right
  SPACE      – zero throttle (coast)
  R          – reset episode
  Q / Esc    – quit and save

Usage:
  python teleop.py [--out teleop_buffer.pkl] [--max-steps 10000]
"""

import argparse
import os
import pickle
import sys
import time

import cv2
import numpy as np
import pygame
import torch
import gymnasium as gym

# ── Same wrappers the training script uses ────────────────────────────────────
from racer_imu_env import RacerEnv, StackingWrapper,RewardWrapper
from wrappers import MobileNetFeatureWrapper
from jaxrl2.data import ReplayBuffer

# ─────────────────────────────────────────────────────────────────────────────
# Config (mirrors train.py defaults)
# ─────────────────────────────────────────────────────────────────────────────
NUM_STACK        = 3
MOBILENET_BLOCKS = 4
MOBILENET_INPUT  = 84
FPS              = 30

# Steering / throttle step sizes per key-press
STEER_STEP    = 0.05
THROTTLE_STEP = 0.02
STEER_DECAY   = 0.85   # auto-centre when no left/right key held
THROTTLE_DECAY = 0.90  # auto-coast when no fwd/back key held


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_env(device: str) -> gym.Env:
    """Construct the exact same wrapper stack as train.py."""
    env = RacerEnv(render_mode="human")
    env = StackingWrapper(env, num_stack=NUM_STACK)
    env = RewardWrapper(env)
    env = MobileNetFeatureWrapper(
        env,
        device=device,
        num_blocks=MOBILENET_BLOCKS,
        input_size=MOBILENET_INPUT,
    )
    return env


def save_buffer(replay_buffer: ReplayBuffer, path: str, step: int):
    """Save replay buffer in the same format load_replay_buffer() expects."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    data = {
        "data":         replay_buffer.dataset_dict,
        "insert_index": replay_buffer._insert_index,
        "size":         replay_buffer._size,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n[teleop] Saved {replay_buffer._size} transitions → {path}")


def load_existing_buffer(replay_buffer: ReplayBuffer, path: str) -> int:
    """
    If *path* already exists, load it into replay_buffer and return the
    number of transitions loaded, otherwise return 0.
    """
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        replay_buffer.dataset_dict      = data["data"]
        replay_buffer._insert_index = data["insert_index"]
        replay_buffer._size         = data["size"]
        print(f"[teleop] Resumed existing buffer: {replay_buffer._size} transitions")
        return replay_buffer._size
    except Exception as e:
        print(f"[teleop] Could not load existing buffer ({e}), starting fresh.")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay (drawn onto the pygame surface already managed by RacerEnv)
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(screen: pygame.Surface,
             steering: float, throttle: float,
             step: int, ep: int, ep_steps: int,
             total_reward: float):
    font = pygame.font.SysFont("monospace", 16, bold=True)

    def txt(msg, x, y, color=(255, 255, 0)):
        screen.blit(font.render(msg, True, color), (x, y))

    # Semi-transparent bar at the bottom
    bar = pygame.Surface((screen.get_width(), 80), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 160))
    screen.blit(bar, (0, screen.get_height() - 80))

    y0 = screen.get_height() - 75
    txt(f"Steer  {steering:+.3f}   Throttle {throttle:+.3f}", 10, y0)
    txt(f"Step {step:>6d}   Ep {ep:>4d}   Ep-step {ep_steps:>4d}", 10, y0 + 20)
    txt(f"Ep reward {total_reward:+.2f}", 10, y0 + 40)
    txt("W/S=throttle  A/D=steer  SPC=coast  R=reset  Q=quit", 10, y0 + 58,
        color=(200, 200, 200))

    # Steering bar
    bar_w, bar_h = 200, 10
    bx = screen.get_width() - bar_w - 10
    by = y0
    pygame.draw.rect(screen, (80, 80, 80), (bx, by, bar_w, bar_h))
    centre = bx + bar_w // 2
    fill_w = int(abs(steering) * (bar_w // 2))
    if steering >= 0:
        pygame.draw.rect(screen, (0, 200, 100), (centre, by, fill_w, bar_h))
    else:
        pygame.draw.rect(screen, (200, 80, 0),  (centre - fill_w, by, fill_w, bar_h))
    pygame.draw.line(screen, (255, 255, 255), (centre, by), (centre, by + bar_h), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main teleop loop
# ─────────────────────────────────────────────────────────────────────────────

def teleop(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[teleop] Device: {device}")

    # ── Build env (no changes to any env class) ───────────────────────────────
    print("[teleop] Building environment…")
    env = build_env(device)

    # ── Replay buffer — same observation/action spaces as train.py ────────────
    replay_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        args.buffer_size,
    )
    replay_buffer.seed(0)

    # Resume from existing file if present
    prior_steps = load_existing_buffer(replay_buffer, args.out)

    # ── pygame for HUD (RacerEnv already owns a pygame window; we reuse it) ──
    if not pygame.get_init():
        pygame.init()
    clock = pygame.time.Clock()

    # ── Control state ─────────────────────────────────────────────────────────
    steering  = 0.0
    throttle  = 0.120
    alow, ahigh = env.action_space.low, env.action_space.high

    # ── Episode state ─────────────────────────────────────────────────────────
    obs, _   = env.reset()
    done     = False
    step     = 0
    ep       = 0
    ep_steps = 0
    ep_reward = 0.0

    print("\n[teleop] Ready.  Controls: W/S=throttle  A/D=steer  SPC=coast  "
          "R=reset  Q/Esc=quit\n")

    try:
        while step < args.max_steps:
            clock.tick(FPS)

            # ── Event / key handling ──────────────────────────────────────────
            quit_requested  = False
            reset_requested = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        quit_requested = True
                    if event.key == pygame.K_r:
                        reset_requested = True
                    if event.key == pygame.K_SPACE:
                        throttle = 0.120

            if quit_requested:
                break

            keys = pygame.key.get_pressed()

            # Steering
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                steering = max(alow[0],  steering - STEER_STEP)
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                steering = min(ahigh[0], steering + STEER_STEP)

            # Throttle
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                throttle = min(ahigh[1], throttle + THROTTLE_STEP)
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                throttle = max(alow[1],  throttle - THROTTLE_STEP)


            action = np.array([steering, throttle], dtype=np.float32)

            # ── Step environment ──────────────────────────────────────────────
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            mask = 0.0 if done else 1.0

            replay_buffer.insert(dict(
                observations      = obs,
                actions           = action,
                rewards           = float(reward),
                masks             = mask,
                dones             = bool(done),
                next_observations = next_obs,
            ))

            obs        = next_obs
            step      += 1
            ep_steps  += 1
            ep_reward += float(reward)

            # ── HUD ───────────────────────────────────────────────────────────
            screen = pygame.display.get_surface()
            if screen is not None:
                draw_hud(screen, steering, throttle,
                         prior_steps + step, ep, ep_steps, ep_reward)
                pygame.display.flip()

            # ── Episode end ───────────────────────────────────────────────────
            if done or reset_requested:
                print(f"[teleop] Ep {ep:>4d} | steps {ep_steps:>4d} | "
                      f"reward {ep_reward:+.2f} | "
                      f"buffer {prior_steps + step:>6d}/{args.buffer_size}")
                ep        += 1
                ep_steps   = 0
                ep_reward  = 0.0
                steering   = 0.0
                throttle=0.120
                obs, _     = env.reset()
                done       = False

            # ── Periodic auto-save ────────────────────────────────────────────
            # if step % args.save_every == 0:
            #     save_buffer(replay_buffer, args.out, prior_steps + step)

    except KeyboardInterrupt:
        print("\n[teleop] Interrupted.")
    finally:
        save_buffer(replay_buffer, args.out, prior_steps + step)
        env.close()
        print(f"[teleop] Done. Collected {step} new transitions "
              f"({prior_steps + step} total).")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Donkey Car teleoperation collector")
    parser.add_argument("--out",         default="teleop_buffer.pkl",
                        help="Path to save (and resume) the replay buffer pickle")
    parser.add_argument("--max-steps",   type=int, default=20_000,
                        help="Stop collecting after this many new steps")
    parser.add_argument("--buffer-size", type=int, default=100_000,
                        help="Maximum replay buffer capacity")
    parser.add_argument("--save-every",  type=int, default=500,
                        help="Auto-save buffer every N steps")
    args = parser.parse_args()
    teleop(args)