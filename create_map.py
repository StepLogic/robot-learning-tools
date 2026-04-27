#!/usr/bin/env python3
"""
Teleoperation image collector for Donkey Car.
Drives the robot with keyboard/joystick, saves camera frames only into ./topomap.

Controls (keyboard):
  W / Up     – increase throttle
  S / Down   – decrease throttle / brake
  A / Left   – steer left
  D / Right  – steer right
  SPACE      – zero throttle (coast)
  R          – reset episode
  Q / Esc    – quit
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pygame
import torch
import gymnasium as gym

# Same wrappers the training script uses
from racer_imu_env import RacerEnv, StackingWrapper, RewardWrapper
from wrappers import MobileNetFeatureWrapper


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
NUM_STACK        = 3
MOBILENET_BLOCKS = 4
MOBILENET_INPUT  = 84
FPS              = 30

STEER_STEP       = 0.05
THROTTLE_STEP    = 0.02


def build_env(device: str) -> gym.Env:
    """Construct the exact same wrapper stack as train.py."""
    env = RacerEnv(render_mode="human")
    # env = StackingWrapper(env, num_stack=NUM_STACK)
    # env = RewardWrapper(env)
    # env = MobileNetFeatureWrapper(
    #     env,
    #     device=device,
    #     num_blocks=MOBILENET_BLOCKS,
    #     input_size=MOBILENET_INPUT,
    # )
    return env


# HUD overlay (unchanged)
def draw_hud(screen: pygame.Surface,
             steering: float, throttle: float,
             step: int, ep: int, ep_steps: int,
             total_reward: float):
    font = pygame.font.SysFont("monospace", 16, bold=True)

    def txt(msg, x, y, color=(255, 255, 0)):
        screen.blit(font.render(msg, True, color), (x, y))

    bar = pygame.Surface((screen.get_width(), 80), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 160))
    screen.blit(bar, (0, screen.get_height() - 80))

    y0 = screen.get_height() - 75
    txt(f"Steer  {steering:+.3f}   Throttle {throttle:+.3f}", 10, y0)
    txt(f"Step {step:>6d}   Ep {ep:>4d}   Ep-step {ep_steps:>4d}", 10, y0 + 20)
    txt(f"Ep reward {total_reward:+.2f}", 10, y0 + 40)
    txt("W/S=throttle  A/D=steer  SPC=coast  R=reset  Q=quit", 10, y0 + 58,
        color=(200, 200, 200))

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
# Main teleop loop — image-only logging
# ─────────────────────────────────────────────────────────────────────────────
def teleop(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[teleop] Device: {device}")

    # Build env
    print("[teleop] Building environment…")
    env = build_env(device)

    # Image output directory
    out_dir = "topomap"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[teleop] Saving camera images to: {out_dir}")

    # pygame / HUD
    if not pygame.get_init():
        pygame.init()
    clock = pygame.time.Clock()

    steering  = 0.0
    throttle  = 0.120
    alow, ahigh = env.action_space.low, env.action_space.high

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

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Extract current camera image from observation
            # Assuming first channel/part of obs corresponds to the latest frame
            # and is a HxWx3 uint8 RGB image; adapt if your wrapper differs.
            img = obs["image"]  # if your RacerEnv exposes it
            # Fallback: try to infer from observation tensor
            if img is None and isinstance(obs, np.ndarray):
                # Example: obs shape (num_stack, H, W, C) or (H, W, C)
                arr = obs
                if arr.ndim == 4:
                    frame = arr[-1]  # last in stack
                elif arr.ndim == 3:
                    frame = arr
                else:
                    frame = None
                if frame is not None:
                    img = frame

            if img is not None:
                # Ensure uint8 HxWx3 BGR for OpenCV
                img_np = np.array(img)
                if img_np.ndim == 3 and img_np.shape[2] == 3:
                    # Assume RGB, convert to BGR for imwrite
                    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    fname = os.path.join(out_dir, f"frame_{step:06d}.png")
                    cv2.imwrite(fname, bgr)

            obs        = next_obs
            step      += 1
            ep_steps  += 1
            ep_reward += float(reward)

            # HUD
            screen = pygame.display.get_surface()
            if screen is not None:
                draw_hud(screen, steering, throttle,
                         step, ep, ep_steps, ep_reward)
                pygame.display.flip()

            # Episode end
            if done or reset_requested:
                print(f"[teleop] Ep {ep:>4d} | steps {ep_steps:>4d} | "
                      f"reward {ep_reward:+.2f}")
                ep        += 1
                ep_steps   = 0
                ep_reward  = 0.0
                steering   = 0.0
                throttle   = 0.120
                obs, _     = env.reset()
                done       = False

    except KeyboardInterrupt:
        print("\n[teleop] Interrupted.")
    finally:
        env.close()
        print(f"[teleop] Done. Saved {step} frames to {out_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Donkey Car teleoperation image collector")
    parser.add_argument("--max-steps",   type=int, default=20_000,
                        help="Stop collecting after this many steps")
    args = parser.parse_args()
    teleop(args)
    