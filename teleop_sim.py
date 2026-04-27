#!/usr/bin/env python
"""
DonkeyCar data collection with pygame teleoperation.

- Pygame teleop (WASD) for steering/throttle
- Shows camera image in a pygame window
- Stores (image, position, action, reward, timestamp, episode, step) in pickle

This script is for data collection only (no RL training).
"""

import os
import uuid
import time
import pickle
from datetime import datetime

import numpy as np
from absl import app, flags

import gymnasium as gym
from gymnasium import spaces

import torch
import pygame
import pygame.surfarray as surfarray  # for numpy->Surface conversion[web:40]

from gym_donkeycar.envs.donkey_env import DonkeyEnv  # DonkeyCar env[web:42]

from wrappers import (
    EnvCompatibility,
    Logger,
    MobileNetFeatureWrapper,
    RewardWrapper,
    StackingWrapper,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "donkey-warehouse-v0", "Environment name.")
flags.DEFINE_integer("port", 9091, "Port to use for tcp.")
flags.DEFINE_string("save_dir", "./data/", "Directory to save collected data.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("max_steps", int(5e4), "Max env steps for data collection.")
flags.DEFINE_integer("frame_stack", 3, "Number of frames to stack.")
flags.DEFINE_integer("mobilenet_blocks", 4, "Number of MobileNetV3 blocks to use.")
flags.DEFINE_integer("mobilenet_input_size", 84, "Input size for MobileNetV3.")
flags.DEFINE_integer("save_interval", 1000, "Save pickle every N steps.")
flags.DEFINE_string("run_name", None, "Optional name for this data collection run.")
flags.DEFINE_integer("display_scale", 4, "Scale factor for camera display.")


# ---------------------------------------------------------------------------
# Custom Donkey envs (same as in your training script)
# ---------------------------------------------------------------------------
class OfficeEnvCorner(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="OfficeBox-Corner", *args, **kwargs)


class BarrowsHallEnvCorner(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="BarrowsHall", *args, **kwargs)


class OfficeEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="OfficeBox", *args, **kwargs)


# ---------------------------------------------------------------------------
# Pygame teleop helper
# ---------------------------------------------------------------------------
def pygame_get_action(prev_action, dt):
    """
    Continuous keyboard control using pygame.

    Keys:
      A / D : steer left / right
      W / S : throttle up / down
      SPACE : brake (throttle -> 0)
      ESC   : quit script

    prev_action: np.array([steer, throttle])
    dt: time since last frame (seconds)
    """
    steer, throttle = prev_action

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return prev_action, True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return prev_action, True

    keys = pygame.key.get_pressed()

    steer_delta = 1.0 * dt
    throttle_delta = 0.7 * dt

    if keys[pygame.K_a]:
        steer -= steer_delta
    if keys[pygame.K_d]:
        steer += steer_delta

    if keys[pygame.K_w]:
        throttle += throttle_delta
    if keys[pygame.K_s]:
        throttle -= throttle_delta

    if keys[pygame.K_SPACE]:
        throttle = 0.0

    steer = float(np.clip(steer, -1.0, 1.0))
    throttle = float(np.clip(throttle, -1.0, 1.0))

    return np.array([steer, throttle], dtype=np.float32), False


# ---------------------------------------------------------------------------
# Helper: draw camera frame in pygame window
# ---------------------------------------------------------------------------
def draw_camera_frame(screen, image, scale):
    """
    image: numpy array H x W x C (RGB), stacked frames OK if you slice.
    scale: integer scale factor for display.
    """
    # If your StackingWrapper packs frames along channel dim, we take the latest
    # Example: pixels shape (H, W, 3 * num_stack)
    img = image
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, -3:]  # last RGB frame of stack

    # Ensure uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w, _ = img.shape

    # Make surface from numpy array[web:32][web:41]
    surf = surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    if scale != 1:
        surf = pygame.transform.scale(surf, (w * scale, h * scale))

    screen.blit(surf, (0, 0))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Main data-collection function
# ---------------------------------------------------------------------------
def main(_):
    print("\n" + "=" * 70)
    print("DonkeyCar Data Collection with Pygame Teleop")
    print("=" * 70 + "\n")

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    conf = {
        "host": "127.0.0.1",
        "port": FLAGS.port,
        "body_rgb": (128, 128, 128),
        "car_name": "",
        "font_size": 100,
        "racer_name": "",
        "country": "USA",
        "bio": "Data collection for DonkeyCar",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
        "frame_skip": 3,
    }

    # Choose env variant
    env = BarrowsHallEnvCorner(conf=conf)

    env = EnvCompatibility(env)
    # env = StackingWrapper(env, num_stack=FLAGS.frame_stack)
    # env = RewardWrapper(env)
    # env = MobileNetFeatureWrapper(
    #     env,
    #     device=device,
    #     num_blocks=FLAGS.mobilenet_blocks,
    #     input_size=FLAGS.mobilenet_input_size,
    # )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Get one sample obs to size the pygame window
    obs, info = env.reset()
    sample_img = obs 
    if sample_img.ndim == 3 and sample_img.shape[2] > 3:
        sample_img = sample_img[:, :, -3:]
    h, w, _ = sample_img.shape

    # Init pygame
    pygame.init()
    screen = pygame.display.set_mode(
        (w * FLAGS.display_scale, h * FLAGS.display_scale)
    )
    pygame.display.set_caption("DonkeyCar Teleop (WASD)")
    clock = pygame.time.Clock()

    # Data buffer
    collected = []
    run_id = FLAGS.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.join(FLAGS.save_dir, f"donkey_data_{run_id}")

    def save_data(suffix):
        if not collected:
            return
        fname = f"{base_filename}_{suffix}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(collected, f)
        print(f"[SAVE] Saved {len(collected)} samples to {fname}")

    prev_action = np.array([0.0, 0.0], dtype=np.float32)
    step = 0
    episode = 0

    try:
        while step < FLAGS.max_steps:
            dt = clock.tick(60) / 1000.0

            action, quit_flag = pygame_get_action(prev_action, dt)
            prev_action = action.copy()
            if quit_flag:
                print("Quitting data collection.")
                break

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = False

            image = obs 
            pos = info.get("pos", [0.0, 0.0, 0.0])

            sample = {
                "image": image,
                "position": np.asarray(pos, dtype=np.float32),
                "action": action.copy(),
                "reward": reward,
                "timestamp": time.time(),
                "episode": episode,
                "step": step,
            }
            collected.append(sample)

            # Draw camera frame
            draw_camera_frame(screen, image, FLAGS.display_scale)

            step += 1
            obs = next_obs

            if step % FLAGS.save_interval == 0:
                save_data(f"step{step:07d}")
                collected.clear()

            if done:
                print(f"Episode {episode} done at step {step}. Resetting.")
                obs, info = env.reset()
                episode += 1
                prev_action = np.array([0.0, 0.0], dtype=np.float32)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopping data collection.")

    if collected:
        save_data("final")

    env.close()
    pygame.quit()
    print("Data collection finished.")


if __name__ == "__main__":
    app.run(main)
