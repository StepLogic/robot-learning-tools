#!/usr/bin/env python
"""
Teleoperation data collector for Donkey Car with Hindsight Experience Replay (HER).
Drives the robot with keyboard/joystick, stores transitions in a replay
buffer. 

Modifications:
- Automatically relabels 100% of the goals for every episode to match
  the final state reached by the human expert (Positive Samples).
- Simultaneously preserves and inserts the original environment transitions 
  as unachieved, random goals (Negative Samples).
- Uses the exact same pseudo-odometry polar math as the training loop.

Controls (keyboard):
  W / Up     – increase throttle
  S / Down   – decrease throttle / brake
  A / Left   – steer left
  D / Right  – steer right
  SPACE      – zero throttle (coast)
  R          – reset episode
  Q / Esc    – quit and save
"""

import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pygame
import torch
from absl import app, flags
import gymnasium as gym
from gymnasium import spaces
import gymnasium as gym
from jaxrl2.data import ReplayBuffer

# ── Environment & Wrappers ───────────────────────────────────────────────────
from racer_imu_env import RacerEnv, StackingWrapper, RewardWrapper
from wrappers import MobileNetFeatureWrapper

# IMPORTANT: Ensure this matches the import path for your custom wrapper
# from train_her_robot import GoalRelObservationWrapper 

# ─────────────────────────────────────────────────────────────────────────────
# Flags
# ─────────────────────────────────────────────────────────────────────────────
FLAGS = flags.FLAGS
flags.DEFINE_string("out", "teleop_buffer.pkl", "Path to save the replay buffer pickle")
flags.DEFINE_integer("max_steps", 20_000, "Stop collecting after this many new steps")
flags.DEFINE_integer("buffer_size", 100_000, "Maximum replay buffer capacity")
flags.DEFINE_integer("save_interval", 2000, "Save the teleop buffer every N steps")

# Wrapper Configs
NUM_STACK        = 3
MOBILENET_BLOCKS = 4
MOBILENET_INPUT  = 84
FPS              = 30

# Steering / throttle step sizes per key-press
STEER_STEP     = 0.05
THROTTLE_STEP  = 0.02

# HER Configuration
GOAL_RANGE       = 20.0
GOAL_THRESHOLD   = 1.0
USE_GOAL_MASKING = True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
# Goal-Relative Wrapper & HER Buffer
# ═══════════════════════════════════════════════════════════════════════════════
class GoalRelObservationWrapper(gym.Wrapper):
    """
    Goal-relative observation wrapper modified for real robot pseudo-odometry.
    - Operates natively in polar (angle, distance) space.
    - Tracks absolute pseudo-odometry internally and passes it via 'info' for HER.
    """
    def __init__(
        self,
        env,
        goal_range: float = 20.0,
        goal_threshold: float = 1.0,
        use_goal_masking: bool = True,
        mask_probability: float = 0.3,
    ):
        super().__init__(env)
        self.goal_range = goal_range 
        self.goal_threshold = goal_threshold 
        self.use_goal_masking = use_goal_masking
        self.mask_probability = mask_probability
        
        # Internal state tracking
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_yaw = 0.0
        self.current_goal_abs = np.zeros(2, dtype=np.float32)
        
        self.stop_count = 0
        self.goals_reached = 0  
        self.total_distance = 0.0  
        self.goal_state_value = 1.0  

        if use_goal_masking:
            goal_rel_space = spaces.Box(
                low=-np.array([np.pi, goal_range, 0], dtype=np.float32),
                high=np.array([np.pi, goal_range, 1], dtype=np.float32),
                dtype=np.float32
            )
        else:
            goal_rel_space = spaces.Box(
                low=-np.array([np.pi, goal_range, 0], dtype=np.float32),
                high=np.array([np.pi, goal_range, 0.1], dtype=np.float32),
                dtype=np.float32
            )

        if isinstance(self.observation_space, spaces.Dict):
            new_spaces = dict(self.observation_space.spaces)
            new_spaces["goal_rel"] = goal_rel_space
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = spaces.Dict(
                {"obs": self.observation_space, "goal_rel": goal_rel_space}
            )

    def _get_obs_dict(self, base_obs):
        # Calculate relative polar state
        dx = self.current_goal_abs[0] - self.agent_x
        dy = self.current_goal_abs[1] - self.agent_y
        dist = np.hypot(dx, dy)
        
        abs_angle = np.arctan2(dy, dx)
        rel_angle = (abs_angle - self.agent_yaw + np.pi) % (2 * np.pi) - np.pi
        
        mask = 1.0
        if self.use_goal_masking and np.random.random() < self.mask_probability:
            mask = 0.0
            
        goal_rel = np.array([rel_angle, dist, mask], dtype=np.float32)

        if isinstance(base_obs, dict) and "goal_rel" not in base_obs:
            out = dict(base_obs)
        elif isinstance(base_obs, dict) and "goal_rel" in base_obs:
            out = dict(base_obs) 
        else:
            out = {"obs": base_obs}
            
        out["goal_rel"] = goal_rel
        return out

    def _update_info(self, info):
        info["agent_pos"] = np.array([self.agent_x, self.agent_y], dtype=np.float32)
        info["agent_yaw"] = np.array([self.agent_yaw], dtype=np.float32)
        info["distance"] = self.total_distance
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_yaw = info.get("yaw", 0.0)
        self.total_distance = 0.0
        self.stop_count = 0

        # Sample initial goal
        rel_angle = np.random.uniform(-np.pi, np.pi)
        dist = np.random.uniform(0.0, self.goal_range)
        abs_angle = self.agent_yaw + rel_angle
        self.current_goal_abs = np.array([
            self.agent_x + dist * np.cos(abs_angle),
            self.agent_y + dist * np.sin(abs_angle)
        ], dtype=np.float32)

        return self._get_obs_dict(obs), self._update_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vel = info.get("forward_vel", 0.0)
        self.agent_yaw = info.get("yaw", 0.0)
        
        if vel > 0.01:
            dist_step = 0.01
            self.total_distance += dist_step
            self.agent_x += dist_step * np.cos(self.agent_yaw)
            self.agent_y += dist_step * np.sin(self.agent_yaw)

        dx = self.current_goal_abs[0] - self.agent_x
        dy = self.current_goal_abs[1] - self.agent_y
        dist = np.hypot(dx, dy)
        
        reward = -0.1 + vel

        if dist < self.goal_threshold:
            self.goals_reached += 1
            reward += 10.0
            
            # Resample goal
            rel_angle = np.random.uniform(-np.pi, np.pi)
            new_dist = np.random.uniform(0.0, self.goal_range)
            abs_angle = self.agent_yaw + rel_angle
            self.current_goal_abs = np.array([
                self.agent_x + new_dist * np.cos(abs_angle),
                self.agent_y + new_dist * np.sin(abs_angle)
            ], dtype=np.float32)

        self.stop_count = self.stop_count + 1 if vel < 0.01 else 0
        if self.stop_count > 20:
            terminated = True
            reward -= 100
        if info.get("hit", "none") != "none":
            terminated = True
            reward -= 100

        return self._get_obs_dict(obs), reward, terminated, truncated, self._update_info(info)



def build_env(device: str) -> gym.Env:
    """Construct the exact same wrapper stack as train.py, including GoalRel."""
    env = RacerEnv(render_mode="human")
    env = StackingWrapper(env, num_stack=NUM_STACK)
    env = RewardWrapper(env)
    env = MobileNetFeatureWrapper(
        env,
        device=device,
        num_blocks=MOBILENET_BLOCKS,
        input_size=MOBILENET_INPUT,
    )
    env = GoalRelObservationWrapper(
        env,
        goal_range=GOAL_RANGE,
        goal_threshold=GOAL_THRESHOLD,
        use_goal_masking=USE_GOAL_MASKING,
        mask_probability=0.0, # 0.0 mask prob during teleop for pure expert states
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
    print(f"\n[Async Save] Saved {replay_buffer._size} transitions → {path}")


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
        replay_buffer.dataset_dict  = data["data"]
        replay_buffer._insert_index = data["insert_index"]
        replay_buffer._size         = data["size"]
        print(f"[Init] Resumed existing buffer: {replay_buffer._size} transitions from {path}")
        return replay_buffer._size
    except Exception as e:
        print(f"[Init] Could not load existing buffer ({e}), starting fresh.")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────────────────────────────────────

def _draw_hud(screen: pygame.Surface,
              steering: float, throttle: float,
              step: int, ep: int, ep_steps: int,
              total_reward: float, buf_size: int, max_buf_size: int):
    
    font = pygame.font.SysFont("monospace", 16, bold=True)

    def txt(msg, x, y, color=(255, 255, 0)):
        screen.blit(font.render(msg, True, color), (x, y))

    bar = pygame.Surface((screen.get_width(), 80), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 160))
    screen.blit(bar, (0, screen.get_height() - 80))

    y0 = screen.get_height() - 75
    txt(f"Steer  {steering:+.3f}   Throttle {throttle:+.3f}", 10, y0)
    txt(f"Step {step:>6d}   Ep {ep:>4d}   Ep-step {ep_steps:>4d} (Pending Relabel)", 10, y0 + 20)
    txt(f"Ep reward {total_reward:+.2f}    Buffer {buf_size}/{max_buf_size}", 10, y0 + 40)
    txt("W/S=throttle  A/D=steer  SPC=coast  R=reset  Q=quit", 10, y0 + 58, color=(200, 200, 200))

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
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────

def main(_):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[teleop] Device: {device}")

    print("[teleop] Building environment…")
    env = build_env(device)

    # Standard ReplayBuffer (we do the HER logic manually before inserting)
    replay_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        FLAGS.buffer_size,
    )
    replay_buffer.seed(0)

    prior_steps = load_existing_buffer(replay_buffer, FLAGS.out)
    
    saver = ThreadPoolExecutor(max_workers=1)

    if not pygame.get_init():
        pygame.init()
    clock = pygame.time.Clock()

    steering  = 0.0
    throttle  = 0.120
    alow, ahigh = env.action_space.low, env.action_space.high

    obs, info = env.reset()
    
    done      = False
    step      = 0
    ep        = 0
    ep_steps  = 0
    ep_reward = 0.0

    episode_buffer = []

    print("\n" + "="*70)
    print("Teleoperation Ready")
    print("Controls: W/S=throttle  A/D=steer  SPC=coast  R=reset  Q/Esc=quit")
    print("="*70 + "\n")

    try:
        while step < FLAGS.max_steps:
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

            keys = pygame.key.get_pressed()

            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                steering = max(alow[0],  steering - STEER_STEP)
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                steering = min(ahigh[0], steering + STEER_STEP)

            if keys[pygame.K_w] or keys[pygame.K_UP]:
                throttle = min(ahigh[1], throttle + THROTTLE_STEP)
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                throttle = max(alow[1],  throttle - THROTTLE_STEP)

            action = np.array([steering, throttle], dtype=np.float32)

            # ── Step Environment ──────────────────────────────────────────────
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            mask = 0.0 if done else 1.0

            # Stage transition using pseudo-odometry provided by GoalRel wrapper
            episode_buffer.append({
                'obs': obs,
                'action': action,
                'reward': float(reward),
                'mask': mask,
                'done': bool(done),
                'next_obs': next_obs,
                'pos': info.get('agent_pos', np.zeros(2)),
                'yaw': info.get('agent_yaw', np.zeros(1)),
                'next_pos': next_info.get('agent_pos', np.zeros(2)),
                'next_yaw': next_info.get('agent_yaw', np.zeros(1))
            })

            obs = next_obs
            info = next_info
            
            step += 1
            ep_steps += 1
            ep_reward += float(reward)

            # ── HUD rendering ─────────────────────────────────────────────────
            screen = pygame.display.get_surface()
            if screen is not None:
                _draw_hud(screen, steering, throttle,
                          prior_steps + step, ep, ep_steps, ep_reward, 
                          replay_buffer._size, FLAGS.buffer_size)
                pygame.display.flip()

            # ── Async Save Checkpoint ─────────────────────────────────────────
            if step > 0 and step % FLAGS.save_interval == 0:
                 saver.submit(save_buffer, replay_buffer, FLAGS.out, prior_steps + step)

            # ── Episode end & HER Relabeling ──────────────────────────────────
            if done or reset_requested or quit_requested:
                if len(episode_buffer) > 0:
                    
                    # 1. Final position becomes the new HER target
                    final_pos = episode_buffer[-1]['next_pos']

                    # 2. Relabel the entire episode buffer
                    for i, t in enumerate(episode_buffer):
                        
                        # =======================================================
                        # --- 1. NEGATIVE SAMPLE (Original Goal) ---
                        # =======================================================
                        o_orig = dict(t['obs'])
                        no_orig = dict(t['next_obs'])
                        
                        replay_buffer.insert(dict(
                            observations      = o_orig,
                            actions           = t['action'],
                            rewards           = t['reward'],  
                            masks             = t['mask'],
                            dones             = t['done'],
                            next_observations = no_orig,
                        ))

                        # =======================================================
                        # --- 2. POSITIVE SAMPLE (HER Relabeled Goal) ---
                        # Matches exact polar coordinate math from train loop
                        # =======================================================
                        o_her = dict(t['obs'])
                        no_her = dict(t['next_obs'])
                        
                        # Calculate relabeled obs goal_rel
                        curr_pos = t['pos']
                        curr_yaw = t['yaw'][0]
                        dx = final_pos[0] - curr_pos[0]
                        dy = final_pos[1] - curr_pos[1]
                        rel_dist = np.hypot(dx, dy)
                        abs_angle = np.arctan2(dy, dx)
                        rel_angle = (abs_angle - curr_yaw + np.pi) % (2 * np.pi) - np.pi
                        o_her['goal_rel'] = np.array([rel_angle, rel_dist, 1.0], dtype=np.float32)

                        # Calculate relabeled next_obs goal_rel
                        n_curr_pos = t['next_pos']
                        n_curr_yaw = t['next_yaw'][0]
                        n_dx = final_pos[0] - n_curr_pos[0]
                        n_dy = final_pos[1] - n_curr_pos[1]
                        n_rel_dist = np.hypot(n_dx, n_dy)
                        n_abs_angle = np.arctan2(n_dy, n_dx)
                        n_rel_angle = (n_abs_angle - n_curr_yaw + np.pi) % (2 * np.pi) - np.pi
                        no_her['goal_rel'] = np.array([n_rel_angle, n_rel_dist, 1.0], dtype=np.float32)

                        # Adjust Reward identically to training loop 
                        her_reward = t['reward']
                        if n_rel_dist < GOAL_THRESHOLD:
                            her_reward += 10.0

                        replay_buffer.insert(dict(
                            observations      = o_her,
                            actions           = t['action'],
                            rewards           = her_reward,
                            masks             = t['mask'],
                            dones             = t['done'],
                            next_observations = no_her,
                        ))

                print(f"[teleop] Ep {ep:>4d} | steps {ep_steps:>4d} | "
                      f"reward {ep_reward:+.2f} | "
                      f"buffer {prior_steps + step:>6d}/{FLAGS.buffer_size}")
                
                episode_buffer.clear()
                
                if quit_requested:
                    break

                ep        += 1
                ep_steps   = 0
                ep_reward  = 0.0
                steering   = 0.0
                throttle   = 0.120
                
                obs, info  = env.reset()
                done       = False

    except KeyboardInterrupt:
        print("\n[teleop] Interrupted.")
    finally:
        saver.shutdown(wait=True)
        save_buffer(replay_buffer, FLAGS.out, prior_steps + step)
        env.close()
        
        print("\n" + "=" * 70)
        print("Collection complete")
        print(f"  New steps  : {step:,}")
        print(f"  Total buf  : {prior_steps + step:,} / {FLAGS.buffer_size:,}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    app.run(main)