# Sim2Real Habitat-to-Robot Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Habitat sim-trained DrQ checkpoints to be deployed and fine-tuned on the real robot by bridging the IMU dimension mismatch and adding an evaluation script.

**Architecture:** Add `Sim2RealIMUWrapper` that maps real 6D IMU to Habitat 10D format, fix `GoalImageWrapper` IMU masking, add `deterministic` flag to `MobileNetV3Encoder`, create `eval_habitat_robot.py`, and update `train_image_goal.py` to use the sim2real wrapper when loading sim checkpoints.

**Tech Stack:** Python 3.9+, gymnasium, numpy, JAX/Flax (Orbax checkpoints), PyTorch (MobileNetV3), absl-py

---

### Task 1: Create `Sim2RealIMUWrapper` in `sim2real_wrappers.py`

**Files:**
- Create: `sim2real_wrappers.py`
- Create: `tests/test_sim2real_wrappers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sim2real_wrappers.py`:

```python
"""Tests for sim2real_wrappers.py — Sim2RealIMUWrapper."""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from sim2real_wrappers import Sim2RealIMUWrapper


class FakeRacerEnv(gym.Env):
    """Minimal env that mimics RacerEnv observation/action spaces."""

    observation_space = spaces.Dict({
        "image": spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
        "imu": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
    })
    action_space = spaces.Box(
        low=np.array([-1.0, 0.120], dtype=np.float32),
        high=np.array([1.0, 0.2], dtype=np.float32),
        dtype=np.float32,
    )

    def __init__(self):
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = {
            "image": np.zeros((120, 160, 3), dtype=np.uint8),
            "imu": np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03], dtype=np.float32),
        }
        return obs, {}

    def step(self, action):
        self._step_count += 1
        imu = np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03], dtype=np.float32)
        obs = {"image": np.zeros((120, 160, 3), dtype=np.uint8), "imu": imu}
        return obs, -1.0, False, False, {"velocity": {"ms": 0.1}}


class TestSim2RealIMUWrapper:

    def test_observation_space_imu_is_10d(self):
        env = Sim2RealIMUWrapper(FakeRacerEnv())
        assert env.observation_space["imu"].shape == (10,), \
            f"Expected (10,), got {env.observation_space['imu'].shape}"

    def test_image_space_unchanged(self):
        env = Sim2RealIMUWrapper(FakeRacerEnv())
        assert env.observation_space["image"].shape == (120, 160, 3)

    def test_reset_imu_fields(self):
        env = Sim2RealIMUWrapper(FakeRacerEnv())
        obs, info = env.reset()
        imu = obs["imu"]
        assert imu.shape == (10,)
        assert imu[0] == 0.0  # angular_vel_cmd (no last action)
        assert imu[1] == 0.0  # linear_vel_cmd
        assert imu[8] == -1.0  # geodesic_distance
        assert imu[9] == 1.0   # distance_mask

    def test_step_maps_real_imu_to_habitat_format(self):
        env = Sim2RealIMUWrapper(FakeRacerEnv())
        obs, info = env.reset()
        action = np.array([0.5, 0.15], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        imu = obs["imu"]
        assert imu.shape == (10,)
        # imu[0] and imu[1] should be the PREVIOUS action (zeros on first step)
        assert imu[0] == 0.0  # angular_vel_cmd from reset
        assert imu[1] == 0.0  # linear_vel_cmd from reset
        # Real sensor values mapped to indices 2-5
        assert imu[2] == 0.1   # ax
        assert imu[3] == 0.2   # ay
        assert imu[4] == 0.01  # gx (roll_rate)
        assert imu[5] == 0.02  # gy (pitch_rate)
        # Rolling means
        assert imu[6] >= 0.0   # mean_resultant_accel_20
        assert 0.0 <= imu[7] <= 0.2  # mean_throttle_20
        # Fixed fields
        assert imu[8] == -1.0  # geodesic_distance
        assert imu[9] == 1.0   # distance_mask

    def test_second_step_uses_first_action(self):
        """On the 2nd step, imu[0:2] should reflect the 1st action."""
        env = Sim2RealIMUWrapper(FakeRacerEnv())
        env.reset()
        action1 = np.array([0.3, 0.18], dtype=np.float32)
        env.step(action1)
        action2 = np.array([-0.2, 0.15], dtype=np.float32)
        obs, _, _, _, _ = env.step(action2)
        imu = obs["imu"]
        assert abs(imu[0] - 0.3) < 1e-6   # angular_vel_cmd from action1
        assert abs(imu[1] - 0.18) < 1e-6   # linear_vel_cmd from action1

    def test_rolling_mean_accel_updates(self):
        """mean_resultant_accel_20 should reflect real acceleration magnitude."""
        env = Sim2RealIMUWrapper(FakeRacerEnv())
        env.reset()
        obs, _, _, _, _ = env.step(np.array([0.0, 0.15], dtype=np.float32))
        imu = obs["imu"]
        # With ax=0.1, ay=0.2, az=9.8 from FakeRacerEnv
        expected_accel = np.sqrt(0.1**2 + 0.2**2 + 9.8**2)
        assert abs(imu[6] - expected_accel) < 0.1

    def test_reset_clears_state(self):
        env = Sim2RealIMUWrapper(FakeRacerEnv())
        env.reset()
        env.step(np.array([0.5, 0.15], dtype=np.float32))
        obs, info = env.reset()
        imu = obs["imu"]
        assert imu[0] == 0.0  # last action cleared
        assert imu[8] == -1.0
        assert imu[9] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -m pytest tests/test_sim2real_wrappers.py -v 2>&1 | tail -20`
Expected: FAIL — `ModuleNotFoundError: No module named 'sim2real_wrappers'`

- [ ] **Step 3: Write `Sim2RealIMUWrapper` implementation**

Create `sim2real_wrappers.py`:

```python
"""
Sim2Real wrappers for deploying Habitat sim checkpoints on the real robot.

Sim2RealIMUWrapper maps the real robot's 6D IMU to Habitat's 10D format
so that Habitat-trained agent weights can be loaded without reshaping
the input projection layers.
"""

from collections import deque

import gymnasium as gym
import numpy as np


class Sim2RealIMUWrapper(gym.Wrapper):
    """
    Maps real robot 6D IMU to Habitat 10D format.

    Habitat IMU: [angular_vel_cmd, linear_vel_cmd, ax, ay, gx, gy,
                  mean_resultant_accel_20, mean_throttle_20,
                  geodesic_distance, distance_mask]

    Real robot IMU: [ax, ay, az, roll_rate, pitch_rate, yaw_rate]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        self._accel_history = deque(maxlen=20)
        self._throttle_history = deque(maxlen=20)

        # Update observation space: imu from (6,) to (10,)
        new_spaces = dict(env.observation_space.spaces)
        new_spaces["imu"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def _map_imu(self, real_imu: np.ndarray) -> np.ndarray:
        ax, ay, az = real_imu[0], real_imu[1], real_imu[2]
        roll_rate, pitch_rate = real_imu[3], real_imu[4]

        self._accel_history.append(np.sqrt(ax*ax + ay*ay + az*az))
        self._throttle_history.append(float(self._last_action[1]))

        mean_accel = float(np.mean(self._accel_history)) if self._accel_history else 0.0
        mean_throttle = float(np.mean(self._throttle_history)) if self._throttle_history else 0.0

        return np.array([
            self._last_action[0],  # angular_vel_cmd
            self._last_action[1],  # linear_vel_cmd
            ax,                    # forward accel
            ay,                    # lateral accel
            roll_rate,             # gx
            pitch_rate,            # gy
            mean_accel,            # mean_resultant_accel_20
            mean_throttle,         # mean_throttle_20
            -1.0,                  # geodesic_distance (unknown)
            1.0,                   # distance_mask (always True)
        ], dtype=np.float32)

    def reset(self, **kwargs):
        self._last_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        self._accel_history.clear()
        self._throttle_history.clear()
        obs, info = self.env.reset(**kwargs)
        obs = dict(obs)
        obs["imu"] = self._map_imu(obs["imu"])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = action.copy().astype(np.float32)
        obs = dict(obs)
        obs["imu"] = self._map_imu(obs["imu"])
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -m pytest tests/test_sim2real_wrappers.py -v 2>&1 | tail -20`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git add sim2real_wrappers.py tests/test_sim2real_wrappers.py
git commit -m "feat: add Sim2RealIMUWrapper for habitat-to-robot IMU mapping

Maps real robot 6D IMU [ax,ay,az,roll,pitch,yaw] to Habitat 10D format
[angular_vel_cmd,linear_vel_cmd,ax,ay,gx,gy,mean_accel,mean_throttle,
geodesic_dist,distance_mask] so Habitat-trained agent weights can be
loaded without reshaping input projection layers."
```

---

### Task 2: Fix `GoalImageWrapper.step()` IMU masking check

**Files:**
- Modify: `wrappers.py:439-447`

- [ ] **Step 1: Fix the IMU mask check**

In `wrappers.py`, change `GoalImageWrapper.step()` (around line 439-447) from:

```python
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        goal_features = self._encode_goal(info)
        obs = dict(obs)
        obs["goal_features"] = goal_features
        obs["imu"] = obs["imu"]
        if obs["imu"][5] < 0.0:
                obs["goal_features"] = np.zeros_like(obs["goal_features"])
        return obs, reward, terminated, truncated, info
```

to:

```python
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        goal_features = self._encode_goal(info)
        obs = dict(obs)
        obs["goal_features"] = goal_features
        # Zero goal features when distance_mask is True (Habitat 10D IMU only).
        # For 6D real-robot IMU (no mask field), goal features are always kept.
        if obs["imu"].shape[0] >= 10 and obs["imu"][-1] > 0.5:
            obs["goal_features"] = np.zeros_like(obs["goal_features"])
        return obs, reward, terminated, truncated, info
```

Also remove the now-unnecessary `obs["imu"] = obs["imu"]` line.

- [ ] **Step 2: Verify the file parses**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -c "from wrappers import GoalImageWrapper; print('OK')" 2>&1`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git add wrappers.py
git commit -m "fix: GoalImageWrapper IMU mask now checks distance_mask field

Previously checked imu[5] which in 10D Habitat format is 'gy' (angular
velocity y), causing spurious goal feature zeroing. Now checks the last
IMU element (distance_mask) only when IMU is 10D. For 6D real-robot
IMU, goal features are always visible."
```

---

### Task 3: Add `deterministic` flag to `MobileNetV3Encoder`

**Files:**
- Modify: `wrappers.py:247-279` (MobileNetV3Encoder class)
- Modify: `wrappers.py:124-136` (add deterministic transform)

- [ ] **Step 1: Add a deterministic transform alongside `sim2real_a`**

In `wrappers.py`, after the `sim2real_a` lambda (around line 136), add:

```python
sim2real_deterministic = lambda input_size: T.Compose([
    T.ToPILImage(),
    T.Resize((input_size, input_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

- [ ] **Step 2: Add `deterministic` parameter to `MobileNetV3Encoder`**

Change the `MobileNetV3Encoder.__init__` signature and transform selection (around lines 250-260) from:

```python
    def __init__(self, device: str = "cuda", num_blocks: int = 4, input_size: int = 84):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder = torch_nn.Sequential(*list(model.features[:num_blocks]))
        self.encoder.to(self.device)
        self.encoder.eval()

        self.transform = sim2real_a(input_size)
```

to:

```python
    def __init__(self, device: str = "cuda", num_blocks: int = 4, input_size: int = 84,
                 deterministic: bool = False):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder = torch_nn.Sequential(*list(model.features[:num_blocks]))
        self.encoder.to(self.device)
        self.encoder.eval()

        self.transform = sim2real_deterministic(input_size) if deterministic else sim2real_a(input_size)
```

- [ ] **Step 3: Verify the module imports correctly**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -c "from wrappers import MobileNetV3Encoder; e = MobileNetV3Encoder(deterministic=True); print('OK, deterministic=True')" 2>&1`
Expected: `OK, deterministic=True`

- [ ] **Step 4: Commit**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git add wrappers.py
git commit -m "feat: add deterministic flag to MobileNetV3Encoder

When deterministic=True, uses resize+normalize only (no random crops,
blur, color jitter) for reproducible inference. Default is False
(training-time augmentation)."
```

---

### Task 4: Create `eval_habitat_robot.py`

**Files:**
- Create: `eval_habitat_robot.py`

- [ ] **Step 1: Write the evaluation script**

Create `eval_habitat_robot.py`:

```python
#!/usr/bin/env python
"""
Deploy Habitat sim-trained checkpoint on the real robot for evaluation.

Wrapper stack (mirrors Habitat training):
  RacerEnv → Sim2RealIMUWrapper → StackingWrapper → MobileNetFeatureWrapper
  → GoalImageWrapper

Usage:
  python eval_habitat_robot.py --checkpoint_path ./robot_policy/checkpoint_200000
  python eval_habitat_robot.py --checkpoint_path ./checkpoints/habitat_best --num_episodes 20
"""

import os
import numpy as np
from absl import app, flags

import gymnasium as gym
import torch
import pygame

from racer_imu_env import RacerEnv, StackingWrapper
from sim2real_wrappers import Sim2RealIMUWrapper
from wrappers import (
    MobileNetV3Encoder,
    MobileNetFeatureWrapper,
    GoalImageWrapper,
)
from train_image_goal import GoalImagePool, HumanController, _draw_hud, _load_checkpoint
from jaxrl2.agents import DrQLearner
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", "", "Path to Habitat checkpoint directory.")
flags.DEFINE_string("host", "10.42.0.1", "Robot server hostname.")
flags.DEFINE_integer("port", 9000, "Robot TCP port.")
flags.DEFINE_integer("ws_port", 9001, "Robot WebSocket port.")
flags.DEFINE_integer("num_episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_integer("max_episode_steps", 500, "Max steps per episode.")
flags.DEFINE_integer("mobilenet_blocks", 13, "MobileNetV3 blocks (must match sim training).")
flags.DEFINE_integer("mobilenet_input_size", 84, "MobileNetV3 input size.")
flags.DEFINE_integer("frame_stack", 3, "Frame stack depth.")
flags.DEFINE_float("goal_feature_threshold", 1.0, "Feature-distance threshold for goal reached.")
flags.DEFINE_boolean("enable_hitl", True, "Enable Human-in-the-Loop keyboard override.")
flags.DEFINE_string("goal_pool_path", "robot_policy/goal_image_pool.pkl", "Goal image pool path.")
flags.DEFINE_boolean("deterministic_encoder", True, "Use deterministic MobileNet encoding (no augmentation).")
flags.DEFINE_string("config", "./configs/drq_default.py", "DrQ config (must match sim training).")

POLICY_FOLDER = "robot_policy"


def main(_):
    print("\n" + "=" * 70)
    print("Eval: Habitat Checkpoint on Real Robot")
    print("=" * 70 + "\n")

    assert FLAGS.checkpoint_path, "Must provide --checkpoint_path"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Environment ───────────────────────────────────────────────────────────
    print("Building real robot environment with sim2real wrapper…")
    env = RacerEnv(render_mode="human")
    env = Sim2RealIMUWrapper(env)
    env = StackingWrapper(env, num_stack=FLAGS.frame_stack, image_format="bgr")

    shared_encoder = MobileNetV3Encoder(
        device=device,
        num_blocks=FLAGS.mobilenet_blocks,
        input_size=FLAGS.mobilenet_input_size,
        deterministic=FLAGS.deterministic_encoder,
    )

    goal_pool = GoalImagePool(capacity=5000, save_path=FLAGS.goal_pool_path)
    from train_image_goal import RealRobotGoalWrapper
    env = RealRobotGoalWrapper(env, goal_pool=goal_pool)
    env = MobileNetFeatureWrapper(env, encoder=shared_encoder)
    env = GoalImageWrapper(env, encoder=shared_encoder)
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=FLAGS.max_episode_steps)

    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    print(f"IMU dim           : {env.observation_space['imu'].shape[0]} (should be 30 for 10D * 3 frames)")

    # ── Agent ─────────────────────────────────────────────────────────────────
    kwargs = dict(FLAGS.config) if FLAGS.config else {}
    agent = DrQLearner(
        FLAGS.seed if hasattr(FLAGS, 'seed') else 42,
        env.observation_space.sample(),
        env.action_space.sample(),
        **kwargs,
    )
    agent, step = _load_checkpoint(agent, FLAGS.checkpoint_path)
    print(f"Loaded checkpoint from {FLAGS.checkpoint_path} (step {step:,})\n")

    # ── Pygame / HitL ─────────────────────────────────────────────────────────
    if FLAGS.enable_hitl:
        pygame.init()
        pygame.font.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((640, 480))
            pygame.display.set_caption("Eval: Habitat → Real Robot")
        clock = pygame.time.Clock()
        human = HumanController(env.action_space.low, env.action_space.high)
    else:
        clock = None
        human = None

    # ── Eval loop ─────────────────────────────────────────────────────────────
    episode_returns = []
    episode_lengths = []
    successes = []

    try:
        for ep in range(FLAGS.num_episodes):
            obs, info = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            done = False

            while not done:
                if clock is not None:
                    clock.tick(30)

                if FLAGS.enable_hitl and human is not None:
                    quit_req, reset_req, _ = human.process_events()
                    if quit_req:
                        print("\nQuit requested.")
                        raise KeyboardInterrupt
                    human_action, human_active = human.read()
                    if human_active or human.paused:
                        action = human_action
                    else:
                        action = agent.sample_actions(obs, deterministic=True)
                else:
                    action = agent.sample_actions(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += float(reward)
                ep_steps += 1

                # Print diagnostics
                feature_dist = info.get("feature_distance", float("inf"))
                vel = info.get("velocity", {}).get("ms", 0.0)
                collision = info.get("collision", {}).get("detected", False)
                if ep_steps % 10 == 0 or done:
                    print(f"  Ep {ep+1} step {ep_steps:4d} | "
                          f"dist={feature_dist:.2f} vel={vel:.3f}m/s "
                          f"collision={collision} reward={ep_reward:+.1f}")

                if FLAGS.enable_hitl:
                    _draw_hud(
                        pygame.display.get_surface(),
                        float(action[0]), float(action[1]),
                        step, ep, ep_steps, ep_reward,
                        human_active if human else False,
                        human.paused if human else False,
                        0, 0,
                    )

            episode_returns.append(ep_reward)
            episode_lengths.append(ep_steps)
            success = info.get("feature_distance", float("inf")) < FLAGS.goal_feature_threshold if terminated else False
            successes.append(float(success))

            print(f"  Episode {ep+1}: reward={ep_reward:+.1f} steps={ep_steps} success={success}")

            if human is not None:
                human.reset_controls()

    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")

    finally:
        # ── Results ─────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        if episode_returns:
            print(f"  Episodes     : {len(episode_returns)}")
            print(f"  Mean return  : {np.mean(episode_returns):+.2f} ± {np.std(episode_returns):.2f}")
            print(f"  Mean length  : {np.mean(episode_lengths):.1f}")
            print(f"  Success rate : {np.mean(successes)*100:.1f}%")
            print(f"  Best return  : {np.max(episode_returns):+.2f}")
        print("=" * 70 + "\n")

        env.close()


if __name__ == "__main__":
    app.run(main)
```

- [ ] **Step 2: Verify the script parses correctly**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -c "import eval_habitat_robot; print('OK')" 2>&1`
Expected: `OK` (or import errors for hardware-dependent modules like `witmotion`, which is acceptable — the absl flags should at least register)

- [ ] **Step 3: Commit**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git add eval_habitat_robot.py
git commit -m "feat: add eval_habitat_robot.py for deploying sim checkpoints

Evaluation script that loads a Habitat sim-trained DrQ checkpoint and
runs it on the real robot. Uses Sim2RealIMUWrapper to bridge the IMU
dimension mismatch, deterministic MobileNetV3 encoding for reproducible
inference, and optional Human-in-the-Loop override."
```

---

### Task 5: Update `train_image_goal.py` to use `Sim2RealIMUWrapper`

**Files:**
- Modify: `train_image_goal.py:62-108` (add `--sim2real_imu` flag)
- Modify: `train_image_goal.py:479-500` (wrapper stack construction)

- [ ] **Step 1: Add `--sim2real_imu` flag**

In `train_image_goal.py`, after the checkpoint flags (around line 108), add:

```python
flags.DEFINE_boolean("sim2real_imu", True,
                     "Use Sim2RealIMUWrapper (6D→10D IMU) for Habitat checkpoint compatibility.")
```

- [ ] **Step 2: Add import and insert `Sim2RealIMUWrapper` in the wrapper stack**

Add the import at the top (after line 48):

```python
from sim2real_wrappers import Sim2RealIMUWrapper
```

Modify the environment construction (around lines 480-482) from:

```python
    env = RacerEnv(render_mode="human")
    env = StackingWrapper(env, num_stack=FLAGS.frame_stack, image_format="bgr")
```

to:

```python
    env = RacerEnv(render_mode="human")
    if FLAGS.sim2real_imu:
        env = Sim2RealIMUWrapper(env)
    env = StackingWrapper(env, num_stack=FLAGS.frame_stack, image_format="bgr")
```

- [ ] **Step 3: Add checkpoint compatibility diagnostics**

After the checkpoint loading (around line 538), add:

```python
    print(f"IMU dimension    : {env.observation_space['imu'].shape[0]} "
          f"({'10D (sim2real)' if FLAGS.sim2real_imu else '6D (native)'})")
```

- [ ] **Step 4: Verify the script parses correctly**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -c "from train_image_goal import Sim2RealIMUWrapper; print('OK')" 2>&1`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git add train_image_goal.py
git commit -m "feat: add Sim2RealIMUWrapper to train_image_goal.py

When --sim2real_imu is True (default), wraps RacerEnv with
Sim2RealIMUWrapper to produce 10D Habitat-format IMU, enabling
Habitat-trained checkpoints to be loaded and fine-tuned on the real
robot. Adds IMU dimension diagnostic on startup."
```

---

### Task 6: Run full test suite and final verification

**Files:**
- All modified/created files

- [ ] **Step 1: Run all project tests**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -m pytest tests/test_sim2real_wrappers.py tests/test_robot_event_client.py -v 2>&1 | tail -30`
Expected: All tests PASS.

- [ ] **Step 2: Verify import chain for all changed files**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -c "from sim2real_wrappers import Sim2RealIMUWrapper; from wrappers import MobileNetV3Encoder, GoalImageWrapper; print('All imports OK')" 2>&1`
Expected: `All imports OK`

- [ ] **Step 3: Commit any remaining changes**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git status
git diff
# Only commit if there are uncommitted changes
```