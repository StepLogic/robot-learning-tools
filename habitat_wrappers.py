

import math
import os
from collections import deque
import cv2
import numpy as np
import gymnasium as gym

# ═══════════════════════════════════════════════════════════════════════════════
# Video recording
# ═══════════════════════════════════════════════════════════════════════════════

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

    Episodic curiosity (Savinov et al. 2019, https://arxiv.org/abs/1810.02274):
      + Intrinsic reward for novel states within each episode via NN cosine
        distance in MobileNetV3 feature space.

    Commented-out terms (steering penalty, stall detection, raw movement bonus)
    can be re-enabled by uncommenting the corresponding lines in step().
    """

    def __init__(self, env, goal_threshold=0.5,
                 k_dist=1.0, k_delta_x=1.0, k_throttle=1.0,
                 k_goal=10.0, k_collision=0.001, k_steering=0.01,
                 k_explore_vel=1.0, k_curiosity=0.01,
                 curiosity_memory_size=1000):
        super().__init__(env)
        self.goal_threshold = goal_threshold
        self.k_dist = k_dist
        self.k_delta_x = k_delta_x
        self.k_throttle = k_throttle
        self.k_goal = k_goal
        self.k_collision = k_collision
        self.k_steering = k_steering
        self.k_explore_vel = k_explore_vel
        self.k_curiosity = k_curiosity
        self.curiosity_memory = deque(maxlen=curiosity_memory_size)
        self.curiosity_memory_size = curiosity_memory_size
        self._per_frame_dim = None
        self.deltas = deque(maxlen=5)  
        self.steering_hist = deque(maxlen=10)
        
        self.throttle_hist = deque(maxlen=10)
        self._prev_distance = float("inf")
        self.distance_covered = 0
        self._start_position = None
        self.eval_mode = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.distance_covered = 0
        self._prev_distance = info.get("distance_to_goal", float("inf"))
        self._start_position = info.get("pos", None)
        self.curiosity_memory.clear()
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        reward = -1.0

        # Proximity penalty: smoothly escalate as agent nears obstacles
        proximity = float(obs["imu"][-1])
        if proximity >= 0:
            reward -= math.exp(-proximity)

        goal_masked = obs["imu"][-1] > 0.0
        info["curiosity"] = 0.0
        has_goal = not goal_masked
        mean_throttle = float(np.mean(self.unwrapped._throttle_history))
        curiosity_bonus = self._compute_episodic_curiosity(obs["pixels"])
        # if not has_goal:
        reward -= curiosity_bonus
        # print(curiosity_bonus)
        info["curiosity"] = curiosity_bonus

        # else:
        #     # ── Goal-directed reward ──────────────────────────────────────────
        #     curr_distance = info["distance_to_goal"]
        #     delta_dist = self._prev_distance - curr_distance
        # reward += sel  # positive when getting closer
        #     self._prev_distance = curr_distance

        # Small throttle penalty for standing still

        if mean_throttle < 0.1:
            reward -= 1.0

        # Steering penalty: discourage excessive turning / circling
        reward -= abs(action[0])

        # Circularity detection: high variance in steering with low net progress
        self.steering_hist.append(action[0])
        self.throttle_hist.append(action[1])

        # Goal reached
        if info["habitat_success"] > 0.0:
            reward += self.k_goal
            print("Goal Reached")
            terminated = True

        # Collision penalty (applies in both modes)
        if info.get("hit", False):
            reward -= 1.0
            if self.eval_mode:
                terminated = True

        return obs, reward, terminated, truncated, info

    def set_eval_mode(self, enabled: bool = True):
        """Toggle eval mode where collisions terminate the episode."""
        self.eval_mode = enabled

    def _compute_episodic_curiosity(self, stacked_pixels):
        """Episodic curiosity via nearest-neighbor cosine distance in feature space.

        Stores L2-normalised features in the episodic buffer and uses a
        vectorised matrix multiply for the NN search (no Python loop).

        Args:
            stacked_pixels: (3 * per_frame_dim,) concatenated per-frame features.
        Returns:
            float: cosine distance (0-2 range, higher = more novel).
        """
        # breakpoint()
        if self._per_frame_dim is None:
            self._per_frame_dim = stacked_pixels.shape[-1] // 3
        feat_norm = stacked_pixels[-self._per_frame_dim:]
        feat_norm = feat_norm / (np.linalg.norm(feat_norm) + 1e-8)

        if not self.curiosity_memory:
            self.curiosity_memory.append(feat_norm.copy())
            return 0.0

        # Vectorised NN search: (N, D) @ (D,) → (N,)
        memory = np.array(self.curiosity_memory, dtype=np.float32)
        similarities = memory @ feat_norm
        nn_sim = float(np.mean(similarities))
        nn_sim = max(0.0, min(nn_sim, 1.0))
        self.curiosity_memory.append(feat_norm.copy())
        return nn_sim

    # ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers (reuse from wrappers.py)
# ═══════════════════════════════════════════════════════════════════════════════

class VideoRecorder(gym.Wrapper):
    """Records episode frames and writes MP4 videos to disk.

    Supports two modes:
      - Manual: start/stop via start_recording() / stop_and_save().
      - Periodic: call record_next_episode() to capture the next
        complete episode from reset to termination, auto-saving on
        episode end.

    In headless (rgb_array) mode, captures the raw RGB observation.
    In human-render mode, captures the composed debug view.
    """

    def __init__(self, env, video_dir: str, fps: int = 30):
        super().__init__(env)
        self.video_dir = video_dir
        self.fps = fps
        os.makedirs(video_dir, exist_ok=True)
        self._recording = False
        self._record_next = False
        self._frames = []
        self._episode_count = 0

    def start_recording(self):
        """Start recording immediately (captures frames from next step/reset)."""
        self._recording = True
        self._frames = []

    def record_next_episode(self):
        """Record the next complete episode from reset to termination."""
        self._record_next = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._recording and len(self._frames) < 3000:
            frame = self._get_display_frame(obs, info)
            if frame is not None:
                self._frames.append(frame)
        if self._recording and (terminated or truncated):
            self._episode_count += 1
            self.save(f"episode_{self._episode_count:04d}.mp4")
            self._frames = []
            self._recording = False
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._recording and len(self._frames) < 3000:
            frame = self._get_display_frame(obs, info)
            if frame is not None:
                self._frames.append(frame)
        if self._record_next:
            self._recording = True
            self._frames = []
            self._record_next = False
            frame = self._get_display_frame(obs, info)
            if frame is not None:
                self._frames.append(frame)
        return obs, info

    def _get_display_frame(self, obs, info):
        try:
            raw = self.env.unwrapped.render()
            if raw is not None:
                if raw.ndim == 3 and raw.shape[2] == 4:
                    raw = raw[:, :, :3]
                return raw
        except Exception:
            pass
        # Fallback: reconstruct from observation if render() fails
        pixels = obs.get("pixels", None) if isinstance(obs, dict) else None
        if pixels is not None and isinstance(pixels, np.ndarray):
            if pixels.ndim == 3 and pixels.shape[-1] in (1, 3, 4):
                vis = (pixels[:, :, :3] * 255).astype(np.uint8)
                h, w = vis.shape[:2]
                vis = cv2.resize(vis, (w * 4, h * 4),
                                 interpolation=cv2.INTER_NEAREST)
                return vis
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
        self._recording = False
        self._frames = []


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
