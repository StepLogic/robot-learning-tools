

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

        reward = -1

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
        mean_throttle = float(np.mean(self.unwrapped._throttle_history))
        if  mean_throttle < 0.1:
            reward -= 1

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
            reward += 1000
            print("Goal Reached")
            terminated = True

        # # # Collision penalty
        if info.get("hit", False):
            reward -= 1.0
            # truncated = True

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
