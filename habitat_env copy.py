"""
Habitat-Lab environment for image-goal navigation.

Pure habitat-lab pipeline — no direct habitat_sim calls.

All simulator access goes through habitat-lab's Env / Task / Simulator
abstractions. Stepping uses the VelocityControl action registered in the
task config. Observations come from habitat-lab sensors (rgb, imagegoal).
Metrics come from habitat-lab measures (distance_to_goal,
distance_to_goal_reward, success, spl, collisions).

Observation:  {"image": (H,W,3) uint8 RGB, "imu": (10,) float32}
              imu = [ang_cmd, lin_cmd, ax, ay, gx, gy,
                     mean_resultant_accel_20, mean_throttle_20,
                     geo_dist (or -1 if masked), mask_flag]
Action:       [angular_velocity, linear_velocity]  in [-0.5,0.5] x [0.0,0.5]
Info keys:    velocity, collision, blocked, forward_vel, actual_vel, hit,
              accel, gyro, pos, yaw, goal_image, distance_to_goal,
              goal_pos, delta_x, habitat_success, habitat_spl,
              habitat_distance_to_goal_reward
"""

from collections import deque
import logging
import math
import os
import random
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf

from configs.habitat_config import HabitatNavConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# habitat-lab imports — all simulator access goes through these.
# ---------------------------------------------------------------------------
import habitat
from habitat import Env as HabitatEnv
from habitat.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import VelocityControlActionConfig
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal


# ============================================================================
# Geometry helpers (pure numpy/math — no habitat_sim imports)
# ============================================================================

def _quat_to_xyzw(q) -> np.ndarray:
    """Convert any quaternion-like to [x,y,z,w]."""
    if hasattr(q, 'x') and hasattr(q, 'w'):
        return np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    arr = np.asarray(q, dtype=np.float64).ravel()
    return arr


def _quat_to_yaw(q) -> float:
    """Yaw (rotation around Y-up) from unit quaternion [x,y,z,w]."""
    a = _quat_to_xyzw(q)
    x, y, z, w = a[0], a[1], a[2], a[3]
    siny = 2.0 * (w * y + x * z)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def _yaw_to_quat_list(yaw: float) -> list:
    """Yaw (rad) → [x,y,z,w] as a plain list (for NavigationEpisode)."""
    half = yaw * 0.5
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] → 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y+z*z),  2*(x*y-w*z),     2*(x*z+w*y)],
        [2*(x*y+w*z),      1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),      2*(y*z+w*x),     1 - 2*(x*x+y*y)],
    ], dtype=np.float64)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ============================================================================
# Config builder
# ============================================================================

def _build_habitat_config(cfg: HabitatNavConfig, scene_path: str) -> "OmegaConf.DictConfig":
    """Build an OmegaConf DictConfig for habitat.Env from HabitatNavConfig."""
    from habitat.config.default import register_configs
    register_configs()

    base_cfg = get_habitat_config(
        "benchmark/nav/imagenav/imagenav_gibson.yaml"
    )
    OmegaConf.set_struct(base_cfg, False)
    OmegaConf.set_readonly(base_cfg, False)

    sim = base_cfg.habitat.simulator
    agent = sim.agents.main_agent

    # Scene
    sim.scene = scene_path
    sim.scene_dataset = cfg.scene_dataset_path or "default"
    sim.default_agent_id = 0
    sim.habitat_sim_v0.gpu_device_id = cfg.gpu_device_id
    sim.habitat_sim_v0.allow_sliding = cfg.allow_sliding

    # Agent geometry
    agent.height = cfg.agent_height
    agent.radius = 0.1

    # RGB sensor
    rgb = agent.sim_sensors.rgb_sensor
    rgb.height = cfg.image_height
    rgb.width = cfg.image_width
    rgb.position = [0.0, cfg.agent_height, 0.0]
    rgb.hfov = int(cfg.hfov)

    # Drop depth sensor (not needed)
    if hasattr(agent.sim_sensors, "depth_sensor"):
        del agent.sim_sensors.depth_sensor

    # Physics — keep enabled so VelocityControl works
    # base_cfg.habitat.simulator.enable_physics = True

    # Remove heavy unused measurements
    if hasattr(base_cfg.habitat.task.measurements, "top_down_map"):
        del base_cfg.habitat.task.measurements.top_down_map

    # Continuous velocity action
    max_ang_deg = math.degrees(cfg.max_angular_velocity)
    vc = OmegaConf.structured(VelocityControlActionConfig(
        lin_vel_range=[0.0, cfg.max_linear_velocity],
        ang_vel_range=[-max_ang_deg, max_ang_deg],
        min_abs_lin_speed=0.001,
        min_abs_ang_speed=0.001,
        time_step=1.0 / cfg.control_frequency,
    ))
    base_cfg.habitat.task.actions = {"velocity_control": vc}

    # Let our wrapper handle episode length
    base_cfg.habitat.environment.max_episode_steps = 1_000_000

    # Dataset placeholder (episodes injected programmatically)
    base_cfg.habitat.dataset.type = "PointNav-v1"
    base_cfg.habitat.dataset.data_path = ""
    base_cfg.habitat.dataset.scenes_dir = "data"
    base_cfg.habitat.seed = cfg.seed

    return base_cfg


# ============================================================================
# HabitatNavEnv — pure habitat-lab
# ============================================================================

class HabitatNavEnv(gym.Env):
    """
    Gymnasium wrapper around habitat-lab for image-goal navigation.

    Stepping:     habitat-lab VelocityControl action (no raw habitat_sim calls)
    Observations: habitat-lab RGB sensor + ImageGoalSensor
    Metrics:      habitat-lab DistanceToGoal, Collisions, success, SPL
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Optional[HabitatNavConfig] = None,
        render_mode: str = "rgb_array",
    ):
        self._cfg = config or HabitatNavConfig()
        self.render_mode = render_mode
        self._imu_dim = 10

        # ── Headless HPC support ───────────────────────────────────────
        if self._cfg.headless:
            os.environ.pop("DISPLAY", None)
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        H, W = self._cfg.image_height, self._cfg.image_width

        # ── Build habitat-lab env ──────────────────────────────────────
        self._current_scene: str = self._cfg.scene_path
        hab_cfg = _build_habitat_config(self._cfg, self._current_scene)

        # Placeholder dataset (one episode; replaced on every reset)
        self._dataset = self._make_placeholder_dataset(self._current_scene)

        self._env: HabitatEnv = HabitatEnv(hab_cfg, dataset=self._dataset)
        self._hab_cfg = hab_cfg  # kept for scene switching

        # Expose the habitat-lab simulator wrapper (not raw habitat_sim)
        # All sim access goes through self._hsim (habitat's HabitatSimV2)
        self._hsim = self._env._sim  # habitat.sims.habitat_simulator.HabitatSim

        # Pathfinder from habitat-lab sim wrapper (safe to use for navmesh queries)
        self._pathfinder = self._hsim.pathfinder

        self._rng = np.random.default_rng(self._cfg.seed)
        self._scene_paths = self._cfg.get_scene_paths()

        # ── Spaces ────────────────────────────────────────────────────
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (H, W, 3), dtype=np.uint8),
            "imu":   spaces.Box(-np.inf, np.inf, (self._imu_dim,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-0.5, 0.0], dtype=np.float32),
            high=np.array([0.5,  0.5], dtype=np.float32),
            dtype=np.float32,
        )

        # ── Internal state ─────────────────────────────────────────────
        self._dt = 1.0 / self._cfg.control_frequency
        self._step_count = 0
        self._episode_counter = 0

        self._prev_pos: np.ndarray = np.zeros(3)
        self._prev_rot: np.ndarray = np.array([0., 0., 0., 1.])
        self._prev_vel: np.ndarray = np.zeros(3)

        self._current_accel = np.zeros(3, dtype=np.float32)
        self._current_gyro  = np.zeros(3, dtype=np.float32)
        self._forward_vel: float = 0.0
        self._actual_vel:  float = 0.0
        self._delta_x = np.zeros(3, dtype=np.float64)
        self._collision_detected: bool = False

        self._current_image = np.zeros((H, W, 3), dtype=np.uint8)
        self._current_imu   = np.zeros(self._imu_dim, dtype=np.float32)
        self._goal_image:    Optional[np.ndarray] = None
        self._goal_position: np.ndarray = np.zeros(3)

        self._accel_hist   = deque(maxlen=20)
        self._throttle_hist = deque(maxlen=20)

        self.action   = np.zeros(2, dtype=np.float32)
        self.geo_dist: float = 0.0
        self.goals = []

        # Top-down map cache
        self._topdown_map:    Optional[np.ndarray] = None
        self._topdown_bounds = None
        self._topdown_res: float = 0.02
        self._trajectory: list = []

    # ------------------------------------------------------------------
    # Dataset / episode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_placeholder_dataset(scene_path: str) -> PointNavDatasetV1:
        ds = PointNavDatasetV1()
        ds.episodes = [
            NavigationEpisode(
                episode_id="0",
                scene_id=scene_path,
                start_position=[0.0, 0.0, 0.0],
                start_rotation=[0.0, 0.0, 0.0, 1.0],
                goals=[NavigationGoal(position=[1.0, 0.0, 0.0])],
            )
        ]
        return ds

    def _sample_navigable(self) -> np.ndarray:
        return np.array(self._pathfinder.get_random_navigable_point())

    def _sample_episode(self) -> NavigationEpisode:
        """Sample start + goal positions on the navmesh."""
        max_dist = 10.0
        start = self._sample_navigable()
        yaw   = self._rng.uniform(0, 2 * math.pi)
        rot   = _yaw_to_quat_list(yaw)

        for _ in range(50):
            dist  = self._rng.uniform(0.5, max_dist)
            angle = self._rng.uniform(0, 2 * math.pi)
            # Candidate goal in world space (flat plane, Y from navmesh snap)
            candidate = start + np.array([
                dist * math.sin(angle),
                0.0,
                dist * math.cos(angle),
            ])
            if not self._pathfinder.is_navigable(candidate):
                max_dist *= 1.1
                continue
            geo = self._hsim.geodesic_distance(start, candidate)
            if 0.5 <= geo <= max_dist:
                ep = NavigationEpisode(
                    episode_id=str(self._episode_counter),
                    scene_id=self._current_scene,
                    start_position=start.tolist(),
                    start_rotation=rot,
                    goals=[NavigationGoal(position=candidate.tolist())],
                )
                self._episode_counter += 1
                return ep
            max_dist *= 1.1

        # Fallback: retry with two fresh navigable points
        return self._sample_episode()

    # ------------------------------------------------------------------
    # Scene switching
    # ------------------------------------------------------------------

    def _switch_scene(self, scene_path: str):
        if scene_path == self._current_scene:
            return
        logger.info("Switching scene: %s → %s", self._current_scene, scene_path)
        hab_cfg = _build_habitat_config(self._cfg, scene_path)
        self._hab_cfg = hab_cfg
        self._hsim.reconfigure(hab_cfg.habitat.simulator)
        self._pathfinder = self._hsim.pathfinder
        self._current_scene = scene_path
        self._topdown_map = None
        self._topdown_bounds = None
        self.goals = []

    # ------------------------------------------------------------------
    # IMU synthesis
    # ------------------------------------------------------------------

    def _synthesize_imu(self) -> np.ndarray:
        agent_state = self._hsim.get_agent_state()
        curr_pos = np.array(agent_state.position, dtype=np.float64)
        curr_rot = _quat_to_xyzw(agent_state.rotation)

        curr_yaw = _quat_to_yaw(curr_rot)
        prev_yaw = _quat_to_yaw(self._prev_rot)
        yaw_rate = _wrap(curr_yaw - prev_yaw) / self._dt

        curr_vel   = (curr_pos - self._prev_pos) / self._dt
        world_accel = (curr_vel - self._prev_vel) / self._dt

        R = _quat_to_R(curr_rot)
        local_accel = R.T @ world_accel

        ax = -local_accel[2]   # forward
        ay =  local_accel[0]   # lateral
        gx, gy = 0.0, 0.0
        gz = yaw_rate

        if self._cfg.imu_noise_std > 0:
            n = self._rng.normal(0, self._cfg.imu_noise_std, 6)
            ax += n[0]; ay += n[1]
            gx += n[3]; gy += n[4]

        self._prev_pos = curr_pos
        self._prev_rot = curr_rot
        self._prev_vel = curr_vel
        self._current_accel = np.array([ax, ay, 0.0], dtype=np.float32)
        self._current_gyro  = np.array([gx, gy, gz],  dtype=np.float32)

        mean_accel    = float(np.mean(self._accel_hist))    if self._accel_hist    else 0.0
        mean_throttle = float(np.mean(self._throttle_hist)) if self._throttle_hist else 0.0

        gd   = self.geo_dist
        mask = random.random() < 0.3
        if mask:
            gd = -1.0

        return np.array(
            [self.action[0], self.action[1],
             ax, ay, gx, gy,
             mean_accel, mean_throttle,
             gd, float(mask)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Observation / info helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        return {"image": self._current_image, "imu": self._current_imu}

    def _get_info(self) -> dict:
        agent_state = self._hsim.get_agent_state()
        pos = np.array(agent_state.position, dtype=np.float64)
        self.geo_dist = self._hsim.geodesic_distance(pos, self._goal_position)

        metrics = {}
        try:
            metrics = self._env.get_metrics()
        except Exception:
            pass

        return {
            "velocity":   {"cms": self._forward_vel * 100, "ms": self._forward_vel},
            "collision":  {"detected": self._collision_detected},
            "blocked":     self._collision_detected,
            "forward_vel": self._forward_vel,
            "actual_vel":  self._actual_vel,
            "hit":         self._collision_detected,
            "accel":       self._current_accel,
            "gyro":        self._current_gyro,
            "pos":         pos,
            "delta_x":     self._delta_x,
            "yaw":         _quat_to_yaw(agent_state.rotation),
            "goal_image":  self._goal_image,
            "distance_to_goal": self.geo_dist,
            "goal_pos":    self._goal_position,
            "habitat_success": metrics.get("success", 0.0),
            "habitat_spl":     metrics.get("spl", 0.0),
            "habitat_distance_to_goal_reward": metrics.get("distance_to_goal_reward", 0.0),
        }

    # ------------------------------------------------------------------
    # Gym API — reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self.action = np.zeros(2, dtype=np.float32)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Optional scene randomization
        if self._cfg.randomize_scenes and random.randint(0, 100) == 1:
            self._switch_scene(self._random_scene())

        # Sample episode and inject it before habitat.Env.reset()
        episode = self._sample_episode()
        self._env._current_episode = episode
        self._env._episode_over = False
        self._env.reset()

        # Teleport agent to sampled start (task.reset doesn't move agent
        # when we bypass Env.reconfigure)
        self._hsim.set_agent_state(
            position=episode.start_position,
            rotation=episode.start_rotation,
            agent_id=0,
            reset_sensors=True,
        )

        # Re-fetch observations from correct position
        self._hsim.get_observations_at(
            position=episode.start_position,
            rotation=episode.start_rotation,
            keep_agent_at_new_pose=True,
        )

        # Reset measures with correct starting position
        self._env._task.measurements.reset_measures(
            episode=episode,
            task=self._env._task,
            observations=obs,
        )

        # Extract image obs
        self._current_image = np.array(obs["rgb"][:, :, :3], dtype=np.uint8)
        self._goal_image     = np.array(obs["imagegoal"][:, :, :3], dtype=np.uint8) \
                               if "imagegoal" in obs else np.zeros_like(self._current_image)
        self._goal_position  = np.array(episode.goals[0].position, dtype=np.float64)

        # Reset internal state
        agent_state = self._hsim.get_agent_state()
        self._prev_pos = np.array(agent_state.position, dtype=np.float64)
        self._prev_rot = _quat_to_xyzw(agent_state.rotation)
        self._prev_vel = np.zeros(3)
        self._current_accel = np.zeros(3, dtype=np.float32)
        self._current_gyro  = np.zeros(3, dtype=np.float32)
        self._collision_detected = False
        self._forward_vel = 0.0
        self._actual_vel  = 0.0
        self._delta_x     = np.zeros(3, dtype=np.float64)
        self._step_count  = 0
        self._current_imu = np.zeros(self._imu_dim, dtype=np.float32)
        self.geo_dist     = 0.0
        self._accel_hist.clear()
        self._throttle_hist.clear()
        self._trajectory = [self._prev_pos.copy()]

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------
    # Gym API — step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        self.action = action
        ang_cmd = float(np.clip(action[0], -1.0, 1.0))
        lin_cmd = float(np.clip(action[1],  0.0, 1.0))

        # Map normalised [-1,1] / [0,1] → physical velocities
        ang_vel_deg = math.degrees(ang_cmd * self._cfg.max_angular_velocity)
        lin_vel_ms  = lin_cmd * self._cfg.max_linear_velocity

        # ── Step through habitat-lab VelocityControl action ──────────────
        # habitat-lab expects: {"velocity_control": {"angular_velocity": deg,
        #                                            "linear_velocity":  m/s}}
        initial_pos = np.array(self._hsim.get_agent_state().position, dtype=np.float64)

        # for _ in range(self._cfg.frame_skip):
        step_action = {
            "action": "velocity_control",
            "action_args": {
                "angular_velocity": ang_vel_deg,
                "linear_velocity":  lin_vel_ms,
            },
        }
        obs = self._env.step(step_action)

        # Check if habitat-lab flagged a collision this sub-step
        metrics = {}
        try:
            metrics = self._env.get_metrics()
        except Exception:
            pass
        if metrics.get("collisions", {}).get("is_collision", False):
            self._collision_detected = True

        # ── Extract image from last step's obs ────────────────────────────
        self._current_image = np.array(obs["rgb"][:, :, :3], dtype=np.uint8)

        # ── Velocity from displacement ─────────────────────────────────────
        final_pos = np.array(self._hsim.get_agent_state().position, dtype=np.float64)
        total_time = self._cfg.frame_skip * self._dt
        self._delta_x     = final_pos - initial_pos
        self._forward_vel = lin_vel_ms
        self._actual_vel  = float(np.linalg.norm(self._delta_x) / total_time)

        # ── IMU ───────────────────────────────────────────────────────────
        self._current_imu = self._synthesize_imu()
        resultant = float(np.linalg.norm(self._current_accel[:2]))
        self._accel_hist.append(resultant)
        self._throttle_hist.append(action[1])

        self._step_count += 1
        self._trajectory.append(final_pos.copy())

        # ── Reward ────────────────────────────────────────────────────────
        try:
            all_metrics = self._env.get_metrics()
        except Exception:
            all_metrics = {}

        reward  = float(all_metrics.get("distance_to_goal_reward", 0.0))
        reward += float(np.linalg.norm(self._delta_x))
        if self._collision_detected:
            reward -= 1.0

        # ── Termination ───────────────────────────────────────────────────
        pos = np.array(self._hsim.get_agent_state().position, dtype=np.float64)
        self.geo_dist  = self._hsim.geodesic_distance(pos, self._goal_position)
        terminated = self.geo_dist < self._cfg.goal_radius
        truncated  = False

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def _random_scene(self) -> str:
        if len(self._scene_paths) <= 1:
            return self._scene_paths[0] if self._scene_paths else self._current_scene
        return str(self._rng.choice(self._scene_paths))

    # ------------------------------------------------------------------
    # Top-down map / rendering
    # ------------------------------------------------------------------

    def _build_topdown_map(self):
        bounds = self._pathfinder.get_bounds()
        min_b = np.array([bounds[0][0], bounds[0][2]], dtype=np.float64)
        max_b = np.array([bounds[1][0], bounds[1][2]], dtype=np.float64)
        self._topdown_bounds = (min_b, max_b)
        td = self._pathfinder.get_topdown_view(
            meters_per_pixel=self._topdown_res, height=0.09
        )
        rgb = np.zeros((*td.shape, 3), dtype=np.uint8)
        rgb[td]  = [200, 200, 200]
        rgb[~td] = [40,  40,  40]
        self._topdown_map = rgb

    def _world_to_px(self, xz: np.ndarray):
        if self._topdown_bounds is None:
            return (0, 0)
        min_b, _ = self._topdown_bounds
        col = int((xz[0] - min_b[0]) / self._topdown_res)
        row = int((xz[1] - min_b[1]) / self._topdown_res)
        row = np.clip(row, 0, self._topdown_map.shape[0] - 1)
        col = np.clip(col, 0, self._topdown_map.shape[1] - 1)
        return (row, col)

    def _render_topdown(self) -> np.ndarray:
        if self._topdown_map is None:
            self._build_topdown_map()
        img = self._topdown_map.copy()

        if len(self._trajectory) > 1:
            for i in range(1, len(self._trajectory)):
                p1 = self._world_to_px(self._trajectory[i-1][[0, 2]])
                p2 = self._world_to_px(self._trajectory[i][[0, 2]])
                cv2.line(img, (p1[1], p1[0]), (p2[1], p2[0]), (0, 200, 0), 1)

        gp = self._world_to_px(self._goal_position[[0, 2]])
        cv2.circle(img, (gp[1], gp[0]), 5, (0, 0, 255), -1)

        state = self._hsim.get_agent_state()
        ap = self._world_to_px(np.array(state.position)[[0, 2]])
        cv2.circle(img, (ap[1], ap[0]), 4, (255, 0, 0), -1)
        yaw = _quat_to_yaw(state.rotation)
        ex = int(ap[1] + 10 * math.sin(yaw))
        ey = int(ap[0] - 10 * math.cos(yaw))
        cv2.line(img, (ap[1], ap[0]), (ex, ey), (255, 100, 0), 2)
        return img

    def render(self):
        if self.render_mode == "rgb_array":
            return self._current_image
        self._render_frame()

    def _render_frame(self):
        VH, VW = 240, 320
        agent_view = cv2.cvtColor(
            cv2.resize(self._current_image, (VW, VH)), cv2.COLOR_RGB2BGR
        )
        if self._goal_image is not None:
            goal_view = cv2.cvtColor(
                cv2.resize(self._goal_image, (VW, VH)), cv2.COLOR_RGB2BGR
            )
        else:
            goal_view = np.zeros((VH, VW, 3), dtype=np.uint8)

        td = self._render_topdown()
        th = VH * 2
        scale = th / td.shape[0]
        tw = int(td.shape[1] * scale)
        td_bgr = cv2.cvtColor(
            cv2.resize(td, (tw, th), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_RGB2BGR,
        )

        state = self._hsim.get_agent_state()
        pos   = np.array(state.position, dtype=np.float64)
        dist  = self._hsim.geodesic_distance(pos, self._goal_position)
        for i, line in enumerate([
            f"step:{self._step_count}  dist:{dist:.2f}m",
            f"coll:{'YES' if self._collision_detected else 'no'}  vel:{self._forward_vel:.2f}m/s",
        ]):
            cv2.putText(td_bgr, line, (10, 20 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        left     = np.concatenate([agent_view, goal_view], axis=0)
        combined = np.concatenate([left, td_bgr], axis=1)
        cv2.imshow("HabitatNav", combined)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    def close(self):
        cv2.destroyAllWindows()
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()
            self._env = None