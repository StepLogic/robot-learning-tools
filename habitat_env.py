"""
Habitat-Lab environment for image-goal navigation.

Uses habitat.Env (from habitat-lab) instead of raw habitat_sim.Simulator.
Leverages habitat-lab's built-in:
  - ImageGoalSensor for goal image rendering
  - DistanceToGoal measure for geodesic distance
  - Collisions measure for collision tracking
  - NavigationEpisode for episode management
  - VelocityAction for continuous control

Preserves the same obs/info/action interface as the previous raw-sim version
so the existing wrapper stack (StackingWrapper → MobileNetFeatureWrapper →
GoalImageWrapper → HabitatRewardWrapper) works unchanged.

Observation:  {"image": (H,W,3) uint8 RGB, "imu": (6,) float32}
              imu = [ax, ay, gx, gy, mean_resultant_accel_20, mean_throttle_20]
Action:       [angular_velocity, linear_velocity]  normalised [-1,1] x [0,1]
Info keys:    velocity, collision, blocked, forward_vel, hit, accel, gyro,
              pos, yaw, goal_image, distance_to_goal, goal_pos
"""

from collections import deque
import logging
import math
from dataclasses import asdict
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
# Try importing habitat-lab — set a flag so we can provide a dummy fallback.
# ---------------------------------------------------------------------------
# try:
import habitat
from habitat import Env as HabitatEnv
from habitat.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    VelocityControlActionConfig,
    ProximitySensorConfig,
)
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
import habitat_sim
import magnum as mn
HAS_HABITAT_LAB = True
# except ImportError:
#     HAS_HABITAT_LAB = False
#     logger.warning("habitat-lab not installed — HabitatNavEnv will not be usable.")


# ============================================================================
# Quaternion / geometry helpers
# ============================================================================

def _quat_to_xyzw(q) -> np.ndarray:
    """Convert any quaternion-like object to [x,y,z,w] numpy array.

    Handles numpy-quaternion (has .w/.x/.y/.z) and plain arrays.
    """
    if hasattr(q, 'w') and hasattr(q, 'x') and hasattr(q, 'y') and hasattr(q, 'z'):
        return np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    return np.asarray(q, dtype=np.float64).ravel()


def _quat_to_yaw(q) -> float:
    """Extract yaw (rotation around Y-up) from a unit quaternion [x,y,z,w]."""
    a = _quat_to_xyzw(q)
    w, x, y, z = a[3], a[0], a[1], a[2]
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert yaw angle (rad, around Y-up) to [x,y,z,w] quaternion."""
    half = yaw * 0.5
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)],
                    dtype=np.float64)


def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] to 3x3 rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ============================================================================
# Config builder
# ============================================================================

def _build_habitat_config(cfg: HabitatNavConfig) -> "OmegaConf.DictConfig":
    """Build an OmegaConf DictConfig for habitat.Env from HabitatNavConfig."""
    from habitat.config.default import register_configs
    register_configs()

    base_cfg = get_habitat_config(
        "benchmark/nav/imagenav/imagenav_gibson.yaml"
    )
    OmegaConf.set_struct(base_cfg, False)
    OmegaConf.set_readonly(base_cfg, False)

    # Simulator settings
    base_cfg.habitat.simulator.scene = cfg.scene_path
    base_cfg.habitat.simulator.scene_dataset = (
        cfg.scene_dataset_path if cfg.scene_dataset_path else "default"
    )
    base_cfg.habitat.simulator.default_agent_id = 0
    base_cfg.habitat.simulator.agents.main_agent.height = cfg.agent_height
    base_cfg.habitat.simulator.agents.main_agent.radius = 0.1
    base_cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = cfg.image_height
    base_cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = cfg.image_width
    base_cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [
        0.0, cfg.agent_height, 0.0
    ]
    base_cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = int(cfg.hfov)
    base_cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = cfg.gpu_device_id
    base_cfg.habitat.simulator.habitat_sim_v0.allow_sliding = cfg.allow_sliding
    base_cfg.habitat.seed = cfg.seed

    # Remove depth sensor (we only need RGB + imagegoal)
    del base_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    # Remove top_down_map (heavy, unused)
    del base_cfg.habitat.task.measurements.top_down_map
    # Keep: distance_to_goal, success, spl, distance_to_goal_reward

    # Add proximity sensor
    if cfg.proximity_sensor:
        base_cfg.habitat.task.lab_sensors.proximity_sensor = OmegaConf.structured(ProximitySensorConfig())

    # Use VelocityAction for continuous control
    max_ang_deg = math.degrees(cfg.max_angular_velocity)
    vc = OmegaConf.structured(VelocityControlActionConfig(
        lin_vel_range=[0.0, cfg.max_linear_velocity],
        ang_vel_range=[-max_ang_deg, max_ang_deg],
        min_abs_lin_speed=0.001,
        min_abs_ang_speed=0.001,
        time_step=1.0 / cfg.control_frequency,
    ))
    base_cfg.habitat.task.actions = {"velocity_control": vc}

    # Environment
    base_cfg.habitat.environment.max_episode_steps = 100000  # Handled by TimeLimit wrapper

    # Dataset: will be provided programmatically
    base_cfg.habitat.dataset.type = "PointNav-v1"
    base_cfg.habitat.dataset.data_path = ""
    base_cfg.habitat.dataset.scenes_dir = "data"

    return base_cfg


# ============================================================================
# HabitatNavEnv
# ============================================================================

class HabitatNavEnv(gym.Env):
    """
    Gymnasium environment wrapping habitat-lab for image-goal navigation.

    Uses habitat.Env for simulator management, ImageGoalSensor for goal
    rendering, DistanceToGoal for geodesic distance, and Collisions for
    collision detection. Continuous control via VelocityAction with custom
    frame_skip integration.

    Mirrors RacerEnv interface:
        obs  = {"image": (H,W,3) uint8 RGB, "imu": (6,) float32}
               imu = [ax, ay, gx, gy, mean_resultant_accel_20, mean_throttle_20]
        info = {velocity, collision, blocked, forward_vel, hit, accel, gyro,
                pos, yaw, goal_image, distance_to_goal, goal_pos}
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: Optional[HabitatNavConfig] = None,
        render_mode: str = "rgb_array",
    ):
        if not HAS_HABITAT_LAB:
            raise RuntimeError(
                "habitat-lab is not installed. Install with: "
                "pip install habitat-lab"
            )

        self._cfg = config or HabitatNavConfig()
        self.render_mode = render_mode

        H, W = self._cfg.image_height, self._cfg.image_width
        self._imu_dimension: int = 11
        # ── Build habitat config and dataset ──────────────────────────────────
        hab_cfg = _build_habitat_config(self._cfg)

        # Create placeholder dataset with one episode.
        # Start and goal are 1m apart so SPL's _start_end_episode_distance
        # is non-zero during env init (avoids ZeroDivisionError).
        self._dataset = PointNavDatasetV1()
        self._dataset.episodes = [
            NavigationEpisode(
                episode_id="0",
                scene_id=self._cfg.scene_path,
                start_position=[0.0, 0.0, 0.0],
                start_rotation=[0.0, 0.0, 0.0, 1.0],
                goals=[NavigationGoal(position=[1.0, 0.0, 0.0])],
            )
        ]

        # Create habitat.Env directly (composition, not RLEnv inheritance)
        self._env = HabitatEnv(hab_cfg, dataset=self._dataset)
        self._sim = self._env._sim
        self._pathfinder = self._sim.pathfinder
        self._rng = np.random.default_rng(self._cfg.seed)
        self._scene_paths = self._cfg.get_scene_paths()
        self._current_scene = self._cfg.scene_path

        # Define observation and action spaces
        H, W = self._cfg.image_height, self._cfg.image_width
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
            "imu":   spaces.Box(low=-np.inf, high=np.inf, shape=(self._imu_dimension,),
                               dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-0.5, 0.0], dtype=np.float32),
            high=np.array([0.5, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

        # Velocity controller for frame_skip stepping
        self._vel_control = habitat_sim.physics.VelocityControl()
        self._vel_control.controlling_lin_vel = True
        self._vel_control.lin_vel_is_local = True
        self._vel_control.controlling_ang_vel = True
        self._vel_control.ang_vel_is_local = True
        self._dt = 1.0 / self._cfg.control_frequency

        # ── Internal state ────────────────────────────────────────────────
        self._step_count = 0
        self._prev_position: np.ndarray = np.zeros(3)
        self._prev_rotation: np.ndarray = np.array([0, 0, 0, 1], dtype=np.float64)
        self._prev_velocity: np.ndarray = np.zeros(3)
        self._current_accel: np.ndarray = np.zeros(3, dtype=np.float32)
        self._current_gyro: np.ndarray = np.zeros(3, dtype=np.float32)
        self._forward_vel: float = 0.0
        self._actual_vel: float = 0.0
        self._delta_x: np.ndarray = np.zeros(3, dtype=np.float64)
        self._collision_detected: bool = False
        self._proximity: float = -1.0
        self._current_image: np.ndarray = np.zeros((H, W, 3), dtype=np.uint8)
     
        self._current_imu: np.ndarray = np.zeros(self._imu_dimension, dtype=np.float32)
        self._goal_image: Optional[np.ndarray] = None
        self._goal_position: np.ndarray = np.zeros(3)
        self._episode_counter = 0
        self._goals = []
        self._accel_resultant_history = deque(maxlen=20)
        self._throttle_history = deque(maxlen=20)
        # ── Top-down map (cached on first access) ──────────────────────────
        self._topdown_map: Optional[np.ndarray] = None  # RGB map image
        self._topdown_bounds = None  # (min_xz, max_xz) in world coords
        self._topdown_resolution: float = 0.02  # meters per pixel
        self._trajectory: list = []  # agent positions for trail
        self.goals=[]
        self.enable_random_masking=False
        self.random_mask_prob=0.1
        self.geo_dist=0.0
        self.action =np.zeros(2,dtype=np.float32)

    def set_random_masking(self, enabled: bool = True, mask_prob: float = 0.1):
        """Toggle random goal masking for curriculum learning."""
        self.enable_random_masking = enabled
        self.random_mask_prob = mask_prob
    # ── Scene randomization ─────────────────────────────────────────────

    def _switch_scene(self, scene_path: str):
        """Reconfigure the simulator to load a different scene."""
        if scene_path == self._current_scene:
            return
        logger.info("Switching scene: %s → %s", self._current_scene, scene_path)
        # Build fresh config with the new scene, then reconfigure the simulator
        # directly (bypassing Env.reconfigure which has config-struct issues).
        hab_cfg = _build_habitat_config(self._cfg)
        hab_cfg.habitat.simulator.scene = scene_path
        if not self._cfg.scene_dataset_path:
            hab_cfg.habitat.simulator.scene_dataset = "default"
        # Update the env's stored config so episode generation uses the new scene
        self._env._config = hab_cfg
        # Reconfigure the simulator backend directly
        self._sim.reconfigure(hab_cfg.habitat.simulator)
        self._pathfinder = self._sim.pathfinder
        self._current_scene = scene_path
        # Invalidate cached top-down map
        self._topdown_map = None
        self._topdown_bounds = None
        self._goals=[]

    def _random_scene(self) -> str:
        """Pick a random scene from the scene pool."""
        if len(self._scene_paths) <= 1:
            return self._scene_paths[0] if self._scene_paths else self._cfg.scene_path
        return self._rng.choice(self._scene_paths)

    # ── Episode sampling ─────────────────────────────────────────────────

    def _sample_navigable_point(self) -> np.ndarray:
        """Sample a random navigable point on the navmesh."""
        # for _ in range(100):
        point = np.array(self._pathfinder.get_random_navigable_point())
            # if self._pathfinder.is_navigable(point):
            # return point
        # Fallback: snap a random point to navmesh
        point = np.array(self._pathfinder.get_random_navigable_point())
        return point

    # ── Top-down map ──────────────────────────────────────────────────────

    def _build_topdown_map(self):
        """Build and cache a top-down map from the navmesh."""
        bounds = self._pathfinder.get_bounds()
        min_b = np.array([bounds[0][0], bounds[0][2]], dtype=np.float64)  # x, z
        max_b = np.array([bounds[1][0], bounds[1][2]], dtype=np.float64)
        self._topdown_bounds = (min_b, max_b)

        # Get navmesh top-down view at agent height
        td = self._pathfinder.get_topdown_view(
            meters_per_pixel=self._topdown_resolution,
            height=0.09,
        )
        # td is bool (H, W): True = navigable
        # Convert to RGB: navigable=light gray, walls=dark
        h, w = td.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[td] = [200, 200, 200]   # navigable = light gray
        rgb[~td] = [40, 40, 40]     # walls = dark
        self._topdown_map = rgb

    def _world_to_map_px(self, world_xz: np.ndarray) -> tuple:
        """Convert world (x, z) coordinates to top-down map pixel (row, col)."""
        if self._topdown_bounds is None:
            return (0, 0)
        min_b, max_b = self._topdown_bounds
        res = self._topdown_resolution
        col = int((world_xz[0] - min_b[0]) / res)
        row = int((world_xz[1] - min_b[1]) / res)
        row = np.clip(row, 0, self._topdown_map.shape[0] - 1)
        col = np.clip(col, 0, self._topdown_map.shape[1] - 1)
        return (row, col)

    def _render_topdown(self) -> np.ndarray:
        """Render top-down map with agent, goal, and trajectory overlay."""
        if self._topdown_map is None:
            self._build_topdown_map()

        map_img = self._topdown_map.copy()
        h, w = map_img.shape[:2]

        # Draw trajectory trail (green)
        if len(self._trajectory) > 1:
            for i in range(1, len(self._trajectory)):
                p1 = self._world_to_map_px(self._trajectory[i - 1][[0, 2]])
                p2 = self._world_to_map_px(self._trajectory[i][[0, 2]])
                cv2.line(map_img, (p1[1], p1[0]), (p2[1], p2[0]),
                         (0, 200, 0), 1, cv2.LINE_AA)

        # Draw goal (red circle)
        goal_px = self._world_to_map_px(self._goal_position[[0, 2]])
        cv2.circle(map_img, (goal_px[1], goal_px[0]), 5, (0, 0, 255), -1)

        # Draw agent (blue circle + direction line)
        agent_state = self._sim.get_agent_state()
        agent_pos = np.array(agent_state.position, dtype=np.float64)
        agent_px = self._world_to_map_px(agent_pos[[0, 2]])
        cv2.circle(map_img, (agent_px[1], agent_px[0]), 4, (255, 0, 0), -1)
        # Direction indicator
        yaw = _quat_to_yaw(agent_state.rotation)
        dir_len = 10  # pixels
        end_col = int(agent_px[1] + dir_len * math.sin(yaw))
        end_row = int(agent_px[0] - dir_len * math.cos(yaw))
        cv2.line(map_img, (agent_px[1], agent_px[0]), (end_col, end_row),
                 (255, 100, 0), 2, cv2.LINE_AA)

        return map_img
    def _sample_episode(self) -> NavigationEpisode:
        """Sample a NavigationEpisode with goal distance exponentially biased toward start."""

        start_pos = self._sample_navigable_point()
        start_yaw = self._rng.uniform(0, 2 * math.pi)
        start_rot = _yaw_to_quat(start_yaw).tolist()
        # print(start_pos)
        # Sample a goal within 40 meters of the agent
        max_attempts = 50
        goal_pos = None
        max_distance = 10.0  # meters
        
        for _ in range(max_attempts):
            # Sample random distance and angle in agent's local frame
            distance = self._rng.uniform(self._cfg.goal_radius + 0.5, max_distance)
            angle = self._rng.uniform(0, 2 * math.pi)  # full 360 degrees
            
            # Convert polar to Cartesian in agent's local frame
            # In Habitat, typically: +X is right, +Z is forward, +Y is up
            local_offset = mn.Vector3(
                distance * math.sin(angle),  # X offset
                0.0,                          # Y offset (keep at same height)
                distance * math.cos(angle)   # Z offset
            )
            
            # Rotate offset by agent's yaw to get world-space offset
            rotation = mn.Quaternion.rotation(mn.Rad(start_yaw), mn.Vector3(0, 1, 0))
            world_offset = rotation.transform_vector(local_offset)
            
            # Add to start position
            candidate = start_pos + world_offset
            
            # Snap to navigable mesh
            # snapped_goal = self._pathfinder.snap_point(candidate)
            snapped_goal = candidate

            # Verify it's navigable and within distance constraint
            if self._pathfinder.is_navigable(snapped_goal):
                geodesic_dist = self._sim.geodesic_distance(start_pos, snapped_goal)
                if geodesic_dist <= max_distance and geodesic_dist >= self._cfg.goal_radius + 0.5:
                    goal_pos = list(snapped_goal)
                    break
            max_distance *= 1.2
        # Fallback: if no valid goal found, sample any navigable point
        if goal_pos is None:
            return self._sample_episode()
        
        ep_id = str(self._episode_counter)
        self._episode_counter += 1
        
        return NavigationEpisode(
            episode_id=ep_id,
            scene_id=self._current_scene,
            start_position=start_pos.tolist(),
            start_rotation=start_rot,
            goals=[NavigationGoal(position=goal_pos)],
        )    # ── IMU synthesis ─────────────────────────────────────────────────────

    def _synthesize_imu(self) -> np.ndarray:
        """Synthesize IMU observation from agent state deltas.

        Returns [ax, ay, gx, gy, mean_resultant_accel_20, mean_throttle_20]
        where ax,ay are body-frame x,y acceleration, gx,gy are x,y angular
        velocity, mean_resultant_accel_20 is the rolling mean of sqrt(ax²+ay²)
        over the last 20 steps, and mean_throttle_20 is the rolling mean of
        throttle (linear velocity command) over the last 20 steps.
        """
        agent_state = self._sim.get_agent_state()
        curr_pos = np.array(agent_state.position, dtype=np.float64)
        curr_rot = _quat_to_xyzw(agent_state.rotation)

        dt = self._dt

        # Angular velocity from yaw change
        curr_yaw = _quat_to_yaw(curr_rot)
        prev_yaw = _quat_to_yaw(self._prev_rotation)
        yaw_rate = _wrap_angle(curr_yaw - prev_yaw) / dt

        # Linear velocity from position change
        curr_vel = (curr_pos - self._prev_position) / dt
        world_accel = (curr_vel - self._prev_velocity) / dt

        # Transform to body frame
        R = _quat_to_rotation_matrix(curr_rot)
        local_accel = R.T @ world_accel

        # Map Habitat body frame to robot convention
        ax = -local_accel[2]  # forward
        ay = local_accel[0]    # lateral
        az = local_accel[1]    # vertical

        gx, gy, gz = 0.0, 0.0, yaw_rate

        # Add optional noise
        if self._cfg.imu_noise_std > 0:
            noise = self._rng.normal(0, self._cfg.imu_noise_std, size=6)
            ax += noise[0]; ay += noise[1]; az += noise[2]
            gx += noise[3]; gy += noise[4]; gz += noise[5]

        self._prev_position = curr_pos
        self._prev_rotation = curr_rot
        self._prev_velocity = curr_vel
        self._current_accel = np.array([ax, ay, az], dtype=np.float32)
        self._current_gyro = np.array([gx, gy, gz], dtype=np.float32)

        # Rolling means (0.0 if history is empty)
        mean_resultant = float(np.mean(self._accel_resultant_history)) if self._accel_resultant_history else 0.0
        mean_throttle = float(np.mean(self._throttle_history)) if self._throttle_history else 0.0
        gd=self.geo_dist
        # print("geo_dist:",gd)
        # mask=random.uniform(0,1)<0.1
        mask_img=random.uniform(0,1)<0.1
        # mask= False
        if self.enable_random_masking and mask_img:
            gd = -1.0
        return np.array([self.action[0],self.action[1], ax,ay,gx,gy,mean_resultant, mean_throttle,gd,float(int(mask_img)), self._proximity], dtype=np.float32)

    # ── Observation extraction ────────────────────────────────────────────

    # Observation extraction — habitat-lab returns RGB directly

    def _get_obs(self) -> dict:
        # print(self._current_image.shape)
        return {"image": self._current_image, "imu": self._current_imu}

    # ── Geodesic distance ─────────────────────────────────────────────────

    def _geodesic_distance(self, start: np.ndarray, end: np.ndarray) -> float:
        """Compute geodesic (navmesh shortest path) distance."""
        return self._sim.geodesic_distance(start, end)

    # ── Habitat-lab metrics ────────────────────────────────────────────────

    def _update_measures(self):
        """Update habitat-lab measures after a physics step."""
        episode = self._env._current_episode
        action = {"velocity_control": np.array([0.0, 0.0], dtype=np.float32)}
        self._env._task.measurements.update_measures(
            episode=episode, action=action, task=self._env._task
        )

    def _get_habitat_metrics(self) -> dict:
        """Get habitat-lab measure metrics (success, SPL, distance_to_goal_reward)."""
        try:
            return self._env.get_metrics()
        except Exception:
            return {}

    # ── Info dict ──────────────────────────────────────────────────────────

    def _get_info(self) -> dict:
        agent_state = self._sim.get_agent_state()
        pos = np.array(agent_state.position, dtype=np.float64)
        self.geo_dist = self._geodesic_distance(pos, self._goal_position)

        # Get habitat-lab metrics (success, SPL, distance_to_goal_reward)
        hab_metrics = self._get_habitat_metrics()

        return {
            "velocity": {"cms": self._forward_vel * 100,
                         "ms": self._forward_vel,
                         "method": "habitat_lab"},
            "collision": {"detected": self._collision_detected,
                          "distance_cm": float("inf"),
                          "threshold_cm": 15.0},
            "blocked": self._collision_detected,
            "forward_vel": self._forward_vel,
            "actual_vel": self._actual_vel,
            "hit":  self._collision_detected,
            "accel": self._current_accel,
            "gyro": self._current_gyro,
            "pos": pos,
            "delta_x": self._delta_x,
            "yaw": _quat_to_yaw(agent_state.rotation),
            "goal_image": self._goal_image,
            "distance_to_goal": self.geo_dist,
            "goal_pos": self._goal_position,
            "habitat_success": float(self.geo_dist < self._cfg.goal_radius),
            "habitat_spl": hab_metrics.get("spl", 0.0),
            "habitat_distance_to_goal_reward": hab_metrics["distance_to_goal_reward"],
        }

    # ── Gym API ────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self.action = np.zeros(2, dtype=np.float32)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Randomly select a scene and reconfigure if it changed
        if random.randint(0,100)==1 and self._cfg.randomize_scenes:
            new_scene = self._random_scene()
            self._switch_scene(new_scene)

        # Sample a new episode and set it BEFORE resetting the habitat env,
        # so measures compute correct distances (not the placeholder episode).
        episode = self._sample_episode()

        self._env._current_episode = episode
        self._env._episode_over = False
        obs = self._env._task.reset(episode=episode)
        self.geo_dist=0.0
        # Manually reposition agent to episode's start position.
        # task.reset() doesn't move the agent because we bypass Env.reconfigure()
        # (which sets is_set_start_state=True via overwrite_sim_config).
        new_state = habitat_sim.AgentState()
        new_state.position = np.array(episode.start_position, dtype=np.float32)
        new_state.rotation = np.array(episode.start_rotation, dtype=np.float32)
        self._sim.get_agent(0).set_state(new_state)

        # Re-get observations from the correct position
        sim_obs = self._sim.get_sensor_observations()
        rgb = sim_obs["rgb"]
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        obs["rgb"] = rgb

        # Extract proximity sensor reading (Habitat-Lab key is "proximity")
        prox = obs.get("proximity", np.array([-1.0], dtype=np.float32))
        self._proximity = float(prox.item() if hasattr(prox, "item") else prox)

        # Reset habitat-lab measures (e.g. SPL needs _previous_position set)
        self._env._task.measurements.reset_measures(
            episode=episode, task=self._env._task, observations=obs
        )

        # Extract RGB observation (habitat-lab returns RGB directly)
        self._current_image = obs["rgb"]
        # cv2.imwrite("current_image.png", self._current_image)
        # Extract goal image from ImageGoalSensor
        # if "imagegoal" in obs:
        self._goal_image = obs["imagegoal"]
        # cv2.imwrite("goal_image.png", self._goal_image)
        # else:
        #     self._goal_image = np.zeros_like(self._current_image)
        # print(episode.goals[0].position)
        self._goal_position = np.array(episode.goals[0].position, dtype=np.float64)

        # Reset internal state
        agent_state = self._sim.get_agent_state()
        self._prev_position = np.array(agent_state.position, dtype=np.float64)
        self._prev_rotation = _quat_to_xyzw(agent_state.rotation)
        self._prev_velocity = np.zeros(3)
        self._current_accel = np.zeros(3, dtype=np.float32)
        self._current_gyro = np.zeros(3, dtype=np.float32)
        self._collision_detected = False
        self._forward_vel = 0.0
        self._actual_vel = 0.0
        self._delta_x = np.zeros(3, dtype=np.float64)
        self._step_count = 0
        self._current_imu = np.zeros(self._imu_dimension, dtype=np.float32)
        self._accel_resultant_history.clear()
        self._throttle_history.clear()
        self._trajectory = [self._prev_position.copy()]

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.action = action
        angular_vel_cmd = float(np.clip(action[0], -1.0, 1.0))
        linear_vel_cmd = float(np.clip(action[1], 0.0, 1.0))

        # Map normalised commands to physical velocities
        ang_vel = angular_vel_cmd * self._cfg.max_angular_velocity  # rad/s
        lin_vel = linear_vel_cmd * self._cfg.max_linear_velocity     # m/s

        # ── Custom frame_skip velocity integration ─────────────────────────
        # We use VelocityControl manually (same as raw habitat-sim approach)
        # to get frame_skip sub-steps with navmesh snapping between each.
        self._vel_control.linear_velocity = np.array([0.0, 0.0, -lin_vel])
        self._vel_control.angular_velocity = np.array([0.0, ang_vel, 0.0])

        agent_state = self._sim.get_agent_state()
        initial_position = np.array(agent_state.position, dtype=np.float64)
        prev_rot_xyzw = _quat_to_xyzw(agent_state.rotation)
        prev_quat = mn.Quaternion(
            mn.Vector3(prev_rot_xyzw[0], prev_rot_xyzw[1], prev_rot_xyzw[2]),
            prev_rot_xyzw[3],
        )
        prev_rigid_state = habitat_sim.RigidState(
            prev_quat, mn.Vector3(*np.array(agent_state.position, dtype=np.float32))
        )

        collision_detected = False
        for _ in range(self._cfg.frame_skip):
            target_rigid_state = self._vel_control.integrate_transform(
                self._dt, prev_rigid_state
            )
            end_pos = self._sim.step_filter(
                prev_rigid_state.translation,
                target_rigid_state.translation,
            )
            # Collision detection: agent moved less than expected
            # Compare per-sub-step displacement, not cumulative from origin
            prev_pos = np.array(prev_rigid_state.translation, dtype=np.float32)
            dist_moved_before = float(np.linalg.norm(
                np.array(target_rigid_state.translation, dtype=np.float32) - prev_pos
            ) ** 2)
            dist_moved_after = float(np.linalg.norm(
                np.array(end_pos, dtype=np.float32) - prev_pos
            ) ** 2)
            eps = 1e-6
            if dist_moved_after + eps < dist_moved_before:
                collision_detected = True

            prev_rigid_state = habitat_sim.RigidState(
                target_rigid_state.rotation, end_pos
            )

        self._collision_detected = collision_detected

        # Set agent state from integrated position/rotation
        new_state = habitat_sim.AgentState()
        new_state.position = np.array(prev_rigid_state.translation, dtype=np.float32)
        rot_xyzw = prev_rigid_state.rotation.xyzw
        new_state.rotation = np.array(list(rot_xyzw), dtype=np.float32)
        self._sim.get_agent(0).set_state(new_state)

        # ── Update habitat-lab measures after physics step ────────────────────
        self._update_measures()

        # ── Get observation ─────────────────────────────────────────────────
        sim_obs = self._sim.get_sensor_observations()
        rgb = sim_obs["rgb"]
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        self._current_image = rgb

        # Extract proximity sensor reading
        if "proximity_sensor" in self._env._task.sensor_suite.sensors:
            prox_sensor = self._env._task.sensor_suite.sensors["proximity_sensor"]
            val = prox_sensor.get_observation(
                observations=sim_obs,
                episode=self._env._current_episode,
            )
            self._proximity = float(val[0])
        else:
            self._proximity = -1.0

        # ── Compute actual velocity from displacement ────────────────────────
        final_position = np.array(prev_rigid_state.translation, dtype=np.float64)
        total_time = self._cfg.frame_skip * self._dt
        self._forward_vel = lin_vel  # commanded velocity (backward compat)
        self._actual_vel = float(
            np.linalg.norm(final_position - initial_position) / total_time
        )
        self._delta_x = final_position - initial_position

        # ── Synthesize IMU ──────────────────────────────────────────────────
        self._current_imu = self._synthesize_imu()

        # Update rolling histories for IMU observation features
        resultant_accel = float(np.sqrt(self._current_accel[0]**2 + self._current_accel[1]**2))
        self._accel_resultant_history.append(resultant_accel)
        self._throttle_history.append(action[1])

        self._step_count += 1
        self._trajectory.append(np.array(self._sim.get_agent_state().position, dtype=np.float64))

        # ── Reward & termination ────────────────────────────────────────────
        hab_metrics = self._get_habitat_metrics()
        reward = float(hab_metrics.get("distance_to_goal_reward", 0.0)) + float(np.linalg.norm(self._delta_x))
        if self._collision_detected:
            reward -= 1.0

        # Terminate if agent reaches goal (distance < goal_radius)
        geo_dist = self._geodesic_distance(
            np.array(self._sim.get_agent_state().position, dtype=np.float64),
            self._goal_position,
        )
        terminated = geo_dist < self._cfg.goal_radius
        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ── Rendering ──────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "rgb_array":
            return self._current_image
        self._render_frame()

    def _render_frame(self):
        # ── Agent first-person view (left panel) ────────────────────────────
        VIEW_H, VIEW_W = 240, 320
        agent_view = cv2.cvtColor(cv2.resize(self._current_image, (VIEW_W, VIEW_H)),
                                  cv2.COLOR_RGB2BGR)
        if self._goal_image is not None:
            goal_view = cv2.cvtColor(cv2.resize(self._goal_image, (VIEW_W, VIEW_H)),
                                     cv2.COLOR_RGB2BGR)
        else:
            goal_view = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)

        # ── Top-down map (right panel) ──────────────────────────────────────
        topdown = self._render_topdown()
        # Scale top-down to match height of the two stacked views
        target_h = VIEW_H * 2
        td_h, td_w = topdown.shape[:2]
        scale = target_h / td_h
        target_w = int(td_w * scale)
        topdown_resized = cv2.resize(topdown, (target_w, target_h),
                                     interpolation=cv2.INTER_NEAREST)

        # Convert RGB to BGR for cv2
        topdown_bgr = cv2.cvtColor(topdown_resized, cv2.COLOR_RGB2BGR)

        # Draw info text on top-down map
        pos = np.array(self._sim.get_agent_state().position, dtype=np.float64)
        dist = self._geodesic_distance(pos, self._goal_position)
        info_lines = [
            f"step: {self._step_count}  dist: {dist:.2f}m",
            f"coll: {'YES' if self._collision_detected else 'no'}  vel: {self._forward_vel:.2f}m/s",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(topdown_bgr, line, (10, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ── Layout: [agent_view   | top-down map]
        #           [goal_view    |             ] ──────────────────────────────
        left = np.concatenate([agent_view, goal_view], axis=0)
        combined = np.concatenate([left, topdown_bgr], axis=1)
        cv2.imshow("HabitatNav", combined)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        if hasattr(self, '_env') and self._env is not None:
            self._env.close()
            self._env = None