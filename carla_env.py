from __future__ import annotations
import random
import subprocess
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

import carla
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    width: int = 84
    height: int = 84
    fov: float = 90.0


@dataclass
class CarlaConfig:
    server_path: str = "/opt/carla-simulator/CarlaUE4.sh"
    host: str = "localhost"
    port: int = 2000
    timeout: float = 20.0
    town: str = "Town01"
    fps: int = 20
    agent_blueprints: tuple[str, ...] = ("vehicle.micro.microlino", "vehicle.citroen.c3")
    camera: CameraConfig = field(default_factory=CameraConfig)
    spawn_index: int = 0


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RewardFunction:
    """
    All weights/penalties are class-level constants.
    Subclass or edit these to reshape the reward without touching env logic.
    """

    SPEED_TARGET_KMH = 30.0
    SPEED_WEIGHT = 1.0
    COLLISION_PENALTY = -200.0
    LANE_INVASION_PENALTY = -10.0
    OBSTACLE_PROXIMITY_PENALTY = -5.0
    OBSTACLE_TRIGGER_DISTANCE = 5.0

    def __call__(self, agent: carla.Actor, sensor_data: dict, done: bool) -> float:
        if done and "collision" in sensor_data:
            return self.COLLISION_PENALTY

        v = agent.get_velocity()
        speed_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        reward = self.SPEED_WEIGHT * max(0.0, self.SPEED_TARGET_KMH - abs(speed_kmh - self.SPEED_TARGET_KMH))

        if "lane_invasion" in sensor_data:
            reward += self.LANE_INVASION_PENALTY

        obstacle = sensor_data.get("obstacle")
        if obstacle is not None and obstacle["distance"] < self.OBSTACLE_TRIGGER_DISTANCE:
            reward += self.OBSTACLE_PROXIMITY_PENALTY

        return reward


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------

class SensorManager:
    """Attaches and manages every sensor CARLA supports on a single actor."""

    _DEFINITIONS = [
        {
            "key": "rgb_camera",
            "blueprint": "sensor.camera.rgb",
            "transform": carla.Transform(carla.Location(x=1.5, z=2.4)),
            "attributes": {},
        },
        {
            "key": "depth_camera",
            "blueprint": "sensor.camera.depth",
            "transform": carla.Transform(carla.Location(x=1.5, z=2.4)),
            "attributes": {},
        },
        # {
        #     "key": "semantic_camera",
        #     "blueprint": "sensor.camera.semantic_segmentation",
        #     "transform": carla.Transform(carla.Location(x=1.5, z=2.4)),
        #     "attributes": {},
        # },
        # {
        #     "key": "instance_camera",
        #     "blueprint": "sensor.camera.instance_segmentation",
        #     "transform": carla.Transform(carla.Location(x=1.5, z=2.4)),
        #     "attributes": {},
        # },
        # {
        #     "key": "dvs_camera",
        #     "blueprint": "sensor.camera.dvs",
        #     "transform": carla.Transform(carla.Location(x=1.5, z=2.4)),
        #     "attributes": {},
        # },
        # {
        #     "key": "optical_flow_camera",
        #     "blueprint": "sensor.camera.optical_flow",
        #     "transform": carla.Transform(carla.Location(x=1.5, z=2.4)),
        #     "attributes": {},
        # },
        # {
        #     "key": "lidar",
        #     "blueprint": "sensor.lidar.ray_cast",
        #     "transform": carla.Transform(carla.Location(x=0.0, z=2.6)),
        #     "attributes": {
        #         "channels": "32",
        #         "range": "50",
        #         "points_per_second": "56000",
        #         "rotation_frequency": "20",
        #     },
        # },
        # {
        #     "key": "semantic_lidar",
        #     "blueprint": "sensor.lidar.ray_cast_semantic",
        #     "transform": carla.Transform(carla.Location(x=0.0, z=2.6)),
        #     "attributes": {
        #         "channels": "32",
        #         "range": "50",
        #         "points_per_second": "56000",
        #         "rotation_frequency": "20",
        #     },
        # },
        # {
        #     "key": "radar",
        #     "blueprint": "sensor.other.radar",
        #     "transform": carla.Transform(carla.Location(x=2.0, z=1.0)),
        #     "attributes": {"horizontal_fov": "30", "vertical_fov": "10", "range": "100"},
        # },
        # {
        #     "key": "gnss",
        #     "blueprint": "sensor.other.gnss",
        #     "transform": carla.Transform(carla.Location(x=0.0, z=0.0)),
        #     "attributes": {},
        # },
        {
            "key": "imu",
            "blueprint": "sensor.other.imu",
            "transform": carla.Transform(carla.Location(x=0.0, z=0.0)),
            "attributes": {},
        },
        {
            "key": "collision",
            "blueprint": "sensor.other.collision",
            "transform": carla.Transform(),
            "attributes": {},
        },
        {
            "key": "lane_invasion",
            "blueprint": "sensor.other.lane_invasion",
            "transform": carla.Transform(),
            "attributes": {},
        },
        {
            "key": "obstacle",
            "blueprint": "sensor.other.obstacle",
            "transform": carla.Transform(carla.Location(x=1.5, z=1.0)),
            "attributes": {"distance": "5", "hit_radius": "0.5"},
        },
    ]

    def __init__(self, world: carla.World, agent: carla.Actor, camera_cfg: CameraConfig):
        self._world = world
        self._agent = agent
        self._camera_cfg = camera_cfg
        self._sensors: dict[str, carla.Actor] = {}
        self.data: dict[str, Any] = {}
        self.collision_detected = False
        self.lane_invasion_detected = False

    def spawn_all(self) -> None:
        bp_lib = self._world.get_blueprint_library()
        for defn in self._DEFINITIONS:
            bp = bp_lib.find(defn["blueprint"])
            if bp is None:
                continue
            self._configure_blueprint(bp, defn["blueprint"], defn["attributes"])
            sensor = self._world.spawn_actor(bp, defn["transform"], attach_to=self._agent)
            self._sensors[defn["key"]] = sensor
            self._register_callback(defn["key"], sensor)

    def _configure_blueprint(self, bp: carla.ActorBlueprint, name: str, attributes: dict) -> None:
        if "camera" in name:
            for attr, val in (("image_size_x", str(self._camera_cfg.width)),
                              ("image_size_y", str(self._camera_cfg.height)),
                              ("fov", str(self._camera_cfg.fov))):
                if bp.has_attribute(attr):
                    bp.set_attribute(attr, val)
        for attr, val in attributes.items():
            if bp.has_attribute(attr):
                bp.set_attribute(attr, val)

    def _register_callback(self, key: str, sensor: carla.Actor) -> None:
        weak = weakref.ref(self)

        if key == "collision":
            sensor.listen(lambda e: (s := weak()) and s._on_collision(e))
        elif key == "lane_invasion":
            sensor.listen(lambda e: (s := weak()) and s._on_lane_invasion(e))
        elif "camera" in key:
            sensor.listen(lambda img, k=key: (s := weak()) and s._on_image(k, img))
        elif "lidar" in key:
            sensor.listen(lambda pc, k=key: (s := weak()) and s._on_lidar(k, pc))
        elif key == "radar":
            sensor.listen(lambda m, k=key: (s := weak()) and s._on_radar(k, m))
        elif key == "gnss":
            sensor.listen(lambda m: (s := weak()) and s._on_gnss(m))
        elif key == "imu":
            sensor.listen(lambda m: (s := weak()) and s._on_imu(m))
        elif key == "obstacle":
            sensor.listen(lambda e: (s := weak()) and s._on_obstacle(e))

    def _on_collision(self, event) -> None:
        self.collision_detected = True
        self.data["collision"] = {
            "other_actor": event.other_actor.type_id,
            "impulse": np.array([event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z]),
        }

    def _on_lane_invasion(self, event) -> None:
        self.lane_invasion_detected = True
        self.data["lane_invasion"] = [str(m) for m in event.crossed_lane_markings]

    def _on_image(self, key: str, image) -> None:

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.data[key] = array

    def _on_lidar(self, key: str, point_cloud) -> None:
        self.data[key] = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape((-1, 4))

    def _on_radar(self, key: str, measurement) -> None:
        self.data[key] = np.array(
            [[d.depth, d.azimuth, d.altitude, d.velocity] for d in measurement], dtype=np.float32
        )

    def _on_gnss(self, measurement) -> None:
        self.data["gnss"] = np.array(
            [measurement.latitude, measurement.longitude, measurement.altitude], dtype=np.float64
        )

    def _on_imu(self, measurement) -> None:
        self.data["imu"] = {
            "accelerometer": np.array([measurement.accelerometer.x, measurement.accelerometer.y, measurement.accelerometer.z]),
            "gyroscope": np.array([measurement.gyroscope.x, measurement.gyroscope.y, measurement.gyroscope.z]),
            "compass": measurement.compass,
        }

    def _on_obstacle(self, event) -> None:
        self.data["obstacle"] = {"other_actor": event.other_actor.type_id, "distance": event.distance}

    def destroy(self) -> None:
        for sensor in self._sensors.values():
            if sensor.is_alive:
                sensor.destroy()
        self._sensors.clear()
        self.data.clear()
        self.collision_detected = False
        self.lane_invasion_detected = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

TerminationCondition = Callable[[carla.Actor, dict], bool]


def _collision_condition(agent: carla.Actor, sensor_data: dict) -> bool:
    return "collision" in sensor_data


class CarlaEnv:
    """
    Gym-style CARLA environment.

    - Server is launched automatically from cfg.server_path.
    - Agent alternates between micro.microlino and citroen.c3 each episode.
    - Episode ends on collision by default; pass extra_termination_conditions
      to add more without modifying this class.
    - Swap reward_fn with any callable(agent, sensor_data, done) -> float.
    """

    def __init__(
        self,
        config: CarlaConfig | None = None,
        reward_fn: RewardFunction | None = None,
        extra_termination_conditions: list[TerminationCondition] | None = None,
    ):
        self.cfg = config or CarlaConfig()
        self.reward_fn = reward_fn or RewardFunction()
        self.termination_conditions: list[TerminationCondition] = [_collision_condition]
        if extra_termination_conditions:
            self.termination_conditions.extend(extra_termination_conditions)

        self._server_process: subprocess.Popen | None = None
        self._client: carla.Client | None = None
        self._world: carla.World | None = None
        self._agent: carla.Actor | None = None
        self._sensor_manager: SensorManager | None = None
        self._blueprint_index = 0
        self._step_count = 0

        self._launch_server()
        self._connect()

    def reset(self) -> dict:
        self._destroy_episode()
        self._world.set_weather(carla.WeatherParameters.ClearNoon)

        spawn_points = self._world.get_map().get_spawn_points()
        spawn_tf = spawn_points[self.cfg.spawn_index % len(spawn_points)]

        self._agent = self._world.spawn_actor(self._next_blueprint(), spawn_tf)
        self._sensor_manager = SensorManager(self._world, self._agent, self.cfg.camera)
        self._sensor_manager.spawn_all()

        self._world.tick()
        time.sleep(0.5)
        self._step_count = 0
        return self._observation()

    def step(self, control: carla.VehicleControl) -> tuple[dict, float, bool, dict]:
        self._agent.apply_control(control)
        self._world.tick()
        self._step_count += 1
        self._world.get_spectator().set_transform(
            carla.Transform(
                self._agent.get_location() + carla.Location(z=50),
                carla.Rotation(pitch=-90, yaw=0, roll=0)
            )
        )

        sensor_data = self._sensor_manager.data
        done = any(cond(self._agent, sensor_data) for cond in self.termination_conditions)
        reward = self.reward_fn(self._agent, sensor_data, done)
        info = {
            "step": self._step_count,
            "collision": self._sensor_manager.collision_detected,
            "lane_invasion": self._sensor_manager.lane_invasion_detected,
        }
        return self._observation(), reward, done, info

    def close(self) -> None:
        self._destroy_episode()
        if self._server_process is not None:
            self._server_process.terminate()
            self._server_process.wait()
            self._server_process = None

    def _launch_server(self) -> None:
        # self._server_process = subprocess.Popen(
        #     [self.cfg.server_path, ""],
        #     # stdout=subprocess.DEVNULL,
        #     # stderr=subprocess.DEVNULL,
        # )
        # time.sleep(8.0)
        pass

    def _connect(self) -> None:
        self._client = carla.Client(self.cfg.host, self.cfg.port)
        self._client.set_timeout(self.cfg.timeout)
        self._world = self._client.load_world(self.cfg.town)
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.cfg.fps
        self._world.apply_settings(settings)

    def _next_blueprint(self) -> carla.ActorBlueprint:
        name = self.cfg.agent_blueprints[self._blueprint_index % len(self.cfg.agent_blueprints)]
        self._blueprint_index += 1
        matches = list(self._world.get_blueprint_library().filter(name))
        if not matches:
            raise RuntimeError(f"Blueprint not found: {name}")
        bp = random.choice(matches)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        return bp

    def _observation(self) -> dict:
        return dict(self._sensor_manager.data) if self._sensor_manager else {}

    def _destroy_episode(self) -> None:
        if self._sensor_manager is not None:
            self._sensor_manager.destroy()
            self._sensor_manager = None
        if self._agent is not None and self._agent.is_alive:
            self._agent.destroy()
            self._agent = None


# # ---------------------------------------------------------------------------
# # Example usage
# # ---------------------------------------------------------------------------

def _obstacle_too_close(agent: carla.Actor, sensor_data: dict) -> bool:
    obstacle = sensor_data.get("obstacle")
    return obstacle is not None and obstacle["distance"] < 1.5


if __name__ == "__main__":
    cfg = CarlaConfig(
        server_path="/home/kojogyaase/Apps/CARLA_0.9.16/CarlaUE4.sh",
        town="Town03",
        camera=CameraConfig(width=128, height=128, fov=110.0),
    )

    env = CarlaEnv(
        config=cfg,
        reward_fn=RewardFunction(),
        extra_termination_conditions=[_obstacle_too_close],
    )

    try:
        obs = env.reset()
        done = False
        while not done:
            control = carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
            obs, reward, done, info = env.step(control)
            print(f"step={info['step']}  reward={reward:.2f}  done={done}  "
                  f"collision={info['collision']}  lane_inv={info['lane_invasion']}")
    finally:
        env.close()