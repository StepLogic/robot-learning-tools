import glob
import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class HabitatNavConfig:
    scene_path: str = "data/gibson/Cantwell.glb"
    scene_dataset_path: str = ""   # empty = standalone GLB (uses "default" dataset)
    scene_paths: List[str] = field(default_factory=list)  # pool for random scene selection
    image_height: int = 120
    image_width: int = 160
    hfov: float = 110.0
    agent_height: float = 0.1
    control_frequency: int = 10        # actions per second
    frame_skip: int = 3               # physics integration steps per action
    max_linear_velocity: float = 0.5  # m/s
    max_angular_velocity: float = 1.5  # rad/s
    allow_sliding: bool = True
    seed: int = 42
    gpu_device_id: int = 0
    imu_noise_std: float = 0.0       # Gaussian noise std for synthesized IMU
    debug_render: bool = False        # Show cv2 debug window with agent view
    headless: bool = False           # Force EGL headless rendering (unset DISPLAY, suppress QT windows)
    goal_radius: float = 1.0         # Success radius for DistanceToGoal
    goal_distance_scale: float = 3.0  # Exponential rate for goal distance (meters, lower = closer goals)
    goal_max_distance: float = 10.0   # Cap on sampled goal distance (meters)
    randomize_scenes: bool = False     # Randomly switch scenes across episodes
    held_out_scenes: List[str] = field(default_factory=list)  # scene names excluded from training pool

    def get_scene_paths(self) -> List[str]:
        """Return scene_paths if set, otherwise glob for .glb files next to scene_path.

        Scenes whose basename (without extension) appears in held_out_scenes
        are excluded from the returned list.
        """
        exclude = set(self.held_out_scenes)

        if self.scene_paths:
            paths = list(self.scene_paths)
        else:
            scene_dir = os.path.dirname(self.scene_path)
            paths = sorted(glob.glob(os.path.join(scene_dir, "*.glb")))
            if not paths:
                for fallback in ["data/gibson", "data/scene_datasets/gibson",
                                 "data/versioned_data/habitat_test_scenes"]:
                    fb_paths = sorted(glob.glob(os.path.join(fallback, "*.glb")))
                    if fb_paths:
                        paths = fb_paths
                        break
            if not paths:
                paths = [self.scene_path]

        if exclude:
            filtered = [p for p in paths
                        if os.path.splitext(os.path.basename(p))[0] not in exclude]
            return filtered or paths  # fall back to unfiltered if all were excluded
        return paths