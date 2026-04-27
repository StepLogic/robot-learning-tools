"""
Stage 1: Collect dense images from Habitat scenes.

Samples navigable positions on the navmesh, renders RGB images at multiple
yaw angles per position, and saves them for graph construction (Stage 2).

Usage:
    python collect_habitat_images.py --scene skokloster-castle --num_positions 200 --num_yaws 8
    # Smoke test:
    python collect_habitat_images.py --scene van-gogh-room --num_positions 5 --num_yaws 2
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.habitat_config import HabitatNavConfig

# Habitat imports — will fail gracefully if not installed
try:
    import habitat_sim
    import magnum as mn
    HAS_HABITAT_SIM = True
except ImportError:
    HAS_HABITAT_SIM = False


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert yaw (rad, around Y-up) to quaternion [x,y,z,w]."""
    half = yaw * 0.5
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float64)


def _rgb_to_bgr(rgba: np.ndarray) -> np.ndarray:
    """Convert RGBA (H,W,4) or RGB (H,W,3) → BGR (H,W,3) uint8."""
    rgb = rgba[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def build_sim(scene_path: str, scene_dataset_path: str = "",
              image_height: int = 120, image_width: int = 160,
              hfov: float = 90.0, agent_height: float = 0.88,
              gpu_device_id: int = 0, allow_sliding: bool = True):
    """Construct a minimal Habitat-Sim instance for rendering."""
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene_path
    if scene_dataset_path:
        cfg.scene_dataset_config_file = scene_dataset_path
    cfg.allow_sliding = allow_sliding
    cfg.gpu_device_id = gpu_device_id

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [image_height, image_width]
    rgb_spec.hfov = hfov
    rgb_spec.position = [0.0, agent_height, 0.0]
    rgb_spec.orientation = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]
    agent_cfg.action_space = {}

    backend_cfg = habitat_sim.Configuration(cfg, [agent_cfg])
    return habitat_sim.Simulator(backend_cfg)


def render_at(sim, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Render RGB from a given pose, restoring agent state after."""
    agent = sim.get_agent(0)
    original_state = agent.get_state()

    target_state = habitat_sim.AgentState()
    target_state.position = np.asarray(position, dtype=np.float32)
    target_state.rotation = np.asarray(rotation, dtype=np.float32)
    agent.set_state(target_state)

    obs = sim.get_sensor_observations()

    agent.set_state(original_state)
    return obs["rgb"]


def sample_navigable_points(pathfinder, num_points: int,
                            rng: np.random.Generator) -> list:
    """Sample navigable points on the navmesh."""
    points = []
    for _ in range(num_points * 10):  # oversample to get enough
        if len(points) >= num_points:
            break
        pt = pathfinder.get_random_navigable_point()
        if pathfinder.is_navigable(pt):
            points.append(np.array(pt, dtype=np.float64))
    if len(points) < num_points:
        print(f"Warning: only found {len(points)} navigable points "
              f"(requested {num_points})")
    return points


def main():
    parser = argparse.ArgumentParser(description="Collect dense images from Habitat scenes")
    parser.add_argument("--scene", type=str, default="skokloster-castle",
                        help="Scene name (without .glb extension)")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "scene_datasets"),
                        help="Root directory for scene datasets")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "corl-2026"),
                        help="Root output directory")
    parser.add_argument("--num_positions", type=int, default=200,
                        help="Number of navigable positions to sample")
    parser.add_argument("--num_yaws", type=int, default=8,
                        help="Number of yaw angles per position")
    parser.add_argument("--image_height", type=int, default=120)
    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--agent_height", type=float, default=0.88)
    parser.add_argument("--gpu_device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not HAS_HABITAT_SIM:
        print("Error: habitat_sim is not installed. Install with:")
        print("  conda install habitat-sim -c conda-forge -c aihabitat")
        sys.exit(1)

    # Resolve scene path
    data_root = os.path.dirname(args.data_dir)  # parent of scene_datasets = data/
    scene_candidates = [
        os.path.join(args.data_dir, "habitat-test-scenes", f"{args.scene}.glb"),
        os.path.join(args.data_dir, "gibson_habitat", f"{args.scene}.glb"),
        os.path.join(args.data_dir, "gibson", f"{args.scene}.glb"),
        os.path.join(data_root, "gibson", f"{args.scene}.glb"),
        os.path.join(args.data_dir, f"{args.scene}.glb"),
    ]
    scene_path = None
    for candidate in scene_candidates:
        if os.path.exists(candidate):
            scene_path = candidate
            break
    if scene_path is None:
        print(f"Error: scene '{args.scene}' not found. Searched:")
        for c in scene_candidates:
            print(f"  {c}")
        sys.exit(1)
    print(f"Scene: {scene_path}")

    # Build simulator
    print("Building Habitat simulator...")
    sim = build_sim(
        scene_path=scene_path,
        image_height=args.image_height,
        image_width=args.image_width,
        hfov=args.hfov,
        agent_height=args.agent_height,
        gpu_device_id=args.gpu_device_id,
    )

    # Load navmesh for pathfinder
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        pathfinder.load_nav_mesh(scene_path)

    # Output directory
    scene_output = os.path.join(args.output_dir, args.scene, "dense_images")
    images_dir = os.path.join(scene_output, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Sample positions
    rng = np.random.default_rng(args.seed)
    print(f"Sampling {args.num_positions} navigable positions...")
    positions = sample_navigable_points(pathfinder, args.num_positions, rng)
    print(f"Sampled {len(positions)} positions")

    # Generate yaw angles
    yaw_step = 2 * math.pi / args.num_yaws
    yaws = [i * yaw_step for i in range(args.num_yaws)]

    # Collect images
    metadata = []
    node_id = 0
    total = len(positions) * len(yaws)
    print(f"Rendering {total} images...")

    for pos_idx, position in enumerate(positions):
        for yaw_idx, yaw in enumerate(yaws):
            rotation = _yaw_to_quat(yaw)
            rgba = render_at(sim, position, rotation)
            bgr = _rgb_to_bgr(rgba)

            img_filename = f"{node_id:06d}.png"
            img_path = os.path.join(images_dir, img_filename)
            cv2.imwrite(img_path, bgr)

            metadata.append({
                "id": node_id,
                "position": position.tolist(),
                "yaw": float(yaw),
                "position_index": pos_idx,
                "yaw_index": yaw_idx,
                "image_path": img_filename,
            })

            node_id += 1
            if node_id % 100 == 0:
                print(f"  {node_id}/{total} images rendered")

    # Save metadata
    meta_path = os.path.join(scene_output, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    sim.close()
    print(f"Done. {node_id} images saved to {scene_output}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()