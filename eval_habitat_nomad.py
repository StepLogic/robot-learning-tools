"""
Stage 4: Evaluate graph-based NoMaD navigation in Habitat.

Compares three navigation strategies:
  A) NoMaD with dense topological graph
  B) NoMaD with sparse (sparsified) topological graph
  C) DrQ policy from checkpoint

Usage:
    # Evaluate NoMaD with sparse graph
    python eval_habitat_nomad.py --scene skokloster-castle --mode nomad_sparse --num_episodes 100

    # Evaluate NoMaD with dense graph
    python eval_habitat_nomad.py --scene skokloster-castle --mode nomad_dense --num_episodes 100

    # Evaluate DrQ policy
    python eval_habitat_nomad.py --scene skokloster-castle --mode drq --drq_checkpoint checkpoints/drq_100k --num_episodes 100

    # Smoke test
    python eval_habitat_nomad.py --scene van-gogh-room --mode nomad_sparse --num_episodes 5
"""

import argparse
import heapq
import json
import math
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.habitat_config import HabitatNavConfig
from navigation_policies.navigation_policies.baseline_policies.misc import (
    load_model, transform_images, to_numpy, get_action,
)
from navigation_policies.navigation_policies.baseline_policies.baselines_config import nomad_config
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

try:
    import habitat_sim
    HAS_HABITAT_SIM = True
except ImportError:
    HAS_HABITAT_SIM = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Utility functions (shared with collect_habitat_images.py)
# ============================================================================

def _yaw_to_quat(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float64)


def _quat_to_yaw(q) -> float:
    """Extract yaw from quaternion [x,y,z,w]."""
    a = np.asarray(q, dtype=np.float64).ravel()
    x, y, z, w = a[0], a[1], a[2], a[3]
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rgb_to_bgr(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ============================================================================
# PD Controller for Habitat action space
# ============================================================================

def habitat_pd_controller(waypoint: np.ndarray) -> list:
    """PD controller adapted for Habitat's [-1,1]x[0,1] action space.

    waypoint: 2D (dx, dy) or 4D (dx, dy, hx, hy) predicted action from NoMaD.
    Returns: [angular_vel, linear_vel] in normalized range.
    """
    if len(waypoint) == 4:
        dx, dy, hx, hy = waypoint
    else:
        dx, dy = waypoint[0], waypoint[1]

    EPS = 1e-8

    if abs(dx) < EPS and abs(dy) < EPS and len(waypoint) == 4:
        # Use heading direction when position delta is negligible
        angular = np.clip(np.arctan2(hy, hx) / (np.pi / 2), -1, 1)
        linear = 0.05  # creep forward slowly
    elif abs(dx) < EPS:
        angular = np.sign(dy) * 1.0  # full turn
        linear = 0.0
    else:
        # Steering: angle to the waypoint
        angular = np.clip(np.arctan2(dy, dx) / np.pi, -1, 1)
        # Throttle: proportional to forward distance
        linear = np.clip(np.sqrt(dx**2 + dy**2) / 4.0, 0.0, 1.0)
        # Slow down when turning sharply
        linear *= max(0.2, 1.0 - abs(angular))

    return [float(angular), float(linear)]


# ============================================================================
# Graph-based NoMaD Navigator
# ============================================================================

class GraphNoMaDNavigator:
    """NoMaD navigation using a topological graph with Dijkstra path planning.

    Replaces NoMaD's linear topomap with a general graph structure.
    Localization uses NoMaD's dist_pred_net to find the closest graph node.
    Path planning uses Dijkstra's algorithm on the graph.
    Action generation uses NoMaD's diffusion model.
    """

    def __init__(self, graph_path: str, images_dir: str,
                 ckpt_path: str, distance_threshold: float = 3.0,
                 search_radius: int = 4, close_threshold: float = 3.0):
        # Load graph
        with open(graph_path) as f:
            if "threshold_3.0" in f.read()[:200]:
                # sparse_graph.json format
                f.seek(0)
                data = json.load(f)
                # Find the appropriate threshold key
                key = f"threshold_{distance_threshold}"
                if key not in data:
                    key = list(data.keys())[0]
                self.node_ids = data[key]["retained_nodes_indices"]
            else:
                # Regular graph.json — use all nodes
                f.seek(0)
                data = json.load(f)
                from networkx import node_link_graph
                G = node_link_graph(data, directed=True)
                if G.is_directed():
                    G = G.to_undirected()
                self.node_ids = list(G.nodes())

        self.graph_data = data
        self.distance_threshold = distance_threshold
        self.search_radius = search_radius
        self.close_threshold = close_threshold

        # Load node metadata and images
        self.node_info = {}  # node_id -> {position, yaw, image_path}
        self.node_images = {}  # node_id -> PIL Image
        self.node_positions = {}  # node_id -> np.array position

        # Try loading from graph metadata first, then from Stage 1 metadata
        self._load_graph_data(graph_path, images_dir)

        # Build adjacency for Dijkstra
        self._build_adjacency()

        # Load NoMaD model
        print(f"Loading NoMaD model from {ckpt_path}...")
        self.model = load_model(ckpt_path, nomad_config, device)
        self.model.eval()
        self.context_size = nomad_config["context_size"]
        self.image_size = nomad_config["image_size"]
        self.num_diffusion_iters = nomad_config["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=nomad_config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        # Navigation state
        self.context_queue = []
        self.closest_node = None
        self.current_path = []
        self.path_index = 0

    def _load_graph_data(self, graph_path: str, images_dir: str):
        """Load node data from graph.json or metadata.json."""
        # Check if we have node data in the graph file
        if isinstance(self.graph_data, dict):
            # NetworkX node_link_data format
            if "nodes" in self.graph_data:
                for node in self.graph_data["nodes"]:
                    nid = node["id"]
                    if nid in self.node_ids:
                        self.node_info[nid] = node
                        if "position" in node:
                            self.node_positions[nid] = np.array(node["position"])

        # Load images from dense_images directory
        meta_path = os.path.join(os.path.dirname(images_dir), "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)

            # Build lookup from id to metadata
            id_to_meta = {e["id"]: e for e in metadata}

            for nid in self.node_ids:
                if nid in id_to_meta:
                    meta = id_to_meta[nid]
                    img_path = os.path.join(images_dir, meta["image_path"])
                    bgr = cv2.imread(img_path)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    self.node_images[nid] = Image.fromarray(rgb)
                    if nid not in self.node_info:
                        self.node_info[nid] = meta
                    if "position" in meta and nid not in self.node_positions:
                        self.node_positions[nid] = np.array(meta["position"])

    def _build_adjacency(self):
        """Build adjacency list from graph edges for Dijkstra."""
        self.adj = defaultdict(list)  # node_id -> [(neighbor_id, nomad_distance)]

        if isinstance(self.graph_data, dict) and "links" in self.graph_data:
            for link in self.graph_data["links"]:
                src, dst = link["source"], link["target"]
                if src in self.node_ids and dst in self.node_ids:
                    dist = link.get("nomad_distance", link.get("weight", 1.0))
                    self.adj[src].append((dst, dist))
                    self.adj[dst].append((src, dist))

    def dijkstra(self, start: int, goal: int) -> list:
        """Find shortest path from start to goal using NoMaD distances."""
        if start == goal:
            return [start]

        dist = {start: 0.0}
        prev = {}
        pq = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if u == goal:
                break
            if d > dist.get(u, float('inf')):
                continue
            for v, w in self.adj.get(u, []):
                new_dist = d + w
                if new_dist < dist.get(v, float('inf')):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))

        if goal not in prev:
            return []  # No path found

        # Reconstruct path
        path = []
        node = goal
        while node != start:
            path.append(node)
            node = prev[node]
        path.append(start)
        path.reverse()
        return path

    def localize(self, obs: np.ndarray) -> int:
        """Find the closest graph node to the current observation.

        Uses NoMaD's dist_pred_net to compute distances to nearby nodes.
        """
        obs_img = Image.fromarray(obs)
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(obs_img)
            return self.node_ids[0] if self.node_ids else 0

        # Build context
        obs_images = transform_images(self.context_queue, self.image_size,
                                       center_crop=False)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(device)
        mask = torch.zeros(1).long().to(device)

        # Compute distances to candidate nodes
        # Start with nodes near previous closest, expand if needed
        candidates = self._get_localization_candidates()

        goal_images = [self.node_images[nid] for nid in candidates]
        goal_tensor = torch.cat([
            transform_images([img], self.image_size, center_crop=False).to(device)
            for img in goal_images
        ], dim=0)

        obs_batch = obs_images.repeat(len(candidates), 1, 1, 1)
        mask_batch = mask.repeat(len(candidates))

        with torch.no_grad():
            obsgoal_cond = self.model('vision_encoder',
                                       obs_img=obs_batch,
                                       goal_img=goal_tensor,
                                       input_goal_mask=mask_batch)
            dists = self.model('dist_pred_net', obsgoal_cond=obsgoal_cond)
            dists = to_numpy(dists).flatten()

        min_idx = np.argmin(dists)
        self.closest_node = candidates[min_idx]
        return self.closest_node

    def _get_localization_candidates(self) -> list:
        """Get candidate nodes for localization (nearby in graph)."""
        if self.closest_node is None:
            return self.node_ids[:min(20, len(self.node_ids))]

        # BFS to find nodes within search_radius hops
        candidates = [self.closest_node]
        visited = {self.closest_node}
        frontier = [self.closest_node]

        for _ in range(self.search_radius):
            next_frontier = []
            for node in frontier:
                for neighbor, _ in self.adj.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                        candidates.append(neighbor)
            frontier = next_frontier

        return candidates

    def set_goal(self, goal_node: int):
        """Set navigation goal and plan initial path."""
        start = self.closest_node or self.node_ids[0]
        self.current_path = self.dijkstra(start, goal_node)
        self.path_index = 0

    def act(self, obs: np.ndarray) -> list:
        """Compute action for current observation.

        1. Localize on graph
        2. Follow Dijkstra path to goal
        3. Use NoMaD diffusion to generate waypoint actions
        """
        # Update context
        obs_img = Image.fromarray(obs)
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(obs_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(obs_img)

        if len(self.context_queue) < self.context_size + 1:
            return [0.0, 0.0]  # not enough context yet

        # Localize
        current_node = self.localize(obs)

        # Get subgoal from path
        if not self.current_path or self.path_index >= len(self.current_path):
            # No path or reached end — go towards goal directly
            subgoal_node = self.current_path[-1] if self.current_path else current_node
        else:
            # Advance along path if we've reached the next waypoint
            if current_node == self.current_path[self.path_index]:
                self.path_index = min(self.path_index + 1, len(self.current_path) - 1)
            subgoal_node = self.current_path[self.path_index]

        # Use NoMaD diffusion to generate action
        obs_images = transform_images(self.context_queue, self.image_size,
                                      center_crop=False)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(device)

        goal_img = self.node_images[subgoal_node]
        goal_tensor = transform_images([goal_img], self.image_size,
                                       center_crop=False).to(device)

        mask = torch.zeros(1).long().to(device)

        with torch.no_grad():
            obsgoal_cond = self.model('vision_encoder',
                                        obs_img=obs_images,
                                        goal_img=goal_tensor,
                                        input_goal_mask=mask)
            dist = self.model('dist_pred_net', obsgoal_cond=obsgoal_cond)
            dist_val = float(to_numpy(dist).flatten()[0])

            # If close to subgoal, advance along path
            if dist_val < self.close_threshold and self.path_index < len(self.current_path) - 1:
                self.path_index += 1
                subgoal_node = self.current_path[self.path_index]
                goal_tensor = transform_images(
                    [self.node_images[subgoal_node]], self.image_size,
                    center_crop=False).to(device)
                obsgoal_cond = self.model('vision_encoder',
                                           obs_img=obs_images,
                                           goal_img=goal_tensor,
                                           input_goal_mask=mask)

            # Diffusion action generation
            obs_cond = obsgoal_cond
            num_samples = 8
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(num_samples, 1, 1)

            noisy_action = torch.randn(
                (num_samples, nomad_config["len_traj_pred"], 2), device=device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model('noise_pred_net',
                                         sample=naction,
                                         timestep=k,
                                         global_cond=obs_cond)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        naction = to_numpy(get_action(naction))
        waypoint = naction[0]

        # Extract 2D waypoint (dx, dy) from trajectory prediction
        # NoMaD predicts len_traj_pred waypoints; use waypoint index 2 (3rd)
        # which corresponds to ~1.5m ahead
        wp_idx = min(2, len(waypoint) - 1)
        dx, dy = waypoint[wp_idx * 2], waypoint[wp_idx * 2 + 1] if waypoint.shape[0] > 2 else waypoint[0], waypoint[1]

        return habitat_pd_controller(waypoint)


# ============================================================================
# Habitat environment wrapper for evaluation
# ============================================================================

class HabitatEvalEnv:
    """Lightweight wrapper around HabitatNavEnv for evaluation."""

    def __init__(self, scene_path: str, config: HabitatNavConfig = None,
                 goal_threshold: float = 1.0, max_steps: int = 500):
        from habitat_env import HabitatNavEnv
        self.config = config or HabitatNavConfig()
        self.config.scene_path = scene_path
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.env = HabitatNavEnv(config=self.config, render_mode="rgb_array")

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def get_goal_distance(self):
        """Get geodesic distance to goal using habitat_env's info dict."""
        info = self.env._get_info()
        return info.get("distance_to_goal", float("inf"))

    def close(self):
        self.env.close()


# ============================================================================
# Evaluation loop
# ============================================================================

def find_closest_node(position: np.ndarray, node_positions: dict) -> int:
    """Find the graph node closest to a 3D position."""
    min_dist = float('inf')
    closest = None
    for nid, npos in node_positions.items():
        dist = np.linalg.norm(position - np.array(npos))
        if dist < min_dist:
            min_dist = dist
            closest = nid
    return closest


def evaluate_nomad(navigator: GraphNoMaDNavigator, env: HabitatEvalEnv,
                    num_episodes: int, goal_threshold: float,
                    num_map_nodes: int = 0) -> dict:
    """Evaluate NoMaD navigation with graph-based planning."""
    results = {
        "successes": 0,
        "collisions": 0,
        "timeouts": 0,
        "spl_values": [],
        "path_lengths": [],
        "distances_to_goal": [],
        "num_map_nodes": num_map_nodes,
    }

    for ep in range(num_episodes):
        obs, info = env.reset()
        initial_dist = env.get_goal_distance()
        optimal_dist = initial_dist

        # Set goal: find closest graph node to the goal position
        goal_pos = info.get("goal_pos", info.get("pos", np.zeros(3)))
        if "goal_pos" in info:
            goal_pos = np.array(info["goal_pos"])
        goal_node = find_closest_node(goal_pos, navigator.node_positions)

        if goal_node is None:
            print(f"  Episode {ep}: No goal node found, skipping")
            continue

        navigator.context_queue = []
        navigator.closest_node = None
        navigator.current_path = []
        navigator.path_index = 0

        path_length = 0.0
        prev_pos = np.array(info["pos"]) if "pos" in info else None
        success = False

        for step in range(env.max_steps):
            # Get observation image
            obs_img = obs["image"]  # BGR
            obs_rgb = cv2.cvtColor(obs_img, cv2.COLOR_BGR2RGB)

            # Compute action
            action = navigator.act(obs_rgb)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Track path length
            curr_pos = np.array(info["pos"]) if "pos" in info else None
            if prev_pos is not None and curr_pos is not None:
                path_length += np.linalg.norm(curr_pos - prev_pos)
            prev_pos = curr_pos

            # Check success
            dist_to_goal = env.get_goal_distance()
            if dist_to_goal < goal_threshold:
                success = True
                break

            # Check collision
            if info.get("hit") == "collision" or info.get("collision", {}).get("detected"):
                results["collisions"] += 1
                break

            if terminated or truncated:
                break

        if success:
            results["successes"] += 1
            spl = optimal_dist / max(path_length, optimal_dist)
            results["spl_values"].append(spl)
        else:
            results["spl_values"].append(0.0)

        results["distances_to_goal"].append(dist_to_goal if not success else 0.0)
        results["path_lengths"].append(path_length)

        if not success and step >= env.max_steps - 1:
            results["timeouts"] += 1

        print(f"  Episode {ep}: {'SUCCESS' if success else 'FAIL'} "
              f"dist={dist_to_goal:.2f}m steps={step} path={path_length:.1f}m")

    total = num_episodes
    results["success_rate"] = results["successes"] / total if total > 0 else 0
    results["spl"] = np.mean(results["spl_values"]) if results["spl_values"] else 0
    results["collision_rate"] = results["collisions"] / total if total > 0 else 0
    results["timeout_rate"] = results["timeouts"] / total if total > 0 else 0
    return results


def evaluate_drq(env: HabitatEvalEnv, checkpoint_dir: str,
                   num_episodes: int, goal_threshold: float) -> dict:
    """Evaluate DrQ policy from checkpoint."""
    # Import DrQ components
    from wrappers import StackingWrapper, MobileNetFeatureWrapper, GoalImageWrapper
    from habitat_env import HabitatNavEnv

    # Load checkpoint
    from flax.training import checkpoints
    from jaxrl2.train import DrQLearner

    # This is a placeholder — actual DrQ evaluation depends on the
    # specific checkpoint format and wrapper configuration
    print(f"DrQ evaluation from {checkpoint_dir} not yet fully implemented.")
    print("This requires the full DrQ wrapper stack (StackingWrapper, "
          "MobileNetFeatureWrapper, etc.) and checkpoint loading.")
    results = {
        "success_rate": 0.0,
        "spl": 0.0,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "note": "DrQ evaluation not yet implemented — see train_habitat_her.py for setup"
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate graph-based NoMaD navigation in Habitat")
    parser.add_argument("--scene", type=str, default="skokloster-castle")
    parser.add_argument("--mode", type=str, default="nomad_sparse",
                        choices=["nomad_dense", "nomad_sparse", "drq", "all"],
                        help="Navigation mode. 'all' runs dense + every sparse threshold in sparse_graph.json")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "corl-2026"))
    parser.add_argument("--scene_data_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "scene_datasets"))
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--goal_threshold", type=float, default=1.0,
                        help="Distance threshold (m) for success")
    parser.add_argument("--ckpt_path", type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             "navigation_policies",
                                             "navigation_policies",
                                             "pretrained_models", "nomad.pth"))
    parser.add_argument("--drq_checkpoint", type=str, default=None,
                        help="DrQ checkpoint directory (for --mode drq)")
    parser.add_argument("--distance_threshold", type=float, default=3.0,
                        help="Distance threshold for sparse graph (used in --mode nomad_sparse)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not HAS_HABITAT_SIM:
        print("Error: habitat_sim is not installed.")
        sys.exit(1)

    # Resolve scene path
    data_root = os.path.dirname(args.scene_data_dir)  # parent of scene_datasets = data/
    scene_candidates = [
        os.path.join(args.scene_data_dir, "habitat-test-scenes", f"{args.scene}.glb"),
        os.path.join(args.scene_data_dir, "gibson_habitat", f"{args.scene}.glb"),
        os.path.join(args.scene_data_dir, "gibson", f"{args.scene}.glb"),
        os.path.join(data_root, "gibson", f"{args.scene}.glb"),
        os.path.join(args.scene_data_dir, f"{args.scene}.glb"),
    ]
    scene_path = None
    for c in scene_candidates:
        if os.path.exists(c):
            scene_path = c
            break
    if scene_path is None:
        print(f"Error: scene '{args.scene}' not found.")
        sys.exit(1)

    # Create evaluation environment
    config = HabitatNavConfig(scene_path=scene_path, seed=args.seed)
    env = HabitatEvalEnv(scene_path=scene_path, config=config,
                         goal_threshold=args.goal_threshold,
                         max_steps=args.max_steps)

    all_results = {}  # mode -> results dict, for LaTeX table

    modes_to_run = []
    if args.mode == "all":
        # Run dense + every sparse threshold found in sparse_graph.json
        modes_to_run.append("nomad_dense")
        sparse_path = os.path.join(args.data_dir, args.scene, "sparse_graph.json")
        if os.path.exists(sparse_path):
            with open(sparse_path) as f:
                sparse_data = json.load(f)
            for key in sparse_data:
                modes_to_run.append(f"nomad_sparse_{key}")
    else:
        modes_to_run.append(args.mode)

    for mode in modes_to_run:
        if mode == "nomad_dense":
            graph_path = os.path.join(args.data_dir, args.scene, "graph.json")
            threshold = args.distance_threshold
        elif mode.startswith("nomad_sparse"):
            graph_path = os.path.join(args.data_dir, args.scene, "sparse_graph.json")
            # Extract threshold from mode name like "nomad_sparse_threshold_3.0"
            if "threshold_" in mode:
                try:
                    threshold = float(mode.split("threshold_")[-1])
                except ValueError:
                    threshold = args.distance_threshold
            else:
                threshold = args.distance_threshold
        elif mode == "drq":
            if args.drq_checkpoint is None:
                print("Error: --drq_checkpoint required for --mode drq")
                sys.exit(1)
            results = evaluate_drq(env, args.drq_checkpoint, args.num_episodes,
                                   args.goal_threshold)
            all_results[mode] = results
            continue
        else:
            print(f"Unknown mode: {mode}")
            continue

        images_dir = os.path.join(args.data_dir, args.scene, "dense_images", "images")

        if not os.path.exists(graph_path):
            print(f"Error: graph file not found at {graph_path}")
            print(f"Run build_topo_graph.py (and sparsify_graph.py if sparse) first.")
            sys.exit(1)

        navigator = GraphNoMaDNavigator(
            graph_path=graph_path,
            images_dir=images_dir,
            ckpt_path=args.ckpt_path,
            distance_threshold=threshold,
        )
        num_map_nodes = len(navigator.node_ids)
        print(f"\n{'='*40}")
        print(f"Evaluating: {mode} ({num_map_nodes} nodes)")
        print(f"{'='*40}")

        results = evaluate_nomad(navigator, env, args.num_episodes,
                                  args.goal_threshold,
                                  num_map_nodes=num_map_nodes)
        all_results[mode] = results

    # Print results for each mode
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Scene:            {args.scene}")
    print(f"  Episodes:         {args.num_episodes}")
    print(f"  Goal threshold:   {args.goal_threshold}m")
    print()
    print(f"  {'Method':<25} {'Map Nodes':>10} {'SR (%)':>8} {'SPL':>6}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*6}")
    for mode, results in all_results.items():
        sr = results.get('success_rate', 0) * 100
        spl = results.get('spl', 0)
        nodes = results.get('num_map_nodes', '—')
        print(f"  {mode:<25} {str(nodes):>10} {sr:>7.1f}% {spl:>6.3f}")
    print("=" * 60)

    # Save individual results JSONs
    for mode, results in all_results.items():
        results_path = os.path.join(args.data_dir, args.scene,
                                    f"eval_{mode}_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        serializable = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                v = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                v = int(v)
            serializable[k] = v
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {results_path}")

    # Save LaTeX table with SR, SPL, Map Size across all evaluated modes
    _save_latex_table(all_results, args.data_dir, args.scene, args.num_episodes)

    env.close()


def _save_latex_table(all_results: dict, data_dir: str, scene: str,
                       num_episodes: int):
    """Save a LaTeX table showing the effect of map size on SR and SPL.

    Reads any existing results JSONs from prior runs to build a complete table
    across all map sizes (dense, sparse at various thresholds, drq).
    """
    table_dir = os.path.join(data_dir, scene)
    os.makedirs(table_dir, exist_ok=True)

    # Collect results from all existing eval JSONs in the scene directory
    collected = dict(all_results)  # start with current run

    for fname in os.listdir(table_dir):
        if fname.startswith("eval_") and fname.endswith("_results.json"):
            mode_key = fname.replace("eval_", "").replace("_results.json", "")
            if mode_key not in collected:
                with open(os.path.join(table_dir, fname)) as f:
                    collected[mode_key] = json.load(f)

    # Sort: drq first, then nomad_dense, then sparse thresholds ascending
    def sort_key(mode):
        if mode == "drq":
            return (0, 0)
        if mode == "nomad_dense":
            return (1, 0)
        # nomad_sparse — try to extract threshold for ordering
        try:
            threshold = float(mode.split("_")[-1])
        except ValueError:
            threshold = 0
        return (2, threshold)

    sorted_modes = sorted(collected.keys(), key=sort_key)

    # Build LaTeX table
    scene_escaped = scene.replace("_", r"\_")
    cap_text = (f"Effect of topological map size on navigation performance "
                f"on {scene_escaped} scene. "
                f"Evaluated over {num_episodes} episodes.")
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + cap_text + r"}",
        r"\label{tab:map_size_ablation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Map Nodes & SR (\%) & SPL \\",
        r"\midrule",
    ]

    for mode in sorted_modes:
        r = collected[mode]
        sr = r.get("success_rate", 0) * 100
        spl = r.get("spl", 0)
        num_nodes = r.get("num_map_nodes", "—")

        # Pretty-print mode name
        if mode == "drq":
            label = "DrQ (ours)"
        elif mode == "nomad_dense":
            label = "NoMaD (dense)"
        elif mode.startswith("nomad_sparse"):
            # Include threshold info if available
            if "threshold_" in mode:
                thr = mode.split("threshold_")[-1]
                label = f"NoMaD (sparse, d={thr})"
            else:
                label = "NoMaD (sparse)"
        else:
            label = mode

        if isinstance(num_nodes, int):
            latex_lines.append(f"  {label} & {num_nodes} & {sr:.1f} & {spl:.3f} \\\\")
        else:
            latex_lines.append(f"  {label} & {num_nodes} & {sr:.1f} & {spl:.3f} \\\\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = os.path.join(table_dir, "table_map_size_ablation.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")

    # Also save a machine-readable CSV for plotting
    csv_path = os.path.join(table_dir, "map_size_results.csv")
    with open(csv_path, "w") as f:
        f.write("method,num_map_nodes,sr,spl\n")
        for mode in sorted_modes:
            r = collected[mode]
            sr = r.get("success_rate", 0)
            spl = r.get("spl", 0)
            num_nodes = r.get("num_map_nodes", 0)
            f.write(f"{mode},{num_nodes},{sr:.4f},{spl:.4f}\n")

    print(f"LaTeX table saved to {latex_path}")
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()