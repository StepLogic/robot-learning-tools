"""
Stage 2: Build topological graph using NoMaD distance predictions.

Loads dense images from Stage 1, computes pairwise NoMaD distances for
nearby nodes (filtered by geodesic distance), and saves a NetworkX graph.

Usage:
    python build_topo_graph.py --scene skokloster-castle --geodesic_threshold 3.0
    # Smoke test:
    python build_topo_graph.py --scene van-gogh-room --geodesic_threshold 5.0 --batch_size 32
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import networkx as nx
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navigation_policies.navigation_policies.baseline_policies.misc import (
    load_model, transform_images, to_numpy,
)
from navigation_policies.navigation_policies.baseline_policies.baselines_config import nomad_config

# Habitat imports for geodesic distance
try:
    import habitat_sim
    HAS_HABITAT_SIM = True
except ImportError:
    HAS_HABITAT_SIM = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _yaw_to_quat(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float64)


def compute_geodesic_distances(scene_path: str, positions: list,
                                threshold: float) -> dict:
    """Compute geodesic distances between all position pairs within threshold.

    Returns dict mapping (i, j) -> geodesic_distance for pairs within threshold.
    """
    if not HAS_HABITAT_SIM:
        print("Warning: habitat_sim not installed, falling back to Euclidean distance")
        return _compute_euclidean_distances(positions, threshold)

    # Build a minimal simulator just for pathfinding
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene_path
    cfg.gpu_device_id = 0

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    agent_cfg.action_space = {}

    backend_cfg = habitat_sim.Configuration(cfg, [agent_cfg])
    sim = habitat_sim.Simulator(backend_cfg)
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        pathfinder.load_nav_mesh(scene_path)

    geodesic_pairs = {}
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            pi = np.array(positions[i], dtype=np.float64)
            pj = np.array(positions[j], dtype=np.float64)

            path = habitat_sim.ShortestPath()
            path.requested_start = pi.tolist()
            path.requested_end = pj.tolist()
            if pathfinder.find_path(path):
                geo_dist = path.geodesic_distance
            else:
                geo_dist = float(np.linalg.norm(pi - pj))

            if geo_dist <= threshold:
                geodesic_pairs[(i, j)] = geo_dist
                geodesic_pairs[(j, i)] = geo_dist

    sim.close()
    return geodesic_pairs


def _compute_euclidean_distances(positions: list, threshold: float) -> dict:
    """Fallback: compute Euclidean distances when habitat_sim is unavailable."""
    pairs = {}
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(
                np.array(positions[i]) - np.array(positions[j])
            ))
            if dist <= threshold:
                pairs[(i, j)] = dist
                pairs[(j, i)] = dist
    return pairs


def compute_nomad_distance(model: torch.nn.Module, obs_img: Image.Image,
                            goal_img: Image.Image, image_size: list,
                            context_size: int) -> float:
    """Compute NoMaD predicted distance between an observation and goal image.

    Since NoMaD's vision encoder uses self-attention over obs+goal tokens,
    each (obs, goal) pair must be passed through the full encoder.
    For single-frame observations, we repeat the image to fill the context queue.
    """
    # Build context: repeat the observation image (context_size + 1) times
    obs_images = [obs_img] * (context_size + 1)
    obs_tensor = transform_images(obs_images, image_size, center_crop=False)
    obs_tensor = obs_tensor.split(3, dim=1)
    obs_tensor = torch.cat(obs_tensor, dim=1).to(device)

    # Transform goal image
    goal_tensor = transform_images([goal_img], image_size, center_crop=False).to(device)

    mask = torch.zeros(1).long().to(device)

    with torch.no_grad():
        obsgoal_cond = model('vision_encoder',
                             obs_img=obs_tensor,
                             goal_img=goal_tensor,
                             input_goal_mask=mask)
        dist = model('dist_pred_net', obsgoal_cond=obsgoal_cond)

    return float(to_numpy(dist).flatten()[0])


def compute_nomad_distances_batch(model: torch.nn.Module,
                                   obs_imgs: list,
                                   goal_imgs: list,
                                   image_size: list,
                                   context_size: int) -> np.ndarray:
    """Compute NoMaD distances for multiple (obs, goal) pairs in a batch.

    Args:
        obs_imgs: list of PIL Images (observations)
        goal_imgs: list of PIL Images (goals), same length as obs_imgs
    Returns:
        numpy array of distances
    """
    assert len(obs_imgs) == len(goal_imgs)

    # Build context for each observation
    obs_tensors = []
    goal_tensors = []
    for obs_img, goal_img in zip(obs_imgs, goal_imgs):
        ctx = [obs_img] * (context_size + 1)
        obs_t = transform_images(ctx, image_size, center_crop=False)
        obs_t = obs_t.split(3, dim=1)
        obs_t = torch.cat(obs_t, dim=1)
        obs_tensors.append(obs_t)

        goal_t = transform_images([goal_img], image_size, center_crop=False)
        goal_tensors.append(goal_t)

    obs_batch = torch.cat(obs_tensors, dim=0).to(device)
    goal_batch = torch.cat(goal_tensors, dim=0).to(device)
    mask = torch.zeros(len(obs_imgs)).long().to(device)

    with torch.no_grad():
        obsgoal_cond = model('vision_encoder',
                             obs_img=obs_batch,
                             goal_img=goal_batch,
                             input_goal_mask=mask)
        dists = model('dist_pred_net', obsgoal_cond=obsgoal_cond)

    return to_numpy(dists).flatten()


def main():
    parser = argparse.ArgumentParser(description="Build topological graph with NoMaD distances")
    parser.add_argument("--scene", type=str, default="skokloster-castle")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "corl-2026"))
    parser.add_argument("--scene_data_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "scene_datasets"))
    parser.add_argument("--geodesic_threshold", type=float, default=3.0,
                        help="Max geodesic distance (m) to create an edge")
    parser.add_argument("--ckpt_path", type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             "navigation_policies",
                                             "navigation_policies",
                                             "pretrained_models", "nomad.pth"),
                        help="Path to NoMaD checkpoint")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for NoMaD distance computation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load metadata from Stage 1
    scene_input = os.path.join(args.data_dir, args.scene, "dense_images")
    meta_path = os.path.join(scene_input, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"Error: metadata not found at {meta_path}")
        print("Run collect_habitat_images.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} nodes from {meta_path}")

    # Load images
    images_dir = os.path.join(scene_input, "images")
    node_images = {}
    for entry in metadata:
        img_path = os.path.join(images_dir, entry["image_path"])
        # Convert BGR (saved by cv2) back to RGB for NoMaD
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        node_images[entry["id"]] = Image.fromarray(rgb)
    print(f"Loaded {len(node_images)} images")

    # Resolve scene path for geodesic distance
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

    # Compute geodesic distances for edge filtering
    # Group by position_index to get unique positions
    unique_positions = {}
    for entry in metadata:
        pidx = entry["position_index"]
        if pidx not in unique_positions:
            unique_positions[pidx] = entry["position"]

    pos_indices = sorted(unique_positions.keys())
    pos_list = [unique_positions[i] for i in pos_indices]

    print(f"Computing geodesic distances between {len(pos_list)} unique positions "
          f"(threshold={args.geodesic_threshold}m)...")
    t0 = time.time()
    geo_pairs = compute_geodesic_distances(scene_path, pos_list, args.geodesic_threshold)
    print(f"Geodesic filtering done in {time.time() - t0:.1f}s, "
          f"{len(geo_pairs) // 2} position pairs within threshold")

    # Build edge list: connect nodes at nearby positions
    # For each position pair within threshold, connect all yaw combinations
    edge_candidates = []
    for (pi_idx, pj_idx), geo_dist in geo_pairs.items():
        if pi_idx >= pj_idx:
            continue
        # Find all nodes at these positions
        nodes_i = [e["id"] for e in metadata if e["position_index"] == pi_idx]
        nodes_j = [e["id"] for e in metadata if e["position_index"] == pj_idx]
        for ni in nodes_i:
            for nj in nodes_j:
                if ni < nj:
                    edge_candidates.append((ni, nj, geo_dist))

    # Also connect same-position different-yaw nodes (geodesic=0, always)
    for pidx in pos_indices:
        nodes_at_pos = [e["id"] for e in metadata if e["position_index"] == pidx]
        for i_idx in range(len(nodes_at_pos)):
            for j_idx in range(i_idx + 1, len(nodes_at_pos)):
                edge_candidates.append((nodes_at_pos[i_idx], nodes_at_pos[j_idx], 0.0))

    print(f"Edge candidates: {len(edge_candidates)} pairs to evaluate with NoMaD")

    # Load NoMaD model
    print(f"Loading NoMaD model from {args.ckpt_path}...")
    model = load_model(args.ckpt_path, nomad_config, device)
    model.eval()
    image_size = nomad_config["image_size"]
    context_size = nomad_config["context_size"]

    # Compute NoMaD distances in batches
    print("Computing NoMaD distances...")
    t0 = time.time()
    G = nx.Graph()

    # Add all nodes with attributes
    for entry in metadata:
        G.add_node(entry["id"],
                   position=entry["position"],
                   yaw=entry["yaw"],
                   position_index=entry["position_index"],
                   image_path=entry["image_path"])

    # Process edge candidates in batches
    batch_obs = []
    batch_goals = []
    batch_meta = []

    for ni, nj, geo_dist in edge_candidates:
        batch_obs.append(node_images[ni])
        batch_goals.append(node_images[nj])
        batch_meta.append((ni, nj, geo_dist))

        if len(batch_obs) >= args.batch_size:
            dists = compute_nomad_distances_batch(model, batch_obs, batch_goals,
                                                   image_size, context_size)
            for (ni, nj, geo_dist), nomad_dist in zip(batch_meta, dists):
                G.add_edge(ni, nj,
                           nomad_distance=float(nomad_dist),
                           geodesic_distance=geo_dist)
            batch_obs.clear()
            batch_goals.clear()
            batch_meta.clear()

    # Flush remaining
    if batch_obs:
        dists = compute_nomad_distances_batch(model, batch_obs, batch_goals,
                                               image_size, context_size)
        for (ni, nj, geo_dist), nomad_dist in zip(batch_meta, dists):
            G.add_edge(ni, nj,
                       nomad_distance=float(nomad_dist),
                       geodesic_distance=geo_dist)

    elapsed = time.time() - t0
    print(f"NoMaD distance computation done in {elapsed:.1f}s")

    # Also compute reverse distances (i→j and j→i may differ due to NoMaD asymmetry)
    # Add reverse edges if they don't exist
    reverse_candidates = [(nj, ni, geo_dist) for ni, nj, geo_dist in edge_candidates
                          if not G.has_edge(nj, ni)]
    if reverse_candidates:
        print(f"Computing {len(reverse_candidates)} reverse distances...")
        batch_obs.clear()
        batch_goals.clear()
        batch_meta.clear()

        for ni, nj, geo_dist in reverse_candidates:
            batch_obs.append(node_images[ni])
            batch_goals.append(node_images[nj])
            batch_meta.append((ni, nj, geo_dist))

            if len(batch_obs) >= args.batch_size:
                dists = compute_nomad_distances_batch(model, batch_obs, batch_goals,
                                                       image_size, context_size)
                for (ni, nj, geo_dist), nomad_dist in zip(batch_meta, dists):
                    G.add_edge(ni, nj,
                               nomad_distance=float(nomad_dist),
                               geodesic_distance=geo_dist)
                batch_obs.clear()
                batch_goals.clear()
                batch_meta.clear()

        if batch_obs:
            dists = compute_nomad_distances_batch(model, batch_obs, batch_goals,
                                                   image_size, context_size)
            for (ni, nj, geo_dist), nomad_dist in zip(batch_meta, dists):
                G.add_edge(ni, nj,
                           nomad_distance=float(nomad_dist),
                           geodesic_distance=geo_dist)

    # Save graph
    output_path = os.path.join(args.data_dir, args.scene, "graph.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    graph_data = nx.node_link_data(G)
    with open(output_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    print(f"Graph saved to {output_path}")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if G.number_of_edges() > 0:
        nomad_dists = [d["nomad_distance"] for _, _, d in G.edges(data=True)]
        geo_dists = [d["geodesic_distance"] for _, _, d in G.edges(data=True)]
        print(f"  NoMaD distances: mean={np.mean(nomad_dists):.2f}, "
              f"min={np.min(nomad_dists):.2f}, max={np.max(nomad_dists):.2f}")
        print(f"  Geodesic distances: mean={np.mean(geo_dists):.2f}, "
              f"min={np.min(geo_dists):.2f}, max={np.max(geo_dists):.2f}")
        degrees = [d for _, d in G.degree()]
        print(f"  Average degree: {np.mean(degrees):.1f}")


if __name__ == "__main__":
    main()