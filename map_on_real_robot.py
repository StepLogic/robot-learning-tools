#!/usr/bin/env python
"""
Real-robot execution script for RacerEnv: Exploration and Topological Map Building.
1. Mapping Mode (--mode map): Runs an exploring agent, detects entropy spikes via SMA filter, and saves map nodes.
2. Build Mode (--mode build): Loads nodes and builds a FAISS-based SIFT+VLAD topological graph.
"""

import os
import pickle
import random
import argparse
from collections import deque

import numpy as np
import networkx as nx
import faiss
import flax
from flax.training import checkpoints

# JaxRL2 & Agent Imports
from jaxrl2.agents import DrQLearner
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# Configs & Graph Builder
from src.configs.eval_config_v2 import config
from src.configs.drq_default import get_config
from eval_graph_builder_vlad import TopologicalGraphBuilder, GraphBuilderConfig, SIFTVLADConfig

# Import your real robot environment
from racer_env import RacerEnv

flax.config.update('flax_use_orbax_checkpointing', True)

# ============================================================================
# CONFIGURATION
# ============================================================================
CHECKPOINT_PATH = "./checkpoints/final_drq/checkpoint_2"
CHECKPOINT_STEP = 4950000
SEED = 42
RESULTS_DIR = "./racer_results"
CONFIG = get_config()

# SMA Filter settings for spike detection
MAD_WINDOW = 200
MAD_THRESHOLD = 3.5

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# UTILITIES & FILTERS
# ============================================================================

class Filter:
    """
    Implements a 'First-Strike' Simple Moving Average (SMA) filter.
    Triggers if the new value exceeds the moving average by a 
    specified threshold of standard deviations.
    """
    def __init__(self, window_size, threshold, lockout_samples=30):
        self.window_size = window_size
        self.threshold = threshold  
        self.history = deque(maxlen=window_size)
        self.lockout_samples = lockout_samples
        self.current_lockout = 0

    def update_and_check(self, new_value):
        self.history.append(new_value)
        
        if self.current_lockout > 0:
            self.current_lockout -= 1
            return False

        if len(self.history) < self.window_size:
            return False

        data = np.array(self.history)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            std = 1e-6 
            
        deviation = new_value - mean
        
        if (deviation > 0) and (deviation > self.threshold * std):
            self.current_lockout = self.lockout_samples
            return True

        return False

def compute_bearing_between_points(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    bearing = np.arctan2(dx, dy)
    if bearing < 0:
        bearing += 2 * np.pi
    return bearing

def load_checkpoint(agent, checkpoint_path):
    """Load agent parameters from checkpoint."""
    state_dict = {
        'actor_params': agent._actor,
        'critic_params': agent._critic,
        'target_critic_params': agent._target_critic_params,
        'temp': agent._temp,
        'rng': agent._rng,
    }
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=state_dict)
    agent._actor = state_dict['actor_params']
    agent._critic = state_dict['critic_params']
    agent._target_critic_params = state_dict['target_critic_params']
    agent._temp = state_dict['temp']
    agent._rng = state_dict['rng']
    print("Checkpoint loaded successfully.")
    return agent

def get_action_distribution_stats(agent, obs):
    """Get mean, std, and entropy of the agent's action distribution."""
    dist = agent.action_dist(obs)
    mean = np.array(dist.mode())
    std = np.array(dist.stddev())
    return mean, std, float(dist.entropy())

# ============================================================================
# PHASE 1: EXPLORATION & MAPPING
# ============================================================================

def collect_map_data(env, agent, num_episodes=5):
    """Run agent on real robot with exploration noise, compute entropy, and save map nodes."""
    print(f"\n--- Starting Map Collection for {num_episodes} episodes ---")
    unwrapped_env = env.unwrapped
    all_spike_data = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_length = 0
        spike_tracker = Filter(window_size=MAD_WINDOW, threshold=MAD_THRESHOLD)
        
        # Action noise for active exploration
        noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=np.array([0.05, 0.3]))

        while not done:
            action_mean, action_std, entropy = get_action_distribution_stats(agent, obs)
            is_spike = spike_tracker.update_and_check(entropy)
            
            # Format observation vector (adjust bounds for real robot)
            vec = np.copy(obs["vector"])
            vec[2] = np.clip(unwrapped_env.experiment.speed / 1.0, 0, 1.0) if hasattr(unwrapped_env.experiment, 'speed') else 0.0
            vec[3:7] = [-1.0, -1.0, -1.0, 0.0]
            obs["vector"] = vec
            
            action = agent.sample_actions(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, done, truncated, info = env.step(action)
            episode_length += 1
            
            if is_spike:
                # TODO: Ensure these methods map correctly to RacerEnv's odometry API
                current_location = unwrapped_env.get_current_location() # Expected: (x, y, z)
                current_compass = unwrapped_env.get_compass_bearing()   # Expected: radians
                current_yaw = unwrapped_env.get_yaw()                   # Expected: radians
                
                current_obs_image = unwrapped_env.experiment.obs_img
                current_latent = getattr(unwrapped_env.experiment, 'current_image_latent', None)
                
                node_data = {
                    'location': current_location,
                    'compass': current_compass,
                    'yaw': current_yaw,
                    'latent': current_latent,
                    'obs_img': current_obs_image,
                    'entropy': entropy,
                    'step': episode_length,
                    'episode': episode + 1,
                }
                all_spike_data.append(node_data)
                print(f"Node saved at step {episode_length} | Entropy: {entropy:.3f}")

            obs = next_obs
            if done or truncated:
                noise.reset()
                break

    # Post-process to add bearing to next
    for i in range(len(all_spike_data) - 1):
        x1, y1, _ = all_spike_data[i]['location']
        x2, y2, _ = all_spike_data[i+1]['location']
        all_spike_data[i]['bearing_to_next'] = compute_bearing_between_points(x1, y1, x2, y2)
    
    if all_spike_data:
        all_spike_data[-1]['bearing_to_next'] = all_spike_data[-1]['compass']

    map_path = os.path.join(RESULTS_DIR, "map.pkl")
    with open(map_path, "wb") as f:
        pickle.dump(all_spike_data, f)
    print(f"Map data saved to {map_path} with {len(all_spike_data)} nodes.")

# ============================================================================
# PHASE 2: GRAPH BUILDING
# ============================================================================

def setup_graph_builder():
    sift_vlad_config = SIFTVLADConfig(
        n_clusters=64,
        n_keypoints=500,
        vlad_normalize=True,
        intra_normalize=True,
        resize_height=240,
        resize_width=320,
        verbose=True
    )
    gconfig = GraphBuilderConfig(
        feature_type='sift_vlad',
        image_key='obs_img',
        sift_vlad_config=sift_vlad_config,
        verbose=True
    )
    return TopologicalGraphBuilder(gconfig)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Real Robot Exploration and Topological Map Building")
    parser.add_argument("--mode", type=str, choices=["map", "build"], required=True, 
                        help="Choose execution mode: 'map' for exploration/data collection, 'build' to generate the FAISS graph.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of exploration episodes for map mode.")
    args = parser.parse_args()

    # Set seeds
    np.random.seed(SEED)
    random.seed(SEED)

    # Run selected mode
    if args.mode == "map":
        print("Initializing RacerEnv for Mapping...")
        env = RacerEnv(config=config)
        env = TimeLimit(env, max_episode_steps=10000)
        env = RecordEpisodeStatistics(env)
        
        agent = DrQLearner(
            SEED,
            env.observation_space.sample(),
            env.action_space.sample(),
            **dict(CONFIG)
        )
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        agent = load_checkpoint(agent, CHECKPOINT_PATH)
        collect_map_data(env, agent, num_episodes=args.episodes)
        env.close()
        
    elif args.mode == "build":
        builder = setup_graph_builder()
        map_path = os.path.join(RESULTS_DIR, "map.pkl")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map file {map_path} not found. Run '--mode map' first to collect data.")
        
        print(f"Building topological graph from {map_path}...")
        builder.build_from_file(map_path)
        print(f"Graph successfully built and indexed. Artifacts are handled by TopologicalGraphBuilder.")

if __name__ == "__main__":
    main()