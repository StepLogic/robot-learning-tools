#!/usr/bin/env python3
"""
SLAM Reward Wrapper for ORB_SLAM3 Integration
==============================================

Gymnasium wrapper that integrates ORB_SLAM3 for accurate position estimation
and replaces HER-based rewards with SLAM-based distance rewards.
"""

import os
import time
import threading
import queue
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
import json
import yaml

# Mock ORB_SLAM3 import - in practice this would be the actual Python binding
try:
    import orbslam3
    ORB_SLAM3_AVAILABLE = True
except ImportError:
    ORB_SLAM3_AVAILABLE = False
    print("ORB_SLAM3 not available - using mock for development")


class SLAMThread:
    """
    Background thread for ORB_SLAM3 processing.
    """

    def __init__(self, vocab_path: str, settings_path: str):
        """
        Initialize SLAM thread.

        Args:
            vocab_path: Path to ORB vocabulary file
            settings_path: Path to ORB_SLAM3 settings YAML
        """
        self.vocab_path = vocab_path
        self.settings_path = settings_path

        # Shared state with thread-safe access
        self.state_lock = threading.Lock()
        self.shared_state = {
            'latest_pose': np.eye(4),  # 4x4 transformation matrix
            'latest_velocity': np.zeros(3),  # 3D velocity
            'latest_state': 'NOT_INITIALIZED',  # Tracking state
            'latest_timestamp': 0.0,
            'frame_count': 0,
            'tracking_ok_count': 0
        }

        # Communication queues
        self.image_queue = queue.Queue(maxsize=10)
        self.imu_queue = queue.Queue(maxsize=10)
        self.shutdown_event = threading.Event()

        # SLAM system
        self.slam_system = None
        self.thread = None

    def initialize_slam(self):
        """Initialize ORB_SLAM3 system."""
        if not ORB_SLAM3_AVAILABLE:
            print("Using mock SLAM system")
            return True

        try:
            # Load settings
            with open(self.settings_path, 'r') as f:
                settings = yaml.safe_load(f)

            # Initialize SLAM system
            self.slam_system = orbslam3.System(
                self.vocab_path,
                self.settings_path,
                orbslam3.SensorType.MONOCULAR_IMU
            )

            print("ORB_SLAM3 initialized successfully")
            return True

        except Exception as e:
            print(f"Failed to initialize ORB_SLAM3: {e}")
            return False

    def process_frame(self, image: np.ndarray, imu_data: np.ndarray, timestamp: float):
        """
        Process a frame through ORB_SLAM3.

        Args:
            image: Raw BGR image (640x480)
            imu_data: Nx7 IMU data [ax, ay, az, gx, gy, gz, timestamp]
            timestamp: Frame timestamp
        """
        if not ORB_SLAM3_AVAILABLE:
            # Mock SLAM - simulate reasonable outputs
            with self.state_lock:
                # Simulate movement
                self.shared_state['frame_count'] += 1

                if self.shared_state['frame_count'] < 30:
                    state = 'NOT_INITIALIZED'
                elif self.shared_state['frame_count'] % 100 < 95:
                    state = 'OK'
                    # Simulate some movement
                    self.shared_state['latest_pose'][0, 3] += 0.01
                    self.shared_state['latest_pose'][1, 3] += 0.005
                    self.shared_state['latest_velocity'] = np.array([0.01, 0.005, 0.0])
                    self.shared_state['tracking_ok_count'] += 1
                else:
                    state = 'LOST'

                self.shared_state['latest_state'] = state
                self.shared_state['latest_timestamp'] = timestamp

            return state == 'OK'

        try:
            # Process with real ORB_SLAM3
            pose, velocity, state = self.slam_system.track_monocular_imu_with_velocity(
                image, timestamp, imu_data
            )

            with self.state_lock:
                self.shared_state['latest_pose'] = pose
                self.shared_state['latest_velocity'] = velocity
                self.shared_state['latest_state'] = state
                self.shared_state['latest_timestamp'] = timestamp

                if state == 'OK':
                    self.shared_state['tracking_ok_count'] += 1
                else:
                    self.shared_state['tracking_ok_count'] = 0

            return state == 'OK'

        except Exception as e:
            print(f"SLAM processing error: {e}")
            with self.state_lock:
                self.shared_state['latest_state'] = 'ERROR'
            return False

    def processing_loop(self):
        """Main processing loop for the SLAM thread."""
        print("SLAM thread started")

        # Initialize SLAM system
        slam_initialized = self.initialize_slam()

        while not self.shutdown_event.is_set():
            try:
                # Get next frame
                frame_data = self.image_queue.get(timeout=0.1)
                image, timestamp = frame_data

                # Get accumulated IMU data
                imu_data_list = []
                while not self.imu_queue.empty():
                    imu_data_list.append(self.imu_queue.get_nowait())

                if imu_data_list:
                    imu_data = np.vstack(imu_data_list)
                else:
                    # No IMU data available, create empty array
                    imu_data = np.zeros((0, 7))

                # Process frame
                if slam_initialized:
                    self.process_frame(image, imu_data, timestamp)
                else:
                    # Fallback: use pseudo-odometry
                    with self.state_lock:
                        self.shared_state['latest_state'] = 'NOT_INITIALIZED'
                        self.shared_state['latest_timestamp'] = timestamp
                        self.shared_state['frame_count'] += 1

            except queue.Empty:
                continue
            except Exception as e:
                print(f"SLAM thread error: {e}")
                time.sleep(0.1)

        # Cleanup
        if self.slam_system:
            self.slam_system.shutdown()

        print("SLAM thread stopped")

    def start(self):
        """Start the SLAM processing thread."""
        if self.thread is None or not self.thread.is_alive():
            self.shutdown_event.clear()
            self.thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the SLAM processing thread."""
        self.shutdown_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)

    def add_frame(self, image: np.ndarray, timestamp: float):
        """Add a frame to the processing queue."""
        try:
            self.image_queue.put_nowait((image.copy(), timestamp))
        except queue.Full:
            # Drop oldest frame if queue is full
            try:
                self.image_queue.get_nowait()
                self.image_queue.put_nowait((image.copy(), timestamp))
            except queue.Empty:
                pass

    def add_imu_data(self, imu_data: np.ndarray):
        """Add IMU data to the processing queue."""
        try:
            self.imu_queue.put_nowait(imu_data.copy())
        except queue.Full:
            # Drop oldest IMU data if queue is full
            try:
                self.imu_queue.get_nowait()
                self.imu_queue.put_nowait(imu_data.copy())
            except queue.Empty:
                pass

    def get_latest_state(self) -> Dict[str, Any]:
        """Get the latest SLAM state (thread-safe)."""
        with self.state_lock:
            return self.shared_state.copy()


class SLAMRewardWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that integrates ORB_SLAM3 for RL training.

    Replaces HER-based rewards with SLAM-based distance rewards.
    """

    def __init__(self, env: gym.Env, vocab_path: str, settings_path: str,
                 goal_threshold: float = 0.5, k_dist: float = 5.0,
                 k_goal: float = 50.0, k_step: float = 0.1):
        """
        Initialize SLAM reward wrapper.

        Args:
            env: Base environment
            vocab_path: Path to ORB vocabulary file
            settings_path: Path to ORB_SLAM3 settings YAML
            goal_threshold: Distance threshold for goal completion (meters)
            k_dist: Weight for distance delta reward
            k_goal: Weight for goal completion reward
            k_step: Weight for per-step time penalty
        """
        super().__init__(env)

        # SLAM configuration
        self.vocab_path = vocab_path
        self.settings_path = settings_path

        # Reward parameters
        self.goal_threshold = goal_threshold
        self.k_dist = k_dist
        self.k_goal = k_goal
        self.k_step = k_step

        # SLAM thread
        self.slam_thread = SLAMThread(vocab_path, settings_path)
        self.slam_thread.start()

        # Goal management
        self.current_goal = None
        self.last_slam_position = None
        self.last_distance_to_goal = None

        # Fallback state (for when SLAM not available)
        self.fallback_position = np.zeros(3)
        self.fallback_velocity = np.zeros(3)
        self.fallback_frame_count = 0

        # Statistics
        self.slam_ok_count = 0
        self.slam_lost_count = 0

    def _get_slam_position(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get current SLAM position and velocity.

        Returns:
            position: 3D position (x, y, z)
            velocity: 3D velocity (vx, vy, vz)
            state: SLAM tracking state
        """
        slam_state = self.slam_thread.get_latest_state()

        if slam_state['latest_state'] == 'OK':
            # Extract position from 4x4 pose matrix (translation component)
            position = slam_state['latest_pose'][:3, 3]
            velocity = slam_state['latest_velocity']
            state = slam_state['latest_state']

            self.slam_ok_count += 1
            return position, velocity, state
        elif slam_state['latest_state'] == 'NOT_INITIALIZED':
            # Use fallback pseudo-odometry during initialization
            self.fallback_frame_count += 1
            self.fallback_position[0] += 0.01  # Simulate forward movement
            return self.fallback_position.copy(), self.fallback_velocity.copy(), 'FALLBACK'
        else:
            # SLAM lost or error
            self.slam_lost_count += 1
            return self.fallback_position.copy(), self.fallback_velocity.copy(), 'LOST'

    def _sample_goal(self, current_position: np.ndarray) -> np.ndarray:
        """
        Sample a new goal position relative to current position.

        Args:
            current_position: Current 3D position

        Returns:
            goal_position: 3D goal position
        """
        # Sample in polar coordinates
        distance = np.random.uniform(0.5, 20.0)  # 0.5m to 20m
        angle = np.random.uniform(-np.pi, np.pi)  # -180 to 180 degrees

        # Convert to Cartesian coordinates relative to current position
        goal_offset = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
            0.0  # Keep same height
        ])

        goal_position = current_position + goal_offset
        return goal_position

    def _compute_distance_to_goal(self, position: np.ndarray, goal: np.ndarray) -> float:
        """
        Compute 2D Euclidean distance to goal (ignoring Z axis).

        Args:
            position: Current 3D position
            goal: Goal 3D position

        Returns:
            distance: 2D distance in meters
        """
        delta = position[:2] - goal[:2]  # Only X and Y
        return np.linalg.norm(delta)

    def reset(self, **kwargs):
        """
        Reset the environment and sample a new goal.
        """
        # Reset base environment
        obs, info = self.env.reset(**kwargs)

        # Get current position (use fallback if SLAM not ready)
        current_position, _, _ = self._get_slam_position()

        # Sample new goal
        self.current_goal = self._sample_goal(current_position)

        # Initialize distance tracking
        self.last_slam_position = current_position.copy()
        self.last_distance_to_goal = self._compute_distance_to_goal(
            current_position, self.current_goal
        )

        # Reset fallback state
        self.fallback_position = current_position.copy()
        self.fallback_velocity = np.zeros(3)
        self.fallback_frame_count = 0

        # Add SLAM info to info dict
        info['slam_pos'] = current_position.tolist()
        info['slam_vel'] = [0.0, 0.0, 0.0]
        info['slam_state'] = 'RESET'
        info['dist_to_goal'] = float(self.last_distance_to_goal)
        info['goal_pos'] = self.current_goal.tolist()

        return obs, info

    def step(self, action):
        """
        Step the environment and compute SLAM-based reward.
        """
        # Step base environment
        obs, _, terminated, truncated, info = self.env.step(action)

        # Get raw image and IMU data from info dict
        raw_image = info.get('raw_image')
        imu_data = info.get('imu_raw')
        timestamp = info.get('timestamp', time.time())

        # Process with SLAM if data available
        if raw_image is not None and imu_data is not None:
            # Add to SLAM thread queues
            self.slam_thread.add_frame(raw_image, timestamp)

            # Format IMU data for ORB_SLAM3 (Nx7 array)
            imu_array = np.array([
                imu_data[0], imu_data[1], imu_data[2],  # ax, ay, az
                imu_data[3], imu_data[4], imu_data[5],  # gx, gy, gz (already in rad/s)
                timestamp
            ]).reshape(1, 7)
            self.slam_thread.add_imu_data(imu_array)

        # Get current SLAM state
        current_position, current_velocity, slam_state = self._get_slam_position()

        # Compute distance to goal
        current_distance = self._compute_distance_to_goal(current_position, self.current_goal)

        # Compute reward
        if self.last_distance_to_goal is not None:
            delta_dist = self.last_distance_to_goal - current_distance
        else:
            delta_dist = 0.0

        # Check if goal reached
        goal_reached = current_distance < self.goal_threshold

        # Compute reward components
        step_penalty = -self.k_step
        distance_reward = self.k_dist * delta_dist
        goal_reward = self.k_goal if goal_reached else 0.0

        reward = step_penalty + distance_reward + goal_reward

        # Check termination conditions
        slam_terminated = False
        slam_reward_penalty = 0.0

        if slam_state == 'LOST':
            # SLAM tracking lost - terminate with penalty
            slam_terminated = True
            slam_reward_penalty = -30.0
            reward += slam_reward_penalty
        elif slam_state == 'ERROR':
            # SLAM error - terminate with larger penalty
            slam_terminated = True
            slam_reward_penalty = -50.0
            reward += slam_reward_penalty

        # Update state for next step
        self.last_slam_position = current_position.copy()
        self.last_distance_to_goal = current_distance

        # If goal reached, sample new goal
        if goal_reached:
            current_position, _, _ = self._get_slam_position()
            self.current_goal = self._sample_goal(current_position)
            current_distance = self._compute_distance_to_goal(
                current_position, self.current_goal
            )
            self.last_distance_to_goal = current_distance

        # Add SLAM info to info dict
        info['slam_pos'] = current_position.tolist()
        info['slam_vel'] = current_velocity.tolist()
        info['slam_state'] = slam_state
        info['dist_to_goal'] = float(current_distance)
        info['goal_pos'] = self.current_goal.tolist()
        info['reward_components'] = {
            'step_penalty': float(step_penalty),
            'distance_reward': float(distance_reward),
            'goal_reward': float(goal_reward),
            'slam_penalty': float(slam_reward_penalty)
        }

        # Update termination status
        terminated = terminated or slam_terminated

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources."""
        self.slam_thread.stop()
        super().close()

    def get_slam_stats(self) -> Dict[str, int]:
        """Get SLAM statistics."""
        return {
            'slam_ok_count': self.slam_ok_count,
            'slam_lost_count': self.slam_lost_count
        }


# Mock ORB_SLAM3 classes for development
if not ORB_SLAM3_AVAILABLE:
    class SensorType:
        MONOCULAR_IMU = 1

    class System:
        def __init__(self, vocab_path, settings_path, sensor_type):
            self.vocab_path = vocab_path
            self.settings_path = settings_path
            self.sensor_type = sensor_type
            print(f"Mock ORB_SLAM3 system initialized with {vocab_path}")

        def track_monocular_imu_with_velocity(self, image, timestamp, imu_data):
            # Mock tracking - return simulated pose and velocity
            pose = np.eye(4)
            pose[0, 3] = np.random.normal(1.0, 0.1)  # X position
            pose[1, 3] = np.random.normal(0.5, 0.1)  # Y position

            velocity = np.array([
                np.random.normal(0.01, 0.005),
                np.random.normal(0.005, 0.005),
                0.0
            ])

            # Simulate occasional tracking loss
            if np.random.random() < 0.05:
                state = 'LOST'
            else:
                state = 'OK'

            return pose, velocity, state

        def shutdown(self):
            print("Mock ORB_SLAM3 system shut down")

    # Add mock classes to orbslam3 module
    import sys
    orbslam3 = type(sys)('orbslam3')
    orbslam3.SensorType = SensorType
    orbslam3.System = System
    sys.modules['orbslam3'] = orbslam3