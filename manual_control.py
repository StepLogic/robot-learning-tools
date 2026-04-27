import json
import os
from urllib.request import Request, urlopen
import sys
import numpy as np
import cv2
import pygame
import pickle
from datetime import datetime

# Initial arbitrary screen dimensions. These will be updated dynamically
# once the first image from the robot camera is received.
INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT = 640, 480 
FPS = 30 # Frames per second for the game loop and image display

def send_command_and_get_observation(steering, throttle, server="http://10.42.0.1:8000"):
    """
    Sends control commands (steering and throttle) to the specified server
    and attempts to retrieve the latest camera image observation and IMU data.

    Args:
        steering (float): The steering value, typically in the range [-1.0, 1.0].
        throttle (float): The throttle value, typically in the range [-1.0, 1.0].
        server (str): The base URL of the robot's control server.

    Returns:
        tuple: (image, imu_data, success) where:
            - image (np.array): The camera image as a NumPy array (in RGB format)
            - imu_data (dict): Dictionary containing IMU sensor data
            - success (bool): True if successful, False otherwise
    """
    try:
        # Construct the JSON payload for the control command
        data = f'{{"steering":{steering},"throttle":{throttle}}}'.encode()
        req = Request(f"{server}/control", data=data, 
                     headers={'Content-Type': 'application/json'})
        
        # Send the request and get the response, with a short timeout for responsiveness
        with urlopen(req, timeout=1) as response:
            result = json.loads(response.read())
            
            # Extract the image data from the observation
            image = np.array(result["observation"]["img"])
            
            # Ensure the image data type is unsigned 8-bit integer (uint8)
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Convert the image to RGB format if it's in BGR or BGRA (common for OpenCV)
            if len(image.shape) == 3:
                if image.shape[2] == 3: # Assuming BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4: # Assuming BGRA
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            
            # Extract IMU data
            imu_data = result["observation"]["imu"]
            
            # Print current control values to the console for feedback
            print(f"\rS={steering:+.2f} T={throttle:+.2f} | "
                  f"Acc: ({imu_data['acceleration']['x']:.2f}, "
                  f"{imu_data['acceleration']['y']:.2f}, "
                  f"{imu_data['acceleration']['z']:.2f})", 
                  end='', flush=True)
            
            return image, imu_data, True
            
    except Exception as e:
        # Catch and print any errors during the process
        print(f"\rError: {e}", end='', flush=True)
        return None, None, False


def save_trajectory_data(trajectory_data, trajectory_num, base_dir="topomap"):
    """
    Save trajectory data to a pickle file.
    
    Args:
        trajectory_data (dict): Dictionary containing trajectory information
        trajectory_num (int): Current trajectory number
        base_dir (str): Base directory for saving data
    """
    filename = f"{base_dir}/trajectory_{trajectory_num:03d}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(trajectory_data, f)
    print(f"\n✓ Saved trajectory {trajectory_num} to {filename}")


def manual_control():
    """
    Provides live manual control of the robot using Pygame for keyboard input
    and displaying the camera feed. Collects and saves trajectory data including
    images, actions, IMU data, and position information.
    """
    print("=" * 60)
    print("=== Pygame Topomap Data Collection ===")
    print("=" * 60)
    print("\nControls:")
    print("  W/S = Throttle forward/backward")
    print("  A/D = Steering left/right")
    print("  SPACE = Stop (zero controls)")
    print("  R = Start new trajectory (saves current one)")
    print("  Q = Quit and save")
    print("\nData Collection:")
    print("  - Images saved to topomap/traj_XXX/")
    print("  - Trajectory data saved to topomap/trajectory_XXX.pkl")
    print("=" * 60)
    
    # Initialize Pygame modules
    pygame.init()
    
    # Create the Pygame display surface. Initial size is a placeholder.
    screen = pygame.display.set_mode((INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT))
    pygame.display.set_caption("Donkey Car - Topomap Collection")
    
    # Create a clock object to control the frame rate
    clock = pygame.time.Clock()

    # Control variables
    steering = 0.0
    throttle = 0.0
    
    # Flags to track continuous key presses for smoother control
    move_forward = False
    move_backward = False
    turn_left = False
    turn_right = False

    # Trajectory tracking
    trajectory_num = 0
    step_count = 0
    running_throttle = 0.0
    running_steering = 0.0
    
    # Current trajectory data structure
    trajectory_data = {
        "trajectory_id": trajectory_num,
        "start_time": datetime.now().isoformat(),
        "steps": [],
        "positions": [],
        "actions": [],
        "imu_data": [],
        "image_paths": [],
        "metadata": {
            "total_steps": 0,
            "total_distance": 0.0,
            "end_time": None
        }
    }
    
    # Create base directory
    base_dir = "topomap"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create trajectory-specific directory
    traj_dir = f"{base_dir}/traj_{trajectory_num:03d}"
    os.makedirs(traj_dir, exist_ok=True)

    running = True
    first_image_received = False
    
    print(f"\n🚗 Trajectory {trajectory_num} started! Drive to collect data...")
    
    while running:
        # Event handling loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_w:
                    move_forward = True
                elif event.key == pygame.K_s:
                    move_backward = True
                elif event.key == pygame.K_a:
                    turn_left = True
                elif event.key == pygame.K_d:
                    turn_right = True
                elif event.key == pygame.K_SPACE:
                    # Emergency stop
                    steering = 0.0
                    throttle = 0.0
                    move_forward = move_backward = turn_left = turn_right = False
                elif event.key == pygame.K_r:
                    # Start new trajectory
                    if step_count > 0:  # Only save if we have data
                        # Finalize current trajectory
                        trajectory_data["metadata"]["total_steps"] = step_count
                        trajectory_data["metadata"]["end_time"] = datetime.now().isoformat()
                        
                        # Calculate total distance
                        if len(trajectory_data["positions"]) > 1:
                            positions = np.array(trajectory_data["positions"])
                            distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                            trajectory_data["metadata"]["total_distance"] = float(np.sum(distances))
                        
                        # Save current trajectory
                        save_trajectory_data(trajectory_data, trajectory_num, base_dir)
                        
                        # Start new trajectory
                        trajectory_num += 1
                        step_count = 0
                        running_throttle = 0.0
                        running_steering = 0.0
                        
                        # Create new trajectory directory
                        traj_dir = f"{base_dir}/traj_{trajectory_num:03d}"
                        os.makedirs(traj_dir, exist_ok=True)
                        
                        # Initialize new trajectory data
                        trajectory_data = {
                            "trajectory_id": trajectory_num,
                            "start_time": datetime.now().isoformat(),
                            "steps": [],
                            "positions": [],
                            "actions": [],
                            "imu_data": [],
                            "image_paths": [],
                            "metadata": {
                                "total_steps": 0,
                                "total_distance": 0.0,
                                "end_time": None
                            }
                        }
                        
                        print(f"\n🚗 Trajectory {trajectory_num} started!")
                    
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    move_forward = False
                elif event.key == pygame.K_s:
                    move_backward = False
                elif event.key == pygame.K_a:
                    turn_left = False
                elif event.key == pygame.K_d:
                    turn_right = False

        # Update steering and throttle based on active key states
        if move_forward:
            throttle = min(1.0, throttle + 0.05) 
        if move_backward:
            throttle = max(-1.0, throttle - 0.05)
        if turn_left:
            steering = max(-1.0, steering - 0.05)
        if turn_right:
            steering = min(1.0, steering + 0.05)

        # Send commands and get observation
        actual_throttle = throttle / 2  # Apply throttle scaling
        image, imu_data, success = send_command_and_get_observation(steering, actual_throttle)
        
        if success and image is not None:
            # Update position tracking (cumulative)
            running_throttle += actual_throttle
            running_steering += steering
            
            # Save image to disk
            image_filename = f"{traj_dir}/step_{step_count:05d}.jpg"
            # Convert back to BGR for OpenCV saving
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_filename, image_bgr)
            
            # Store step data
            step_data = {
                "step": step_count,
                "timestamp": datetime.now().isoformat(),
                "position": [float(running_throttle), float(running_steering)],
                "action": [float(steering), float(actual_throttle)],
                "imu": {
                    "acceleration": {
                        "x": float(imu_data["acceleration"]["x"]),
                        "y": float(imu_data["acceleration"]["y"]),
                        "z": float(imu_data["acceleration"]["z"]),
                        "resultant": float(imu_data["acceleration"]["resultant"])
                    },
                    "angular_velocity": {
                        "roll_rate": float(imu_data["angular_velocity"]["roll_rate"]),
                        "pitch_rate": float(imu_data["angular_velocity"]["pitch_rate"]),
                        "yaw_rate": float(imu_data["angular_velocity"]["yaw_rate"]),
                        "magnitude": float(imu_data["angular_velocity"]["magnitude"])
                    },
                    "orientation": {
                        "roll": float(imu_data["orientation"]["roll"]),
                        "pitch": float(imu_data["orientation"]["pitch"]),
                        "yaw": float(imu_data["orientation"]["yaw"])
                    }
                },
                "image_path": image_filename
            }
            
            # Append to trajectory data
            trajectory_data["steps"].append(step_data)
            trajectory_data["positions"].append([float(running_throttle), float(running_steering)])
            trajectory_data["actions"].append([float(steering), float(actual_throttle)])
            trajectory_data["imu_data"].append(step_data["imu"])
            trajectory_data["image_paths"].append(image_filename)
            
            step_count += 1
            
            # Display info on screen
            if not first_image_received:
                screen = pygame.display.set_mode((INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT))
                first_image_received = True

            # Convert image to Pygame surface and display
            img_surface = pygame.surfarray.make_surface(
                cv2.resize(image.swapaxes(0, 1), 
                          (INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT),
                          interpolation=cv2.INTER_AREA)
            )
            screen.blit(img_surface, (0, 0))
            
            # Add text overlay with trajectory info
            font = pygame.font.Font(None, 24)
            info_texts = [
                f"Trajectory: {trajectory_num}",
                f"Steps: {step_count}",
                f"Steering: {steering:+.2f}",
                f"Throttle: {actual_throttle:+.2f}",
                f"Press R for new trajectory"
            ]
            
            y_offset = 10
            for text in info_texts:
                text_surface = font.render(text, True, (255, 255, 0))
                text_rect = text_surface.get_rect()
                # Draw background rectangle for better visibility
                bg_rect = text_rect.copy()
                bg_rect.inflate_ip(10, 5)
                bg_rect.topleft = (10, y_offset)
                pygame.draw.rect(screen, (0, 0, 0), bg_rect)
                screen.blit(text_surface, (10, y_offset))
                y_offset += 25
        
        # Update the display
        pygame.display.flip()
        
        # Limit the frame rate
        clock.tick(FPS)

    # Save final trajectory before exiting
    if step_count > 0:
        trajectory_data["metadata"]["total_steps"] = step_count
        trajectory_data["metadata"]["end_time"] = datetime.now().isoformat()
        
        if len(trajectory_data["positions"]) > 1:
            positions = np.array(trajectory_data["positions"])
            distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            trajectory_data["metadata"]["total_distance"] = float(np.sum(distances))
        
        save_trajectory_data(trajectory_data, trajectory_num, base_dir)

    # Clean up
    pygame.quit()
    send_command_and_get_observation(0.0, 0.0)  # Stop the robot
    
    print("\n" + "=" * 60)
    print("Data Collection Summary")
    print("=" * 60)
    print(f"Total trajectories collected: {trajectory_num + 1}")
    print(f"Data saved in: {base_dir}/")
    print(f"Total steps: {step_count}")
    print("=" * 60)
    print("\n✓ Collection complete! Use this data for topological mapping.")


if __name__ == '__main__':
    manual_control()