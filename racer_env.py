import json
import os
from urllib.request import Request, urlopen
import sys
import numpy as np
import cv2
import pygame # Import the Pygame library
import pickle
import gymnasium as gym # Import Gymnasium
from gymnasium import spaces # For defining observation and action spaces


INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT = 640, 480 
FPS = 30 # Frames per second for the game loop and image display
ROBOT_SERVER_URL = "http://10.42.0.1:8000"

class RacerEnv(gym.Env):
    """
    A Gymnasium environment for controlling a Donkey Car robot.
    The environment provides camera observations and accepts steering and throttle actions.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode="human"):
        super().__init__()

        # Define the observation space: a Box representing the camera image
        # The image will be 3 channels (RGB) with shape (height, width)
        # We assume the image will be uint8 values (0-255)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(INITIAL_SCREEN_HEIGHT, INITIAL_SCREEN_WIDTH, 3), 
            dtype=np.uint8
        )


        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32), 
            high=np.array([1.0, 5.0], dtype=np.float32), 
            dtype=np.float32
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Pygame related variables
        self.screen = None
        self.clock = None
        self.first_image_received = False
        self.trajectory = {"position": []}
        self.frame_count = 0
        
        # Internal state to track if the environment should terminate due to Pygame QUIT
        self._terminated_by_pygame_quit = False 
        self.image=np.zeros_like((120,160,3),dtype=np.uint8)

    def _send_command_and_get_image(self, steering, throttle):
    
        try:
            # Construct the JSON payload for the control command
            data = f'{{"steering":{np.clip(steering,-1.0,1.0)},"throttle":{np.clip(throttle,0,0.15)}}}'.encode()
            # print(data)
            req = Request(f"{ROBOT_SERVER_URL}/control", data=data, 
                         headers={'Content-Type': 'application/json'})
            
            # Send the request and get the response, with a short timeout for responsiveness
            with urlopen(req, timeout=1) as response:
                result = json.loads(response.read())
                
                # Extract the image data from the observation
                image = np.array(result["observation"]["img"])
                
                # Ensure the image data type is unsigned 8-bit integer (uint8)
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                
                # # Convert the image to RGB format if it's in BGR or BGRA (common for OpenCV)
                # if len(image.shape) == 3:
                #     if image.shape[2] == 3: # Assuming BGR
                #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #     elif image.shape[2] == 4: # Assuming BGRA
                #         image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                image = cv2.flip(image,0)
                self.image=image
                # Print current control values to the console for feedback
                # print(f"\rS={steering:+.2f} T={throttle:+.2f}", end='', flush=True)
                return image
        except Exception as e:
            # Catch and print any errors during the process
            # print(f"\rError: {e}", end='', flush=True)
            return self.image

    def _get_obs(self):
        """
        Returns the current observation. In this case, it's the camera image.
        """
        return self.current_image

    def _get_info(self):
        """
        Returns auxiliary information. Could include sensor data, robot status, etc.
        For now, we'll return the recorded trajectory data.
        """
        return {"trajectory": self.trajectory}

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        Args:
            seed (int): An optional seed for reproducibility.
            options (dict): An optional dictionary of options for the reset.

        Returns:
            observation (object): The initial observation.
            info (dict): A dictionary containing auxiliary diagnostic information.
        """
        super().reset(seed=seed)

        # Stop the robot
        self._send_command_and_get_image(0.0, 0.0) 
        
        # Get the first observation after stopping
        self.current_image = self._send_command_and_get_image(0.0, 0.0)
        
        # Reset internal state variables
        self.trajectory = {"position": []}
        self.frame_count = 0
        self.running_throttle = 0
        self.running_steering = 0
        self.first_image_received = False # Reset for render
        self._terminated_by_pygame_quit = False # Reset termination flag

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Performs one step in the environment.
        Args:
            action (np.array): An array containing [steering, throttle].

        Returns:
            observation (object): The observation of the environment after the action.
            reward (float): The amount of reward received by the agent after the action.
            terminated (bool): Whether the episode has ended (e.g., robot crashed, reached goal).
            truncated (bool): Whether the episode has been truncated (e.g., time limit reached).
            info (dict): A dictionary containing auxiliary diagnostic information.
        """
        steering, throttle = action[0], action[1]
        
        # Initialize terminated and reward
        terminated = False
        reward = 0.1 # Default positive reward

        # --- Pygame Event Handling within the step method ---
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        # running = False # Quit the application
                            terminated = True


        if terminated:
            steering = 0.0
            throttle = 0.0
            reward = -10.0 # Maintain penalty
            terminated = True


        # Send the command to the robot. Note: throttle is divided by 2 as in the original script.
        self.current_image = self._send_command_and_get_image(steering, throttle)

        truncated = False # You would define conditions for truncation (e.g., time limit)

        # observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        self._send_command_and_get_image(0.0, 0.0)

        return self.current_image, reward, terminated, truncated, info

    def render(self):
    
        if self.render_mode == "rgb_array":
            return self._get_obs()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT))
            pygame.display.set_caption("Donkey Car Control")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.current_image is not None:
            # If this is the first image, resize the Pygame window to match its dimensions
            if not self.first_image_received:
                self.first_image_received = True

            # Convert the NumPy image array to a Pygame Surface.
            img_surface = pygame.surfarray.make_surface(cv2.resize(self.current_image.swapaxes(0, 1), 
                                                                    (INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT), 
                                                                    interpolation=cv2.INTER_AREA)) 
            
            # Draw the image onto the Pygame screen at position (0,0)
            self.screen.blit(img_surface, (0, 0))
        
        # We need to process Pygame events for the window to remain responsive.
        # However, for an RL agent, the agent itself shouldn't be handling manual events.
        # The key for this request is to check for pygame.QUIT within the step method.
        pygame.display.flip() # Update the entire screen
        self.clock.tick(self.metadata["render_fps"]) # Limit frame rate

    def close(self):
        """
        Cleans up resources.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
        # Send a final stop command to the robot
        self._send_command_and_get_image(0.0, 0.0)
        print("\nEnvironment Closed.")

# Example of how to use the Gymnasium environment
if __name__ == '__main__':
    # Create the environment in human render mode
    env = DonkeyCarEnv(render_mode="human")

    # Reset the environment to get the initial observation
    observation, info = env.reset()

    running = True
    manual_steering = 0.0
    manual_throttle = 0.0

    while running:
        # Check for Pygame events here for manual control during example usage,
        # but the environment's step method will also check for QUIT.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False # Exit the main loop
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False # Quit the application
                elif event.key == pygame.K_w:
                    manual_throttle = min(1.0, manual_throttle + 0.05)
                elif event.key == pygame.K_s:
                    manual_throttle = max(-1.0, manual_throttle - 0.05)
                elif event.key == pygame.K_a:
                    manual_steering = max(-1.0, manual_steering - 0.05)
                elif event.key == pygame.K_d:
                    manual_steering = min(1.0, manual_steering + 0.05)
                elif event.key == pygame.K_SPACE:
                    manual_steering = 0.0
                    manual_throttle = 0.0
            elif event.type == pygame.KEYUP:
                # You might want to add more sophisticated release logic
                pass

        # Create the action array [steering, throttle]
        action = np.array([manual_steering, manual_throttle], dtype=np.float32)
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode finished. Reward: {reward}")
            # If the episode terminated, stop the main loop
            running = False 

    env.close()