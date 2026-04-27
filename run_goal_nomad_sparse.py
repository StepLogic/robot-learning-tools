# ROS
import glob
import threading
from multiprocessing import Process
from typing import Tuple
import os
import time
import numpy as np
import torch
import json

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from navigation_policies.navigation_policies.baseline_policies.misc import get_action
from PIL import Image
from navigation_policies.navigation_policies.baseline_policies.misc import Args, to_numpy, transform_images, load_model
from navigation_policies.navigation_policies.baseline_policies.baselines_config import nomad_config
from racer_imu_env import RacerEnv

EPS = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Args(model="nomad", waypoint=3, num_samples=8, radius=4, close_threshold=3)

def clip_angle(theta) -> float:
    """Clip angle to [-pi, pi]"""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi
      
def pd_controller(waypoint: np.ndarray):
    """PD controller for the robot"""
    assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint
    # this controller only uses the predicted heading if dx and dy near zero
    if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
        v = 0
        w = clip_angle(np.arctan2(hy, hx))  
    elif np.abs(dx) < EPS:
        v =  0
        w = np.sign(dy) * np.pi/(2)
    else:
        v = dx 
        w = np.arctan(dy/dx)
    v = np.clip(v/4, 0, 0.2)
    w = np.clip(-0.18+w, -0.2, 0.3)
    return w, v


class NoMaD():
    def __init__(self, 
                 ckpt_path="/home/kojogyaase/Projects/Research/robot-server/navigation_policies/navigation_policies/pretrained_models/nomad.pth",
                 mode='navigate', 
                 goal=-1, 
                 map_dir="topomap", 
                 skip_index=1, 
                 search_radius=4,
                 json_path=None, 
                 traj_id="traj_1", 
                 threshold_key="threshold_3.0"):
        
        self.mode = mode
        self.goal_node = goal
        self.closest_node = 0
        self.context_queue = []
        self.args = args
        self.args = self.args._replace(radius=search_radius)
        
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Please specify a valid checkpoint path: {ckpt_path}")
            
        # Load model parameters
        self.context_size = nomad_config["context_size"]
        self.model_params = nomad_config
        self.model = load_model(ckpt_path, nomad_config, device)
        self.model = self.model.to(device)
        self.model.eval()

        self.num_diffusion_iters = nomad_config["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=nomad_config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        # Map Loading Logic
        self.topomap = []
        self.map_dir = map_dir
        self.nodes_used = []
        
        # Get all map images (adjust .png to .jpg if your map uses jpegs)
        topomap_filenames = sorted(glob.glob(os.path.join(map_dir, "*.png")))
        if not topomap_filenames:
            topomap_filenames = sorted(glob.glob(os.path.join(map_dir, "*.jpg")))
            
        if not topomap_filenames:
            raise FileNotFoundError(f"No images found in {map_dir}")

        # JSON Loading Logic
        if json_path and os.path.exists(json_path):
            print(f"Loading sparsified map from {json_path}...")
            with open(json_path, 'r') as f:
                map_data = json.load(f)
                
            try:
                indices = map_data[traj_id][threshold_key]["retained_nodes_indices"]
                print(f"Loaded {len(indices)} retained nodes out of {len(topomap_filenames)} original.")
                for ix in indices:
                    self.topomap.append(Image.open(topomap_filenames[ix]))
            except KeyError as e:
                raise KeyError(f"Could not find valid keys in JSON. Make sure {traj_id} and {threshold_key} exist. Error: {e}")
        else:
            print("No valid JSON provided. Loading standard map with skip_index...")
            num_nodes = len(topomap_filenames)
            for ix in range(0, num_nodes, skip_index):
                path = topomap_filenames[ix]
                self.topomap.append(Image.open(path))
                
        self.goal_node = len(self.topomap) - 1

             
    def callback_obs(self, msg):
        obs_img = Image.fromarray((msg[...,-1] * 255).astype(np.uint8))
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)


    def eval_action(self, obs):
        self.callback_obs(obs)
        if len(self.context_queue) > self.model_params["context_size"]:
            obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1) 
            obs_images = obs_images.to(device)
            mask = torch.zeros(1).long().to(device)  

            start = max(self.closest_node - self.args.radius, 0)
            end = min(self.closest_node + self.args.radius + 1, self.goal_node)
            goal_image = [transform_images(g_img, self.model_params["image_size"], center_crop=False).to(device) for g_img in self.topomap[start:end + 1]]
            goal_image = torch.concat(goal_image, dim=0)

            obsgoal_cond = self.model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
            dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists = to_numpy(dists.flatten())
            min_idx = np.argmin(dists)
            self.closest_node = min_idx + start

            sg_idx = min(min_idx + int(dists[min_idx] < self.args.close_threshold), len(obsgoal_cond) - 1)
            obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
            print(f"Closest Node Index: {self.closest_node}")
            
            # infer action
            self.nodes_used.append(self.closest_node)
            with torch.no_grad():
                # encoder vision features
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(self.args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)
                
                # initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (self.args.num_samples, self.model_params["len_traj_pred"], 2), device=device)
                naction = noisy_action

                # init scheduler
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                for k in self.noise_scheduler.timesteps[:]:
                    # predict noise
                    noise_pred = self.model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    # inverse diffusion step (remove noise)
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample


            naction = to_numpy(get_action(naction))
            naction = naction[0] 
            return pd_controller(naction[2])
          
        else:
            return [0.0, 0.0]

import urllib.request

IP = "http://10.42.0.1"
def quick_send(steering, throttle, server=f"{IP}:8000"):
    """Send single command without saving image"""
    try:
        data = f'{{"steering":{steering},"throttle":{throttle}}}'.encode()
        req = urllib.request.Request(f"{server}/control", data=data, headers={'Content-Type': 'application/json'})
        image = None
        with urllib.request.urlopen(req, timeout=3) as response:
            result = json.loads(response.read())
            image = np.array(result["observation"]["img"])
            
            # Convert data type to uint8 if needed
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] >= 3:
                import cv2
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            
            print(f"Sent: S={steering:+.2f} T={throttle:+.2f}")
            return image  
    except Exception as e:
        print(f"Error in quick_send: {e}")
        return None
     
if __name__ == '__main__':
    steering, throttle = 0.0, 0.0
    
    # Example usage: Pass the path to your previously generated JSON
    policy = NoMaD(
        map_dir="/home/kojogyaase/Projects/Research/recovery-from-failure/traj_5",
        json_path="sparsified_map.json",    # <--- Point to your generated JSON
        traj_id="traj_1",                   # <--- Adjust to match the JSON key
        threshold_key="threshold_3.0"       # <--- Adjust to match the JSON key
    )
    
    env = RacerEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    try:
        while not done:
            img = quick_send(steering, throttle)
            action = policy.eval_action(obs["image"][..., None])
            obs, _, done, truncated, _ = env.step(action)
            if truncated: 
                done = True
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Nodes used during run:", len(set(policy.nodes_used)))
        print("Total number of nodes in map:", len(policy.topomap))
        env.close()