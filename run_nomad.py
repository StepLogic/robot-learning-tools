# ROS
import glob
import threading
from multiprocessing import Process
from typing import Tuple
import os
import time
import numpy as np
import torch

# import yaml

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from navigation_policies.navigation_policies.baseline_policies.misc import get_action
from PIL import Image
# from navigation_policies.bas.misc import Args,to_numpy, transform_images, load_model
from navigation_policies.navigation_policies.baseline_policies.misc import Args,to_numpy, transform_images, load_model
from navigation_policies.navigation_policies.baseline_policies.baselines_config import nomad_config
from PIL import Image
from navigation_policies.navigation_policies.baseline_policies.misc import load_model, to_numpy, transform_images,Args
from racer_imu_env import RacerEnv
EPS = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Args(model="nomad", waypoint=3, num_samples=8,radius=4,close_threshold=3)
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
    # print(v)
    v = np.clip(v/4, 0, 0.2)
    w = np.clip(-0.18+w, -0.2, 0.3)
    return  w,v


class NoMaD():
    def __init__(self,ckpt_path="/home/kojogyaase/Projects/Research/robot-server/navigation_policies/navigation_policies/pretrained_models/nomad.pth",mode='navigate',goal=-1,map_dir="topomap",skip_index=1,search_radius=4):
        self.mode=mode
        self.goal_node=goal
        self.closest_node = 0
        self.context_queue = []
        self.args=args
        self.args._replace(radius=search_radius)
        if ckpt_path is None:
            raise FileNotFoundError(f"Please specify checkpoint path")
        # Load model parameters
        self.context_size = nomad_config["context_size"]
        self.model_params = nomad_config
        self.model = load_model(ckpt_path, nomad_config, device)
        # breakpoint()
        self.model = self.model.to(device)
        self.model.eval()

        self.num_diffusion_iters = nomad_config["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=nomad_config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        self.topomap = []
        self.map_dir=None
        # if not self.mode =="explore":
        #     # breakpoint()
        #     topomap_filenames = sorted(glob.glob(f"{os.path.join(map_dir)}/*.jpg"),key=lambda x: int(x.split("/")[-1].split(".")[0]))
        #     num_nodes = len(topomap_filenames)-1
        #     for ix in range(0,num_nodes,skip_index):
        #         # if ix%skip_index!=0:
        #             path=topomap_filenames[ix]
        #             self.topomap.append(Image.open(path))
        #     self.goal_node=len(self.topomap)-1
        # else:
        #     os.makedirs(map_dir,exist_ok=True)
        #     # self.map_dir=map_dir
        #     # self.index=0
        #     # self.count=0
        #     # self.skip_index=skip_index
      
        print("===============.....Model Loaded .... Waiting for Observations.....==============================")

             
    def callback_obs(self, msg):
        # breakpoint()
        obs_img = Image.fromarray((msg[...,-1]).astype(np.uint8),mode='RGB')
        # self.store_map(obs_img)
        # obs_img.save("debug.jpg")
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)

    def eval_action(self,obs):
        self.callback_obs(obs)
        if len(self.context_queue) > self.model_params["context_size"]:
            # if self.mode =="explore":
                obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1)
                obs_images = obs_images.to(device)
                fake_goal = torch.randn((1, 3, *self.model_params["image_size"])).to(device)
                mask = torch.ones(1).long().to(device)  # ignore the goal

                with torch.no_grad():
                    obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=fake_goal,
                                            input_goal_mask=mask)

                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(self.args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)
                    noisy_action = torch.randn(
                        (self.args.num_samples, self.model_params["len_traj_pred"], 2), device=device)
                    naction = noisy_action

                    # print("her", np.concatenate((np.array([0]), naction.flatten())))
                    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                    start_time = time.time()
                    for k in self.noise_scheduler.timesteps[:]:
                        noise_pred = self.model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample
                    # print("time elapsed:", time.time() - start_time)

                naction = to_numpy(get_action(naction))
                naction = naction[0]
                return pd_controller(naction[2])
                # return naction[2]
          
        else:
                return [0.0,0.0]
        




# Quick functions for direct use
import json
from urllib.request import Request, urlopen
import cv2
import numpy as np

IP="http://10.42.0.1"
def quick_send(steering, throttle, server=f"{IP}:8000"):
    """Send single command without saving image"""
    try:
        data = f'{{"steering":{steering},"throttle":{throttle}}}'.encode()
        req = Request(f"{server}/control", data=data,
                     headers={'Content-Type': 'application/json'})
        image=None
        with urlopen(req, timeout=3) as response:
            result = json.loads(response.read())
            image = np.array(result["observation"]["img"])
            
            # Convert data type to uint8 if needed
            if image.dtype != np.uint8:
                # Assuming values are in 0-255 range but wrong data type
                image = image.astype(np.uint8)
            
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] >= 3:
                if image.shape[2] == 3:
                    # Assuming BGR, convert to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4:
                    # Assuming BGRA, convert to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            
            # Save as RGB image (convert back to BGR for saving)
            cv2.imwrite("response.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Sent: S={steering:+.2f} T={throttle:+.2f}")
            return image  # Returns RGB format
    except Exception as e:
        print(f"Error: {e}")
        return None
     
if __name__ == '__main__':
    steering,throttle=0.0,0.0
    policy=NoMaD(map_dir="/home/kojogyaase/Projects/Research/robot-server/topomap")
    env = RacerEnv(render_mode="human")
    obs,info=env.reset()
    done =False
    try:
        while not done:
            img=quick_send(steering,throttle)
            action=policy.eval_action(obs["image"][...,None])
            obs,_,done,done,_=env.step(action)    
    except:
            env.close()
          
        