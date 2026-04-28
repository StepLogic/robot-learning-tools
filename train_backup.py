#!/usr/bin/env python
"""
DrQ Training — Habitat Image-Goal Navigation
=============================================
Uses Habitat-Sim with continuous VelocityControl for navigation.
Goal images are encoded via a Siamese MobileNetV3 encoder (shared
with the current-observation encoder).

Wrapper stack:
  HabitatNavEnv → StackingWrapper → MobileNetFeatureWrapper
  → GoalImageWrapper → HabitatRewardWrapper → RecordEpisodeStatistics → TimeLimit
"""
import os

from habitat_wrappers import HabitatRewardWrapper, VideoRecorder

# ── Headless / HPC env vars (before any habitat imports) ──────────────────
os.environ.pop("DISPLAY", None)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
import faulthandler
import flax
import importlib.util
import torch

from jaxrl2.agents import DrQLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

from habitat_env import HabitatNavEnv
from configs.habitat_config import HabitatNavConfig
from racer_imu_env import StackingWrapper
from wrappers import (
    Logger,
    MobileNetFeatureWrapper,
    MobileNetV3Encoder,
    GoalImageWrapper,
    load_checkpoint,
    save_checkpoint,
)


faulthandler.enable()
flax.config.update("flax_use_orbax_checkpointing", True)
POLICY_FOLDER = "robot_policy"


def _str_to_bool(v) -> bool:
    """Convert a string to bool, for argparse flags that accept 'True'/'False'."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True/False, got '{v}'")


def _parse_args():
    """Parse command-line arguments and load the ml_collections config file."""
    parser = argparse.ArgumentParser(
        description="DrQ Habitat Image-Goal Navigation"
    )
    parser.add_argument("--scene_path", default="data/gibson/Cantwell.glb")
    parser.add_argument("--scene_dataset_path", default="")
    parser.add_argument("--randomize_scenes", type=_str_to_bool, default=False)
    parser.add_argument("--control_frequency", type=int, default=5)
    parser.add_argument("--frame_skip", type=int, default=6)
    parser.add_argument("--max_linear_velocity", type=float, default=0.5)
    parser.add_argument("--max_angular_velocity", type=float, default=1.5)
    parser.add_argument("--imu_noise_std", type=float, default=0.0)
    parser.add_argument("--gpu_device_id", type=int, default=0)
    parser.add_argument("--debug_render", type=_str_to_bool, default=False)

    # ── Training flags ───────────────────────────────────────────────────
    parser.add_argument("--save_dir", default="./logs/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=50000)
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--video_interval", type=int, default=50000)
    parser.add_argument("--video_length", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=int(1e6))
    parser.add_argument("--start_training", type=int, default=int(5e3))
    parser.add_argument("--replay_buffer_size", type=int, default=int(1e5))
    parser.add_argument("--tqdm", type=_str_to_bool, default=True,
                        help="Show tqdm progress bar.")
    parser.add_argument("--frame_stack", type=int, default=3)
    parser.add_argument("--mobilenet_blocks", type=int, default=13)
    parser.add_argument("--mobilenet_input_size", type=int, default=84)
    parser.add_argument("--goal_distance_scale", type=float, default=3.0)
    parser.add_argument("--goal_max_distance", type=float, default=10.0)
    parser.add_argument("--max_episode_steps", type=int, default=1000)

    # ── Expert / teleop flags ────────────────────────────────────────────
    parser.add_argument("--pretrained_checkpoint", default="")
    parser.add_argument("--expert_sample_ratio", type=float, default=0.0)
    parser.add_argument("--expert_buffer_path", default="")

    # ── HER flags ────────────────────────────────────────────────────────
    parser.add_argument("--her_ratio", type=float, default=0.0)
    parser.add_argument("--her_goal_threshold", type=float, default=None)

    # ── Config file (ml_collections .py) ─────────────────────────────────
    parser.add_argument("--config", default="./configs/drq_default.py",
                        help="Training hyperparameter config file (.py).")

    args = parser.parse_args()

    # Load ml_collections config from the .py file
    if args.config:
        spec = importlib.util.spec_from_file_location("train_config", args.config)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        args.config = mod.get_config()

    return args


def _find_latest_checkpoint(folder):
    """Find the latest checkpoint in a folder, return (path, step) or (None, 0)."""
    if not os.path.isdir(folder):
        return None, 0
    ckpts = []
    for d in os.listdir(folder):
        if d.startswith("checkpoint_"):
            try:
                step = int(d.split("_")[-1])
                ckpts.append((os.path.abspath(os.path.join(folder, d)), step))
            except ValueError:
                continue
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda x: x[1])
    return ckpts[-1]



def main(args):
    print("\n" + "=" * 70)
    print("DrQ Habitat Image-Goal Nav | Siamese MobileNetV3")
    print("=" * 70 + "\n")

    # np.random.seed(args.seed)
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Environment ───────────────────────────────────────────────────────────
    print("Building Habitat environment…")
    # habitat_cfg = HabitatNavConfig(
    #     scene_path=args.scene_path,
    #     scene_dataset_path=args.scene_dataset_path,
    #     control_frequency=args.control_frequency,
    #     frame_skip=args.frame_skip,
    #     max_linear_velocity=args.max_linear_velocity,
    #     max_angular_velocity=args.max_angular_velocity,
    #     imu_noise_std=args.imu_noise_std,
    #     gpu_device_id=args.gpu_device_id,
    #     seed=args.seed,
    #     debug_render=args.debug_render,
    #     headless=not args.debug_render,
    #     goal_distance_scale=args.goal_distance_scale,
    #     goal_max_distance=args.goal_max_distance,
    #     randomize_scenes=args.randomize_scenes,
    # )
    habitat_cfg = HabitatNavConfig(headless=True)
    if args.randomize_scenes:
        print(f"Scene randomization: {len(habitat_cfg.get_scene_paths())} scenes available")

    # EnvClass = HabitatNavEnv
    render_mode = "rgb_array"
    env = HabitatNavEnv(config=habitat_cfg, render_mode=render_mode)
    env = StackingWrapper(env, num_stack=args.frame_stack, image_format="rgb")

    # Shared MobileNetV3 encoder for current obs and goal
    shared_encoder = MobileNetV3Encoder(
        device=device,
        num_blocks=args.mobilenet_blocks,
        input_size=args.mobilenet_input_size,
    )
    env = MobileNetFeatureWrapper(env, encoder=shared_encoder)
    env = GoalImageWrapper(env, encoder=shared_encoder)
    goal_threshold = 2.0
    env = HabitatRewardWrapper(env, goal_threshold=goal_threshold)
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)

    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")

    # kwargs = dict(args.config) if args.config else {}
    # agent = DrQLearner(
    #     args.seed,
    #     env.observation_space.sample(),
    #     env.action_space.sample(),
    #     **kwargs,
    # )

    # # Resume checkpoint if available
    # os.makedirs(POLICY_FOLDER, exist_ok=True)
    # resume_path, resume_step = _find_latest_checkpoint(POLICY_FOLDER)
    # if resume_path is not None:
    #     agent = load_checkpoint(agent, resume_path)
    #     print(f"[Checkpoint] Resumed from {resume_path} (step {resume_step:,})")
    # elif args.pretrained_checkpoint:
    #     agent = load_checkpoint(agent, args.pretrained_checkpoint)
    #     print(f"[Checkpoint] Loaded pre-trained from {args.pretrained_checkpoint}")
    #     resume_step = 0
    # else:
    #     resume_step = 0
    #     print("[Checkpoint] Training from scratch")

    # start_step = resume_step + 1

    # # ── Replay buffer ─────────────────────────────────────────────────────────
    # replay_buffer = ReplayBuffer(
    #     env.observation_space,
    #     env.action_space,
    #     args.replay_buffer_size,
    # )

    # # Optional expert buffer
    # expert_buf = None

    # # ── Noise ─────────────────────────────────────────────────────────────────
    # ou_noise = OrnsteinUhlenbeckActionNoise(
    #     mean=np.zeros(env.action_space.shape[0]),
    #     theta=0.15,
    #     sigma=0.2,
    # )

    # # ── Logging ───────────────────────────────────────────────────────────────
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_dir = os.path.join(args.save_dir, f"habitat_nav_{timestamp}")
    # logger = Logger(log_dir)

    # # ── Video recording ──────────────────────────────────────────────────────
    # video_rec = None
    # if args.video_interval > 0:
    #     video_dir = os.path.join(log_dir, "videos")
    #     env = VideoRecorder(env, video_dir=video_dir)
    #     # env = video_rec  # wrap so env.step/reset captures frames

    # # ── Training loop ────────────────────────────────────────────────────────
    # obs, info = env.reset()
    # episode_reward = 0.0
    # episode_length = 0
    # episode_distance = 0.0
    # episode_start_pos = info.get("pos", np.zeros(3)).copy()
    # episode_prev_pos = episode_start_pos.copy()
    # video_recording = False
    # video_step_count = 0

    # # Episode buffer for HER
    # episode_transitions = []

    # # Running episode stats
    # episode_successes = deque(maxlen=100)
    # episode_distances = deque(maxlen=100)

    # pbar = tqdm.tqdm(range(start_step, args.max_steps + 1),
    #                  disable=not args.tqdm, desc="Training")

    # for step in pbar:
    #     # ── Action selection ──────────────────────────────────────────────────
    #     if step < args.start_training:
    #         action = env.action_space.sample()
    #     else:
    #         action = agent.sample_actions(obs)
    #         noise = ou_noise()
    #         action = np.clip(action + noise, env.action_space.low,
    #                          env.action_space.high)
            

    #     # ── Environment step ─────────────────────────────────────────────────
    #     next_obs, reward, terminated, truncated, next_info = env.step(action)

    #     # ── Store transition ─────────────────────────────────────────────────
    #     transition = dict(
    #         observations=obs,
    #         actions=action,
    #         rewards=reward,
    #         next_observations=next_obs,
    #         masks=np.float32(1.0 - terminated),
    #         dones=bool(terminated),
    #     )
    #     replay_buffer.insert(transition)
    #     episode_reward += reward
    #     episode_length += 1

     
    #     # ── Distance covered tracking ──────────────────────────────────────────
    #     curr_pos = next_info.get("pos", None)
    #     if curr_pos is not None:
    #         episode_distance += np.linalg.norm(curr_pos - episode_prev_pos)
    #         episode_prev_pos = curr_pos.copy()

    #     # ── Episode end ──────────────────────────────────────────────────────
    #     if terminated or truncated:
    #         success = next_info.get("distance_to_goal", float("inf")) < goal_threshold if terminated else False
    #         episode_transitions = []
    #         hab_success = next_info.get("habitat_success", 0.0)
    #         hab_spl = next_info.get("habitat_spl", 0.0)
    #         episode_successes.append(float(success))
    #         episode_distances.append(episode_distance)
    #         logger.log_episode({
    #             "return": episode_reward,
    #             "length": episode_length,
    #             "success": float(success),
    #             "distance": episode_distance,
    #             "habitat_success": float(hab_success),
    #             "habitat_spl": float(hab_spl),
    #         }, step)
    #         episode_reward = 0.0
    #         episode_length = 0
    #         episode_distance = 0.0
    #         obs, info = env.reset()
    #         episode_start_pos = info.get("pos", np.zeros(3)).copy()
    #         episode_prev_pos = episode_start_pos.copy()
    #     else:
    #         obs = next_obs
    #         info = next_info

    #     # ── Gradient update ──────────────────────────────────────────────────
    #     if step >= args.start_training and replay_buffer._size >= args.batch_size:
    #         batch = replay_buffer.sample(args.batch_size)

    #         # Mixed batch with expert data
    #         if expert_buf is not None and args.expert_sample_ratio > 0:
    #             n_expert = int(args.batch_size * args.expert_sample_ratio)
    #             n_online = args.batch_size - n_expert
    #             online_batch = replay_buffer.sample(n_online)
    #             expert_batch = expert_buf.sample(n_expert)
    #             # Merge: unfreeze FrozenDicts, concatenate, refreeze
    #             from flax.core import frozen_dict
    #             online_unfrozen = frozen_dict.unfreeze(online_batch)
    #             expert_unfrozen = frozen_dict.unfreeze(expert_batch)
    #             merged = {}
    #             for key in online_unfrozen:
    #                 if isinstance(online_unfrozen[key], dict):
    #                     merged[key] = {
    #                         k: np.concatenate(
    #                             [online_unfrozen[key][k], expert_unfrozen[key][k]]
    #                         )
    #                         for k in online_unfrozen[key]
    #                     }
    #                 else:
    #                     merged[key] = np.concatenate(
    #                         [online_unfrozen[key], expert_unfrozen[key]], axis=0
    #                     )
    #             batch = frozen_dict.freeze(merged)

    #         update_info = agent.update(batch)

    #         if step % args.log_interval == 0:
    #             logger.log_training(update_info, step)

    #     # ── Checkpoint ───────────────────────────────────────────────────────
    #     if step % args.checkpoint_interval == 0 and step > args.start_training:
    #         ckpt_dir = os.path.join(POLICY_FOLDER, f"checkpoint_{step}")
    #         save_checkpoint(agent, replay_buffer, ckpt_dir, step)
    #         print(f"[Checkpoint] Saved at step {step:,}")

    #     # # ── Video recording ──────────────────────────────────────────────────
    #     # if video_rec is not None:
    #     #     if not video_recording and step % args.video_interval == 0:
    #     #         video_rec.start_recording()
    #     #         video_recording = True
    #     #         video_step_count = 0
    #     #     if video_recording:
    #     #         video_step_count += 1
    #     #         if video_step_count >= args.video_length:
    #     #             video_rec.stop_and_save(f"step_{step:07d}.mp4")
    #     #             video_recording = False

    #     # ── Progress ──────────────────────────────────────────────────────────
    #     if step % args.log_interval == 0:
    #         pbar.set_postfix({
    #             "step": step,
    #             "buffer": replay_buffer._size,
    #             "ep_rew": f"{np.mean(logger.episode_returns):.1f}" if logger.episode_returns else "0.0",
    #             "sr": f"{np.mean(episode_successes):.0%}" if episode_successes else "0%",
    #             "dist": f"{np.mean(episode_distances):.2f}m" if episode_distances else "0m",
    #         })
    #         logger.print_status(step, args.max_steps, extra_stats={
    #             "Buffer size": replay_buffer._size,
    #             "Goal threshold": goal_threshold,
    #         })

    # # ── Final save ───────────────────────────────────────────────────────────
    # final_dir = os.path.join(POLICY_FOLDER, "final")
    # save_checkpoint(agent, replay_buffer, final_dir, args.max_steps)
    # print(f"\nTraining complete. Final checkpoint saved to {final_dir}")

    # env.close()


if __name__ == "__main__":
    import habitat_sim
    habitat_cfg = HabitatNavConfig(headless=True)
    # EnvClass = HabitatNavEnv
    render_mode = "rgb_array"
    env = HabitatNavEnv(config=habitat_cfg, render_mode=render_mode)
    args = _parse_args()
    main(args)