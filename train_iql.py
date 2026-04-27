#!/usr/bin/env python
"""
Offline IQL Training with JAX, MobileNetV3 encoder, and Temporal Modeling
─────────────────────────────────────────────────────────────────────────
Pure offline training — no environment interaction after the initial
observation/action space sampling.  All gradient updates come from the
expert (teleop) replay buffer.

Evaluation: every `eval_interval` steps we sample a batch from the expert
buffer, run the IQL actor on the observations, and report per-dimension
and aggregate action MSE vs the expert actions.
"""

import random
import os
from datetime import datetime
import tqdm
from typing import Dict
import pickle

import flax
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from ml_collections import config_flags

# PyTorch imports for MobileNetV3
import torch

# JAX RL imports
from jaxrl2.agents import PixelIQLLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

from fake_racer_imu_env import RacerEnv, StackingWrapper, RewardWrapper
from wrappers import EnvCompatibility, Logger, MobileNetFeatureWrapper, save_checkpoint

flax.config.update("flax_use_orbax_checkpointing", True)

FLAGS = flags.FLAGS

# ── Flags ──────────────────────────────────────────────────────────────────────
flags.DEFINE_string("save_dir",            "./logs/",          "Tensorboard log dir.")
flags.DEFINE_integer("seed",               42,                 "Random seed.")
flags.DEFINE_integer("log_interval",       500,                "Logging interval (steps).")
flags.DEFINE_integer("eval_interval",      5000,               "Evaluation interval (steps).")
flags.DEFINE_integer("eval_batches",       20,                 "Batches of expert data used per eval.")
flags.DEFINE_integer("checkpoint_interval",5000,               "Checkpoint interval (steps).")
flags.DEFINE_integer("batch_size",         16,                 "Batch size.")
flags.DEFINE_integer("max_steps",          int(1e6),           "Total training steps.")
flags.DEFINE_integer("replay_buffer_size", int(1e4),           "Expert replay buffer capacity.")
flags.DEFINE_boolean("tqdm",               True,               "Show tqdm bar.")
flags.DEFINE_integer("mobilenet_blocks",   4,                  "MobileNetV3 blocks.")
flags.DEFINE_integer("mobilenet_input_size",84,                "MobileNetV3 input size.")
flags.DEFINE_integer("utd_ratio",          1,                  "Gradient updates per step.")
flags.DEFINE_string("expert_buffer_path",
    "/home/kojogyaase/Projects/Research/recovery-from-failure/teleop_buffer_err.pkl",
    "Path to the expert (teleop) replay buffer pickle.")

# IQL hyperparameters (override config file from CLI if desired)
flags.DEFINE_float("expectile",  0.9,   "IQL expectile.")
flags.DEFINE_float("A_scaling",  10.0,  "IQL advantage scaling.")
flags.DEFINE_float("iql_tau",    0.005, "Soft target update rate.")
flags.DEFINE_float("discount",   0.99,  "Discount factor.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/iql_default.py",
    "IQL hyperparameter config file.",
    lock_config=False,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_replay_buffer(replay_buffer: ReplayBuffer, path: str) -> ReplayBuffer:
    """Load a pickled replay buffer in-place and return it."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert buffer not found at: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        replay_buffer.dataset_dict  = data["data"]
        replay_buffer._insert_index = data["insert_index"]
        replay_buffer._size         = data["size"]
        print(f"[Buffer] Loaded {replay_buffer._size:,} expert transitions from {path}")
    except Exception as exc:
        print(f"[ERROR] Could not load expert buffer: {exc}")
    return replay_buffer


def evaluate_action_mse(
    agent:          PixelIQLLearner,
    buffer_iterator,
    num_batches:    int,
    action_dim:     int,
) -> Dict[str, float]:
    """
    Sample `num_batches` batches from the expert buffer, run the actor on
    each observation, and compute:
        - per-dimension MSE  (mse_dim_0, mse_dim_1, …)
        - mean MSE across all dimensions
        - per-dimension MAE

    Returns a flat dict suitable for TensorBoard / logger.
    """
    all_predicted = []
    all_expert    = []

    for _ in range(num_batches):
        batch = next(buffer_iterator)
        # agent.sample_actions operates on a single obs; use _actor directly
        # for batched inference to avoid a Python loop.
        predicted = agent._actor.apply_fn(
            {"params": agent._actor.params},
            batch["observations"],
            training=False,
        ).mode()                         # deterministic action from IQL actor
        all_predicted.append(np.array(predicted))
        all_expert.append(np.array(batch["actions"]))

    predicted_arr = np.concatenate(all_predicted, axis=0)   # (N, action_dim)
    expert_arr    = np.concatenate(all_expert,    axis=0)   # (N, action_dim)

    diff_sq = (predicted_arr - expert_arr) ** 2
    diff_ab = np.abs(predicted_arr - expert_arr)

    metrics: Dict[str, float] = {}
    for d in range(action_dim):
        metrics[f"eval/action_mse_dim{d}"] = float(diff_sq[:, d].mean())
        metrics[f"eval/action_mae_dim{d}"] = float(diff_ab[:, d].mean())

    metrics["eval/action_mse_mean"] = float(diff_sq.mean())
    metrics["eval/action_mae_mean"] = float(diff_ab.mean())

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main(_):
    print("\n" + "=" * 70)
    print("Offline IQL Training  —  MobileNetV3 features  —  expert buffer only")
    print("=" * 70 + "\n")

    # ── Seeds ─────────────────────────────────────────────────────────────────
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Build env (dummy — used only for space definitions) ───────────────────
    env = RacerEnv(render_mode="human")
    # env = EnvCompatibility(env)
    env = StackingWrapper(env, num_stack=4)
    env = RewardWrapper(env)
    env = MobileNetFeatureWrapper(
        env,
        device=device,
        num_blocks=FLAGS.mobilenet_blocks,
        input_size=FLAGS.mobilenet_input_size,
    )
    # We call reset once to get a concrete sample obs for agent init.
    sample_obs, _ = env.reset()
    sample_action  = env.action_space.sample()
    action_dim     = env.action_space.shape[0]

    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    print(f"Action dim        : {action_dim}\n")

    # ── IQL agent ─────────────────────────────────────────────────────────────
    kwargs = dict(FLAGS.config) if FLAGS.config else {}
    kwargs.setdefault("expectile", FLAGS.expectile)
    kwargs.setdefault("A_scaling", FLAGS.A_scaling)
    kwargs.setdefault("tau",       FLAGS.iql_tau)
    kwargs.setdefault("discount",  FLAGS.discount)

    agent = PixelIQLLearner(
        seed         = FLAGS.seed,
        observations = env.observation_space.sample(),
        actions      = sample_action,
        **kwargs,
    )
    print("PixelIQLLearner initialised.\n")

    # ── Expert replay buffer ──────────────────────────────────────────────────
    expert_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        FLAGS.replay_buffer_size,
    )
    expert_buffer = load_replay_buffer(expert_buffer, FLAGS.expert_buffer_path)
    expert_buffer.seed(FLAGS.seed)
    # Two independent iterators: one for training, one for evaluation
    train_iterator = expert_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size}
    )
    eval_iterator  = expert_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size}
    )
    

    # if expert_buffer._size == 0:
    #     raise RuntimeError(
    #         "Expert buffer is empty — cannot run offline training. "
    #         "Check --expert_buffer_path."
    #     )

    # ── Logging / checkpointing ───────────────────────────────────────────────
    logger = Logger(log_dir=FLAGS.save_dir)
    policy_folder = os.path.join(
        "checkpoints",
        f"offline_iql_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(policy_folder, exist_ok=True)

    # ── Training loop — no env interaction ────────────────────────────────────
    print("=" * 70)
    print(f"Starting offline training for {FLAGS.max_steps:,} steps")
    print(f"  Expert transitions : {expert_buffer._size:,}")
    print(f"  Batch size         : {FLAGS.batch_size}")
    print(f"  UTD ratio          : {FLAGS.utd_ratio}")
    print(f"  Eval every         : {FLAGS.eval_interval:,} steps  "
          f"({FLAGS.eval_batches} batches, {FLAGS.eval_batches * FLAGS.batch_size} samples)")
    print("=" * 70 + "\n")

    update_info: Dict[str, float] = {}

    for step in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        # ── Gradient update(s) on expert data ─────────────────────────────────
        for _ in range(FLAGS.utd_ratio):
            batch       = next(train_iterator)
            update_info = agent.update(batch)

        # ── Logging ───────────────────────────────────────────────────────────
        if step % FLAGS.log_interval == 0:
            logger.log_training(update_info, step)
            logger.print_status(step, FLAGS.max_steps)

        # ── Evaluation: action MSE vs expert ──────────────────────────────────
        if step % FLAGS.eval_interval == 0:
            mse_metrics = evaluate_action_mse(
                agent,
                eval_iterator,
                num_batches=FLAGS.eval_batches,
                action_dim=action_dim,
            )
            # Log to TensorBoard
            logger.log_training(mse_metrics, step)

            # Pretty-print summary
            print(f"\n── Eval @ step {step:,} ──────────────────────────")
            for d in range(action_dim):
                dim_name = ["steering", "throttle"][d] if d < 2 else f"dim{d}"
                print(
                    f"  {dim_name:10s}  "
                    f"MSE={mse_metrics[f'eval/action_mse_dim{d}']:.5f}  "
                    f"MAE={mse_metrics[f'eval/action_mae_dim{d}']:.5f}"
                )
            print(
                f"  {'MEAN':10s}  "
                f"MSE={mse_metrics['eval/action_mse_mean']:.5f}  "
                f"MAE={mse_metrics['eval/action_mae_mean']:.5f}"
            )
            print()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % FLAGS.checkpoint_interval == 0:
            save_checkpoint(
                agent,
                expert_buffer,           # save buffer ref for resume convenience
                os.path.join(policy_folder, f"step_{step}"),
                step,
            )

    # ── Final checkpoint ──────────────────────────────────────────────────────
    save_checkpoint(
        agent,
        expert_buffer,
        os.path.join(policy_folder, "final"),
        FLAGS.max_steps,
    )

    # ── Final eval ────────────────────────────────────────────────────────────
    final_mse = evaluate_action_mse(
        agent, eval_iterator,
        num_batches=FLAGS.eval_batches * 5,   # larger final evaluation
        action_dim=action_dim,
    )
    print("\n" + "=" * 70)
    print("Training complete — Final action MSE vs expert buffer")
    print("=" * 70)
    for d in range(action_dim):
        dim_name = ["steering", "throttle"][d] if d < 2 else f"dim{d}"
        print(
            f"  {dim_name:10s}  "
            f"MSE={final_mse[f'eval/action_mse_dim{d}']:.5f}  "
            f"MAE={final_mse[f'eval/action_mae_dim{d}']:.5f}"
        )
    print(
        f"  {'MEAN':10s}  "
        f"MSE={final_mse['eval/action_mse_mean']:.5f}  "
        f"MAE={final_mse['eval/action_mae_mean']:.5f}"
    )
    print()

    logger.close()
    env.close()


if __name__ == "__main__":
    app.run(main)