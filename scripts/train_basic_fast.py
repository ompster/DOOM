"""HIGH PERFORMANCE training script for VizDoom BASIC scenario.

This version maximizes GPU utilization for RTX 3090:
- 48 parallel environments (3x more data throughput)
- Batch size 1024 (4x larger GPU batches)
- Should achieve 70-90% GPU usage
- 2-3x faster training speed

Requirements:
- 12-16GB RAM (more parallel environments)
- 8+ CPU cores recommended
- NVIDIA GPU with CUDA support
"""
import os
import sys
from typing import Callable

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.vizdoom_env import DoomEnvWrapper
from src.callbacks.checkpoint_callback import CheckpointCallback
from src.callbacks.tensorboard_callback import TensorBoardCallback
from src.callbacks.eval_callback import EvalCallback


# Configuration - OPTIMIZED FOR RTX 3090
SCENARIO = "basic"
TOTAL_TIMESTEPS = 1_000_000  # 1M steps (faster convergence with better batching)
N_ENVS = 48  # 3x more parallel environments = more GPU data
LEARNING_RATE = 3e-4
BATCH_SIZE = 1024  # 4x larger batches = GPU works harder
N_STEPS = 128  # Steps per environment per update
N_EPOCHS = 4
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2  # Larger clip range for more policy updates
ENT_COEF = 0.02  # Higher entropy = MORE exploration
VF_COEF = 0.5  # Value function coefficient
MAX_GRAD_NORM = 0.5
CHECKPOINT_FREQ = 100_000  # Save every 100k steps
EVAL_FREQ = 50_000  # Evaluate every 50k steps
N_EVAL_EPISODES = 10


def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Create a single VizDoom environment.

    Args:
        rank: Environment ID (for seeding)
        seed: Base random seed

    Returns:
        Function that creates the environment
    """
    def _init():
        # Create environment
        env = DoomEnvWrapper(
            scenario=SCENARIO,
            frame_skip=4,
            resolution=(84, 84),
            frame_stack=4,
            render_mode=None  # No rendering during training
        )

        # Wrap with Monitor for episode statistics
        env = Monitor(env)

        # Set seed
        env.reset(seed=seed + rank)

        return env

    return _init


def main():
    """Main training loop."""
    print("=" * 60)
    print(f"HIGH PERFORMANCE Training - {SCENARIO.upper()} Scenario")
    print("=" * 60)

    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create directories
    checkpoint_dir = os.path.join(base_dir, "checkpoints", SCENARIO)
    log_dir = os.path.join(base_dir, "logs", SCENARIO)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nConfiguration (Optimized for RTX 3090):")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS} (HIGH - maxes out CPU)")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE} (LARGE - maxes out GPU)")
    print(f"  Expected GPU usage: 70-90%")
    print(f"  Expected speedup: 2-3x faster than standard config")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"\nCheckpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")

    # Create vectorized training environment
    print(f"\nCreating {N_ENVS} parallel environments...")
    print("(This may take 30-60 seconds with many parallel envs...)")
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    print("Environments created successfully!")

    # Create evaluation environment (single env, no parallelization)
    print("Creating evaluation environment...")
    eval_env = DoomEnvWrapper(scenario=SCENARIO)
    eval_env = Monitor(eval_env)

    # Setup callbacks
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ // N_ENVS,  # Adjust for parallel envs
            save_path=checkpoint_dir,
            name_prefix=f"ppo_{SCENARIO}",
            verbose=1
        ),
        EvalCallback(
            eval_env=eval_env,
            eval_freq=EVAL_FREQ // N_ENVS,  # Adjust for parallel envs
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=1,
            best_model_save_path=checkpoint_dir,
            log_path=log_dir
        ),
        TensorBoardCallback(verbose=1)
    ])

    print("\nCallbacks configured:")
    print(f"  Checkpoint every {CHECKPOINT_FREQ:,} steps")
    print(f"  Evaluate every {EVAL_FREQ:,} steps ({N_EVAL_EPISODES} episodes)")

    # Initialize PPO model
    print("\nInitializing PPO model...")

    # Policy kwargs - tell SB3 that images are already normalized
    policy_kwargs = dict(
        normalize_images=False  # We already normalize to [0, 1] in the wrapper
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
        seed=42
    )

    print("\nModel architecture:")
    print(f"  Policy: CnnPolicy (Convolutional Neural Network)")
    print(f"  Optimizer: Adam")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")

    # Start training
    print("\n" + "=" * 60)
    print("Starting HIGH PERFORMANCE training...")
    print("=" * 60)
    print(f"\nMonitor progress with TensorBoard:")
    print(f"  tensorboard --logdir={log_dir} --port=6007")
    print(f"\nWatch GPU usage with nvidia-smi:")
    print(f"  nvidia-smi -l 1")
    print(f"\nExpected training time: ~15-20 minutes (vs 60+ min standard)")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=False  # Set to True if you install tqdm: pip install tqdm rich
        )

        # Save final model
        final_model_path = os.path.join(checkpoint_dir, f"final_{SCENARIO}")
        model.save(final_model_path)
        print(f"\n{'=' * 60}")
        print("Training completed successfully!")
        print(f"Final model saved to: {final_model_path}.zip")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_path = os.path.join(checkpoint_dir, f"interrupted_{SCENARIO}")
        model.save(interrupted_path)
        print(f"Model saved to: {interrupted_path}.zip")

    finally:
        # Clean up
        env.close()
        eval_env.close()
        print("\nEnvironments closed.")


if __name__ == "__main__":
    # CRITICAL: This wrapper is required for Windows multiprocessing compatibility
    main()
