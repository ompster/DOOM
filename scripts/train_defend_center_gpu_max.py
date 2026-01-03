"""HIGH GPU UTILIZATION training script for VizDoom DEFEND_THE_CENTER.

This version is optimized for high performance without exhausting system memory:
- 16 parallel environments (balanced throughput)
- Batch size 512 (large GPU batches)
- NO rendering (pure performance)
- Target: 60-80% GPU usage
- ~2-3x faster than standard config

Use this when you want fast training speed without paging file errors.
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


# Configuration - MAXIMUM GPU UTILIZATION
SCENARIO = "defend_the_center"
TOTAL_TIMESTEPS = 10_000_000  # 10M steps
N_ENVS = 16  # Reduced from 48 to prevent paging file errors
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 512  # Reduced from 1024 to fit in available memory
N_STEPS = 256  # More steps per update
N_EPOCHS = 10  # More epochs for complex scenarios
GAMMA = 0.995  # Higher discount for long-term planning
GAE_LAMBDA = 0.98
CLIP_RANGE = 0.2
ENT_COEF = 0.02  # Keep exploration high
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CHECKPOINT_FREQ = 200_000  # Save every 200k steps
EVAL_FREQ = 100_000  # Evaluate every 100k steps
N_EVAL_EPISODES = 10


def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Create a single VizDoom environment.

    NO rendering - pure performance!

    Args:
        rank: Environment ID (for seeding)
        seed: Base random seed

    Returns:
        Function that creates the environment
    """
    def _init():
        # NO rendering - all headless for maximum speed
        render_mode = None

        # Create environment
        env = DoomEnvWrapper(
            scenario=SCENARIO,
            frame_skip=4,
            resolution=(84, 84),
            frame_stack=4,
            render_mode=render_mode  # No rendering
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
    print(f"MAXIMUM GPU UTILIZATION - {SCENARIO.upper().replace('_', ' ')}")
    print("=" * 60)

    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create directories
    checkpoint_dir = os.path.join(base_dir, "checkpoints", SCENARIO)
    log_dir = os.path.join(base_dir, "logs", SCENARIO)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nConfiguration (HIGH PERFORMANCE):")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Rendering: NONE (pure performance)")
    print(f"  Gamma: {GAMMA}")
    print(f"  Expected GPU usage: 60-80%")
    print(f"  Expected speedup: 2-3x vs standard config")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"\nCheckpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")

    # Create vectorized training environment
    print(f"\nCreating {N_ENVS} parallel environments...")
    print("(This may take 30-60 seconds...)")
    print("NO rendering - pure performance mode!")
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
    print("Starting MAXIMUM PERFORMANCE training...")
    print("=" * 60)
    print(f"\nMonitor progress with TensorBoard:")
    print(f"  tensorboard --logdir={log_dir} --port=6007")
    print(f"\nWatch GPU usage (should be 60-80%):")
    print(f"  nvidia-smi -l 1")
    print("\nPerformance mode:")
    print(f"  - {N_ENVS} parallel environments")
    print(f"  - Batch size {BATCH_SIZE}")
    print("  - No rendering overhead")
    print("  - Expected training time: ~45-60 minutes (vs 3+ hours standard)")
    print("\nThis is a high-performance configuration optimized for system stability!")
    print("Your GPU fans will spin up - this is normal!")
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
