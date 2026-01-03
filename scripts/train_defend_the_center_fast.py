"""HIGH PERFORMANCE training script for VizDoom DEFEND_THE_CENTER scenario.

This version optimizes GPU utilization for RTX 3090:
- 24 parallel environments (balanced for RAM/GPU)
- Batch size 512 (large GPU batches)
- ONE environment renders so you can watch training live
- Should achieve 50-70% GPU usage
- ~2x faster than standard config

Scenario: DEFEND_THE_CENTER
- Survive waves of enemies from all directions
- Limited ammo: 26 bullets (must aim carefully!)
- Rewards: +1 per kill, -1 per death
- Episode ends when player dies or time runs out
- Difficulty: Intermediate (requires ammo management strategy)
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
SCENARIO = "defend_the_center"
TOTAL_TIMESTEPS = 10_000_000  # 10M steps (more complex scenario)
N_ENVS = 24  # Balanced: good GPU usage without running out of RAM
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 512  # Large batches = GPU works hard
N_STEPS = 256  # More steps per update for longer episodes
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

    The FIRST environment (rank 0) will render so you can watch training!

    Args:
        rank: Environment ID (for seeding)
        seed: Base random seed

    Returns:
        Function that creates the environment
    """
    def _init():
        # First environment renders, others are headless
        render_mode = "human" if rank == 0 else None

        # Create environment
        env = DoomEnvWrapper(
            scenario=SCENARIO,
            frame_skip=4,
            resolution=(84, 84),
            frame_stack=4,
            render_mode=render_mode  # Only first env shows window
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
    print(f"HIGH PERFORMANCE Training - {SCENARIO.upper().replace('_', ' ')} Scenario")
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
    print(f"  Parallel environments: {N_ENVS} (balanced for RAM/GPU)")
    print(f"  Batch size: {BATCH_SIZE} (large batches)")
    print(f"  Rendering: Environment #0 will show game window")
    print(f"  Gamma: {GAMMA} (high discount for long-term planning)")
    print(f"  Expected GPU usage: 50-70%")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"\nCheckpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")

    # Create vectorized training environment
    print(f"\nCreating {N_ENVS} parallel environments...")
    print("(This may take 30-60 seconds with many parallel envs...)")
    print("Watch for the VizDoom window to appear!")
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    print("Environments created successfully!")
    print("VizDoom window shows Environment #0 training in real-time!")

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
    print(f"\nWatch GPU usage:")
    print(f"  nvidia-smi -l 1")
    print("\nThis scenario is more challenging:")
    print("  - Multiple waves of enemies")
    print("  - Limited ammo (26 bullets)")
    print("  - Requires strategic shooting")
    print("  - Expected training time: ~1-1.5 hours (vs 3+ hours standard)")
    print("\nWatch the VizDoom window to see the agent learning!")
    print("Early on: Random actions, wastes ammo")
    print("Later: Strategic aiming, ammo conservation")
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
