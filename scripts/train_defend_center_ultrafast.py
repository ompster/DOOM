"""ULTRA-FAST training script for VizDoom DEFEND_THE_CENTER.

Optimizations for maximum speed:
- Higher frame skip (8 vs 4) = 2x less preprocessing
- Lower resolution (64x64 vs 84x84) = ~1.75x faster preprocessing
- Fewer epochs (4 vs 10) = 2.5x faster updates
- Larger rollout buffer (512 vs 256) = fewer updates
- No eval callback = no evaluation overhead
- Less frequent checkpoints

Expected: 3-4x faster than current config (~20k steps/min vs 8k)
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


# Configuration - ULTRA-FAST SPEED
SCENARIO = "defend_the_center"
TOTAL_TIMESTEPS = 10_000_000  # 10M steps
N_ENVS = 16  # Keep at 16 to avoid memory issues
LEARNING_RATE = 3e-4
BATCH_SIZE = 512
N_STEPS = 512  # DOUBLED - collect more data before update
N_EPOCHS = 4  # REDUCED from 10 - faster updates
GAMMA = 0.99  # Slightly lower for faster convergence
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CHECKPOINT_FREQ = 500_000  # Less frequent checkpoints
FRAME_SKIP = 8  # DOUBLED from 4 - major speedup
RESOLUTION = (64, 64)  # REDUCED from 84x84 - faster preprocessing


def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Create a single VizDoom environment with SPEED optimizations.

    Args:
        rank: Environment ID (for seeding)
        seed: Base random seed

    Returns:
        Function that creates the environment
    """
    def _init():
        # Create environment with speed settings
        env = DoomEnvWrapper(
            scenario=SCENARIO,
            frame_skip=FRAME_SKIP,  # DOUBLED for speed
            resolution=RESOLUTION,  # REDUCED for speed
            frame_stack=4,
            render_mode=None  # No rendering
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
    print(f"ULTRA-FAST TRAINING - {SCENARIO.upper().replace('_', ' ')}")
    print("=" * 60)

    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create directories
    checkpoint_dir = os.path.join(base_dir, "checkpoints", SCENARIO)
    log_dir = os.path.join(base_dir, "logs", SCENARIO)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nConfiguration (ULTRA-FAST MODE):")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Frame skip: {FRAME_SKIP} (2x faster)")
    print(f"  Resolution: {RESOLUTION} (1.75x faster)")
    print(f"  Epochs: {N_EPOCHS} (2.5x faster)")
    print(f"  N_STEPS: {N_STEPS} (fewer updates)")
    print(f"  Expected speedup: 3-4x (20k+ steps/min)")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"\nCheckpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")

    # Create vectorized training environment
    print(f"\nCreating {N_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    print("Environments created successfully!")

    # Setup callbacks - NO EVAL CALLBACK (too slow)
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ // N_ENVS,
            save_path=checkpoint_dir,
            name_prefix=f"ppo_{SCENARIO}",
            verbose=1
        ),
        TensorBoardCallback(verbose=1)
    ])

    print("\nCallbacks configured:")
    print(f"  Checkpoint every {CHECKPOINT_FREQ:,} steps")
    print(f"  NO evaluation callback (for maximum speed)")

    # Initialize PPO model
    print("\nInitializing PPO model...")

    policy_kwargs = dict(
        normalize_images=False
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
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")

    # Start training
    print("\n" + "=" * 60)
    print("Starting ULTRA-FAST training...")
    print("=" * 60)
    print(f"\nMonitor progress with TensorBoard:")
    print(f"  tensorboard --logdir={log_dir} --port=6007")
    print("\nSpeed optimizations:")
    print(f"  - Frame skip: 8 (skip more frames)")
    print(f"  - Resolution: 64x64 (faster preprocessing)")
    print(f"  - Epochs: 4 (faster updates)")
    print(f"  - No evaluation (no overhead)")
    print(f"\nExpected: 3-4x faster = ~20k steps/min")
    print(f"10M steps should take ~8-10 hours (vs 20+ hours)")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=False
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
        print("\nEnvironments closed.")


if __name__ == "__main__":
    main()
