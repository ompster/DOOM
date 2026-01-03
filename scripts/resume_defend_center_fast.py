"""RESUME training with FASTER settings from existing checkpoint.

Loads best_model.zip and continues training with optimized hyperparameters:
- Fewer epochs (4 vs 10) = 2.5x faster updates
- Larger rollout buffer (512 vs 256) = fewer updates
- No eval callback = no evaluation overhead
- Less frequent checkpoints

Note: We CANNOT change frame_skip or resolution when resuming because
the model's neural network is already trained on specific input dimensions.
We can only optimize the training hyperparameters.

Expected: ~2x faster than current config
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


# Configuration - RESUME WITH FASTER TRAINING
SCENARIO = "defend_the_center"
TOTAL_TIMESTEPS = 9_000_000  # Remaining 9M steps (already did 1M)
N_ENVS = 16
LEARNING_RATE = 3e-4  # Slightly increased for faster convergence
BATCH_SIZE = 512
N_STEPS = 256  # MUST KEEP SAME - buffer size is fixed when loading model
N_EPOCHS = 4  # REDUCED from 10 - much faster updates
GAMMA = 0.995
GAE_LAMBDA = 0.98
CLIP_RANGE = 0.2
ENT_COEF = 0.01  # Reduced exploration (model already learned basics)
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CHECKPOINT_FREQ = 500_000  # Less frequent checkpoints

# These MUST match the original model training (cannot change when resuming)
FRAME_SKIP = 4  # KEEP SAME - model expects this
RESOLUTION = (84, 84)  # KEEP SAME - model expects this
FRAME_STACK = 4  # KEEP SAME - model expects this


def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Create environment matching ORIGINAL settings.

    Must match the settings the model was trained on!
    """
    def _init():
        env = DoomEnvWrapper(
            scenario=SCENARIO,
            frame_skip=FRAME_SKIP,  # Same as original
            resolution=RESOLUTION,  # Same as original
            frame_stack=FRAME_STACK,  # Same as original
            render_mode=None
        )

        env = Monitor(env)
        env.reset(seed=seed + rank)

        return env

    return _init


def main():
    """Main training loop - RESUME from checkpoint."""
    print("=" * 60)
    print(f"RESUME TRAINING (FASTER) - {SCENARIO.upper().replace('_', ' ')}")
    print("=" * 60)

    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Checkpoint paths
    checkpoint_dir = os.path.join(base_dir, "checkpoints", SCENARIO)
    log_dir = os.path.join(base_dir, "logs", SCENARIO)
    resume_path = os.path.join(checkpoint_dir, "best_model.zip")

    # Check if checkpoint exists
    if not os.path.exists(resume_path):
        print(f"ERROR: Checkpoint not found at {resume_path}")
        print("Available checkpoints:")
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.zip'):
                    print(f"  - {f}")
        return

    print(f"\nResuming from: {resume_path}")
    print(f"\nConfiguration (OPTIMIZED FOR SPEED):")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Remaining timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  N_STEPS: {N_STEPS} (same - buffer size fixed)")
    print(f"  Epochs: {N_EPOCHS} (was 10 - MAJOR speedup)")
    print(f"  Learning rate: {LEARNING_RATE} (was 2.5e-4)")
    print(f"  Entropy coef: {ENT_COEF} (reduced)")
    print(f"  Expected speedup: ~2x (16k+ steps/min)")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Create vectorized training environment
    print(f"\nCreating {N_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    print("Environments created successfully!")

    # Setup callbacks - NO EVAL CALLBACK (too slow)
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ // N_ENVS,
            save_path=checkpoint_dir,
            name_prefix=f"ppo_{SCENARIO}_resumed",
            verbose=1
        ),
        TensorBoardCallback(verbose=1)
    ])

    print("\nCallbacks configured:")
    print(f"  Checkpoint every {CHECKPOINT_FREQ:,} steps")
    print(f"  NO evaluation callback (for maximum speed)")

    # Load existing model
    print(f"\nLoading model from {resume_path}...")
    model = PPO.load(
        resume_path,
        env=env,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Update hyperparameters for faster training
    print("\nUpdating hyperparameters for faster training...")
    model.learning_rate = LEARNING_RATE
    # Note: n_steps cannot be changed (buffer already allocated)
    model.batch_size = BATCH_SIZE
    model.n_epochs = N_EPOCHS
    model.ent_coef = ENT_COEF

    print(f"  New learning rate: {LEARNING_RATE}")
    print(f"  n_steps: {model.n_steps} (unchanged - buffer fixed)")
    print(f"  New n_epochs: {N_EPOCHS} (was 10 - 2.5x speedup!)")
    print(f"  New ent_coef: {ENT_COEF}")

    # Start training
    print("\n" + "=" * 60)
    print("Resuming training with FASTER settings...")
    print("=" * 60)
    print(f"\nMonitor progress with TensorBoard:")
    print(f"  tensorboard --logdir={log_dir} --port=6007")
    print("\nSpeed optimizations:")
    print(f"  - Epochs: 4 (was 10) = 2.5x faster updates!")
    print(f"  - No evaluation = no overhead")
    print(f"  - Less frequent checkpoints")
    print(f"\nExpected: ~2x faster = ~15-16k steps/min")
    print(f"9M steps should take ~10-11 hours (vs 18+ hours)")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=False  # Continue counting from checkpoint
        )

        # Save final model
        final_model_path = os.path.join(checkpoint_dir, f"final_{SCENARIO}_resumed")
        model.save(final_model_path)
        print(f"\n{'=' * 60}")
        print("Training completed successfully!")
        print(f"Final model saved to: {final_model_path}.zip")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_path = os.path.join(checkpoint_dir, f"interrupted_{SCENARIO}_resumed")
        model.save(interrupted_path)
        print(f"Model saved to: {interrupted_path}.zip")

    finally:
        # Clean up
        env.close()
        print("\nEnvironments closed.")


if __name__ == "__main__":
    main()
