"""Training script for VizDoom DEFEND_THE_CENTER with 8 rendering environments.

This version shows ALL 8 environments rendering:
- 8 parallel environments (ALL rendering)
- Batch size 256 (stable)
- You can watch all 8 agents training simultaneously
- Good for observing different strategies
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


# Configuration - 8 ENVIRONMENTS ALL RENDERING
SCENARIO = "defend_the_center"
TOTAL_TIMESTEPS = 10_000_000  # 10M steps
N_ENVS = 8  # 8 windows = watch 8 agents train!
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 256  # Standard batch size
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

    ALL environments will render - you'll see 8 windows!

    Args:
        rank: Environment ID (for seeding)
        seed: Base random seed

    Returns:
        Function that creates the environment
    """
    def _init():
        # ALL environments render
        render_mode = "human"

        print(f"[Env {rank}] Initializing WITH RENDERING...")

        # Create environment
        env = DoomEnvWrapper(
            scenario=SCENARIO,
            frame_skip=4,
            resolution=(84, 84),
            frame_stack=4,
            render_mode=render_mode  # All envs show windows
        )

        # Wrap with Monitor for episode statistics
        env = Monitor(env)

        # Set seed
        env.reset(seed=seed + rank)

        print(f"[Env {rank}] Ready!")

        return env

    return _init


def main():
    """Main training loop."""
    print("=" * 60)
    print(f"Training - {SCENARIO.upper().replace('_', ' ')} Scenario")
    print("8 ENVIRONMENTS - ALL RENDERING")
    print("=" * 60)

    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create directories
    checkpoint_dir = os.path.join(base_dir, "checkpoints", SCENARIO)
    log_dir = os.path.join(base_dir, "logs", SCENARIO)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Rendering: ALL {N_ENVS} environments will show windows!")
    print(f"  Gamma: {GAMMA} (high discount for long-term planning)")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"\nCheckpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")

    # Create vectorized training environment
    print(f"\nCreating {N_ENVS} parallel environments...")
    print("Watch for initialization messages from each environment...")
    print("You'll see 8 VizDoom windows appear!")
    print("(Should take ~30-60 seconds)")
    print()

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    print()
    print("=" * 60)
    print("ALL ENVIRONMENTS CREATED SUCCESSFULLY!")
    print(f"You should now see {N_ENVS} VizDoom windows!")
    print("Watch different agents learn different strategies!")
    print("=" * 60)

    # Create evaluation environment (single env, no parallelization)
    print("\nCreating evaluation environment...")
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
    print("Starting training...")
    print("=" * 60)
    print(f"\nMonitor progress with TensorBoard:")
    print(f"  tensorboard --logdir={log_dir} --port=6007")
    print("\nThis scenario is challenging:")
    print("  - Multiple waves of enemies")
    print("  - Limited ammo (26 bullets)")
    print("  - Requires strategic shooting")
    print("\nWatch ALL 8 windows to see different agents learning!")
    print("Early on: Random actions, wastes ammo")
    print("Later: Strategic aiming, ammo conservation")
    print("You'll see different strategies emerge!")
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
