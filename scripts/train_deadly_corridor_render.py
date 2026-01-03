"""WATCH training script for VizDoom DEADLY_CORRIDOR (WITH RENDERING).

Same as fast version but with ALL environments visible so you can watch the agents learn.

DEADLY CORRIDOR is MUCH harder than defend_the_center:
- 7 action buttons (movement + turning + shooting)
- Death penalty: -100 (vs -1)
- Must navigate corridors + combat
- Skill 5 (nightmare difficulty)

Will be SLOWER than fast version due to rendering overhead.
"""
import os
import sys
from typing import Callable

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.vizdoom_env import DoomEnvWrapper
from src.callbacks.checkpoint_callback import CheckpointCallback
from src.callbacks.tensorboard_callback import TensorBoardCallback


# Configuration - WITH RENDERING
SCENARIO = "deadly_corridor"
TOTAL_TIMESTEPS = 10_000_000  # 10M steps
N_ENVS = 8  # Fewer envs for better visibility
LEARNING_RATE = 3e-4
BATCH_SIZE = 512
N_STEPS = 256
N_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.02  # Higher exploration for harder scenario
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CHECKPOINT_FREQ = 500_000
FRAME_SKIP = 4
RESOLUTION = (84, 84)


def make_env(rank: int, seed: int = 0, render: bool = False) -> Callable:
    """Create a single deadly corridor environment."""
    def _init():
        env = DoomEnvWrapper(
            scenario=SCENARIO,
            frame_skip=FRAME_SKIP,
            resolution=RESOLUTION,
            frame_stack=4,
            render_mode="human" if render else None
        )

        env = Monitor(env)
        env.reset(seed=seed + rank)

        return env

    return _init


def main():
    """Main training loop with rendering."""
    print("=" * 60)
    print(f"WATCH MODE - {SCENARIO.upper().replace('_', ' ')}")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    checkpoint_dir = os.path.join(base_dir, "checkpoints", SCENARIO)
    log_dir = os.path.join(base_dir, "logs", SCENARIO)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Scenario: {SCENARIO} (MUCH HARDER than defend_the_center)")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS}")
    print(f"  Rendering: ALL {N_ENVS} ENVS VISIBLE (you can watch!)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Entropy coef: {ENT_COEF} (higher for exploration)")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"\nCheckpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")

    print(f"\nCreating {N_ENVS} parallel environments (ALL visible)...")

    # Create env list: ALL environments render
    env_fns = [make_env(i, render=True) for i in range(N_ENVS)]

    env = SubprocVecEnv(env_fns)
    print("Environments created successfully!")
    print(f"Watch ALL {N_ENVS} visible windows to see your agents learning!")

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ // N_ENVS,
            save_path=checkpoint_dir,
            name_prefix=f"ppo_{SCENARIO}",
            verbose=1
        ),
        TensorBoardCallback(verbose=1)
    ])

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
    print(f"  Policy: CnnPolicy")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space} (7 buttons!)")

    print("\n" + "=" * 60)
    print("Starting training with RENDERING...")
    print("=" * 60)
    print(f"\nMonitor progress with TensorBoard:")
    print(f"  tensorboard --logdir={log_dir} --port=6007")
    print("\nDeadly Corridor is HARD:")
    print("  - 7 action buttons (movement + turning)")
    print("  - Death penalty: -100")
    print("  - Must navigate + fight")
    print("  - Skill 5 (nightmare)")
    print(f"\nWatch ALL {N_ENVS} visible windows to see agent behavior!")
    print("Starting reward will likely be very negative (lots of dying)")
    print("Training will be slower due to rendering overhead")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=False
        )

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
        env.close()
        print("\nEnvironments closed.")


if __name__ == "__main__":
    main()
