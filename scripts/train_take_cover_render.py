"""WATCH training for TAKE_COVER (with all envs visible).

Take Cover is the EASIEST VizDoom scenario:
- Only 2 actions (move left, move right)
- Simple objective: dodge fireballs
- Living reward: +1 per step
- Clear learning signal

This should learn FAST with good results!
"""
import os
import sys
from typing import Callable

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.vizdoom_env import DoomEnvWrapper
from src.callbacks.checkpoint_callback import CheckpointCallback
from src.callbacks.tensorboard_callback import TensorBoardCallback


# Configuration
SCENARIO = "take_cover"
TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 8  # Fewer for visibility
LEARNING_RATE = 3e-4
BATCH_SIZE = 512
N_STEPS = 256
N_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CHECKPOINT_FREQ = 250_000


def make_env(rank: int, seed: int = 0, render: bool = False) -> Callable:
    def _init():
        env = DoomEnvWrapper(
            scenario=SCENARIO,
            frame_skip=4,
            resolution=(84, 84),
            frame_stack=4,
            render_mode="human" if render else None
        )

        env = Monitor(env)
        env.reset(seed=seed + rank)

        return env

    return _init


def main():
    print("=" * 60)
    print("WATCH MODE - TAKE COVER (EASIEST SCENARIO)")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    checkpoint_dir = os.path.join(base_dir, "checkpoints", SCENARIO)
    log_dir = os.path.join(base_dir, "logs", SCENARIO)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS} (ALL VISIBLE)")
    print(f"\n  Why this is EASY:")
    print(f"    - Only 2 actions (move left/right)")
    print(f"    - Simple objective (dodge fireballs)")
    print(f"    - Living reward: +1/step")
    print(f"    - Should learn in 1-2M steps!")
    print(f"\n  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    print(f"\nCreating {N_ENVS} parallel environments (ALL visible)...")
    env_fns = [make_env(i, render=True) for i in range(N_ENVS)]
    env = SubprocVecEnv(env_fns)
    print("Environments created!")
    print(f"Watch ALL {N_ENVS} windows to see agents learning to dodge!")

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
        policy_kwargs=dict(normalize_images=False),
        tensorboard_log=log_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
        seed=42
    )

    print(f"\nAction space: {env.action_space} (only 2 actions!)")

    print("\n" + "=" * 60)
    print("Starting training with rendering...")
    print("=" * 60)
    print(f"\nMonitor progress:")
    print(f"  tensorboard --logdir={log_dir} --port=6007")
    print(f"\nWatch the {N_ENVS} windows to see learning!")
    print("Expected progression:")
    print("  Starting: Random left/right movement, gets hit a lot")
    print("  After 500k: Starting to dodge some fireballs")
    print("  After 2M: Smooth dodging, high survival rate")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=False
        )

        final_model_path = os.path.join(checkpoint_dir, f"final_{SCENARIO}")
        model.save(final_model_path)
        print(f"\nTraining completed!")
        print(f"Final model: {final_model_path}.zip")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        interrupted_path = os.path.join(checkpoint_dir, f"interrupted_{SCENARIO}")
        model.save(interrupted_path)
        print(f"Saved: {interrupted_path}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()
