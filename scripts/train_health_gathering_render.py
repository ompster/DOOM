"""WATCH training for HEALTH_GATHERING (with all envs visible).

Health Gathering scenario:
- Navigate and collect health packs on a poison floor
- 3 actions (turn left, turn right, move forward)
- Living reward: +1 per step
- Death penalty: -100
- Must learn navigation + resource collection
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
SCENARIO = "health_gathering"
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 8  # Fewer for visibility
LEARNING_RATE = 3e-4
BATCH_SIZE = 512
N_STEPS = 256
N_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.02
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CHECKPOINT_FREQ = 500_000


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
    print("WATCH MODE - HEALTH GATHERING")
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
    print(f"\n  Scenario details:")
    print(f"    - Navigate and collect health packs")
    print(f"    - Poison floor damages you")
    print(f"    - Living reward: +1/step")
    print(f"    - Death penalty: -100")
    print(f"    - 3 actions (turn left/right, move forward)")
    print(f"\n  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    print(f"\nCreating {N_ENVS} parallel environments (ALL visible)...")
    env_fns = [make_env(i, render=True) for i in range(N_ENVS)]
    env = SubprocVecEnv(env_fns)
    print("Environments created!")
    print(f"Watch ALL {N_ENVS} windows to see agents learning to navigate!")

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

    print(f"\nAction space: {env.action_space} (3 actions)")

    print("\n" + "=" * 60)
    print("Starting training with rendering...")
    print("=" * 60)
    print(f"\nMonitor progress:")
    print(f"  tensorboard --logdir={log_dir} --port=6006")
    print(f"\nWatch the {N_ENVS} windows:")
    print("  - Agents will wander randomly at first")
    print("  - Should learn to move toward health packs")
    print("  - Watch for emergent navigation behavior!")
    print("\nExpected progression:")
    print("  Starting: Random wandering, quick death")
    print("  After 2M: Moving toward health packs")
    print("  After 5M: Efficient navigation patterns")
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
