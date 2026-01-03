"""HARSH reward shaping for deadly corridor.

The problem is severe: Agent gets ~240 reward in 14 steps from rushing forward.
Death penalty of -100 isn't enough.

Solution: MASSIVE death penalty and heavily scale down ALL rewards.
"""
import os
import sys
from typing import Callable

import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.vizdoom_env import DoomEnvWrapper
from src.callbacks.checkpoint_callback import CheckpointCallback
from src.callbacks.tensorboard_callback import TensorBoardCallback


class HarshRewardWrapper(gym.Wrapper):
    """
    HARSH reward shaping to prevent suicide rushing.

    Strategy:
    - Scale down ALL rewards to 10% of original (reduce progress reward impact)
    - MASSIVE death penalty (-500 to -2000 based on when you die)
    - Make dying early absolutely unprofitable
    """

    def __init__(self, env):
        super().__init__(env)
        self.episode_length = 0

    def reset(self, **kwargs):
        self.episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_length += 1

        # HARSH CHANGE 1: Scale down all rewards to 10%
        # This reduces the value of rushing forward
        shaped_reward = reward * 0.1

        # HARSH CHANGE 2: MASSIVE death penalty
        if terminated:
            # Dying early = catastrophic
            # Dying at step 10: -500
            # Dying at step 50: -500
            # Dying at step 100: -1000
            # Dying at step 500: -2000
            death_penalty = -500 - (self.episode_length * 2)
            shaped_reward += death_penalty

        # Small survival bonus
        shaped_reward += 0.2

        return obs, shaped_reward, terminated, truncated, info


# Configuration
SCENARIO = "deadly_corridor"
TOTAL_TIMESTEPS = 5_000_000  # Less training time for test
N_ENVS = 16
LEARNING_RATE = 3e-4
BATCH_SIZE = 512
N_STEPS = 256
N_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.03  # Even higher exploration
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CHECKPOINT_FREQ = 500_000


def make_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        env = DoomEnvWrapper(
            scenario=SCENARIO,
            frame_skip=4,
            resolution=(84, 84),
            frame_stack=4,
            render_mode=None
        )

        env = HarshRewardWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)

        return env

    return _init


def main():
    print("=" * 60)
    print("HARSH REWARD SHAPING - DEADLY CORRIDOR")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    checkpoint_dir = os.path.join(base_dir, "checkpoints", f"{SCENARIO}_harsh")
    log_dir = os.path.join(base_dir, "logs", f"{SCENARIO}_harsh")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nConfiguration (HARSH REWARDS):")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS}")
    print(f"\n  HARSH reward shaping:")
    print(f"    - All rewards scaled to 10%")
    print(f"    - Death penalty: -500 to -2000 (based on survival)")
    print(f"    - Suicide rushing should now be unprofitable")
    print(f"\n  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    print(f"\nCreating {N_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ // N_ENVS,
            save_path=checkpoint_dir,
            name_prefix=f"ppo_{SCENARIO}_harsh",
            verbose=1
        ),
        TensorBoardCallback(verbose=1)
    ])

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

    print("\n" + "=" * 60)
    print("Training with HARSH penalties...")
    print("=" * 60)
    print("\nMean reward will be NEGATIVE at first - that's expected!")
    print("Agent must learn to survive longer to get positive rewards")
    print("Episode lengths should increase dramatically")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=False
        )

        final_model_path = os.path.join(checkpoint_dir, f"final_{SCENARIO}_harsh")
        model.save(final_model_path)
        print(f"\nTraining completed!")
        print(f"Final model saved to: {final_model_path}.zip")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_path = os.path.join(checkpoint_dir, f"interrupted_{SCENARIO}_harsh")
        model.save(interrupted_path)
        print(f"Model saved to: {interrupted_path}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()
