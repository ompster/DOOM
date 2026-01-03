"""FIXED training for Deadly Corridor with reward shaping.

The problem: Agent learned to rush forward and die (reward hacking).
The solution: Heavily penalize death and encourage survival/combat.

Changes:
- Added custom reward wrapper to reshape rewards
- Massive death penalty scaling
- Reward for staying alive longer
- Reward for maintaining health
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


class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper to fix reward hacking in deadly corridor.

    Problem: Agent rushes forward and dies for quick positive rewards.
    Solution: Heavily penalize death, reward survival and health.
    """

    def __init__(self, env):
        super().__init__(env)
        self.episode_length = 0
        self.last_health = 100

    def reset(self, **kwargs):
        self.episode_length = 0
        self.last_health = 100
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_length += 1

        # Original reward (progress forward, kills, etc)
        shaped_reward = reward

        # MAJOR CHANGE: Multiply death penalty by episode length
        # The longer you survive, the worse dying becomes
        if terminated:
            death_penalty = -100 * max(1, self.episode_length / 50)
            shaped_reward += death_penalty
            print(f"Death at step {self.episode_length}, penalty: {death_penalty:.1f}")

        # Small reward for staying alive (each step)
        shaped_reward += 0.1

        # Reward for maintaining health (punish taking damage)
        # This discourages rushing into enemies
        if hasattr(self.env, 'game') and self.env.game.get_state():
            current_health = self.env.game.get_state().game_variables[0]
            health_delta = current_health - self.last_health
            shaped_reward += health_delta * 0.05  # Small penalty for damage
            self.last_health = current_health

        return obs, shaped_reward, terminated, truncated, info


# Configuration
SCENARIO = "deadly_corridor"
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16
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
FRAME_SKIP = 4
RESOLUTION = (84, 84)


def make_env(rank: int, seed: int = 0) -> Callable:
    """Create environment with reward shaping."""
    def _init():
        env = DoomEnvWrapper(
            scenario=SCENARIO,
            frame_skip=FRAME_SKIP,
            resolution=RESOLUTION,
            frame_stack=4,
            render_mode=None
        )

        # Apply reward shaping wrapper
        env = RewardShapingWrapper(env)

        env = Monitor(env)
        env.reset(seed=seed + rank)

        return env

    return _init


def main():
    """Main training loop."""
    print("=" * 60)
    print(f"FIXED TRAINING - {SCENARIO.upper().replace('_', ' ')}")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    checkpoint_dir = os.path.join(base_dir, "checkpoints", f"{SCENARIO}_fixed")
    log_dir = os.path.join(base_dir, "logs", f"{SCENARIO}_fixed")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nConfiguration (WITH REWARD SHAPING):")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS}")
    print(f"  Reward shaping: ENABLED")
    print(f"    - Death penalty scales with survival time")
    print(f"    - Small reward for staying alive (+0.1/step)")
    print(f"    - Penalty for taking damage")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"\nCheckpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")

    print(f"\nCreating {N_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    print("Environments created successfully!")

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ // N_ENVS,
            save_path=checkpoint_dir,
            name_prefix=f"ppo_{SCENARIO}_fixed",
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

    print("\n" + "=" * 60)
    print("Starting training with FIXED rewards...")
    print("=" * 60)
    print(f"\nMonitor progress with TensorBoard:")
    print(f"  tensorboard --logdir={log_dir} --port=6007")
    print("\nReward shaping should prevent reward hacking:")
    print("  - Agent can't profit from rushing forward and dying")
    print("  - Death becomes more costly the longer you survive")
    print("  - Agent must learn to survive AND make progress")
    print("\nEpisode lengths should be MUCH longer now!")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=False
        )

        final_model_path = os.path.join(checkpoint_dir, f"final_{SCENARIO}_fixed")
        model.save(final_model_path)
        print(f"\n{'=' * 60}")
        print("Training completed successfully!")
        print(f"Final model saved to: {final_model_path}.zip")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_path = os.path.join(checkpoint_dir, f"interrupted_{SCENARIO}_fixed")
        model.save(interrupted_path)
        print(f"Model saved to: {interrupted_path}.zip")

    finally:
        env.close()
        print("\nEnvironments closed.")


if __name__ == "__main__":
    main()
