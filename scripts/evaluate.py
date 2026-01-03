"""Evaluation script for testing trained VizDoom agents."""
import argparse
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.vizdoom_env import DoomEnvWrapper


def evaluate_model(
    model_path: str,
    scenario: str = "basic",
    n_episodes: int = 10,
    render: bool = True,
    deterministic: bool = True,
    sleep_time: float = 0.02
):
    """
    Evaluate a trained model on a VizDoom scenario.

    Args:
        model_path: Path to the trained model (.zip file)
        scenario: VizDoom scenario to evaluate on
        n_episodes: Number of episodes to run
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
        sleep_time: Sleep time between steps (for rendering)
    """
    print("=" * 60)
    print("VizDoom Model Evaluation")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"Scenario: {scenario}")
    print(f"Episodes: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print(f"Render: {render}")

    # Load model
    print("\nLoading model...")
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create environment
    print(f"\nCreating {scenario} environment...")
    env = DoomEnvWrapper(
        scenario=scenario,
        render_mode="human" if render else None
    )
    print("Environment created!")

    # Run evaluation
    print("\n" + "=" * 60)
    print("Starting evaluation...")
    print("=" * 60)

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\nEpisode {episode + 1}/{n_episodes}")

        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Render if enabled
            if render:
                env.render()
                time.sleep(sleep_time)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"\nEpisodes: {n_episodes}")
    print(f"\nRewards:")
    print(f"  Mean:   {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min:    {np.min(episode_rewards):.2f}")
    print(f"  Max:    {np.max(episode_rewards):.2f}")
    print(f"\nEpisode Lengths:")
    print(f"  Mean:   {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Min:    {np.min(episode_lengths)}")
    print(f"  Max:    {np.max(episode_lengths)}")
    print()

    # Close environment
    env.close()


def main():
    """Main evaluation function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VizDoom agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate best model for basic scenario
  python evaluate.py checkpoints/basic/final_basic.zip

  # Evaluate with 20 episodes, no rendering
  python evaluate.py checkpoints/basic/ppo_basic_1000000_steps.zip -n 20 --no-render

  # Evaluate with stochastic actions
  python evaluate.py checkpoints/basic/final_basic.zip --no-deterministic
        """
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model (.zip file)"
    )

    parser.add_argument(
        "-s", "--scenario",
        type=str,
        default=None,
        help="VizDoom scenario (default: infer from model path)"
    )

    parser.add_argument(
        "-n", "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate (default: 10)"
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering"
    )

    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Sleep time between steps in seconds (default: 0.02)"
    )

    args = parser.parse_args()

    # Infer scenario from model path if not provided
    scenario = args.scenario
    if scenario is None:
        # Try to extract scenario from path
        if "take_cover" in args.model_path.lower():
            scenario = "take_cover"
        elif "defend_the_center" in args.model_path.lower():
            scenario = "defend_the_center"
        elif "deadly_corridor" in args.model_path.lower():
            scenario = "deadly_corridor"
        elif "basic" in args.model_path.lower():
            scenario = "basic"
        else:
            scenario = "basic"
            print(f"Warning: Could not infer scenario, using default: {scenario}")

    # Run evaluation
    evaluate_model(
        model_path=args.model_path,
        scenario=scenario,
        n_episodes=args.n_episodes,
        render=not args.no_render,
        deterministic=not args.no_deterministic,
        sleep_time=args.sleep
    )


if __name__ == "__main__":
    main()
