"""Record video of a trained VizDoom agent playing.

This script records the agent's gameplay and saves it as an MP4 video.
Requires: pip install opencv-python imageio imageio-ffmpeg
"""
import argparse
import os
import sys
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.vizdoom_env import DoomEnvWrapper


def record_video(
    model_path: str,
    scenario: str,
    output_path: str,
    n_episodes: int = 3,
    deterministic: bool = True,
    fps: int = 35
):
    """
    Record video of trained agent playing.

    Args:
        model_path: Path to trained model (.zip)
        scenario: VizDoom scenario name
        output_path: Path to save video (.mp4)
        n_episodes: Number of episodes to record
        deterministic: Use deterministic actions
        fps: Frames per second for output video
    """
    print("=" * 60)
    print("VizDoom Video Recording")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"Scenario: {scenario}")
    print(f"Output: {output_path}")
    print(f"Episodes: {n_episodes}")
    print(f"FPS: {fps}")

    # Check dependencies
    try:
        import imageio
    except ImportError:
        print("\nError: imageio not installed!")
        print("Install with: pip install imageio imageio-ffmpeg")
        return

    # Load model
    print("\nLoading model...")
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create environment with rgb_array rendering
    print(f"\nCreating {scenario} environment...")
    env = DoomEnvWrapper(
        scenario=scenario,
        render_mode="rgb_array"
    )
    print("Environment created!")

    # Record episodes
    print("\n" + "=" * 60)
    print("Recording gameplay...")
    print("=" * 60)

    frames = []
    episode_rewards = []

    for episode in range(n_episodes):
        print(f"\nRecording Episode {episode + 1}/{n_episodes}...")
        obs, info = env.reset()
        episode_reward = 0
        episode_frames = []
        done = False

        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward

            # Capture frame
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)

        episode_rewards.append(episode_reward)
        frames.extend(episode_frames)
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Frames: {len(episode_frames)}")

    print(f"\nTotal frames recorded: {len(frames)}")

    # Save video
    print(f"\nSaving video to: {output_path}")
    try:
        imageio.mimsave(output_path, frames, fps=fps)
        print("Video saved successfully!")

        # Print statistics
        print("\n" + "=" * 60)
        print("Recording Summary")
        print("=" * 60)
        print(f"\nEpisodes recorded: {n_episodes}")
        print(f"Total frames: {len(frames)}")
        print(f"Video duration: {len(frames) / fps:.1f} seconds")
        print(f"\nMean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Video saved to: {output_path}")
        print()

    except Exception as e:
        print(f"Error saving video: {e}")

    # Close environment
    env.close()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Record video of a trained VizDoom agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record Take Cover gameplay
  python record_video.py checkpoints/take_cover/ppo_take_cover_4000000_steps.zip

  # Record 5 episodes at 60 FPS
  python record_video.py checkpoints/take_cover/final_take_cover.zip -n 5 --fps 60

  # Custom output path and scenario
  python record_video.py checkpoints/basic/best_model.zip -s basic -o my_agent.mp4

Requirements:
  pip install imageio imageio-ffmpeg
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
        "-o", "--output",
        type=str,
        default=None,
        help="Output video path (default: auto-generate from model name)"
    )

    parser.add_argument(
        "-n", "--n-episodes",
        type=int,
        default=3,
        help="Number of episodes to record (default: 3)"
    )

    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=35,
        help="Frames per second for output video (default: 35)"
    )

    args = parser.parse_args()

    # Infer scenario from model path if not provided
    scenario = args.scenario
    if scenario is None:
        if "take_cover" in args.model_path.lower():
            scenario = "take_cover"
        elif "defend_the_center" in args.model_path.lower():
            scenario = "defend_the_center"
        elif "deadly_corridor" in args.model_path.lower():
            scenario = "deadly_corridor"
        elif "basic" in args.model_path.lower():
            scenario = "basic"
        else:
            print("Error: Could not infer scenario from path. Please specify with -s")
            return

    # Generate output path if not provided
    output_path = args.output
    if output_path is None:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        output_path = f"{model_name}_{scenario}_recording.mp4"

    # Run recording
    record_video(
        model_path=args.model_path,
        scenario=scenario,
        output_path=output_path,
        n_episodes=args.n_episodes,
        deterministic=not args.no_deterministic,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
