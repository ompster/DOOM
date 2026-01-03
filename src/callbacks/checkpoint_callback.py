"""Checkpoint callback for saving models during training."""
import json
import os
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints at regular intervals.

    Saves both the model weights and training metadata (timesteps, episodes, rewards).

    Args:
        save_freq: Save checkpoint every `save_freq` steps (per environment)
        save_path: Directory to save checkpoints
        name_prefix: Prefix for checkpoint filenames
        verbose: Verbosity level (0: no output, 1: info, 2: debug)
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called after every step in the environment.

        Returns:
            True to continue training, False to stop
        """
        # Check if it's time to save
        if self.n_calls % self.save_freq == 0:
            # Create checkpoint path
            checkpoint_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps"
            )

            # Save model
            self.model.save(checkpoint_path)

            if self.verbose > 0:
                print(f"Saving checkpoint to {checkpoint_path}.zip")

            # Save training statistics
            self._save_statistics(checkpoint_path)

        return True

    def _save_statistics(self, checkpoint_path: str) -> None:
        """
        Save training statistics to JSON file.

        Args:
            checkpoint_path: Base path for checkpoint (without extension)
        """
        stats = {
            'timesteps': self.num_timesteps,
            'n_calls': self.n_calls,
        }

        # Add episode statistics if available
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            ep_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]

            stats.update({
                'n_episodes': len(self.model.ep_info_buffer),
                'mean_reward': float(np.mean(ep_rewards)),
                'std_reward': float(np.std(ep_rewards)),
                'min_reward': float(np.min(ep_rewards)),
                'max_reward': float(np.max(ep_rewards)),
                'mean_ep_length': float(np.mean(ep_lengths)),
            })

        # Save to JSON
        stats_path = f"{checkpoint_path}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        if self.verbose > 1:
            print(f"Statistics saved to {stats_path}")
