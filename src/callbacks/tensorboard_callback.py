"""Custom TensorBoard callback for VizDoom-specific metrics."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorBoardCallback(BaseCallback):
    """
    Custom callback for logging VizDoom-specific metrics to TensorBoard.

    Logs:
    - Episode rewards and lengths
    - Action distribution
    - Training progress metrics

    Args:
        verbose: Verbosity level (0: no output, 1: info, 2: debug)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        Called after every step in the environment.

        Returns:
            True to continue training, False to stop
        """
        # Log episode statistics if available
        if len(self.model.ep_info_buffer) > 0:
            # Calculate statistics
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            ep_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]

            # Log custom metrics
            self.logger.record('custom/mean_ep_reward', np.mean(ep_rewards))
            self.logger.record('custom/std_ep_reward', np.std(ep_rewards))
            self.logger.record('custom/min_ep_reward', np.min(ep_rewards))
            self.logger.record('custom/max_ep_reward', np.max(ep_rewards))
            self.logger.record('custom/mean_ep_length', np.mean(ep_lengths))

            # Count episodes
            current_episode_count = len(self.model.ep_info_buffer)
            if current_episode_count > self.episode_count:
                new_episodes = current_episode_count - self.episode_count
                self.episode_count = current_episode_count
                self.logger.record('custom/episodes', self.episode_count)

        # Log training progress percentage (if total timesteps is known)
        if hasattr(self.model, '_total_timesteps') and self.model._total_timesteps > 0:
            progress = (self.num_timesteps / self.model._total_timesteps) * 100
            self.logger.record('custom/training_progress_percent', progress)

        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of a rollout (after collecting experiences).

        Log action distribution and other rollout-specific metrics.
        """
        # Get rollout buffer if available
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            buffer = self.model.rollout_buffer

            # Log action distribution
            if hasattr(buffer, 'actions') and buffer.actions is not None:
                actions = buffer.actions.flatten()
                unique_actions, counts = np.unique(actions, return_counts=True)

                # Log each action's frequency
                for action, count in zip(unique_actions, counts):
                    action_freq = count / len(actions) * 100
                    self.logger.record(
                        f'actions/action_{int(action)}_percent',
                        action_freq
                    )

            # Log average rewards in buffer
            if hasattr(buffer, 'rewards') and buffer.rewards is not None:
                avg_reward = np.mean(buffer.rewards)
                self.logger.record('rollout/avg_reward', avg_reward)

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        if self.verbose > 0:
            print("TensorBoard callback initialized")
            print(f"Logs will be written to: {self.logger.dir}")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print("Training completed")
            print(f"Total episodes: {self.episode_count}")
            print(f"Total timesteps: {self.num_timesteps}")
