"""Evaluation callback for periodic model evaluation."""
from stable_baselines3.common.callbacks import EvalCallback as SB3EvalCallback


class EvalCallback(SB3EvalCallback):
    """
    Wrapper around Stable Baselines 3's EvalCallback for consistency.

    Evaluates the agent's performance periodically during training on a
    separate evaluation environment.

    Args:
        eval_env: Environment used for evaluation
        eval_freq: Evaluate the agent every `eval_freq` steps (per environment)
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic or stochastic actions
        render: Whether to render the evaluation episodes
        verbose: Verbosity level (0: no output, 1: info, 2: debug)
        best_model_save_path: Path to save the best model (optional)
        log_path: Path to save evaluation logs (optional)
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        best_model_save_path: str = None,
        log_path: str = None,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
        )

    def _on_step(self) -> bool:
        """
        Called after every step. Performs evaluation if needed.

        Returns:
            True to continue training, False to stop
        """
        continue_training = super()._on_step()

        # Log best mean reward to TensorBoard
        if self.best_mean_reward > -float('inf'):
            self.logger.record('eval/best_mean_reward', self.best_mean_reward)

        return continue_training
