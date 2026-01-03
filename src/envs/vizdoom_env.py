"""VizDoom Environment Wrapper with preprocessing and frame stacking."""
import os
from collections import deque
from typing import Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd


class DoomEnvWrapper(gym.Env):
    """
    VizDoom environment wrapper with preprocessing and frame stacking.

    Features:
    - Grayscale conversion
    - Resizing to 84x84 (standard Atari preprocessing)
    - Frame stacking (4 frames for temporal context)
    - Frame skipping for faster training
    - Observation normalization to [0, 1]
    - Compatible with SubprocVecEnv for parallel training

    Args:
        scenario: VizDoom scenario name (e.g., 'basic', 'defend_the_center')
        frame_skip: Number of frames to skip (default: 4)
        resolution: Target resolution for observations (default: (84, 84))
        frame_stack: Number of frames to stack (default: 4)
        render_mode: Rendering mode ('human' or None)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 35}

    SCENARIOS = {
        'basic': 'basic.cfg',
        'deadly_corridor': 'deadly_corridor.cfg',
        'defend_the_center': 'defend_the_center.cfg',
        'defend_the_line': 'defend_the_line.cfg',
        'health_gathering': 'health_gathering.cfg',
        'my_way_home': 'my_way_home.cfg',
        'predict_position': 'predict_position.cfg',
        'take_cover': 'take_cover.cfg',
        'deathmatch': 'deathmatch.cfg',
    }

    def __init__(
        self,
        scenario: str = 'basic',
        frame_skip: int = 4,
        resolution: Tuple[int, int] = (84, 84),
        frame_stack: int = 4,
        render_mode: Optional[str] = None,
    ):
        """Initialize the VizDoom environment wrapper."""
        super().__init__()

        self.scenario = scenario
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.frame_stack_size = frame_stack
        self.render_mode = render_mode

        # Get scenario config path
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario}. "
                f"Available scenarios: {list(self.SCENARIOS.keys())}"
            )

        config_file = os.path.join(
            vzd.scenarios_path,
            self.SCENARIOS[scenario]
        )

        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Scenario config not found: {config_file}. "
                f"Make sure VizDoom is installed correctly."
            )

        # Initialize VizDoom game
        self.game = vzd.DoomGame()
        self.game.load_config(config_file)

        # Set window visibility based on render mode
        self.game.set_window_visible(render_mode == "human")

        # Set screen format and resolution
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

        # Initialize the game
        self.game.init()

        # Get available actions
        self.available_actions = []
        n_buttons = self.game.get_available_buttons_size()

        # Create action space - all possible button combinations
        # For simplicity, we'll use single button presses
        self.available_actions = [[False] * n_buttons for _ in range(n_buttons)]
        for i in range(n_buttons):
            self.available_actions[i][i] = True

        # Add a "do nothing" action
        self.available_actions.append([False] * n_buttons)

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(len(self.available_actions))

        # Frame buffer for stacking
        self.frames = deque(maxlen=frame_stack)

        # Observation space - stacked grayscale frames
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(frame_stack, resolution[0], resolution[1]),
            dtype=np.float32
        )

        # Episode statistics
        self.episode_reward = 0.0
        self.episode_length = 0

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame.

        Steps:
        1. Convert to grayscale if needed
        2. Resize to target resolution
        3. Normalize to [0, 1]

        Args:
            frame: Raw frame from VizDoom (H, W) or (H, W, C)

        Returns:
            Preprocessed frame (H, W) normalized to [0, 1]
        """
        # Handle different frame formats
        if frame is None:
            # Return black frame if None
            return np.zeros(self.resolution, dtype=np.float32)

        # Convert to grayscale if RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif len(frame.shape) == 3:
            # If already single channel, squeeze
            frame = frame[:, :, 0]

        # Resize to target resolution
        frame = cv2.resize(
            frame,
            self.resolution,
            interpolation=cv2.INTER_AREA
        )

        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        return frame

    def _get_observation(self) -> np.ndarray:
        """
        Get stacked observation from frame buffer.

        Returns:
            Stacked frames with shape (frame_stack, H, W)
        """
        # Stack frames from deque
        stacked = np.array(list(self.frames), dtype=np.float32)
        return stacked

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        # Set seed if provided
        if seed is not None:
            self.game.set_seed(seed)

        # Reset the game
        self.game.new_episode()

        # Get initial frame
        state = self.game.get_state()
        if state is not None and state.screen_buffer is not None:
            frame = state.screen_buffer
        else:
            frame = np.zeros((480, 640), dtype=np.uint8)

        # Preprocess initial frame
        processed_frame = self._preprocess_frame(frame)

        # Fill frame buffer with initial frame
        for _ in range(self.frame_stack_size):
            self.frames.append(processed_frame)

        # Reset episode statistics
        self.episode_reward = 0.0
        self.episode_length = 0

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute action with frame skipping.

        Args:
            action: Action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action index to button list
        action_buttons = self.available_actions[action]

        # Execute action with frame skipping
        total_reward = 0.0

        for _ in range(self.frame_skip):
            reward = self.game.make_action(action_buttons)
            total_reward += reward

            # Stop if episode ended
            if self.game.is_episode_finished():
                break

        # Get current state
        terminated = self.game.is_episode_finished()

        if not terminated:
            state = self.game.get_state()
            if state is not None and state.screen_buffer is not None:
                frame = state.screen_buffer
            else:
                frame = np.zeros((480, 640), dtype=np.uint8)
        else:
            # Episode ended, use black frame
            frame = np.zeros((480, 640), dtype=np.uint8)

        # Preprocess and add latest frame to buffer
        processed_frame = self._preprocess_frame(frame)
        self.frames.append(processed_frame)

        # Update episode statistics
        self.episode_reward += total_reward
        self.episode_length += 1

        # Info dictionary
        info = {}

        # Add episode statistics to info when episode ends
        if terminated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length
            }

        # Gymnasium uses terminated and truncated
        truncated = False  # VizDoom doesn't have truncation

        return self._get_observation(), total_reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            state = self.game.get_state()
            if state is not None and state.screen_buffer is not None:
                return state.screen_buffer
            return np.zeros((480, 640, 3), dtype=np.uint8)
        # For "human" mode, VizDoom window is already visible
        return None

    def close(self):
        """Close the environment."""
        if hasattr(self, 'game'):
            self.game.close()


def make_doom_env(
    scenario: str = 'basic',
    frame_skip: int = 4,
    resolution: Tuple[int, int] = (84, 84),
    frame_stack: int = 4,
    render_mode: Optional[str] = None,
) -> DoomEnvWrapper:
    """
    Factory function to create a DoomEnvWrapper.

    Args:
        scenario: VizDoom scenario name
        frame_skip: Number of frames to skip
        resolution: Target resolution for observations
        frame_stack: Number of frames to stack
        render_mode: Rendering mode

    Returns:
        Wrapped VizDoom environment
    """
    return DoomEnvWrapper(
        scenario=scenario,
        frame_skip=frame_skip,
        resolution=resolution,
        frame_stack=frame_stack,
        render_mode=render_mode
    )
