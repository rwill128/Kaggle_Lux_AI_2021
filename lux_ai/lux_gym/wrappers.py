import copy
import logging
import pickle

import gym
import numpy as np
import torch
from typing import Dict, List, NoReturn, Optional, Tuple, Union

from .act_spaces import ACTION_MEANINGS
from .lux_env import LuxEnv
from .reward_spaces import BaseRewardSpace
from ..utility_constants import MAX_BOARD_SIZE


class PadFixedShapeEnv(gym.Wrapper):
    """
    Environment wrapper that pads observations to a fixed size.
    
    This wrapper ensures consistent tensor dimensions by:
    1. Padding spatial observations to max_board_size
    2. Maintaining an input mask for valid regions
    3. Handling variable-sized game boards
    
    Key Features:
    - Enables batch processing with fixed dimensions
    - Preserves original game logic within valid regions
    - Handles both 4D and 5D tensors appropriately
    - Maintains compatibility with neural network inputs
    
    Args:
        env: The environment to wrap
        max_board_size: Maximum board dimensions to pad to
    """
    def __init__(self, env: gym.Env, max_board_size: Tuple[int, int] = MAX_BOARD_SIZE):
        super(PadFixedShapeEnv, self).__init__(env)
        self.max_board_size = max_board_size
        self.observation_space = self.unwrapped.obs_space.get_obs_spec(max_board_size)
        self.input_mask = np.zeros((1,) + max_board_size, dtype=bool)
        self.input_mask[:, :self.orig_board_dims[0], :self.orig_board_dims[1]] = True

    def _pad(self, x: Union[Dict, np.ndarray]) -> Union[dict, np.ndarray]:
        """
        Pad observations to fixed dimensions.
        
        Handles:
        1. Dictionary observations recursively
        2. 4D tensors (batch, channels, height, width)
        3. 5D tensors (batch, time, channels, height, width)
        
        Args:
            x: Observation to pad (dict or numpy array)
            
        Returns:
            Padded observation with dimensions matching max_board_size
        """
        if isinstance(x, dict):
            return {key: self._pad(val) for key, val in x.items()}
        elif x.ndim == 4 and x.shape[2:4] == self.orig_board_dims:
            return np.pad(x, pad_width=self.base_pad_width, constant_values=0.)
        elif x.ndim == 5 and x.shape[2:4] == self.orig_board_dims:
            return np.pad(x, pad_width=self.base_pad_width + ((0, 0),), constant_values=0.)
        else:
            return x

    def observation(self, observation: Dict[str, Union[Dict, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            key: self._pad(val) for key, val in observation.items()
        }

    def info(self, info: Dict[str, Union[Dict, np.ndarray]]) -> Dict[str, np.ndarray]:
        info = {
            key: self._pad(val) for key, val in info.items()
        }
        assert "input_mask" not in info.keys()
        info["input_mask"] = self.input_mask
        return info

    def reset(self, **kwargs):
        obs, reward, done, info = super(PadFixedShapeEnv, self).reset(**kwargs)
        self.input_mask[:] = 0
        self.input_mask[:, :self.orig_board_dims[0], :self.orig_board_dims[1]] = 1
        return self.observation(obs), reward, done, self.info(info)

    def step(self, action: Dict[str, np.ndarray]):

        def save_obs(obs, filename="obs.pkl"):
            """
            Save the entire observation dict (with all array values printed) to a pickle file.
            Also turn off print truncation for debugging prints.
            """
            # Turn off the default truncation in printing:
            np.set_printoptions(threshold=np.inf)

            # (Optional) If you just want to print the entire array in the console right now:
            # print(obs)

            # Now pickle and store the full data
            with open(filename, "wb") as f:
                pickle.dump(obs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved obs to {filename}")

        # save_obs(action, "early_actions_in_wrapper.pkl")

        assert action["worker"].shape[2] == action["worker"].shape[3]
        assert action["cart"].shape[2] == action["cart"].shape[3]
        assert action["city_tile"].shape[2] == action["city_tile"].shape[3]
        # logging.warning("shape of early actions.cart: " + str(action["worker"].shape))

        action = {
            key: val[:, :, :self.orig_board_dims[0], :self.orig_board_dims[1], :]
            for key, val in action.items()
        }

        # logging.warning("self.orig_board_dims[0]: " + str(self.orig_board_dims[0]))
        # logging.warning("self.orig_board_dims[1]: " + str(self.orig_board_dims[1]))
        # logging.warning("shape of actions.cart: " + str(action["worker"].shape))

        assert action["worker"].shape[2] == action["worker"].shape[3]
        assert action["cart"].shape[2] == action["cart"].shape[3]
        assert action["city_tile"].shape[2] == action["city_tile"].shape[3]

        obs, reward, done, info = super(PadFixedShapeEnv, self).step(action)
        return self.observation(obs), reward, done, self.info(info)

    @property
    def orig_board_dims(self) -> Tuple[int, int]:
        return self.unwrapped.board_dims

    @property
    def base_pad_width(self):
        return (
            (0, 0),
            (0, 0),
            (0, self.max_board_size[0] - self.orig_board_dims[0]),
            (0, self.max_board_size[1] - self.orig_board_dims[1])
        )


class RewardSpaceWrapper(gym.Wrapper):
    """
    Wrapper for custom reward computation using reward spaces.
    
    This wrapper:
    1. Delegates reward computation to a reward space object
    2. Ensures proper game state synchronization
    3. Handles episode termination conditions
    
    The wrapper maintains consistency between:
    - Lux game engine state
    - Reward space computations
    - Episode termination logic
    
    Args:
        env: LuxEnv instance to wrap
        reward_space: Custom reward computation object
    """
    def __init__(self, env: LuxEnv, reward_space: BaseRewardSpace):
        super(RewardSpaceWrapper, self).__init__(env)
        self.reward_space = reward_space

    def _get_rewards_and_done(self) -> Tuple[Tuple[float, float], bool]:
        """
        Compute rewards and episode completion using reward space.
        
        This method:
        1. Delegates reward computation to reward space
        2. Ensures consistency with game engine state
        3. Validates episode termination conditions
        
        Returns:
            Tuple containing:
            - Tuple[float, float]: Rewards for both players
            - bool: Whether episode is complete
            
        Raises:
            RuntimeError: If reward space and engine state are inconsistent
        """
        rewards, done = self.reward_space.compute_rewards_and_done(self.unwrapped.game_state, self.done)
        if self.unwrapped.done and not done:
            raise RuntimeError("Reward space did not return done, but the lux engine is done.")
        self.unwrapped.done = done
        return rewards, done

    def reset(self, **kwargs):
        obs, _, _, info = super(RewardSpaceWrapper, self).reset(**kwargs)
        return (obs, *self._get_rewards_and_done(), info)

    def step(self, action):
        obs, _, _, info = super(RewardSpaceWrapper, self).step(action)
        return (obs, *self._get_rewards_and_done(), info)


class LoggingEnv(gym.Wrapper):
    """
    Environment wrapper for tracking metrics and game statistics.
    
    Features:
    1. Game State Tracking:
       - Unit counts (workers, carts)
       - City statistics
       - Research progress
       
    2. Performance Metrics:
       - Peak values tracking
       - Cumulative rewards
       - Action distributions
       
    3. Debugging Information:
       - Per-step statistics
       - Resource utilization
       - Strategic indicators
    
    Args:
        env: Environment to wrap
        reward_space: Reward space for additional metrics
    """
    def __init__(self, env: gym.Env, reward_space: BaseRewardSpace):
        super(LoggingEnv, self).__init__(env)
        self.reward_space = reward_space
        self.vals_peak = {}
        self.reward_sums = [0., 0.]
        self.actions_distributions = {
            f"{space}.{meaning}": 0.
            for space, action_meanings in ACTION_MEANINGS.items()
            for meaning in action_meanings
        }
        # TODO: Resource mining % like in visualizer?
        # self.resource_count = {"wood", etc...}
        # TODO: Fuel metric?

    def info(self, info: Dict[str, np.ndarray], rewards: List[int]) -> Dict[str, np.ndarray]:
        info = copy.copy(info)
        game_state = self.env.unwrapped.game_state
        logs = {
            "step": [game_state.turn],
            "city_tiles": [p.city_tile_count for p in game_state.players],
            "separate_cities": [len(p.cities) for p in game_state.players],
            "workers": [sum(u.is_worker() for u in p.units) for p in game_state.players],
            "carts": [sum(u.is_cart() for u in p.units) for p in game_state.players],
            "research_points": [p.research_points for p in game_state.players],
        }
        self.vals_peak = {
            key: np.maximum(val, logs[key]) for key, val in self.vals_peak.items()
        }
        for key, val in self.vals_peak.items():
            logs[f"{key}_peak"] = val.copy()
            logs[f"{key}_final"] = logs[key]
            del logs[key]

        self.reward_sums = [r + s for r, s in zip(rewards, self.reward_sums)]
        logs["mean_cumulative_rewards"] = [np.mean(self.reward_sums)]
        logs["mean_cumulative_reward_magnitudes"] = [np.mean(np.abs(self.reward_sums))]
        logs["max_cumulative_rewards"] = [np.max(self.reward_sums)]

        actions_taken = self.env.unwrapped.action_space.actions_taken_to_distributions(info["actions_taken"])
        self.actions_distributions = {
            f"{space}.{act}": self.actions_distributions[f"{space}.{act}"] + n
            for space, dist in actions_taken.items()
            for act, n in dist.items()
        }
        logs.update({f"ACTIONS_{key}": val for key, val in self.actions_distributions.items()})

        info.update({f"LOGGING_{key}": np.array(val, dtype=np.float32) for key, val in logs.items()})
        # Add any additional info from the reward space
        info.update(self.reward_space.get_info())
        return info

    def reset(self, **kwargs):
        obs, reward, done, info = super(LoggingEnv, self).reset(**kwargs)
        self._reset_peak_vals()
        self.reward_sums = [0., 0.]
        self.actions_distributions = {
            key: 0. for key in self.actions_distributions.keys()
        }
        return obs, reward, done, self.info(info, reward)

    def step(self, action: Dict[str, np.ndarray]):
        obs, reward, done, info = super(LoggingEnv, self).step(action)
        return obs, reward, done, self.info(info, reward)

    def _reset_peak_vals(self) -> NoReturn:
        self.vals_peak = {
            key: np.array([0., 0.])
            for key in [
                "city_tiles",
                "separate_cities",
                "workers",
                "carts",
            ]
        }


class VecEnv(gym.Env):
    """
    Vectorized environment for parallel execution.
    
    This environment:
    1. Manages multiple environments in parallel
    2. Provides batch processing capabilities
    3. Synchronizes environment states
    
    Implementation Features:
    - Efficient dictionary stacking
    - Automatic vectorization of outputs
    - Proper environment lifecycle management
    - Support for forced resets
    
    Args:
        envs: List of environments to run in parallel
    """
    def __init__(self, envs: List[gym.Env]):
        self.envs = envs
        self.last_outs = [() for _ in range(len(self.envs))]

    @staticmethod
    def _stack_dict(x: List[Union[Dict, np.ndarray]]) -> Union[Dict, np.ndarray]:
        """
        Recursively stack dictionary or array elements.
        
        This method:
        1. Handles nested dictionary structures
        2. Stacks numpy arrays along batch dimension
        3. Preserves dictionary structure
        
        Args:
            x: List of dictionaries or arrays to stack
            
        Returns:
            Stacked dictionary or array with batch dimension
        """
        if isinstance(x[0], dict):
            return {key: VecEnv._stack_dict([i[key] for i in x]) for key in x[0].keys()}
        else:
            return np.stack([arr for arr in x], axis=0)

    @staticmethod
    def _vectorize_env_outs(env_outs: List[Tuple]) -> Tuple:
        """
        Convert environment outputs to vectorized format.
        
        Process:
        1. Unpack observation, reward, done, info tuples
        2. Stack observations and info dictionaries
        3. Convert rewards and done flags to arrays
        
        Args:
            env_outs: List of (obs, reward, done, info) tuples
            
        Returns:
            Tuple of vectorized (obs, rewards, dones, infos)
        """
        obs_list, reward_list, done_list, info_list = zip(*env_outs)
        obs_stacked = VecEnv._stack_dict(obs_list)
        reward_stacked = np.array(reward_list)
        done_stacked = np.array(done_list)
        info_stacked = VecEnv._stack_dict(info_list)
        return obs_stacked, reward_stacked, done_stacked, info_stacked

    def reset(self, force: bool = False, **kwargs):
        if force:
            # noinspection PyArgumentList
            self.last_outs = [env.reset(**kwargs) for env in self.envs]
            return VecEnv._vectorize_env_outs(self.last_outs)

        for i, env in enumerate(self.envs):
            # Check if env finished
            if self.last_outs[i][2]:
                # noinspection PyArgumentList
                self.last_outs[i] = env.reset(**kwargs)
        return VecEnv._vectorize_env_outs(self.last_outs)

    def step(self, action: Dict[str, np.ndarray]):
        actions = [
            {key: val[i] for key, val in action.items()} for i in range(len(self.envs))
        ]
        self.last_outs = [env.step(a) for env, a in zip(self.envs, actions)]
        return VecEnv._vectorize_env_outs(self.last_outs)

    def render(self, idx: int, mode: str = "human", **kwargs):
        # noinspection PyArgumentList
        return self.envs[idx].render(mode, **kwargs)

    def close(self):
        return [env.close() for env in self.envs]

    def seed(self, seed: Optional[int] = None) -> list:
        if seed is not None:
            return [env.seed(seed + i) for i, env in enumerate(self.envs)]
        else:
            return [env.seed(seed) for i, env in enumerate(self.envs)]

    @property
    def unwrapped(self) -> List[gym.Env]:
        return [env.unwrapped for env in self.envs]

    @property
    def action_space(self) -> List[gym.spaces.Dict]:
        return [env.action_space for env in self.envs]

    @property
    def observation_space(self) -> List[gym.spaces.Dict]:
        return [env.observation_space for env in self.envs]

    @property
    def metadata(self) -> List[Dict]:
        return [env.metadata for env in self.envs]


class PytorchEnv(gym.Wrapper):
    """
    Environment wrapper for PyTorch tensor conversion.
    
    Features:
    1. Automatic tensor conversion:
       - Numpy arrays to PyTorch tensors
       - Proper device placement
       - Non-blocking transfers
       
    2. Deep dictionary support:
       - Recursive tensor conversion
       - Maintains dictionary structure
       - Preserves metadata
    
    Args:
        env: Environment to wrap (single or vectorized)
        device: PyTorch device for tensor placement
    """
    def __init__(self, env: Union[gym.Env, VecEnv], device: torch.device = torch.device("cpu")):
        super(PytorchEnv, self).__init__(env)
        self.device = device
        
    def reset(self, **kwargs):
        return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).reset(**kwargs)])
        
    def step(self, action: Dict[str, torch.Tensor]):
        action = {
            key: val.cpu().numpy() for key, val in action.items()
        }
        return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).step(action)])

    def _to_tensor(self, x: Union[Dict, np.ndarray]) -> Dict[str, Union[Dict, torch.Tensor]]:
        """
        Convert numpy arrays to PyTorch tensors recursively.
        
        Features:
        1. Handles nested dictionary structures
        2. Non-blocking device transfers
        3. Maintains dictionary hierarchy
        
        Args:
            x: Dictionary or array to convert
            
        Returns:
            Converted structure with PyTorch tensors
        """
        if isinstance(x, dict):
            return {key: self._to_tensor(val) for key, val in x.items()}
        else:
            return torch.from_numpy(x).to(self.device, non_blocking=True)


class DictEnv(gym.Wrapper):
    """
    Wrapper that converts environment outputs to dictionary format.
    
    This wrapper:
    1. Standardizes environment interface
    2. Provides consistent dictionary structure
    3. Ensures key uniqueness
    
    Output Format:
    {
        'obs': observation,
        'reward': reward value,
        'done': completion flag,
        'info': additional information
    }
    """
    @staticmethod
    def _dict_env_out(env_out: tuple) -> dict:
        """
        Convert environment output tuple to standardized dictionary.
        
        Validation:
        1. Ensures no key conflicts
        2. Maintains consistent structure
        3. Preserves all information
        
        Args:
            env_out: (obs, reward, done, info) tuple
            
        Returns: 
            Dictionary with standardized keys
            
        Raises:
            AssertionError: If info dict contains reserved keys
        """
        obs, reward, done, info = env_out
        assert "obs" not in info.keys()
        assert "reward" not in info.keys()
        assert "done" not in info.keys()
        return dict(
            obs=obs,
            reward=reward,
            done=done,
            info=info
        )

    def reset(self, **kwargs):
        return DictEnv._dict_env_out(super(DictEnv, self).reset(**kwargs))

    def step(self, action):
        return DictEnv._dict_env_out(super(DictEnv, self).step(action))
