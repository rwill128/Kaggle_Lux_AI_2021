from abc import ABC, abstractmethod
import numpy as np
import random
from typing import Callable, Dict, Optional, Tuple, Sequence

from .reward_spaces import Subtask
from ..lux.game import Game


class SubtaskSampler(ABC):
    """
    Abstract base class for sampling subtasks in curriculum learning.
    
    This class provides a framework for:
    1. Task Selection:
       - Maintains pool of available subtasks
       - Implements sampling strategy
       - Tracks task performance
       
    2. Curriculum Learning:
       - Progressive task difficulty
       - Performance-based selection
       - Adaptive learning paths
       
    3. Implementation Interface:
       - Abstract sample() method
       - Optional performance tracking
       - Extensible for custom strategies
       
    Args:
        subtask_constructors: Sequence of callable constructors for creating
                            subtask instances. Each constructor should return
                            a Subtask object when called.
    """
    def __init__(self, subtask_constructors: Sequence[Callable[..., Subtask]]):
        self.subtask_constructors = subtask_constructors

    @abstractmethod
    def sample(self, final_rewards: Optional[Tuple[float, float]]) -> Subtask:
        pass

    # noinspection PyMethodMayBeStatic
    def get_info(self) -> Dict[str, np.ndarray]:
        """
        Get debugging and logging information for the sampler.
        
        Returns:
            Dict[str, np.ndarray]: Empty dictionary by default.
                                 Subclasses may override to provide
                                 sampling statistics.
        """
        return {}


class RandomSampler(SubtaskSampler):
    """
    Simple uniform random subtask sampler.
    
    Features:
    1. Sampling Strategy:
       - Uniform random selection
       - No performance tracking
       - Equal probability for all tasks
       
    2. Use Cases:
       - Initial exploration
       - Baseline comparison
       - Task-agnostic training
       
    3. Implementation:
       - Ignores final_rewards
       - Simple random.randrange
       - No state maintenance
    """
    def sample(self, final_rewards: Optional[Tuple[float, float]]) -> Subtask:
        return self.subtask_constructors[random.randrange(len(self.subtask_constructors))]()


class DifficultySampler(SubtaskSampler):
    """
    Performance-based subtask sampler for adaptive curriculum learning.
    
    Features:
    1. Adaptive Sampling:
       - Tracks task performance
       - Weights tasks by difficulty
       - Focuses on challenging tasks
       
    2. Performance Tracking:
       - Maintains success rates
       - Updates per-task statistics
       - Computes sampling weights
       
    3. Implementation Details:
       - Uses running averages
       - Inverse difficulty weighting
       - Normalized probabilities
       
    This sampler implements curriculum learning by:
    - Tracking average rewards per subtask
    - Favoring tasks with lower success rates
    - Automatically adjusting task distribution
    """
    def __init__(self, subtask_constructors: Sequence[Callable[..., Subtask]]):
        super(DifficultySampler, self).__init__(subtask_constructors)
        self.active_subtask_idx = -1
        self.summed_rewards = np.zeros(len(self.subtask_constructors))
        self.n_trials = np.zeros(len(self.subtask_constructors))

    def sample(self, final_rewards: Optional[Tuple[float, float]]) -> Subtask:
        if final_rewards is not None:
            self.n_trials[self.active_subtask_idx] += 1
            self.summed_rewards[self.active_subtask_idx] += np.mean(final_rewards)

        self.active_subtask_idx = np.random.choice(len(self.subtask_constructors), p=self.weights)
        return self.subtask_constructors[self.active_subtask_idx]()

    @property
    def weights(self) -> np.ndarray:
        weights = Subtask.get_reward_spec().reward_max - self.summed_rewards / np.maximum(self.n_trials, 1)
        return weights / weights.sum()

    def get_info(self) -> Dict[str, np.ndarray]:
        """
        Get difficulty-based sampling statistics.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping each subtask name
                                 to its current sampling difficulty weight.
                                 Higher values indicate more challenging tasks.
        """
        return {
            f"LOGGING_{subtask.__name__}_subtask_difficulty": self.weights[i]
            for i, subtask in enumerate(self.subtask_constructors)
        }


class MultiSubtask(Subtask):
    """
    Manager class for handling multiple subtasks with dynamic selection.
    
    This class provides:
    1. Task Management:
       - Maintains subtask pool
       - Handles task transitions
       - Tracks active subtask
       
    2. Reward Processing:
       - Delegates to active subtask
       - Records per-subtask rewards
       - Manages task completion
       
    3. Information Tracking:
       - Logs subtask performance
       - Maintains task statistics
       - Provides debugging info
       
    Args:
        subtask_constructors: Sequence of callable constructors for creating
                            subtask instances
        subtask_sampler_constructor: Constructor for the sampling strategy,
                                   defaults to RandomSampler
        **kwargs: Additional arguments passed to parent Subtask
    """
    def __init__(
            self,
            subtask_constructors: Sequence[Callable[..., Subtask]] = (),
            subtask_sampler_constructor: Callable[..., SubtaskSampler] = RandomSampler,
            **kwargs
    ):
        super(MultiSubtask, self).__init__(**kwargs)
        self.subtask_constructors = subtask_constructors
        self.subtask_sampler = subtask_sampler_constructor(self.subtask_constructors)
        self.active_subtask = self.subtask_sampler.sample(None)
        self.info = {
            f"LOGGING_{subtask.__name__}_subtask_reward": np.array([float("nan"), float("nan")])
            for subtask in self.subtask_constructors
        }

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        reward, done = self.active_subtask.compute_rewards_and_done(game_state, done)
        for subtask in self.subtask_constructors:
            reward_key = f"LOGGING_{subtask.__name__}_subtask_reward"
            if isinstance(self.active_subtask, subtask):
                self.info[reward_key] = np.array(reward)
            else:
                self.info[reward_key] = np.array([float("nan"), float("nan")])
        if done:
            self.active_subtask = self.subtask_sampler.sample(reward)
        return reward, done

    def completed_task(self, game_state: Game) -> np.ndarray:
        raise NotImplementedError

    def get_info(self) -> Dict[str, np.ndarray]:
        """
        Get combined information from active subtask and sampler.
        
        Returns:
            Dict[str, np.ndarray]: Merged dictionary containing:
                                 - Per-subtask reward information
                                 - Sampler statistics and metrics
        """
        return dict(**self.info, **self.subtask_sampler.get_info())

    def get_subtask_encoding(self, subtask_encoding_dict: dict) -> int:
        return self.active_subtask.get_subtask_encoding(subtask_encoding_dict)
