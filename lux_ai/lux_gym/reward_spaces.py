from abc import ABC, abstractmethod
import copy
import logging
import numpy as np
from scipy.stats import rankdata
from typing import Dict, NamedTuple, NoReturn, Tuple

from ..lux.game import Game
from ..lux.game_constants import GAME_CONSTANTS
from ..lux.game_objects import Player


def count_city_tiles(game_state: Game) -> np.ndarray:
    """
    Count the number of city tiles owned by each player.
    
    Args:
        game_state: Current game state containing player information
        
    Returns:
        numpy array of shape (2,) containing city tile counts for each player
    """
    return np.array([player.city_tile_count for player in game_state.players])


def count_units(game_state: Game) -> np.ndarray:
    """
    Count the total number of units (workers and carts) owned by each player.
    
    Args:
        game_state: Current game state containing player information
        
    Returns:
        numpy array of shape (2,) containing unit counts for each player
    """
    return np.array([len(player.units) for player in game_state.players])


def count_total_fuel(game_state: Game) -> np.ndarray:
    """
    Calculate the total fuel stored in all cities for each player.
    
    This is important for:
    - Surviving night cycles
    - Maintaining city tiles
    - Strategic resource management
    
    Args:
        game_state: Current game state containing player information
        
    Returns:
        numpy array of shape (2,) containing total fuel amounts for each player
    """
    return np.array([
        sum([city.fuel for city in player.cities.values()])
        for player in game_state.players
    ])


def count_research_points(game_state: Game) -> np.ndarray:
    """
    Get the current research points accumulated by each player.
    
    Research points are used to:
    - Unlock coal resource collection
    - Unlock uranium resource collection
    - Enable strategic resource progression
    
    Args:
        game_state: Current game state containing player information
        
    Returns:
        numpy array of shape (2,) containing research points for each player
    """
    return np.array([player.research_points for player in game_state.players])


def should_early_stop(game_state: Game) -> bool:
    """
    Determine if the game should end early based on player dominance.
    
    The game ends early if:
    - Any player has lost all city tiles
    - Any player has lost all units
    - Any player controls >= 75% of all city tiles
    - Any player controls >= 75% of all units
    
    This prevents unnecessarily long games when the outcome is clear.
    
    Args:
        game_state: Current game state containing player information
        
    Returns:
        True if early stopping conditions are met, False otherwise
    """
    ct_count = count_city_tiles(game_state)
    unit_count = count_units(game_state)
    ct_pct = ct_count / max(ct_count.sum(), 1)
    unit_pct = unit_count / max(unit_count.sum(), 1)
    return ((ct_count == 0).any() or
            (unit_count == 0).any() or
            (ct_pct >= 0.75).any() or
            (unit_pct >= 0.75).any())


class RewardSpec(NamedTuple):
    """
    Specification for a reward space defining its key characteristics.
    
    Attributes:
        reward_min: Minimum possible reward value
        reward_max: Maximum possible reward value
        zero_sum: Whether rewards sum to zero across players
        only_once: Whether reward is given once (True) or can repeat (False)
    
    This specification helps ensure reward spaces are properly bounded and
    behave consistently with respect to the learning algorithm's expectations.
    """
    reward_min: float  # Minimum possible reward value
    reward_max: float  # Maximum possible reward value
    zero_sum: bool    # Whether rewards sum to zero across players
    only_once: bool   # Whether reward is given once or can repeat


# All reward spaces defined below

class BaseRewardSpace(ABC):
    """
    Abstract base class for defining reward spaces in the Lux environment.
    
    This class provides the interface for:
    1. Reward Calculation:
       - Computing rewards for each player
       - Determining episode termination
       - Handling both full game and subtask rewards
       
    2. Reward Specification:
       - Defining reward bounds
       - Specifying zero-sum properties
       - Indicating one-time vs repeating rewards
       
    3. State Information:
       - Accessing game state
       - Tracking progress
       - Providing debugging info
       
    All reward spaces must implement:
    - get_reward_spec(): Define reward properties
    - compute_rewards_and_done(): Calculate rewards and termination
    
    This abstraction allows for:
    - Consistent reward space interface
    - Flexible reward definitions
    - Clear separation of concerns
    """
    def __init__(self, **kwargs):
        if kwargs:
            logging.warning(f"RewardSpace received unexpected kwargs: {kwargs}")

    @staticmethod
    @abstractmethod
    def get_reward_spec() -> RewardSpec:
        pass

    @abstractmethod
    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        pass

    def get_info(self) -> Dict[str, np.ndarray]:
        return {}


# Full game reward spaces defined below

class FullGameRewardSpace(BaseRewardSpace):
    """
    Base class for reward spaces that span the entire game duration.
    
    This class provides a framework for:
    1. Game-Level Rewards:
       - Victory/defeat conditions
       - Resource accumulation
       - Territory control
       - Research progress
       
    2. Continuous Feedback:
       - Per-step rewards
       - Progress indicators
       - Strategic incentives
       
    3. Terminal Rewards:
       - Final game outcome
       - Achievement bonuses
       - Performance metrics
       
    The distinction between FullGameRewardSpace and Subtask is that
    full game rewards provide continuous feedback throughout the entire
    game, while subtasks focus on specific objectives that can be
    completed before the game ends.
    """
    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        return self.compute_rewards(game_state, done), done

    @abstractmethod
    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        pass


class GameResultReward(FullGameRewardSpace):
    """
    Reward space that focuses on the final game outcome.
    
    This reward space:
    1. Terminal Rewards:
       - +1 for winner, -1 for loser
       - 0 reward during gameplay
       - Uses city tiles as primary victory metric
       - Uses unit count as tiebreaker
       
    2. Early Stopping:
       - Optional early game termination
       - Triggers on clear victory conditions
       - Prevents unnecessarily long games
       
    3. Implementation Details:
       - Normalizes rewards to [-1, 1] range
       - Zero-sum between players
       - Only awarded at game end
       
    This reward space encourages agents to focus on
    winning the game rather than intermediate objectives.
    """
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True,
            only_once=True
        )

    def __init__(self, early_stop: bool = False, **kwargs):
        super(GameResultReward, self).__init__(**kwargs)
        self.early_stop = early_stop

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        if not done:
            return 0., 0.

        # reward here is defined as the sum of number of city tiles with unit count as a tie-breaking mechanism
        rewards = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
        rewards = (rankdata(rewards) - 1.) * 2. - 1.
        return tuple(rewards)

    @staticmethod
    def compute_player_reward(player: Player):
        ct_count = player.city_tile_count
        unit_count = len(player.units)
        # max board size is 32 x 32 => 1024 max city tiles and units,
        # so this should keep it strictly so we break by city tiles then unit count
        return ct_count * 10000 + unit_count


class CityTileReward(FullGameRewardSpace):
    """
    Reward space based on city tile control.
    
    This reward space provides:
    1. Continuous Feedback:
       - Rewards proportional to city tile count
       - Updated every step
       - Normalized to [0, 1] range
       
    2. Strategic Incentives:
       - Encourages city expansion
       - Values territory control
       - Promotes resource gathering for city building
       
    3. Implementation Details:
       - Non-zero sum between players
       - Scales with board size (max 1024 tiles)
       - Independent of other game metrics
       
    This reward helps agents learn the importance of
    city growth and resource management.
    """
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=0.,
            reward_max=1.,
            zero_sum=False,
            only_once=False
        )

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        return tuple(count_city_tiles(game_state) / 1024.)


class StatefulMultiReward(FullGameRewardSpace):
    """
    Complex reward space combining multiple objectives with state tracking.
    
    This reward space provides:
    1. Multi-Objective Rewards:
       - Game result (win/loss)
       - City tile count changes
       - Unit count changes
       - Research progress
       - Fuel management
       - Worker efficiency
       
    2. State Tracking:
       - Maintains previous counts
       - Computes deltas each step
       - Handles night/day transitions
       
    3. Customization:
       - Configurable objective weights
       - Separate positive/negative scaling
       - Optional early stopping
       
    4. Implementation Details:
       - Non-zero sum by default
       - Normalized to small per-step values
       - Balanced for stable learning
       
    This reward space provides rich feedback for learning
    complex strategies while maintaining stable training.
    """
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            reward_max=1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            zero_sum=False,
            only_once=False
        )

    def __init__(
            self,
            positive_weight: float = 1.,
            negative_weight: float = 1.,
            early_stop: bool = False,
            **kwargs
    ):
        assert positive_weight > 0.
        assert negative_weight > 0.
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.early_stop = early_stop

        self.city_count = np.empty((2,), dtype=float)
        self.unit_count = np.empty_like(self.city_count)
        self.research_points = np.empty_like(self.city_count)
        self.total_fuel = np.empty_like(self.city_count)

        self.weights = {
            "game_result": 10.,
            "city": 1.,
            "unit": 0.5,
            "research": 0.1,
            "fuel": 0.005,
            # Penalize workers each step that their cargo remains full
            # "full_workers": -0.01,
            "full_workers": 0.,
            # A reward given each step
            "step": 0.,
        }
        self.weights.update({key: val for key, val in kwargs.items() if key in self.weights.keys()})
        for key in copy.copy(kwargs).keys():
            if key in self.weights.keys():
                del kwargs[key]
        super(StatefulMultiReward, self).__init__(**kwargs)
        self._reset()

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        new_city_count = count_city_tiles(game_state)
        new_unit_count = count_units(game_state)
        new_research_points = count_research_points(game_state)
        new_total_fuel = count_total_fuel(game_state)

        reward_items_dict = {
            "city": new_city_count - self.city_count,
            "unit": new_unit_count - self.unit_count,
            "research": new_research_points - self.research_points,
            # Don't penalize losing fuel at night
            "fuel": np.maximum(new_total_fuel - self.total_fuel, 0),
            "full_workers": np.array([
                sum(unit.get_cargo_space_left() > 0 for unit in player.units if unit.is_worker())
                for player in game_state.players
            ]),
            "step": np.ones(2, dtype=float)
        }

        if done:
            game_result_reward = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
            game_result_reward = (rankdata(game_result_reward) - 1.) * 2. - 1.
            self._reset()
        else:
            game_result_reward = np.array([0., 0.])
            self.city_count = new_city_count
            self.unit_count = new_unit_count
            self.research_points = new_research_points
            self.total_fuel = new_total_fuel
        reward_items_dict["game_result"] = game_result_reward

        assert self.weights.keys() == reward_items_dict.keys()
        reward = np.stack(
            [self.weight_rewards(reward_items_dict[key] * w) for key, w in self.weights.items()],
            axis=0
        ).sum(axis=0)

        return tuple(reward / 500. / max(self.positive_weight, self.negative_weight))

    def weight_rewards(self, reward: np.ndarray) -> np.ndarray:
        reward = np.where(
            reward > 0.,
            self.positive_weight * reward,
            reward
        )
        reward = np.where(
            reward < 0.,
            self.negative_weight * reward,
            reward
        )
        return reward

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)
        self.research_points = np.zeros_like(self.research_points)
        self.total_fuel = np.zeros_like(self.total_fuel)


class ZeroSumStatefulMultiReward(StatefulMultiReward):
    """
    Zero-sum variant of StatefulMultiReward for competitive training.
    
    This reward space:
    1. Competitive Focus:
       - Ensures rewards sum to zero between players
       - Encourages direct competition
       - Prevents reward exploitation
       
    2. Implementation:
       - Inherits StatefulMultiReward tracking
       - Centers rewards around mean
       - Maintains relative performance differences
       
    3. Training Benefits:
       - Stable competitive learning
       - Clear relative progress signals
       - Natural curriculum through opponent improvement
       
    This variant is particularly useful for:
    - Self-play training
    - Tournament-style evaluation
    - Learning robust competitive strategies
    """
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True,
            only_once=False
        )

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        reward = np.array(super(ZeroSumStatefulMultiReward, self).compute_rewards(game_state, done))
        return tuple(reward - reward.mean())


class PunishingExponentialReward(BaseRewardSpace):
    """
    Reward space with strong penalties for losing units/cities.
    
    This reward space provides:
    1. Survival Emphasis:
       - Heavy penalties for losses (-0.1)
       - Encourages conservative play
       - Terminates on unit/city loss
       
    2. Resource Management:
       - Tracks city counts
       - Monitors unit populations
       - Values fuel and research
       
    3. Implementation Details:
       - Non-zero sum
       - Exponential scaling of rewards
       - Early termination on losses
       - State tracking for deltas
       
    4. Learning Objectives:
       - Avoid catastrophic losses
       - Maintain resource efficiency
       - Balance risk and reward
       
    This space helps train agents to:
    - Play more carefully
    - Avoid risky strategies
    - Maintain core assets
    """
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            reward_max=1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            zero_sum=False,
            only_once=False
        )

    def __init__(
            self,
            **kwargs
    ):
        self.city_count = np.empty((2,), dtype=float)
        self.unit_count = np.empty_like(self.city_count)
        self.research_points = np.empty_like(self.city_count)
        self.total_fuel = np.empty_like(self.city_count)

        self.weights = {
            "game_result": 0.,
            "city": 1.,
            "unit": 0.5,
            "research": 0.01,
            "fuel": 0.001,
        }
        self.weights.update({key: val for key, val in kwargs.items() if key in self.weights.keys()})
        for key in copy.copy(kwargs).keys():
            if key in self.weights.keys():
                del kwargs[key]
        super(PunishingExponentialReward, self).__init__(**kwargs)
        self._reset()

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        new_city_count = count_city_tiles(game_state)
        new_unit_count = count_units(game_state)
        new_research_points = count_research_points(game_state)
        new_total_fuel = count_total_fuel(game_state)

        city_diff = new_city_count - self.city_count
        unit_diff = new_unit_count - self.unit_count
        reward_items_dict = {
            "city": new_city_count,
            "unit": new_unit_count,
            "research": new_research_points,
            "fuel": new_total_fuel,
        }

        if done:
            game_result_reward = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
            game_result_reward = (rankdata(game_result_reward) - 1.) * 2. - 1.
            self._reset()
        else:
            game_result_reward = np.array([0., 0.])
            self.city_count = new_city_count
            self.unit_count = new_unit_count
            self.research_points = new_research_points
            self.total_fuel = new_total_fuel
        reward_items_dict["game_result"] = game_result_reward

        assert self.weights.keys() == reward_items_dict.keys()
        reward = np.stack(
            [reward_items_dict[key] * w for key, w in self.weights.items()],
            axis=0
        ).sum(axis=0)

        lost_unit_or_city = (city_diff < 0) | (unit_diff < 0)
        reward = np.where(
            lost_unit_or_city,
            -0.1,
            reward / 1_000.
        )

        return tuple(reward), done or lost_unit_or_city.any()

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        raise NotImplementedError

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)
        self.research_points = np.zeros_like(self.research_points)
        self.total_fuel = np.zeros_like(self.total_fuel)


# Subtask reward spaces defined below
# NB: Subtasks that are "different enough" should be defined separately since each subtask gets its own embedding
# See obs_spaces.SUBTASK_ENCODING

class Subtask(BaseRewardSpace, ABC):
    """
    Abstract base class for specific game objectives/subtasks.
    
    Subtasks provide:
    1. Focused Learning:
       - Single clear objective
       - Binary completion status
       - One-time reward on completion
       
    2. Task Structure:
       - Independent of full game outcome
       - Can complete before game end
       - Specific success criteria
       
    3. Implementation Framework:
       - Consistent reward bounds [0,1]
       - Non-zero sum by default
       - Unique task embeddings
       
    4. Common Subtask Types:
       - Resource collection (wood, coal, uranium)
       - City building (tiles, contiguous)
       - Survival challenges
       - Research milestones
       
    Subtasks help break down complex game strategies
    into learnable components and provide curriculum
    learning opportunities.
    """
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        """
        Don't override reward_spec or you risk breaking classes like multi_subtask.MultiSubtask
        """
        return RewardSpec(
            reward_min=0.,
            reward_max=1.,
            zero_sum=False,
            only_once=True
        )

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        goal_reached = self.completed_task(game_state)
        return tuple(goal_reached.astype(float)), goal_reached.any() or done

    @abstractmethod
    def completed_task(self, game_state: Game) -> np.ndarray:
        pass

    def get_subtask_encoding(self, subtask_encoding: dict) -> int:
        return subtask_encoding[type(self)]


class CollectNWood(Subtask):
    """
    Subtask for collecting a target amount of wood.
    
    Features:
    1. Resource Focus:
       - Wood collection objective
       - Default target = worker capacity
       - Basic resource gathering practice
       
    2. Learning Goals:
       - Worker control
       - Resource identification
       - Inventory management
       
    3. Implementation:
       - Tracks total wood across all units
       - Completes when target reached
       - Independent of other resources
    """
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"], **kwargs):
        super(CollectNWood, self).__init__(**kwargs)
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.wood for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class CollectNCoal(Subtask):
    """
    Subtask for collecting a target amount of coal.
    
    Features:
    1. Advanced Resource:
       - Coal collection objective
       - Requires research progress
       - Higher value than wood
       
    2. Learning Goals:
       - Research progression
       - Resource prioritization
       - Worker specialization
       
    3. Implementation:
       - Default target = half worker capacity
       - Tracks total coal across units
       - Requires coal mining research
    """
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] // 2, **kwargs):
        super(CollectNCoal, self).__init__(**kwargs)
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.coal for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class CollectNUranium(Subtask):
    """
    Subtask for collecting a target amount of uranium.
    
    Features:
    1. End-game Resource:
       - Uranium collection objective
       - Requires maximum research
       - Highest value resource
       
    2. Learning Goals:
       - Late-game strategies
       - Research completion
       - High-value resource management
       
    3. Implementation:
       - Default target = 1/5 worker capacity
       - Tracks total uranium across units
       - Requires uranium mining research
    """
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] // 5, **kwargs):
        super(CollectNUranium, self).__init__(**kwargs)
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.uranium for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class MakeNCityTiles(Subtask):
    """
    Subtask for expanding city size through new tiles.
    
    Features:
    1. City Growth:
       - Build multiple city tiles
       - Any configuration allowed
       - Minimum target > 1 (start city)
       
    2. Learning Goals:
       - Resource to city conversion
       - Territory expansion
       - Worker build actions
       
    3. Implementation:
       - Counts total city tiles
       - Location independent
       - Supports early expansion
    """
    def __init__(self, n_city_tiles: int = 2, **kwargs):
        super(MakeNCityTiles, self).__init__(**kwargs)
        assert n_city_tiles > 1, "Players start with 1 city tile already"
        self.n_city_tiles = n_city_tiles

    def completed_task(self, game_state: Game) -> np.ndarray:
        return count_city_tiles(game_state) >= self.n_city_tiles


class MakeNContiguousCityTiles(MakeNCityTiles):
    """
    Subtask requiring connected city tile placement.
    
    Features:
    1. Strategic Growth:
       - Build adjacent city tiles
       - Form single large city
       - Efficient resource sharing
       
    2. Learning Goals:
       - Spatial planning
       - Defensive positioning
       - Resource optimization
       
    3. Implementation:
       - Tracks largest contiguous city
       - Inherits from MakeNCityTiles
       - Requires strategic placement
    """
    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            # Extra -1 is included to avoid taking max of empty sequence
            max([len(city.citytiles) for city in player.cities.values()] + [0])
            for player in game_state.players
        ]) >= self.n_city_tiles


class CollectNTotalFuel(Subtask):
    """
    Subtask for accumulating total fuel across all cities.
    
    Features:
    1. Night Survival:
       - Fuel stockpiling goal
       - Default = one night's upkeep
       - Critical for city maintenance
       
    2. Learning Goals:
       - Resource conversion
       - Night cycle preparation
       - City sustainability
       
    3. Implementation:
       - Tracks total fuel in cities
       - Scales with city count
       - Based on upkeep constants
    """
    def __init__(self, n_total_fuel: int = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"] *
                                           GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"], **kwargs):
        super(CollectNTotalFuel, self).__init__(**kwargs)
        self.n_total_fuel = n_total_fuel

    def completed_task(self, game_state: Game) -> np.ndarray:
        return count_total_fuel(game_state) >= self.n_total_fuel


class SurviveNNights(Subtask):
    """
    Subtask for surviving multiple night cycles.
    
    Features:
    1. Survival Challenge:
       - Maintain cities through nights
       - No unit/city tile losses
       - Resource management focus
       
    2. Learning Goals:
       - Night cycle preparation
       - Resource stockpiling
       - City/unit preservation
       - Long-term planning
       
    3. Implementation:
       - Tracks game cycles
       - Monitors unit/city counts
       - Partial rewards for progress
       - Fails on any losses
       
    4. Reward Structure:
       - 1.0 for full completion
       - 0.5 for surviving but incomplete
       - 0.0 for any losses
    """
    def __init__(self, n_nights: int = 1, **kwargs):
        super(SurviveNNights, self).__init__(**kwargs)
        cycle_len = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        self.target_step = n_nights * cycle_len
        assert self.target_step <= GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

        self.city_count = np.empty((2,), dtype=int)
        self.unit_count = np.empty_like(self.city_count)

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        failed_task = self.failed_task(game_state)
        completed_task = self.completed_task(game_state)
        if failed_task.any():
            rewards = np.where(
                failed_task,
                0.,
                0.5 + 0.5 * completed_task.astype(float)
            )
        else:
            rewards = completed_task.astype(float)
        done = failed_task.any() or completed_task.any() or done
        if done:
            self._reset()
        return tuple(rewards), done

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            game_state.turn >= self.target_step
        ]).repeat(2)

    def failed_task(self, game_state: Game) -> np.ndarray:
        new_city_count = count_city_tiles(game_state)
        new_unit_count = count_units(game_state)

        failed = np.logical_or(
            new_city_count < self.city_count,
            new_unit_count < self.unit_count
        )
        self.city_count = new_city_count
        self.unit_count = new_unit_count
        return failed

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)


class GetNResearchPoints(Subtask):
    """
    Subtask for achieving research milestones.
    
    Features:
    1. Technology Progress:
       - Research point accumulation
       - Default = coal requirement
       - Unlocks resource access
       
    2. Learning Goals:
       - City tile utilization
       - Research prioritization
       - Strategic progression
       - Resource tier planning
       
    3. Implementation:
       - Tracks research points
       - Configurable target
       - Based on game constants
       - Enables resource tiers
       
    4. Strategic Impact:
       - Coal mining (50 points)
       - Uranium mining (200 points)
       - Advanced resource access
    """
    def __init__(
            self,
            n_research_points: int = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"],
            **kwargs
    ):
        super(GetNResearchPoints, self).__init__(**kwargs)
        self.n_research_points = n_research_points

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([player.research_points for player in game_state.players]) >= self.n_research_points
