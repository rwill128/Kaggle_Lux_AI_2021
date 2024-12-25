from abc import ABC, abstractmethod
from functools import lru_cache
import gym
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union


from ..utility_constants import MAX_CAPACITY, MAX_RESEARCH, MAX_BOARD_SIZE
from ..lux.constants import Constants
from ..lux.game import Game
from ..lux.game_objects import CityTile, Unit

# The maximum number of actions that can be taken by units sharing a square
# All remaining units take the no-op action
MAX_OVERLAPPING_ACTIONS = 4
DIRECTIONS = Constants.DIRECTIONS.astuple(include_center=False)
RESOURCES = Constants.RESOURCE_TYPES.astuple()

ACTION_MEANINGS = {
    "worker": [
        "NO-OP",
    ],
    "cart": [
        "NO-OP",
    ],
    "city_tile": [
        "NO-OP",
        "BUILD_WORKER",
        "BUILD_CART",
        "RESEARCH",
    ]
}
for u in ["worker", "cart"]:
    for d in DIRECTIONS:
        ACTION_MEANINGS[u].append(f"MOVE_{d}")
    for r in RESOURCES:
        for d in DIRECTIONS:
            ACTION_MEANINGS[u].append(f"TRANSFER_{r}_{d}")
ACTION_MEANINGS["worker"].extend(["PILLAGE", "BUILD_CITY"])
ACTION_MEANINGS_TO_IDX = {
    actor: {
        action: idx for idx, action in enumerate(actions)
    } for actor, actions in ACTION_MEANINGS.items()
}


def _no_op(game_object: Union[Unit, CityTile]) -> Optional[str]:
    """
    Generate no-op (do nothing) action for any game object.
    
    Args:
        game_object: Unit or CityTile that will take no action
        
    Returns:
        None to indicate no action
    """
    return None


def _pillage(worker: Unit) -> str:
    """
    Generate pillage action for a worker unit.
    
    Args:
        worker: Worker unit that will pillage current tile
        
    Returns:
        Command string for pillage action
    """
    return worker.pillage()


def _build_city(worker: Unit) -> str:
    """
    Generate city building action for a worker unit.
    
    Args:
        worker: Worker unit that will build a city
        
    Returns:
        Command string for city building action
    """
    return worker.build_city()


def _build_worker(city_tile: CityTile) -> str:
    """
    Generate worker building action for a city tile.
    
    Args:
        city_tile: City tile that will build a worker
        
    Returns:
        Command string for worker building action
    """
    return city_tile.build_worker()


def _build_cart(city_tile: CityTile) -> str:
    """
    Generate cart building action for a city tile.
    
    Args:
        city_tile: City tile that will build a cart
        
    Returns:
        Command string for cart building action
    """
    return city_tile.build_cart()


def _research(city_tile: CityTile) -> str:
    """
    Generate research action for a city tile.
    
    Args:
        city_tile: City tile that will conduct research
        
    Returns:
        Command string for research action
    """
    return city_tile.research()


def _move_factory(action_meaning: str) -> Callable[[Unit], str]:
    """
    Create a function that generates move actions in a specific direction.
    
    Args:
        action_meaning: Action string containing the direction (e.g. "MOVE_NORTH")
        
    Returns:
        Function that takes a unit and returns a move command string
        
    Raises:
        ValueError: If direction is not recognized
    """
    direction = action_meaning.split("_")[1]
    if direction not in DIRECTIONS:
        raise ValueError(f"Unrecognized direction '{direction}' in action_meaning '{action_meaning}'")

    def _move_func(unit: Unit) -> str:
        """Generate move command for unit in specified direction."""
        return unit.move(direction)

    return _move_func


def _transfer_factory(action_meaning: str) -> Callable[..., str]:
    resource, direction = action_meaning.split("_")[1:]
    if resource not in RESOURCES:
        raise ValueError(f"Unrecognized resource '{resource}' in action_meaning '{action_meaning}'")
    if direction not in DIRECTIONS:
        raise ValueError(f"Unrecognized direction '{direction}' in action_meaning '{action_meaning}'")

    def _transfer_func(unit: Unit, pos_to_unit_dict: Dict[Tuple, Optional[Unit]]) -> str:
        dest_pos = unit.pos.translate(direction, 1)
        dest_unit = pos_to_unit_dict.get((dest_pos.x, dest_pos.y), None)
        # If the square is not on the map or there is not an allied unit in that square
        if dest_unit is None:
            return ""
        # NB: Technically, this does limit the agent's action space, particularly in that they cannot transfer anything
        # except the maximum amount. I don't want to deal with continuous action spaces, but perhaps the transfer
        # action could be bucketed if partial transfers become important.
        # The game engine automatically determines the actual maximum legal transfer
        # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/index.ts#L704
        return unit.transfer(dest_id=dest_unit.id, resourceType=resource, amount=MAX_CAPACITY)

    return _transfer_func


ACTION_MEANING_TO_FUNC = {
    "worker": {
        "NO-OP": _no_op,
        "PILLAGE": _pillage,
        "BUILD_CITY": _build_city,
    },
    "cart": {
        "NO-OP": _no_op,
    },
    "city_tile": {
        "NO-OP": _no_op,
        "BUILD_WORKER": _build_worker,
        "BUILD_CART": _build_cart,
        "RESEARCH": _research,
    }
}
for u in ["worker", "cart"]:
    for d in DIRECTIONS:
        a = f"MOVE_{d}"
        ACTION_MEANING_TO_FUNC[u][a] = _move_factory(a)
    for r in RESOURCES:
        for d in DIRECTIONS:
            actions_str = f"TRANSFER_{r}_{d}"
            ACTION_MEANING_TO_FUNC[u][actions_str] = _transfer_factory(actions_str)


class BaseActSpace(ABC):
    @abstractmethod
    def get_action_space(self, board_dims: Tuple[int, int] = MAX_BOARD_SIZE) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def process_actions(
            self,
            action_tensors_dict: Dict[str, np.ndarray],
            game_state: Game,
            board_dims: Tuple[int, int],
            pos_to_unit_dict: Dict[Tuple, Optional[Unit]]
    ) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        pass

    @abstractmethod
    def get_available_actions_mask(
            self,
            game_state: Game,
            board_dims: Tuple[int, int],
            pos_to_unit_dict: Dict[Tuple, Optional[Unit]],
            pos_to_city_tile_dict: Dict[Tuple, Optional[CityTile]]
    ) -> Dict[str, np.ndarray]:
        pass

    @staticmethod
    @abstractmethod
    def actions_taken_to_distributions(actions_taken: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
        pass


class BasicActionSpace(BaseActSpace):
    """
    Standard implementation of the Lux AI action space.
    
    This class implements a multi-discrete action space with:
    - Separate action spaces for workers, carts, and city tiles
    - Support for multiple units sharing the same position
    - Action masking to prevent illegal moves
    - Tracking of taken actions for analysis
    
    The action space is structured as a dictionary of multi-discrete spaces,
    with dimensions (1, num_players, width, height) for each actor type.
    """
    
    def __init__(self, default_board_dims: Optional[Tuple[int, int]] = None):
        """
        Initialize action space with optional custom board dimensions.
        
        Args:
            default_board_dims: Custom board dimensions (width, height), defaults to MAX_BOARD_SIZE
        """
        self.default_board_dims = MAX_BOARD_SIZE if default_board_dims is None else default_board_dims

    @lru_cache(maxsize=None)
    def get_action_space(self, board_dims: Optional[Tuple[int, int]] = None) -> gym.spaces.Dict:
        """
        Get the Gym action space for the current board dimensions.
        
        Creates a dictionary of MultiDiscrete spaces with shape (1, num_players, width, height)
        for each actor type. The last dimension size is the number of possible actions
        for that actor type.
        
        Args:
            board_dims: Optional board dimensions (width, height). Uses default if None.
            
        Returns:
            Dictionary mapping actor types to their MultiDiscrete action spaces
            
        Notes:
            - Uses LRU cache to avoid recreating spaces for same dimensions
            - Shape is (1, 2, width, height) since game always has 2 players
            - Each position can contain multiple units, handled by process_actions
        """
        if board_dims is None:
            board_dims = self.default_board_dims
        x = board_dims[0]
        y = board_dims[1]
        # Player count is always 2 in Lux AI
        p = 2
        return gym.spaces.Dict({
            "worker": gym.spaces.MultiDiscrete(np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["worker"])),
            "cart": gym.spaces.MultiDiscrete(np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["cart"])),
            "city_tile": gym.spaces.MultiDiscrete(
                np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["city_tile"])
            ),
        })

    @lru_cache(maxsize=None)
    def get_action_space_expanded_shape(self, *args, **kwargs) -> Dict[str, Tuple[int, ...]]:
        """
        Get expanded shapes of action spaces including the action dimension.
        
        For each actor type, returns the shape (1, num_players, width, height, num_actions).
        This expanded shape is used for action tracking and masking.
        
        Args:
            *args: Passed to get_action_space
            **kwargs: Passed to get_action_space
            
        Returns:
            Dictionary mapping actor types to their expanded shapes
            
        Notes:
            - Uses LRU cache to avoid recomputation
            - Adds num_actions dimension to base action space shape
            - Used by process_actions and get_available_actions_mask
        """
        action_space = self.get_action_space(*args, **kwargs)
        action_space_expanded = {}
        for key, val in action_space.spaces.items():
            action_space_expanded[key] = val.shape + (len(ACTION_MEANINGS[key]),)
        return action_space_expanded

    def process_actions(
            self,
            action_tensors_dict: Dict[str, np.ndarray],
            game_state: Game,
            board_dims: Tuple[int, int],
            pos_to_unit_dict: Dict[Tuple, Optional[Unit]]
    ) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        """
        Convert network action outputs into game commands and track taken actions.
        
        This method handles the complex task of converting neural network outputs into
        valid game commands while respecting game rules and tracking which actions
        were actually taken. Key features:
        
        1. Multiple Units per Cell:
           - Supports up to MAX_OVERLAPPING_ACTIONS units in same location
           - Units beyond this limit are skipped
           - NO-OP action causes remaining units in cell to be skipped
        
        2. Action Processing:
           - Converts action indices to command strings
           - Handles both unit and city tile actions
           - Tracks which actions were successfully taken
           - Filters out invalid actions (e.g. failed transfers)
        
        Args:
            action_tensors_dict: Network outputs mapping actor types to action tensors
                Shape: (1, num_players, width, height, actions_per_cell)
            game_state: Current game state
            board_dims: Board dimensions (width, height)
            pos_to_unit_dict: Position to unit mapping for transfer validation
            
        Returns:
            Tuple containing:
            - List[List[str]]: Command strings for each player
            - Dict[str, np.ndarray]: Boolean tensors tracking taken actions
            
        Notes:
            - None and empty string ("") both indicate no-op actions
            - Empty string specifically indicates invalid transfer attempts
            - Actions beyond MAX_OVERLAPPING_ACTIONS in a cell are skipped
            - NO-OP action causes all remaining actions in that cell to be skipped
        """
        action_strs = [[], []]
        actions_taken = {
            key: np.zeros(space, dtype=bool) for key, space in self.get_action_space_expanded_shape(board_dims).items()
        }
        for player in game_state.players:
            p_id = player.team
            worker_actions_taken_count = np.zeros(board_dims, dtype=int)
            cart_actions_taken_count = np.zeros_like(worker_actions_taken_count)
            for unit in player.units:
                if unit.can_act():
                    x, y = unit.pos.x, unit.pos.y
                    if unit.is_worker():
                        unit_type = "worker"
                        actions_taken_count = worker_actions_taken_count
                    elif unit.is_cart():
                        unit_type = "cart"
                        actions_taken_count = cart_actions_taken_count
                    else:
                        raise NotImplementedError(f'New unit type: {unit}')
                    # Action plane is selected for stacked units
                    actor_count = actions_taken_count[x, y]
                    if actor_count >= MAX_OVERLAPPING_ACTIONS:
                        action = None
                    else:
                        action_idx = action_tensors_dict[unit_type][0, p_id, x, y, actor_count]
                        action_meaning = ACTION_MEANINGS[unit_type][action_idx]
                        action = get_unit_action(unit, action_idx, pos_to_unit_dict)
                        action_was_taken = action_meaning == "NO-OP" or (action is not None and action != "")
                        actions_taken[unit_type][0, p_id, x, y, action_idx] = action_was_taken
                        # If action is NO-OP, skip remaining actions for units at same location
                        if action_meaning == "NO-OP":
                            actions_taken_count[x, y] += MAX_OVERLAPPING_ACTIONS
                    # None means no-op
                    # "" means invalid transfer action - fed to game as no-op
                    if action is not None and action != "":
                        # noinspection PyTypeChecker
                        action_strs[p_id].append(action)
                    actions_taken_count[x, y] += 1
            for city in player.cities.values():
                for city_tile in city.citytiles:
                    if city_tile.can_act():
                        x, y = city_tile.pos.x, city_tile.pos.y
                        action_idx = action_tensors_dict["city_tile"][0, p_id, x, y, 0]
                        action_meaning = ACTION_MEANINGS["city_tile"][action_idx]
                        action = get_city_tile_action(city_tile, action_idx)
                        action_was_taken = action_meaning == "NO-OP" or (action is not None and action != "")
                        actions_taken["city_tile"][0, p_id, x, y, action_idx] = action_was_taken
                        # None means no-op
                        if action is not None:
                            # noinspection PyTypeChecker
                            action_strs[p_id].append(action)
        return action_strs, actions_taken

    def get_available_actions_mask(
            self,
            game_state: Game,
            board_dims: Tuple[int, int],
            pos_to_unit_dict: Dict[Tuple, Optional[Unit]],
            pos_to_city_tile_dict: Dict[Tuple, Optional[CityTile]]
    ) -> Dict[str, np.ndarray]:
        """
        Generate action masks to prevent illegal actions during training.
        
        Creates boolean tensors indicating which actions are legal for each actor
        at each position. This is critical for RL training to prevent the agent
        from attempting illegal actions. Key rules implemented:
        
        1. Movement Rules:
           - Cannot move off board edges
           - Cannot move onto opposing city tiles
           - Cannot move onto units with cooldown > 0
        
        2. Resource Transfer Rules:
           - Must have allied unit in target direction
           - Must have resource being transferred
           - Target unit must have cargo space
        
        3. Worker-Specific Rules:
           - Pillaging requires road tile and no allied city
           - City building requires sufficient resources and no resource tile
        
        4. City Tile Rules:
           - Research only allowed if below MAX_RESEARCH
           - Unit building only if units < city_tile_count
        
        Args:
            game_state: Current game state
            board_dims: Board dimensions (width, height)
            pos_to_unit_dict: Maps positions to units for collision checks
            pos_to_city_tile_dict: Maps positions to city tiles for rule checks
            
        Returns:
            Dictionary mapping actor types to boolean tensors (True = legal action)
            Shape: (1, num_players, width, height, num_actions)
            
        Notes:
            - NO-OP is always a legal action
            - Masks are critical for policy gradient methods
            - Prevents wasted computation on illegal actions
            - Helps guide exploration to legal actions only
        """
        available_actions_mask = {
            key: np.ones(space.shape + (len(ACTION_MEANINGS[key]),), dtype=bool)
            for key, space in self.get_action_space(board_dims).spaces.items()
        }
        for player in game_state.players:
            p_id = player.team
            for unit in player.units:
                if unit.can_act():
                    x, y = unit.pos.x, unit.pos.y
                    if unit.is_worker():
                        unit_type = "worker"
                    elif unit.is_cart():
                        unit_type = "cart"
                    else:
                        raise NotImplementedError(f"New unit type: {unit}")
                    # No-op is always a legal action
                    # Moving is usually a legal action, except when:
                    #   The unit is at the edge of the board and would try to move off of it
                    #   The unit would move onto an opposing city tile
                    #   The unit would move onto another unit with cooldown > 0
                    # Transferring is only a legal action when:
                    #   There is an allied unit in the target square
                    #   The transferring unit has > 0 cargo of the designated resource
                    #   The receiving unit has cargo space remaining
                    # Workers: Pillaging is only a legal action when on a road tile and is not on an allied city
                    # Workers: Building a city is only a legal action when the worker has the required resources and
                    #       is not on a resource tile
                    for direction in DIRECTIONS:
                        new_pos_tuple = unit.pos.translate(direction, 1)
                        new_pos_tuple = new_pos_tuple.x, new_pos_tuple.y
                        # Moving and transferring - check that the target position exists on the board
                        if new_pos_tuple not in pos_to_unit_dict.keys():
                            available_actions_mask[unit_type][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX[unit_type][f"MOVE_{direction}"]
                            ] = False
                            for resource in RESOURCES:
                                available_actions_mask[unit_type][
                                    :,
                                    p_id,
                                    x,
                                    y,
                                    ACTION_MEANINGS_TO_IDX[unit_type][f"TRANSFER_{resource}_{direction}"]
                                ] = False
                            continue
                        # Moving - check that the target position does not contain an opposing city tile
                        new_pos_city_tile = pos_to_city_tile_dict[new_pos_tuple]
                        if new_pos_city_tile and new_pos_city_tile.team != p_id:
                            available_actions_mask[unit_type][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX[unit_type][f"MOVE_{direction}"]
                            ] = False
                        # Moving - check that the target position does not contain a unit with cooldown > 0
                        new_pos_unit = pos_to_unit_dict[new_pos_tuple]
                        if new_pos_unit and new_pos_unit.cooldown > 0:
                            available_actions_mask[unit_type][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX[unit_type][f"MOVE_{direction}"]
                            ] = False
                        for resource in RESOURCES:
                            if (
                                    # Transferring - check that there is an allied unit in the target square
                                    (new_pos_unit is None or new_pos_unit.team != p_id) or
                                    # Transferring - check that the transferring unit has the designated resource
                                    (unit.cargo.get(resource) <= 0) or
                                    # Transferring - check that the receiving unit has cargo space
                                    (new_pos_unit.get_cargo_space_left() <= 0)
                            ):
                                available_actions_mask[unit_type][
                                    :,
                                    p_id,
                                    x,
                                    y,
                                    ACTION_MEANINGS_TO_IDX[unit_type][f"TRANSFER_{resource}_{direction}"]
                                ] = False
                    if unit.is_worker():
                        # Pillaging - check that worker is on a road tile and not on an allied city tile
                        if game_state.map.get_cell_by_pos(unit.pos).road <= 0 or \
                                pos_to_city_tile_dict[(unit.pos.x, unit.pos.y)] is not None:
                            available_actions_mask[unit_type][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX[unit_type]["PILLAGE"]
                            ] = False
                        # Building a city - check that worker has >= the required resources and is not on a resource
                        if not unit.can_build(game_state.map):
                            available_actions_mask[unit_type][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX[unit_type]["BUILD_CITY"]
                            ] = False
            for city in player.cities.values():
                for city_tile in city.citytiles:
                    if city_tile.can_act():
                        # No-op is always a legal action
                        # Research is a legal action whenever research_points < max_research
                        # Building a new unit is only a legal action when n_units < n_city_tiles
                        x, y = city_tile.pos.x, city_tile.pos.y
                        if player.research_points >= MAX_RESEARCH:
                            available_actions_mask["city_tile"][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX["city_tile"]["RESEARCH"]
                            ] = False
                        if len(player.units) >= player.city_tile_count:
                            available_actions_mask["city_tile"][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX["city_tile"]["BUILD_WORKER"]
                            ] = False
                            available_actions_mask["city_tile"][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX["city_tile"]["BUILD_CART"]
                            ] = False
        return available_actions_mask

    @staticmethod
    def actions_taken_to_distributions(actions_taken: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
        """
        Convert action tracking tensors to action count distributions.
        
        Takes the boolean tensors tracking which actions were taken and converts
        them into a nested dictionary showing how many times each action was taken
        by each actor type.
        
        Args:
            actions_taken: Dictionary mapping actor types to boolean tensors
                         indicating which actions were taken
            
        Returns:
            Nested dictionary:
            {actor_type: {action_name: count}}
            showing distribution of actions taken
            
        Example:
            Input tensor might track that workers took 5 MOVE_NORTH actions
            Output would include {"worker": {"MOVE_NORTH": 5, ...}}
        """
        out = {}
        for space, actions in actions_taken.items():
            out[space] = {
                ACTION_MEANINGS[space][i]: actions[..., i].sum()
                for i in range(actions.shape[-1])
            }
        return out


def get_unit_action(unit: Unit, action_idx: int, pos_to_unit_dict: Dict[Tuple, Optional[Unit]]) -> Optional[str]:
    """
    Generate command string for a unit's action.
    
    Args:
        unit: Unit taking the action
        action_idx: Index into the unit's action space
        pos_to_unit_dict: Dictionary mapping positions to units (for transfer actions)
        
    Returns:
        Command string for the action, or None for no-op
        
    Raises:
        NotImplementedError: If unit type is not worker or cart
        
    Notes:
        Transfer actions require the position dictionary to find target units.
        Other actions only need the unit itself.
    """
    if unit.is_worker():
        unit_type = "worker"
    elif unit.is_cart():
        unit_type = "cart"
    else:
        raise NotImplementedError(f'New unit type: {unit}')
    action = ACTION_MEANINGS[unit_type][action_idx]
    if action.startswith("TRANSFER"):
        # Transfer actions need position dictionary to find target unit
        return ACTION_MEANING_TO_FUNC[unit_type][action](unit, pos_to_unit_dict)
    else:
        return ACTION_MEANING_TO_FUNC[unit_type][action](unit)


def get_city_tile_action(city_tile: CityTile, action_idx: int) -> Optional[str]:
    """
    Generate command string for a city tile's action.
    
    Args:
        city_tile: City tile taking the action
        action_idx: Index into the city tile's action space
        
    Returns:
        Command string for the action, or None for no-op
    """
    action = ACTION_MEANINGS["city_tile"][action_idx]
    return ACTION_MEANING_TO_FUNC["city_tile"][action](city_tile)
