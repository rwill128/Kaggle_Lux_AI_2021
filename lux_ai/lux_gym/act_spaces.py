"""
This module defines an action space for units (worker, cart) and city tiles in a
Lux AI environment. The BasicActionSpace class provides methods to:

1. Generate action spaces (gym.spaces) for each game entity (worker, cart, city tile).
2. Process actions from a dictionary of action tensors into actual game commands.
3. Compute a boolean mask indicating which actions are valid at each position.
4. Convert the final set of actions taken into distribution statistics.

The code heavily relies on the definitions of:
- Units (Worker/Cart)
- CityTiles
- The Game state
- Constants (DIRECTIONS, RESOURCES, etc.)

All actions are enumerated and stored as strings in ACTION_MEANINGS and are mapped to
functions in ACTION_MEANING_TO_FUNC, which calls the relevant environment methods to
perform the action (e.g., move, transfer, pillage, build_city, etc.).
"""

from abc import ABC, abstractmethod
from functools import lru_cache
import gym
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging

# Constants imported from local modules; these constants include things like
# - MAX_CAPACITY: Maximum cargo capacity for units
# - MAX_RESEARCH: Maximum research points a player can have
# - MAX_BOARD_SIZE: Maximum board dimensions
# - DIRECTIONS: Directions the units can move
# - RESOURCE_TYPES: Types of resources available in the game (wood, coal, uranium)
from ..utility_constants import MAX_CAPACITY, MAX_RESEARCH, MAX_BOARD_SIZE
from ..lux.constants import Constants
from ..lux.game import Game
from ..lux.game_objects import CityTile, Unit

# The maximum number of actions that can be taken by units sharing a single cell.
# If more than this number of units occupy the same cell, only this many can do
# something other than NO-OP; the rest automatically default to NO-OP.
MAX_OVERLAPPING_ACTIONS = 4

# Extract a tuple of possible directions (like N, E, S, W) from Constants.
DIRECTIONS = Constants.DIRECTIONS.astuple(include_center=False)

# Extract a tuple of resource types (like wood, coal, uranium).
RESOURCES = Constants.RESOURCE_TYPES.astuple()

# Define the set of all possible actions for each actor type: worker, cart, and city_tile.
# Workers and carts share many move/transfer actions, but workers can also pillage and build cities.
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

# Add move directions (e.g., MOVE_N, MOVE_S, etc.) and transfer actions (TRANSFER_resource_direction)
# to both "worker" and "cart" sets of actions.
for u in ["worker", "cart"]:
    for d in DIRECTIONS:
        ACTION_MEANINGS[u].append(f"MOVE_{d}")
    for r in RESOURCES:
        for d in DIRECTIONS:
            ACTION_MEANINGS[u].append(f"TRANSFER_{r}_{d}")

# Workers have two extra actions: PILLAGE and BUILD_CITY.
ACTION_MEANINGS["worker"].extend(["PILLAGE", "BUILD_CITY"])

# Create a dictionary mapping each actor type and action string to a unique index.
ACTION_MEANINGS_TO_IDX = {
    actor: {
        action: idx for idx, action in enumerate(actions)
    } for actor, actions in ACTION_MEANINGS.items()
}


def _no_op(game_object: Union[Unit, CityTile]) -> Optional[str]:
    """
    Return None, which signifies a "do nothing" action (NO-OP).
    For units, a None action is effectively a no-op in the game engine.
    """
    return None


def _pillage(worker: Unit) -> str:
    """
    Use the worker's built-in method to pillage (destroy a road to gain resources).
    Returns a command string recognized by the game engine.
    """
    return worker.pillage()


def _build_city(worker: Unit) -> str:
    """
    Use the worker's built-in method to build a city tile, if it has enough resources.
    Returns a command string recognized by the game engine.
    """
    return worker.build_city()


def _build_worker(city_tile: CityTile) -> str:
    """
    Build a new worker from a city tile, if permitted.
    Returns a command string recognized by the game engine.
    """
    return city_tile.build_worker()


def _build_cart(city_tile: CityTile) -> str:
    """
    Build a new cart from a city tile, if permitted.
    Returns a command string recognized by the game engine.
    """
    return city_tile.build_cart()


def _research(city_tile: CityTile) -> str:
    """
    Make the city tile perform a research action (increasing the player's research points).
    Returns a command string recognized by the game engine.
    """
    return city_tile.research()


def _move_factory(action_meaning: str) -> Callable[[Unit], str]:
    """
    Create and return a function that moves a unit in the specified direction.
    The direction is parsed from the action_meaning string (e.g., 'MOVE_N' -> 'N').
    """
    direction = action_meaning.split("_")[1]
    if direction not in DIRECTIONS:
        raise ValueError(f"Unrecognized direction '{direction}' in action_meaning '{action_meaning}'")

    def _move_func(unit: Unit) -> str:
        return unit.move(direction)

    return _move_func


def _transfer_factory(action_meaning: str) -> Callable[..., str]:
    """
    Create and return a function that transfers resources from one unit to another.
    The resource type and direction (e.g., 'TRANSFER_wood_N') is parsed from action_meaning.
    """
    resource, direction = action_meaning.split("_")[1:]
    if resource not in RESOURCES:
        raise ValueError(f"Unrecognized resource '{resource}' in action_meaning '{action_meaning}'")
    if direction not in DIRECTIONS:
        raise ValueError(f"Unrecognized direction '{direction}' in action_meaning '{action_meaning}'")

    def _transfer_func(unit: Unit, pos_to_unit_dict: Dict[Tuple, Optional[Unit]]) -> str:
        """
        - unit: The unit performing the transfer.
        - pos_to_unit_dict: A dictionary mapping (x, y) -> Unit to find the target unit.
        We always transfer the maximum capacity of the resource. If no valid target unit is found,
        or if it fails the checks, we return an empty string "" which the engine interprets as no-op.
        """
        dest_pos = unit.pos.translate(direction, 1)
        dest_unit = pos_to_unit_dict.get((dest_pos.x, dest_pos.y), None)
        # If the square is off-map or there is no allied unit in that square, we can't transfer.
        if dest_unit is None:
            return ""
        # We transfer the maximum capacity of the resource. Partial transfers are not handled here.
        return unit.transfer(dest_id=dest_unit.id, resourceType=resource, amount=MAX_CAPACITY)

    return _transfer_func


# Dictionary mapping actor type and action name -> the function to call to produce the command string.
# This is how we get from an "action index" to the actual environment move.
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

# For each direction and resource, we add the appropriate function (move or transfer) to
# both worker and cart. We do this after the dictionary is initially constructed.
for u in ["worker", "cart"]:
    for d in DIRECTIONS:
        a = f"MOVE_{d}"
        ACTION_MEANING_TO_FUNC[u][a] = _move_factory(a)
    for r in RESOURCES:
        for d in DIRECTIONS:
            actions_str = f"TRANSFER_{r}_{d}"
            ACTION_MEANING_TO_FUNC[u][actions_str] = _transfer_factory(actions_str)


class BaseActSpace(ABC):
    """
    Abstract base class defining the interface for an action space:
    1) get_action_space(...) -> Return the gym action space that is used to sample or define actions.
    2) process_actions(...) -> Convert raw action arrays into actual game moves.
    3) get_available_actions_mask(...) -> Return a mask indicating which actions are valid for each cell.
    4) actions_taken_to_distributions(...) -> Aggregates and summarizes how many of each action is taken.
    """

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
        """
        Convert the raw action_tensors_dict (which are NxN arrays of action indices) into actual commands
        the game engine recognizes. Also track which actions were taken for logging or analysis.
        Returns:
            - action_strs: a list of lists of command strings for each player.
            - actions_taken: a dictionary of boolean arrays marking which action indexes were actually used.
        """
        pass

    @abstractmethod
    def get_available_actions_mask(
            self,
            game_state: Game,
            board_dims: Tuple[int, int],
            pos_to_unit_dict: Dict[Tuple, Optional[Unit]],
            pos_to_city_tile_dict: Dict[Tuple, Optional[CityTile]]
    ) -> Dict[str, np.ndarray]:
        """
        Return a mask for each action type (worker, cart, city_tile) specifying which actions are valid
        at each board location (and player/team dimension) from the standpoint of game rules.
        """
        pass

    @staticmethod
    @abstractmethod
    def actions_taken_to_distributions(actions_taken: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
        """
        Given a dictionary of boolean arrays indicating which actions were taken, produce
        a summary dict that counts how many times each action was taken.
        """
        pass


class BasicActionSpace(BaseActSpace):
    """
    A concrete implementation of the BaseActSpace interface. It uses discrete actions per entity type
    (worker, cart, city_tile). Each discrete action corresponds to a string in ACTION_MEANINGS.

    - get_action_space(...) returns a gym.spaces.Dict, where each key is one of ("worker", "cart", "city_tile")
      and the space is a MultiDiscrete over the shape (1, p, x, y) with discrete dimension = number of possible actions.
    - process_actions(...) walks over each player's units and city tiles, looks up the chosen action for that cell,
      and constructs commands for the environment. It also enforces the MAX_OVERLAPPING_ACTIONS limit.
    - get_available_actions_mask(...) sets some actions to invalid based on game rules, e.g., not allowing
      a worker to move off the map or to transfer resources if no ally is present in that target tile.
    - actions_taken_to_distributions(...) aggregates booleans and sums them up.
    """

    def __init__(self, default_board_dims: Optional[Tuple[int, int]] = None):
        """
        If no board dimensions are provided, the class uses MAX_BOARD_SIZE as a default.
        """
        self.default_board_dims = MAX_BOARD_SIZE if default_board_dims is None else default_board_dims

    @lru_cache(maxsize=None)
    def get_action_space(self, board_dims: Optional[Tuple[int, int]] = None) -> gym.spaces.Dict:
        """
        Returns a dictionary containing the discrete action spaces for "worker", "cart", and "city_tile".
        The shape is (1, number_of_players, x, y), and the discrete dimension is the length of the action list
        for that entity type.
        """
        if board_dims is None:
            board_dims = self.default_board_dims
        x = board_dims[0]
        y = board_dims[1]
        # p = number of players (commonly 2 in this environment).
        p = 2

        spaces_dict = gym.spaces.Dict(
            {"worker": gym.spaces.MultiDiscrete(np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["worker"])),
             "cart": gym.spaces.MultiDiscrete(np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["cart"])),
             "city_tile": gym.spaces.MultiDiscrete(
                 np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["city_tile"])), })

        return spaces_dict

    @lru_cache(maxsize=None)
    def get_action_space_expanded_shape(self, *args, **kwargs) -> Dict[str, Tuple[int, ...]]:
        """
        Similar to get_action_space but returns the shapes with the trailing dimension
        representing the total possible actions. This is useful for storing booleans that mark
        which actions were taken or which are valid.
        """
        action_space = self.get_action_space(*args, **kwargs)
        action_space_expanded = {}
        for key, val in action_space.spaces.items():
            # The shape of val is (1, p, x, y). We add a trailing dimension for the # of actions.
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
        1. Iterate over all players.
        2. For each player's units (worker, cart), get the action index from the action_tensors_dict.
        3. Convert it to a command string using get_unit_action(...).
        4. Respect MAX_OVERLAPPING_ACTIONS so that no more than 4 non-NO-OP actions can occur on the same tile.
        5. For city tiles, do something similar, but there's no overlap limit for city tiles.
        6. Return a list of lists of action strings and a dictionary marking which actions were used.
        """

        # Initialize a container that collects the commands (strings) to be executed by each player.
        action_strs = [[], []]

        # Create a boolean array that marks which actions are taken. This uses the expanded shape.
        actions_taken = {
            key: np.zeros(space, dtype=bool) for key, space in self.get_action_space_expanded_shape(board_dims).items()
        }

        for player in game_state.players:
            p_id = player.team

            # These arrays track how many actions have already been selected by overlapping units of the same type.
            # For example, if two workers are on the same cell, the second worker's action is stored at index=1 in
            # the 5D array.
            worker_actions_taken_count = np.zeros(board_dims, dtype=int)
            cart_actions_taken_count = np.zeros_like(worker_actions_taken_count)

            # Process units
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

                    # This 'actor_count' is effectively the index for which plane in the action tensor
                    # is used for this particular overlapping unit.
                    actor_count = actions_taken_count[x, y]

                    # If we've already hit the overlap limit (MAX_OVERLAPPING_ACTIONS),
                    # we force NO-OP for this unit.
                    if actor_count >= MAX_OVERLAPPING_ACTIONS:
                        action = None
                    else:
                        # Retrieve the chosen action index from the precomputed action_tensors_dict.
                        action_idx = action_tensors_dict[unit_type][0, p_id, x, y, actor_count]
                        action_meaning = ACTION_MEANINGS[unit_type][action_idx]
                        # Convert the action index to an actual string command for the game engine.
                        action = get_unit_action(unit, action_idx, pos_to_unit_dict)

                        # Mark the action as taken only if it is valid (or if it is NO-OP).
                        action_was_taken = (action_meaning == "NO-OP") or (action is not None and action != "")
                        actions_taken[unit_type][0, p_id, x, y, action_idx] = action_was_taken

                        # If NO-OP was chosen, we skip further units on the same cell by adding MAX_OVERLAPPING_ACTIONS
                        # to the actor_count. This effectively ensures the rest of the units on this cell do NO-OP as well.
                        if action_meaning == "NO-OP":
                            actions_taken_count[x, y] += MAX_OVERLAPPING_ACTIONS

                    # None means no-op; "" means an invalid action (treated as no-op by the engine).
                    # If the action is valid, add it to the player's action queue.
                    if action is not None and action != "":
                        action_strs[p_id].append(action)

                    # Regardless of whether the action was valid, we increment the overlap counter.
                    actions_taken_count[x, y] += 1

            # Process city tiles
            for city in player.cities.values():
                for city_tile in city.citytiles:
                    if city_tile.can_act():
                        x, y = city_tile.pos.x, city_tile.pos.y
                        # Here we don't use overlapping logic; city tiles do not have the same limit as units.
                        action_idx = action_tensors_dict["city_tile"][0, p_id, x, y, 0]
                        action_meaning = ACTION_MEANINGS["city_tile"][action_idx]
                        action = get_city_tile_action(city_tile, action_idx)

                        # Mark the action as taken if valid.
                        action_was_taken = (action_meaning == "NO-OP") or (action is not None and action != "")
                        actions_taken["city_tile"][0, p_id, x, y, action_idx] = action_was_taken

                        # None means no-op, but if not None, we add it to the commands.
                        if action is not None:
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
        Compute a boolean mask that indicates which actions are valid for each cell, for each player, for each
        possible action. Many checks are performed here:
        - Movement checks (can't move off-board, can't move onto enemy city tile, can't move onto a
          unit with cooldown > 0, etc.)
        - Transfer checks (must be an allied unit in target tile, must have cargo to transfer, etc.)
        - Worker-specific checks (pillage if there's a road, can't be on allied city tile, etc.)
        - City tile checks (e.g., can only do RESEARCH if research_points < MAX_RESEARCH, can only build
          if number of units < number of city tiles)
        Returns a dictionary from actor type -> boolean array.
        """
        # Initialize everything to True; we will set to False if a rule makes the action impossible.
        available_actions_mask = {
            key: np.ones(space.shape + (len(ACTION_MEANINGS[key]),), dtype=bool)
            for key, space in self.get_action_space(board_dims).spaces.items()
        }

        for player in game_state.players:
            p_id = player.team

            # Check each unit
            for unit in player.units:
                if unit.can_act():
                    x, y = unit.pos.x, unit.pos.y
                    if unit.is_worker():
                        unit_type = "worker"
                    elif unit.is_cart():
                        unit_type = "cart"
                    else:
                        raise NotImplementedError(f"New unit type: {unit}")

                    # For each direction, see if movement/transfer is feasible.
                    for direction in DIRECTIONS:
                        new_pos_tuple = unit.pos.translate(direction, 1)
                        new_pos_tuple = (new_pos_tuple.x, new_pos_tuple.y)

                        # If new_pos_tuple is off the board or invalid, disable move and transfer.
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

                        # If the new position is an enemy city tile, can't move onto it.
                        new_pos_city_tile = pos_to_city_tile_dict[new_pos_tuple]
                        if new_pos_city_tile and new_pos_city_tile.team != p_id:
                            available_actions_mask[unit_type][
                            :,
                            p_id,
                            x,
                            y,
                            ACTION_MEANINGS_TO_IDX[unit_type][f"MOVE_{direction}"]
                            ] = False

                        # If the new position has a unit whose cooldown > 0, can't move onto it.
                        new_pos_unit = pos_to_unit_dict[new_pos_tuple]
                        if new_pos_unit and new_pos_unit.cooldown > 0:
                            available_actions_mask[unit_type][
                            :,
                            p_id,
                            x,
                            y,
                            ACTION_MEANINGS_TO_IDX[unit_type][f"MOVE_{direction}"]
                            ] = False

                        # Transfer checks
                        for resource in RESOURCES:
                            # For transfer to be valid:
                            # 1) There's an allied unit in target square.
                            # 2) The transferring unit has that resource in its cargo.
                            # 3) The receiving unit has cargo space available.
                            if (
                                    (new_pos_unit is None or new_pos_unit.team != p_id) or
                                    (unit.cargo.get(resource) <= 0) or
                                    (new_pos_unit.get_cargo_space_left() <= 0)
                            ):
                                available_actions_mask[unit_type][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX[unit_type][f"TRANSFER_{resource}_{direction}"]
                                ] = False

                    # Worker-only actions
                    if unit.is_worker():
                        # PILLAGE: only valid if on a road tile and not on your own city tile.
                        if game_state.map.get_cell_by_pos(unit.pos).road <= 0 or \
                                pos_to_city_tile_dict[(unit.pos.x, unit.pos.y)] is not None:
                            available_actions_mask[unit_type][
                            :,
                            p_id,
                            x,
                            y,
                            ACTION_MEANINGS_TO_IDX[unit_type]["PILLAGE"]
                            ] = False

                        # BUILD_CITY: only valid if the worker has enough resources and is not on a resource tile.
                        if not unit.can_build(game_state.map):
                            available_actions_mask[unit_type][
                            :,
                            p_id,
                            x,
                            y,
                            ACTION_MEANINGS_TO_IDX[unit_type]["BUILD_CITY"]
                            ] = False

            # Check each city tile
            for city in player.cities.values():
                for city_tile in city.citytiles:
                    if city_tile.can_act():
                        x, y = city_tile.pos.x, city_tile.pos.y
                        # RESEARCH is valid if player.research_points < MAX_RESEARCH
                        if player.research_points >= MAX_RESEARCH:
                            available_actions_mask["city_tile"][
                            :,
                            p_id,
                            x,
                            y,
                            ACTION_MEANINGS_TO_IDX["city_tile"]["RESEARCH"]
                            ] = False
                        # BUILD_WORKER or BUILD_CART is only valid if the total number of units < city_tile_count
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
        Given a dict of boolean arrays (indicating which actions were chosen for each discrete possibility),
        this method sums up how many times each action was taken across the board and returns a dictionary
        mapping actor type -> {action_name: count}.
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
    Helper function to convert an action index into an actual command string for a unit (worker or cart).
    If it's a transfer action, we call the specialized function requiring pos_to_unit_dict.
    Otherwise, we call the simpler function from ACTION_MEANING_TO_FUNC.
    """
    if unit.is_worker():
        unit_type = "worker"
    elif unit.is_cart():
        unit_type = "cart"
    else:
        raise NotImplementedError(f'New unit type: {unit}')

    action = ACTION_MEANINGS[unit_type][action_idx]
    if action.startswith("TRANSFER"):
        # For transfer actions, we need the second argument: pos_to_unit_dict
        return ACTION_MEANING_TO_FUNC[unit_type][action](unit, pos_to_unit_dict)
    else:
        return ACTION_MEANING_TO_FUNC[unit_type][action](unit)


def get_city_tile_action(city_tile: CityTile, action_idx: int) -> Optional[str]:
    """
    Helper function to convert an action index into an actual command string for a city tile.
    """
    action = ACTION_MEANINGS["city_tile"][action_idx]
    return ACTION_MEANING_TO_FUNC["city_tile"][action](city_tile)
