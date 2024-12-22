"""
OpenAI Gym environment for the Lux AI competition.

This module implements a Gym environment that interfaces with the Lux AI game engine,
providing a standardized interface for reinforcement learning agents to interact with
the game. The environment handles:

1. Game state management and observation processing
2. Action validation and execution
3. Reward computation
4. Communication with the game engine via Node.js subprocess

Key Features:
- Implements standard OpenAI Gym interface (reset, step, etc.)
- Supports both automatic and manual game execution modes
- Handles unit and city tile position tracking
- Provides action masks for valid moves
- Manages subprocess communication with game engine
"""

import copy
import gym
import itertools
import json
import numpy as np
from kaggle_environments import make
import math
from pathlib import Path
from queue import Queue, Empty
import random
from subprocess import Popen, PIPE
import sys
from threading import Thread
from typing import Any, Dict, List, NoReturn, Optional, Tuple

from ..lux.game import Game
from ..lux.game_objects import Unit, CityTile
from ..lux_gym.act_spaces import BaseActSpace, ACTION_MEANINGS
from ..lux_gym.obs_spaces import BaseObsSpace
from ..lux_gym.reward_spaces import GameResultReward
from ..utility_constants import MAX_BOARD_SIZE

# In case dir_path is removed in production environment
try:
    from kaggle_environments.envs.lux_ai_2021.lux_ai_2021 import dir_path as DIR_PATH
except Exception:
    DIR_PATH = None


"""
def _cleanup_dimensions_factory(dimension_process: Popen) -> NoReturn:
    def cleanup_dimensions():
        if dimension_process is not None:
            dimension_process.kill()
    return cleanup_dimensions
"""


def _enqueue_output(out, queue):
    """
    Helper function to asynchronously read subprocess output.
    
    This function runs in a separate thread to continuously read output from the
    Node.js game engine subprocess and put it into a queue for processing.
    
    Args:
        out: Subprocess output stream to read from
        queue: Queue to store the output lines
    """
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


def _generate_pos_to_unit_dict(game_state: Game) -> Dict[Tuple, Optional[Unit]]:
    """
    Creates a mapping from board positions to units occupying them.
    
    This function generates a dictionary that maps each position on the game board
    to either None (if empty) or the Unit object at that position. This enables
    efficient unit lookup during action processing and collision detection.
    
    Args:
        game_state: Current game state containing map and unit information
        
    Returns:
        Dictionary mapping (x,y) position tuples to Unit objects or None
        
    Note:
        Units are processed in reverse order to ensure the latest unit at each
        position is stored in case of overlapping positions.
    """
    pos_to_unit_dict = {(cell.pos.x, cell.pos.y): None for cell in itertools.chain(*game_state.map.map)}
    for player in game_state.players:
        for unit in reversed(player.units):
            pos_to_unit_dict[(unit.pos.x, unit.pos.y)] = unit

    return pos_to_unit_dict


def _generate_pos_to_city_tile_dict(game_state: Game) -> Dict[Tuple, Optional[CityTile]]:
    """
    Creates a mapping from board positions to city tiles occupying them.
    
    Similar to _generate_pos_to_unit_dict, this function creates a dictionary
    mapping each board position to either None or the CityTile at that position.
    This enables efficient city tile lookup during action processing.
    
    Args:
        game_state: Current game state containing map and city information
        
    Returns:
        Dictionary mapping (x,y) position tuples to CityTile objects or None
    """
    pos_to_city_tile_dict = {(cell.pos.x, cell.pos.y): None for cell in itertools.chain(*game_state.map.map)}
    for player in game_state.players:
        for city in player.cities.values():
            for city_tile in city.citytiles:
                pos_to_city_tile_dict[(city_tile.pos.x, city_tile.pos.y)] = city_tile

    return pos_to_city_tile_dict


# noinspection PyProtectedMember
class LuxEnv(gym.Env):
    """
    OpenAI Gym environment for the Lux AI competition.
    
    This environment provides a standardized interface for reinforcement learning
    agents to interact with the Lux AI game. It handles:
    1. Game state management and observation processing
    2. Action validation and execution
    3. Reward computation
    4. Communication with the game engine
    
    The environment can run in two modes:
    - Automatic: Manages the game engine subprocess internally
    - Manual: Receives game state updates externally
    
    Key Features:
    - Custom observation and action spaces
    - Action masking for valid moves
    - Efficient unit and city tile position tracking
    - Automatic subprocess management with memory leak prevention
    """
    metadata = {"render.modes": []}

    def __init__(
            self,
            act_space: BaseActSpace,
            obs_space: BaseObsSpace,
            configuration: Optional[Dict[str, Any]] = None,
            seed: Optional[int] = None,
            run_game_automatically: bool = True,
            restart_subproc_after_n_resets: int = 100
    ):
        """
        Initialize the Lux AI environment.
        
        This constructor sets up the environment with custom observation and action
        spaces, and optionally initializes the game engine subprocess for automatic
        execution.
        
        Args:
            act_space: Custom action space implementation defining valid actions
            obs_space: Custom observation space implementation for state representation
            configuration: Optional game configuration dictionary
            seed: Optional random seed for reproducibility
            run_game_automatically: Whether to manage game execution internally
            restart_subproc_after_n_resets: Number of episodes before subprocess restart
                                          (helps prevent memory leaks)
        
        The environment can run in two modes:
        - Automatic: Manages game engine subprocess internally (run_game_automatically=True)
        - Manual: Receives game state updates externally (run_game_automatically=False)
        """
        super(LuxEnv, self).__init__()
        self.obs_space = obs_space
        self.action_space = act_space
        self.default_reward_space = GameResultReward()
        self.observation_space = self.obs_space.get_obs_spec()
        self.board_dims = MAX_BOARD_SIZE
        self.run_game_automatically = run_game_automatically
        self.restart_subproc_after_n_resets = restart_subproc_after_n_resets

        self.game_state = Game()
        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = make("lux_ai_2021").configuration
            # 2: warnings, 1: errors, 0: none
            self.configuration["loglevel"] = 0
        if seed is not None:
            self.seed(seed)
        elif "seed" not in self.configuration:
            self.seed()
        self.done = False
        self.info = {}
        self.pos_to_unit_dict = dict()
        self.pos_to_city_tile_dict = dict()
        self.reset_count = 0

        self._dimension_process = None
        self._q = None
        self._t = None
        self._restart_dimension_process()

    def _restart_dimension_process(self) -> NoReturn:
        """
        Restarts the Node.js game engine subprocess.
        
        This method:
        1. Kills any existing subprocess
        2. Starts a new Node.js process for the game engine
        3. Sets up asynchronous output handling
        
        The subprocess is restarted periodically to prevent memory leaks in the
        Node.js game engine. This is managed by tracking reset counts and
        restarting after restart_subproc_after_n_resets episodes.
        """
        if self._dimension_process is not None:
            self._dimension_process.kill()
        if self.run_game_automatically:
            # 1.1: Initialize dimensions in the background
            self._dimension_process = Popen(
                ["node", str(Path(DIR_PATH) / "dimensions/main.js")],
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE
            )
            self._q = Queue()
            self._t = Thread(target=_enqueue_output, args=(self._dimension_process.stdout, self._q))
            self._t.daemon = True
            self._t.start()
            # atexit.register(_cleanup_dimensions_factory(self._dimension_process))

    def reset(self, observation_updates: Optional[List[str]] = None) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        """
        Reset the environment to start a new episode.
        
        This method:
        1. Creates a new game state
        2. Initializes/updates the game engine subprocess if running automatically
        3. Updates observation and action spaces for the new board dimensions
        4. Resets internal state tracking (units, city tiles, etc.)
        
        Args:
            observation_updates: Optional list of observation strings for manual mode
            
        Returns:
            Tuple containing:
            - Game state object
            - Tuple of (reward_player_0, reward_player_1)
            - Done flag
            - Info dictionary with action masks and other metadata
            
        The method handles both automatic and manual execution modes through the
        observation_updates parameter.
        """
        self.game_state = Game()
        self.reset_count = (self.reset_count + 1) % self.restart_subproc_after_n_resets
        # There seems to be a gradual memory leak somewhere, so we restart the dimension process every once in a while
        if self.reset_count == 0:
            self._restart_dimension_process()
        if self.run_game_automatically:
            assert observation_updates is None, "Game is being run automatically"
            # 1.2: Initialize a blank state game if new episode is starting
            self.configuration["seed"] += 1
            initiate = {
                "type": "start",
                "agent_names": [],  # unsure if this is provided?
                "config": self.configuration
            }
            self._dimension_process.stdin.write((json.dumps(initiate) + "\n").encode())
            self._dimension_process.stdin.flush()
            agent1res = json.loads(self._dimension_process.stderr.readline())
            # Skip agent2res and match_obs_meta
            _ = self._dimension_process.stderr.readline(), self._dimension_process.stderr.readline()

            self.game_state._initialize(agent1res)
            self.game_state._update(agent1res[2:])
        else:
            assert observation_updates is not None, "Game is not being run automatically"
            self.game_state._initialize(observation_updates)
            self.game_state._update(observation_updates[2:])

        self.done = False
        self.board_dims = (self.game_state.map_width, self.game_state.map_height)
        self.observation_space = self.obs_space.get_obs_spec(self.board_dims)
        self.info = {
            "actions_taken": {
                key: np.zeros(space.shape + (len(ACTION_MEANINGS[key]),), dtype=bool)
                for key, space in self.action_space.get_action_space(self.board_dims).spaces.items()
            },
            "available_actions_mask": {
                key: np.ones(space.shape + (len(ACTION_MEANINGS[key]),), dtype=bool)
                for key, space in self.action_space.get_action_space(self.board_dims).spaces.items()
            }
        }
        self._update_internal_state()

        return self.get_obs_reward_done_info()

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        """
        Execute one timestep within the environment.
        
        This method:
        1. Processes the agent's actions into game commands
        2. Executes actions in the game engine
        3. Updates internal state tracking
        4. Returns the new state, rewards, done flag, and info
        
        Args:
            action: Dictionary mapping action types to numpy arrays of action indices
            
        Returns:
            Tuple containing:
            - Game state object
            - Tuple of (reward_player_0, reward_player_1)
            - Done flag
            - Info dictionary with action masks and other metadata
            
        The method handles action processing and game state updates, managing the
        interaction between the RL agent and game engine.
        """
        if self.run_game_automatically:
            actions_processed, actions_taken = self.process_actions(action)
            self._step(actions_processed)
            self.info["actions_taken"] = actions_taken
        self._update_internal_state()

        return self.get_obs_reward_done_info()

    def manual_step(self, observation_updates: List[str]) -> NoReturn:
        """
        Update game state from external observations in manual mode.
        
        Args:
            observation_updates: List of observation strings from external game engine
            
        This method is used when run_game_automatically=False to update the
        environment's state based on externally provided game observations.
        """
        assert not self.run_game_automatically
        self.game_state._update(observation_updates)

    def get_obs_reward_done_info(self) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        """
        Get the current environment state, rewards, done flag, and info.
        
        Returns:
            Tuple containing:
            - Game state object
            - Tuple of (reward_player_0, reward_player_1)
            - Done flag
            - Info dictionary with action masks and other metadata
            
        This method computes rewards using the default reward space and returns
        the current environment state information.
        """
        rewards = self.default_reward_space.compute_rewards(game_state=self.game_state, done=self.done)
        return self.game_state, rewards, self.done, copy.copy(self.info)

    def process_actions(self, action: Dict[str, np.ndarray]) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        """
        Convert agent actions into game engine commands.
        
        This method uses the action space to convert the agent's action indices
        into valid game commands, handling action validation and formatting.
        
        Args:
            action: Dictionary mapping action types to numpy arrays of action indices
            
        Returns:
            Tuple containing:
            - List of processed action commands for each player
            - Dictionary tracking which actions were actually taken
        """
        return self.action_space.process_actions(
            action,
            self.game_state,
            self.board_dims,
            self.pos_to_unit_dict
        )

    def _step(self, action: List[List[str]]) -> NoReturn:
        """
        Execute actions in the game engine subprocess.
        
        This internal method:
        1. Sends actions to the Node.js game engine
        2. Receives and processes the resulting observations
        3. Updates game state and checks for episode completion
        4. Handles subprocess output and error logging
        
        Args:
            action: List of action command lists for each player
            
        The method manages the low-level interaction with the game engine
        subprocess, ensuring proper state synchronization.
        """
        # 2.: Pass in actions (json representation along with id of who made that action),
        #       and agent information (id, status) to dimensions via stdin
        assert len(action) == 2
        # TODO: Does dimension process state need to include info other than actions?
        state = [{'action': a} for a in action]
        self._dimension_process.stdin.write((json.dumps(state) + "\n").encode())
        self._dimension_process.stdin.flush()

        # 3.1 : Receive and parse the observations returned by dimensions via stdout
        agent1res = json.loads(self._dimension_process.stderr.readline())
        # Skip agent2res and match_obs_meta
        _ = self._dimension_process.stderr.readline(), self._dimension_process.stderr.readline()
        self.game_state._update(agent1res)

        # Check if done
        match_status = json.loads(self._dimension_process.stderr.readline())
        self.done = match_status["status"] == "finished"

        while True:
            try:
                line = self._q.get_nowait()
            except Empty:
                # no standard error received, break
                break
            else:
                # standard error output received, print it out
                print(line.decode(), file=sys.stderr, end='')

    def _update_internal_state(self) -> NoReturn:
        """
        Update internal state tracking after state changes.
        
        This method:
        1. Updates unit position mapping
        2. Updates city tile position mapping
        3. Updates available action masks
        
        The internal state tracking enables efficient action processing and
        validation by maintaining quick lookup structures for game objects.
        """
        self.pos_to_unit_dict = _generate_pos_to_unit_dict(self.game_state)
        self.pos_to_city_tile_dict = _generate_pos_to_city_tile_dict(self.game_state)
        self.info["available_actions_mask"] = self.action_space.get_available_actions_mask(
            self.game_state,
            self.board_dims,
            self.pos_to_unit_dict,
            self.pos_to_city_tile_dict
        )

    def seed(self, seed: Optional[int] = None) -> NoReturn:
        """
        Set the random seed for the environment.
        
        Args:
            seed: Optional integer seed value. If None, generates random seed.
            
        Note:
            The seed is decremented by 1 since it's incremented on reset().
            This ensures the first episode uses exactly the provided seed.
        """
        if seed is not None:
            # Seed is incremented on reset()
            self.configuration["seed"] = seed - 1
        else:
            self.configuration["seed"] = math.floor(random.random() * 1e9)

    def get_seed(self) -> int:
        """
        Get the current random seed.
        
        Returns:
            Current seed value from the configuration
        """
        return self.configuration["seed"]

    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode (only 'human' supported)
            
        Raises:
            NotImplementedError: This environment doesn't implement rendering
            
        Note: Use the Lux visualizer for game visualization instead.
        """
        raise NotImplementedError('LuxEnv rendering is not implemented. Use the Lux visualizer instead.')
