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
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


def _generate_pos_to_unit_dict(game_state: Game) -> Dict[Tuple, Optional[Unit]]:
    pos_to_unit_dict = {(cell.pos.x, cell.pos.y): None for cell in itertools.chain(*game_state.map.map)}
    for player in game_state.players:
        for unit in reversed(player.units):
            pos_to_unit_dict[(unit.pos.x, unit.pos.y)] = unit

    return pos_to_unit_dict


def _generate_pos_to_city_tile_dict(game_state: Game) -> Dict[Tuple, Optional[CityTile]]:
    pos_to_city_tile_dict = {(cell.pos.x, cell.pos.y): None for cell in itertools.chain(*game_state.map.map)}
    for player in game_state.players:
        for city in player.cities.values():
            for city_tile in city.citytiles:
                pos_to_city_tile_dict[(city_tile.pos.x, city_tile.pos.y)] = city_tile

    return pos_to_city_tile_dict


# noinspection PyProtectedMember
class LuxEnv(gym.Env):
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
        Start or restart the Node.js dimension process used for game simulation.
        
        This process handles the actual game logic and state updates. We communicate
        with it via stdin/stdout to send actions and receive observations. The process
        is restarted periodically to prevent memory leaks.
        
        The process is started with:
        1. Pipes for stdin/stdout/stderr communication
        2. A background thread to handle stdout asynchronously
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
        if self.run_game_automatically:
            actions_processed, actions_taken = self.process_actions(action)
            self._step(actions_processed)
            self.info["actions_taken"] = actions_taken
        self._update_internal_state()

        return self.get_obs_reward_done_info()

    def manual_step(self, observation_updates: List[str]) -> NoReturn:
        assert not self.run_game_automatically
        self.game_state._update(observation_updates)

    def get_obs_reward_done_info(self) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        rewards = self.default_reward_space.compute_rewards(game_state=self.game_state, done=self.done)
        return self.game_state, rewards, self.done, copy.copy(self.info)

    def process_actions(self, action: Dict[str, np.ndarray]) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        """
        Convert network action outputs into game commands.
        
        Processes raw network outputs through the action space to generate valid
        game commands and track which actions were actually taken.
        
        Args:
            action: Dictionary mapping action types to numpy arrays of action logits
            
        Returns:
            Tuple containing:
            - List of game command strings for both players
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
        Execute actions in the game environment via the Node.js process.
        
        This internal method handles the communication protocol with the dimension
        process:
        1. Sends actions as JSON via stdin
        2. Receives and processes game state updates via stderr
        3. Checks for game completion
        4. Handles any error output from the process
        
        Args:
            action: List of action lists for both players (must be length 2)
        """
        # Send actions to dimension process
        assert len(action) == 2
        state = [{'action': a} for a in action]
        self._dimension_process.stdin.write((json.dumps(state) + "\n").encode())
        self._dimension_process.stdin.flush()

        # Receive and process game state updates
        agent1res = json.loads(self._dimension_process.stderr.readline())
        # Skip agent2res and match_obs_meta since we only need player 1's view
        _ = self._dimension_process.stderr.readline(), self._dimension_process.stderr.readline()
        self.game_state._update(agent1res)

        # Check if game is complete
        match_status = json.loads(self._dimension_process.stderr.readline())
        self.done = match_status["status"] == "finished"

        # Process any error output from dimension process
        while True:
            try:
                line = self._q.get_nowait()
            except Empty:
                # No more error output to process
                break
            else:
                # Print error output to stderr
                print(line.decode(), file=sys.stderr, end='')

    def _update_internal_state(self) -> NoReturn:
        """
        Update internal state tracking after game state changes.
        
        This includes:
        1. Regenerating position-to-unit mapping
        2. Regenerating position-to-city mapping
        3. Updating available action masks based on new game state
        
        These updates ensure that action processing and validation use
        the latest game state information.
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
        
        The seed is decremented by 1 because it will be incremented in reset().
        If no seed is provided, generates a random seed.
        
        Args:
            seed: Integer seed for reproducibility (optional)
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
            Current seed value from configuration
        """
        return self.configuration["seed"]

    def render(self, mode='human'):
        """
        Render the environment for visualization.
        
        This method is not implemented as the Lux AI competition provides
        its own visualizer for replays.
        
        Args:
            mode: Rendering mode (unused)
            
        Raises:
            NotImplementedError: Always raises this error
        """
        raise NotImplementedError('LuxEnv rendering is not implemented. Use the Lux visualizer instead.')
