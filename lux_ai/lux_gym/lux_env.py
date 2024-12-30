import copy
import gym
import itertools
import json
import numpy as np
import logging  # <-- Added for minimal logging
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

import pickle
import numpy as np

# Suppose `obs` is your dictionary of arrays (like 'city_tile', etc.)

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

def load_obs(filename="obs.pkl"):
    """
    Load the observation dict back from the pickle file.
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

# Example usage:

# If you already have a variable `obs` that has your arrays:
# obs = {...}

# Save it:
# save_obs(obs, "my_observations.pkl")

# Later on, or in a separate script, you can load it:
# loaded_obs = load_obs("my_observations.pkl")
# print(loaded_obs["city_tile"])  # or whichever key you'd like to inspect


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
        #logging.warning("[LuxEnv __init__] Initializing with run_game_automatically=%s, seed=%s",
        #                run_game_automatically, seed)

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
        #logging.warning("[LuxEnv _restart_dimension_process] Restarting dimension process.")
        if self._dimension_process is not None:
            self._dimension_process.kill()
        if self.run_game_automatically:
            logging.warning("[LuxEnv _restart_dimension_process] Launching Node.js process.")
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

    def reset(self, observation_updates: Optional[List[str]] = None) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        #logging.warning("[LuxEnv reset] Called with observation_updates=%s", observation_updates)
        self.game_state = Game()
        self.reset_count = (self.reset_count + 1) % self.restart_subproc_after_n_resets
        if self.reset_count == 0:
            #logging.warning("[LuxEnv reset] Reached subproc restart threshold, restarting.")
            self._restart_dimension_process()

        if self.run_game_automatically:
            assert observation_updates is None, "Game is being run automatically"
            # 1.2: Initialize a blank state game if new episode is starting
            self.configuration["seed"] += 1
            # logging.warning("[LuxEnv reset] Sending 'start' message with new seed=%d", self.configuration["seed"])

            initiate = {
                "type": "start",
                "agent_names": [],  # unsure if this is provided?
                "config": self.configuration
            }

            # logging.warning("JSON to initiate game: " + str(initiate))

            self._dimension_process.stdin.write((json.dumps(initiate) + "\n").encode())
            self._dimension_process.stdin.flush()
            agent1res = json.loads(self._dimension_process.stderr.readline())
            # Skip agent2res and match_obs_meta
            _ = self._dimension_process.stderr.readline(), self._dimension_process.stderr.readline()

            self.game_state._initialize(agent1res)
            self.game_state._update(agent1res[2:])
        else:
            assert observation_updates is not None, "Game is not being run automatically"
            #logging.warning("[LuxEnv reset] Manually initializing game from observation_updates.")
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

        game_state, rewards, done, info = self.get_obs_reward_done_info()
        #logging.warning("[LuxEnv reset] => returning (game_state, rewards, done, info) = (%s, %s, %s, %s)",
        #                game_state, rewards, done, info)
        return game_state, rewards, done, info

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        #logging.warning("[LuxEnv step] Called with action dict keys: %s", list(action.keys()))

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

        if self.run_game_automatically:
            # logging.warning("Unprocessed actions: " + str(action))

            # save_obs(action, "unprocessed_actions.pkl")

            actions_processed, actions_taken = self.process_actions(action)

            # logging.warning("[LuxEnv step] actions_processed=%s, actions_taken=%s", actions_processed, actions_taken)

            self._step(actions_processed)
            self.info["actions_taken"] = actions_taken
        self._update_internal_state()

        game_state, rewards, done, info = self.get_obs_reward_done_info()
        # logging.warning("[LuxEnv step] => returning (game_state, rewards, done, info) = (%s, %s, %s, %s)",
        #                 game_state, rewards, done, info)
        return game_state, rewards, done, info

    def manual_step(self, observation_updates: List[str]) -> NoReturn:
        #logging.warning("[LuxEnv manual_step] Called with updates=%s", observation_updates)
        assert not self.run_game_automatically
        self.game_state._update(observation_updates)

    def get_obs_reward_done_info(self) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        rewards = self.default_reward_space.compute_rewards(game_state=self.game_state, done=self.done)
        return self.game_state, rewards, self.done, copy.copy(self.info)

    def process_actions(self, action: Dict[str, np.ndarray]) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        #logging.warning("[LuxEnv process_actions] Processing action dict keys: %s", list(action.keys()))
        print(action["worker"].shape[2])
        print(action["worker"].shape[3])
        assert action["worker"].shape[2] == action["worker"].shape[3]
        assert action["cart"].shape[2] == action["cart"].shape[3]
        assert action["city_tile"].shape[2] == action["city_tile"].shape[3]

        return self.action_space.process_actions(
                action,
                self.game_state,
                self.board_dims,
                self.pos_to_unit_dict
            )

    def _step(self, action: List[List[str]]) -> NoReturn:
        #logging.warning("[LuxEnv _step] Called with action=%s", action)
        assert len(action) == 2
        state = [{'action': a} for a in action]

        # logging.warning("Sending actions to node engine: " + str(state))

        self._dimension_process.stdin.write((json.dumps(state) + "\n").encode())
        self._dimension_process.stdin.flush()

        agent1res = json.loads(self._dimension_process.stderr.readline())

        # logging.warning("Reading agent1res: " + str(agent1res))

        _ = self._dimension_process.stderr.readline(), self._dimension_process.stderr.readline()
        self.game_state._update(agent1res)

        match_status = json.loads(self._dimension_process.stderr.readline())
        self.done = match_status["status"] == "finished"

        while True:
            try:
                line = self._q.get_nowait()
            except Empty:
                break
            else:
                logging.warning("[LuxEnv _step] Dimension process stderr: %s", line.decode().strip())

    def _update_internal_state(self) -> NoReturn:
        #logging.warning("[LuxEnv _update_internal_state] Called.")
        self.pos_to_unit_dict = _generate_pos_to_unit_dict(self.game_state)
        self.pos_to_city_tile_dict = _generate_pos_to_city_tile_dict(self.game_state)
        self.info["available_actions_mask"] = self.action_space.get_available_actions_mask(
            self.game_state,
            self.board_dims,
            self.pos_to_unit_dict,
            self.pos_to_city_tile_dict
        )

    def seed(self, seed: Optional[int] = None) -> NoReturn:
        if seed is not None:
            self.configuration["seed"] = seed - 1
            #logging.warning("[LuxEnv seed] Setting seed to %d (will be incremented on reset)", seed)
        else:
            auto_seed = math.floor(random.random() * 1e9)
            self.configuration["seed"] = auto_seed
            #logging.warning("[LuxEnv seed] No seed provided, using random seed=%d", auto_seed)

    def get_seed(self) -> int:
        return self.configuration["seed"]

    def render(self, mode='human'):
        raise NotImplementedError('LuxEnv rendering is not implemented. Use the Lux visualizer instead.')
