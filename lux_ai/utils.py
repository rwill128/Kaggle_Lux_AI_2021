import sys
import time
import torch
from types import SimpleNamespace
from typing import Any, Dict, List, NoReturn, Tuple

from .lux.game_map import Position
from .lux_gym import ACT_SPACES_DICT, OBS_SPACES_DICT, REWARD_SPACES_DICT
from .utility_constants import LOCAL_EVAL


def flags_to_namespace(flags: Dict) -> SimpleNamespace:
    flags = SimpleNamespace(**flags)

    # Env params
    flags.act_space = ACT_SPACES_DICT[flags.act_space]
    flags.obs_space = OBS_SPACES_DICT[flags.obs_space]
    flags.reward_space = REWARD_SPACES_DICT[flags.reward_space]

    # Optimizer params
    flags.optimizer_class = torch.optim.__dict__[flags.optimizer_class]

    # Miscellaneous params
    flags.actor_device = torch.device(flags.actor_device)
    flags.learner_device = torch.device(flags.learner_device)

    return flags


def in_bounds(pos: Position, board_dims: Tuple[int, int]) -> bool:
    return 0 <= pos.x < board_dims[0] and 0 <= pos.y < board_dims[1]


def DEBUG_MESSAGE(msg: Any) -> NoReturn:
    """Print a debug message to stderr.

    Args:
        msg: Message to print (converted to string)
    """
    print(str(msg), file=sys.stderr)


def RUNTIME_DEBUG_MESSAGE(msg: Any) -> NoReturn:
    """Print a debug message only during competition evaluation.

    Used for logging that should only appear in competition logs but not
    during local testing.

    Args:
        msg: Message to print (converted to string)
    """
    if not LOCAL_EVAL:
        DEBUG_MESSAGE(str(msg))


def RUNTIME_ASSERT(statement: bool, msg: Any = "") -> NoReturn:
    """Assert a condition with environment-specific error handling.

    During local evaluation, raises a RuntimeError if the assertion fails.
    During competition evaluation, prints the error message to debug logs
    but allows execution to continue.

    This allows for development-time assertions that won't crash the agent
    during competition matches.

    Args:
        statement: Condition to assert
        msg: Optional error message to display on failure
    
    Raises:
        RuntimeError: If statement is False and in local evaluation mode
    """
    if statement:
        return

    msg = str(msg)
    if LOCAL_EVAL:
        raise RuntimeError(msg)
    else:
        DEBUG_MESSAGE(msg)


class Stopwatch:
    """
    Used to time function calls
    """

    def __init__(self):
        self.durations: Dict[str, Dict] = {}
        self._active_keys: List[str] = []
        self._start_times: List[float] = []

    def __str__(self):
        timing_info = " - ".join([f"{key}: {val['duration']:.2f}" for key, val in self.durations.items()])
        return f"Timing info: {{{timing_info}}}"

    def start(self, key: str):
        self._active_keys.append(key)
        self._start_times.append(time.time())
        current = self.durations
        for key in self._active_keys:
            entry = current.get(key)
            if entry is None:
                entry = {"duration": 0}
                current[key] = entry
            current = entry
        return self

    def stop(self):
        # Get active entry
        current = self.durations
        entry = None
        for key in self._active_keys:
            entry = current.get(key)
            current = entry
        # If there are no entries
        if entry is None and len(self._active_keys) == 0:
            return self

        # Compute time taken
        old_time = entry["duration"]
        diff = time.time() - self._start_times.pop()
        entry["duration"] = old_time + diff
        self._active_keys.pop()
        return self

    def reset(self):
        self.durations = {}
        self._active_keys = []
        self._start_times = []
