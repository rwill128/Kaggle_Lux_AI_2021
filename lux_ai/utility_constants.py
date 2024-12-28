"""Game constants and configuration values for the Lux AI Challenge.

This module provides easy access to important game parameters and derived constants
used throughout the codebase. It includes:
- Day/night cycle lengths
- Resource collection and capacity limits
- Research requirements
- Map size configurations
- Environment-specific behavior flags

Some constants are taken directly from game parameters while others are derived
or sourced from the official game implementation.
"""

import getpass

from .lux.constants import Constants
from .lux.game_constants import GAME_CONSTANTS

# Environment configuration
# Controls whether we're running in local development or competition mode
# This affects debug behavior and assertions
USER = getpass.getuser()
LOCAL_EVAL = USER in ['isaiah']

# Game cycle constants
# These define the fundamental day/night mechanics that affect unit behavior
DAY_LEN = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]      # Length of daytime phase
NIGHT_LEN = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]   # Length of nighttime phase
COLLECTION_RATES = GAME_CONSTANTS["PARAMETERS"]["WORKER_COLLECTION_RATE"]  # Resource gathering speeds

# Derived game cycle values
DN_CYCLE_LEN = DAY_LEN + NIGHT_LEN  # Total length of a full day/night cycle

# Resource and research limits
# Maximum values that constrain resource gathering and technology advancement
MAX_CAPACITY = max(GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"].values())  # Max unit cargo
MAX_RESEARCH = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())  # Highest research level

# Resource generation limits
# Maximum amount of each resource type that can exist in a single cell
# Wood amount from game parameters, Coal/Uranium from game implementation
MAX_RESOURCE = {
    Constants.RESOURCE_TYPES.WOOD: GAME_CONSTANTS["PARAMETERS"]["MAX_WOOD_AMOUNT"],
    # Coal amount from: https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L253
    Constants.RESOURCE_TYPES.COAL: 425.,
    # Uranium amount from: https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L269
    Constants.RESOURCE_TYPES.URANIUM: 350.
}

# Map configuration
# Valid map sizes and maximum dimensions used for observation space normalization
MAP_SIZES = ((12, 12), (16, 16), (24, 24), (32, 32))  # All possible map dimensions
MAX_BOARD_SIZE = (32, 32)  # Largest possible map size, used for tensor dimensions
