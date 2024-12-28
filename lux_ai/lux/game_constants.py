"""
Game constants loader for the Lux AI competition.

This module loads game configuration constants from game_constants.json, which defines:
- Unit types (WORKER, CART) and their properties
- Resource types (WOOD, COAL, URANIUM) and collection rates
- Movement directions (NORTH, SOUTH, EAST, WEST, CENTER)
- Game parameters including:
  - Day/night cycle lengths
  - Unit and city upkeep costs
  - Resource collection and fuel conversion rates
  - Research requirements
  - Action cooldowns
  - Road development mechanics
"""

import json
from os import path

# Get the directory containing this file
dir_path = path.dirname(__file__)

# Construct absolute path to the JSON constants file
constants_path = path.abspath(path.join(dir_path, "game_constants.json"))

# Load game constants from JSON file into a dictionary
with open(constants_path) as f:
    GAME_CONSTANTS = json.load(f)

# The GAME_CONSTANTS dictionary contains the following structure:
# {
#   "UNIT_TYPES": {"WORKER": 0, "CART": 1},
#   "RESOURCE_TYPES": {"WOOD": "wood", "COAL": "coal", "URANIUM": "uranium"},
#   "DIRECTIONS": {"NORTH": "n", "WEST": "w", "EAST": "e", "SOUTH": "s", "CENTER": "c"},
#   "PARAMETERS": {
#     "DAY_LENGTH": 30,              # Length of day phase in turns
#     "NIGHT_LENGTH": 10,            # Length of night phase in turns
#     "MAX_DAYS": 360,               # Maximum game length in days
#     "LIGHT_UPKEEP": {             # Resource upkeep costs during day
#       "CITY": 23,                  # Per city tile
#       "WORKER": 4,                 # Per worker unit
#       "CART": 10                   # Per cart unit
#     },
#     "WOOD_GROWTH_RATE": 1.025,     # Wood regeneration multiplier
#     "MAX_WOOD_AMOUNT": 500,        # Maximum wood per cell
#     "CITY_BUILD_COST": 100,        # Resource cost to build a city
#     "CITY_ADJACENCY_BONUS": 5,     # Bonus for adjacent city tiles
#     "RESOURCE_CAPACITY": {         # Max resource carrying capacity
#       "WORKER": 100,
#       "CART": 2000
#     },
#     "WORKER_COLLECTION_RATE": {    # Resources collected per action
#       "WOOD": 20,
#       "COAL": 5,
#       "URANIUM": 2
#     },
#     "RESOURCE_TO_FUEL_RATE": {     # Fuel value multipliers
#       "WOOD": 1,
#       "COAL": 10,
#       "URANIUM": 40
#     },
#     "RESEARCH_REQUIREMENTS": {     # Research points needed
#       "COAL": 50,                  # To collect coal
#       "URANIUM": 200               # To collect uranium
#     },
#     "CITY_ACTION_COOLDOWN": 10,    # Turns between city actions
#     "UNIT_ACTION_COOLDOWN": {      # Turns between unit actions
#       "CART": 3,
#       "WORKER": 2
#     },
#     "MAX_ROAD": 6,                 # Maximum road development level
#     "MIN_ROAD": 0,                 # Minimum road development level
#     "CART_ROAD_DEVELOPMENT_RATE": 0.75,  # Road improvement per cart move
#     "PILLAGE_RATE": 0.5           # Resource destruction rate when pillaging
#   }
# }
