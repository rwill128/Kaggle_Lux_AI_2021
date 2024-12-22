# lux Directory Overview

## Purpose
This directory implements the core game logic and objects for the Lux AI competition. It provides the fundamental building blocks that the reinforcement learning environment uses to simulate and interact with the game world.

## Key Files and Classes
- `game.py`: Core game state management
  - `Game` class: Manages the overall game state
  - Handles turn processing and game rules
  - Maintains player states and game configuration

- `game_map.py`: Map and resource management
  - `GameMap` class: Represents the game board
  - Handles resource placement and visibility
  - Manages cell states and terrain

- `game_objects.py`: Game entity definitions
  - `Unit` class: Base class for workers and carts
  - `CityTile` class: Represents city components
  - Implements game object behaviors and properties

- `game_constants.py`: Game rules and parameters
  - Defines resource values
  - Sets game constants (day/night cycle, etc.)
  - Configures action costs and limitations

## Role in RL Approach
This directory provides the foundation that enables:

1. State Representation
   - Structured game state for neural network input
   - Resource tracking for strategic decisions
   - Unit and city management interfaces

2. Action Execution
   - Validates and processes unit actions
   - Handles resource collection and city building
   - Manages unit movement and collisions

3. Game Rules Enforcement
   - Day/night cycle mechanics
   - Resource collection rules
   - City fuel consumption
   - Research progression

The clean separation between game logic and RL components allows the agent to focus on learning strategies while the game rules are consistently enforced. This modular design was crucial for efficient training and experimentation with different reward structures.
