# lux_gym Directory Overview

## Purpose
This directory implements the OpenAI Gym environment wrapper for the Lux AI competition. It provides a standardized interface between the game logic and the reinforcement learning agent, handling observation space encoding, action space definition, and reward computation.

## Key Files and Classes
- `lux_env.py`: Core environment implementation
  - `LuxEnv` class: Main environment interface
  - Implements OpenAI Gym interface (reset, step, render)
  - Handles observation encoding and reward computation
  - Manages episode termination conditions

- `act_spaces.py`: Action space definitions
  - Defines discrete action spaces for units and cities
  - Implements action masking for illegal moves
  - Handles action sampling and validation

- `obs_spaces.py`: Observation space implementations
  - Encodes game state into neural network inputs
  - Implements different observation space versions
  - Handles feature normalization and embedding

- `reward_spaces.py`: Reward function definitions
  - Implements different reward schemes:
    - Shaped rewards for basic behaviors
    - Sparse win/loss rewards
    - Multi-component reward functions
  - Manages reward scaling and normalization

## Role in RL Approach
This directory is crucial for effective training:

1. State Representation
   - Converts raw game state to learnable features
   - Implements sophisticated input encoding:
     - Learnable embeddings for discrete features
     - Normalized continuous features
     - Game phase and day/night cycle encoding

2. Action Interface
   - Single network controlling multiple units
   - Masked invalid actions to ensure legal moves
   - Efficient action sampling for stacked units

3. Reward Design
   - Progressive reward shaping:
     - Initial phase: shaped rewards for basic behaviors
     - Later phases: pure win/loss signals
   - Reward normalization for stable training

The clean Gym interface and carefully designed spaces were key to successful training, allowing the agent to learn complex strategies through self-play while maintaining stable optimization.
