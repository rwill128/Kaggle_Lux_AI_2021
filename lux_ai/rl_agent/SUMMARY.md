# rl_agent Directory Overview

## Purpose
This directory contains the core reinforcement learning agent implementation for the Lux AI competition winner. It implements the decision-making logic that interfaces between the neural network policy and the game environment, handling action selection and execution.

## Key Files and Classes
- `rl_agent.py`: Main agent implementation
  - `RLAgent` class: Core decision-making agent
  - Processes observations into network inputs
  - Handles action selection and execution
  - Manages multiple units with single network

- `data_augmentation.py`: Training data enhancement
  - Implements board rotations and flips
  - Handles symmetry-aware action mapping
  - Increases effective training data size

- `policy.py`: Policy implementation
  - Defines action sampling strategies
  - Implements exploration mechanisms
  - Handles action masking and validation

- `utils.py`: Helper functions
  - Observation preprocessing
  - Action space utilities
  - Training helper functions

## Role in RL Approach
This directory implements the core agent logic:

1. Decision Making
   - Single network controlling all units
   - Efficient parallel action computation
   - Sophisticated action masking
   - Handles unit coordination

2. Training Integration
   - Interfaces with IMPALA training
   - Implements UPGO and TD-lambda
   - Manages teacher model distillation
   - Handles reward processing

3. Policy Implementation
   - Progressive exploration reduction
   - Action sampling for stacked units
   - Legal move validation
   - Multi-unit coordination

The RLAgent implementation demonstrates how a single neural network can effectively control multiple units through careful action space design and sophisticated policy implementation. The data augmentation and policy components were crucial for efficient learning from self-play experience.
