# lux_ai Directory Overview

## Purpose
This directory contains the core AI implementation for the Lux AI Season 1 competition winner. It implements a deep reinforcement learning agent that uses a single neural network to control all units simultaneously, trained through self-play with sparse rewards.

## Key Files and Directories
- `lux/`: Core game logic and objects implementation
  - Handles game state, unit actions, and resource management
  - Provides the foundation for the reinforcement learning environment

- `lux_gym/`: Reinforcement learning environment wrapper
  - Implements the OpenAI Gym interface for the Lux game
  - Manages observation spaces, action spaces, and reward computation
  - Key class: `LuxEnv` for environment interaction

- `rl_agent/`: Reinforcement learning agent implementation
  - Contains the main agent logic and policy implementation
  - Handles action selection and execution
  - Key class: `RLAgent` for decision making

- `nns/`: Neural network architecture definitions
  - Implements the ResNet architecture with squeeze-excitation layers
  - Contains actor-critic network implementation
  - Handles input encoding and action/value output heads

- `torchbeast/`: Training framework implementation
  - Based on FAIR's IMPALA algorithm implementation
  - Manages distributed training with multiple actors
  - Implements UPGO and TD-lambda loss computations

## Role in RL Approach
This directory implements the winning approach which features:
1. Single neural network controlling all units (workers, carts, city tiles)
2. Pure self-play training with sparse rewards (-1 for losing, +1 for winning)
3. Progressive training phases:
   - Initial phase with shaped rewards for basic behaviors
   - Later phases with sparse rewards and teacher distillation
4. Sophisticated input encoding including:
   - Learnable embeddings for discrete features
   - Normalized continuous features
   - Game phase and day/night cycle encoding

The implementation demonstrates how a single network can learn complex cooperative behaviors through careful environment design and training progression.
