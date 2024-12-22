# conf Directory Overview

## Purpose
This directory contains the configuration files that define the training progression and hyperparameters for the winning Lux AI agent. The configurations showcase the evolution from shaped rewards to sparse rewards across multiple training phases.

## Key Files
- `conv_phase1_shaped_reward.yaml`: Initial training phase configuration
  - Uses shaped rewards to encourage basic behaviors
  - Smaller network (8 blocks) for faster initial training
  - Higher entropy cost (0.001) to encourage exploration
  - Step-based rewards to encourage activity
  - Uses BasicActionSpace and FixedShapeContinuousObsV2

- `conv_phase5+_final_model.yaml`: Final training phase configuration
  - Pure win/loss rewards (GameResultReward)
  - Larger network (24 blocks) for increased capacity
  - Lower entropy cost (0.0002) for more focused behavior
  - Teacher model KL divergence (0.005) for stability
  - Uses previous phase models as teachers

## Role in RL Approach
The configurations demonstrate the progressive training strategy:
1. Start with shaped rewards to learn basic behaviors
   - City building rewards
   - Unit creation rewards
   - Fuel management rewards
   - Step-based rewards for activity

2. Transition to sparse rewards for advanced strategy
   - Pure win/loss signals (-1/+1)
   - Knowledge distillation from previous models
   - Increased model capacity
   - Fine-tuned hyperparameters

This progression allowed the agent to first learn fundamental gameplay mechanics before developing sophisticated long-term strategies through self-play.

## Training Parameters Evolution
- Network size: 8 blocks → 24 blocks
- Learning rate: 1e-4 → 5e-5
- Entropy cost: 0.001 → 0.0002
- Teacher KL cost: 0.0 → 0.005
- Reward shaping: StatefulMultiReward → GameResultReward
- Lambda (TD/UPGO): 0.8 → 0.9

The configurations show how the training process was carefully tuned to balance exploration, stability, and strategic depth.
