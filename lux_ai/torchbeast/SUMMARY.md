# torchbeast Directory Overview

## Purpose
This directory implements the distributed training framework based on FAIR's IMPALA algorithm. It manages the parallel actor-learner architecture that enabled efficient training of the Lux AI agent through self-play, implementing sophisticated loss computations including UPGO and TD-lambda.

## Key Files and Classes
- `monobeast.py`: Core training implementation
  - Implements IMPALA algorithm
  - Manages distributed actor-learner setup
  - Handles experience collection and learning
  - Coordinates multiple game environments

- `core/vtrace.py`: V-trace implementation
  - Off-policy correction calculations
  - Importance sampling ratios
  - Advantage estimation
  - Policy gradient computation

- `core/td_lambda.py`: TD(λ) implementation
  - Multi-step return calculation
  - Lambda-weighted return estimation
  - Value function learning
  - Advantage computation

- `core/upgo.py`: UPGO loss implementation
  - Upgoing policy gradient
  - Baseline value comparison
  - Policy improvement guarantees
  - Advantage clipping

## Role in RL Approach
The training framework was essential for:

1. Distributed Training
   - Multiple parallel actors
   - Efficient experience collection
   - Synchronized learning updates
   - Resource management

2. Loss Computation
   - IMPALA v-trace losses
   - UPGO policy gradients
   - TD(λ) value learning
   - Teacher model distillation

3. Training Stability
   - Off-policy correction
   - Advantage normalization
   - Gradient clipping
   - Experience replay

The sophisticated training framework enabled efficient learning from self-play, with careful implementation of multiple loss terms and distributed computation that was crucial for developing complex strategies within reasonable training time on personal hardware.
