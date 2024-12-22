# nns Directory Overview

## Purpose
This directory implements the neural network architectures used in the winning Lux AI solution. It defines a sophisticated ResNet with squeeze-excitation layers that processes game state observations and outputs both policy (actor) and value (critic) predictions.

## Key Files and Classes
- `models.py`: Core network architecture
  - ResNet implementation with 24 residual blocks
  - Squeeze-excitation layers for attention
  - Actor-critic output heads
  - ~20 million parameters total

- `in_blocks.py`: Input processing layers
  - Learnable embeddings for discrete features
  - Input normalization for continuous features
  - 1x1 convolutions for feature projection
  - Handles 32x32 padded board inputs

- `attn_blocks.py`: Attention mechanisms
  - Squeeze-excitation block implementation
  - Channel-wise attention computation
  - Feature recalibration logic

- `out_blocks.py`: Output head implementations
  - Three actor heads (workers, carts, cities)
  - One critic head for value estimation
  - Action logit computation
  - Value prediction scaling

## Role in RL Approach
The neural network architecture was crucial for:

1. Model Design
   - Fully convolutional architecture
   - 128-channel 5x5 convolutions
   - No normalization layers
   - Progressive capacity scaling:
     - 8 blocks → 16 blocks → 24 blocks

2. Feature Processing
   - 32-dimensional embeddings
   - Masked convolutions
   - Sophisticated attention mechanisms
   - Efficient parallel computation

3. Output Generation
   - Multi-head action outputs
   - Value estimation for TD learning
   - Action masking integration
   - Policy stabilization support

The careful architecture design, with its balance of capacity and efficiency, was key to learning complex strategies through self-play while maintaining stable training dynamics.
