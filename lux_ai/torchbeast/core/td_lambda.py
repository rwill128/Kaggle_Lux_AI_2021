"""TD(λ) implementation for temporal credit assignment in reinforcement learning.

This module implements the TD(λ) algorithm, which provides a way to balance between:
1. TD(0): Pure temporal difference learning (λ=0)
2. Monte Carlo: Pure returns-based learning (λ=1)

The λ parameter controls this tradeoff:
- Lower λ: More bias, less variance, faster learning
- Higher λ: Less bias, more variance, slower but more accurate learning

Role in Lux AI Training:
1. Credit Assignment:
   - Helps assign credit for rewards to earlier actions
   - Particularly important for long-term strategy games
   - Balances immediate vs future reward attribution

2. Value Estimation:
   - Computes n-step returns with exponential weighting
   - Provides more stable value targets than pure TD(0)
   - Reduces variance compared to pure Monte Carlo

3. Advantage Computation:
   - Calculates advantages for policy gradient updates
   - Uses TD(λ) returns as more accurate value baselines
   - Helps with sparse reward scenarios
"""

import collections
import torch

TDLambdaReturns = collections.namedtuple("TDLambdaReturns", "vs advantages")



@torch.no_grad()
def td_lambda(
        rewards: torch.Tensor,
        values: torch.Tensor,
        bootstrap_value: torch.Tensor,
        discounts: torch.Tensor,
        lmb: float,
) -> TDLambdaReturns:
    """Compute TD(λ) returns and advantages for value-based learning.

    This function implements the TD(λ) algorithm which combines multiple n-step
    returns using exponential weighting controlled by λ. This provides a smooth
    interpolation between TD(0) and Monte Carlo returns.

    Args:
        rewards: Immediate rewards for each timestep [T, B, ...]
        values: Value estimates from critic network [T, B, ...]
        bootstrap_value: Value estimate for terminal state [B, ...]
        discounts: Discount factors for future rewards [T, B, ...]
        lmb: Lambda parameter controlling return mixing (0 ≤ λ ≤ 1)

    Returns:
        TDLambdaReturns containing:
        - vs: TD(λ) value targets [T, B, ...]
        - advantages: TD(λ) advantages [T, B, ...]

    Implementation Details:
    1. Forward View TD(λ):
       - Computes weighted average of n-step returns
       - Weight for n-step return = λ^(n-1)
       - Efficiently implemented through backward recursion

    2. Bootstrapping:
       - Uses critic value estimates for incomplete episodes
       - Combines with actual returns for completed episodes
       - Enables learning from partial trajectories

    3. Advantage Computation:
       - Subtracts critic values from TD(λ) returns
       - Provides centered learning signal for policy
       - Reduces variance in policy gradient updates

    Note:
        Uses @torch.no_grad() for efficiency since targets
        should not require gradients for optimization.
    """
    # Append bootstrapped value to get [v1, ..., v_t+1]
    # This creates a shifted sequence of values for computing TD errors
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
    )
    
    # Initialize target values with bootstrap value for terminal state
    target_values = [bootstrap_value]
    
    # Backward recursion to compute TD(λ) targets
    # This implements the forward view of TD(λ) through backward computation
    for t in range(discounts.shape[0] - 1, -1, -1):
        # noinspection PyUnresolvedReferences
        # Compute weighted combination of 1-step return and future λ-return:
        # - (1-λ) weight on 1-step return: rewards[t] + γ * V(s_{t+1})
        # - λ weight on future return: rewards[t] + γ * λ-return(s_{t+1})
        target_values.append(
            rewards[t] + discounts[t] * ((1 - lmb) * values_t_plus_1[t] + lmb * target_values[-1])
        )
    
    # Reverse to get targets in forward time order
    target_values.reverse()
    
    # Remove bootstrap value from end of target_values list
    # Final shape is [T, B, ...] matching input tensors
    target_values = torch.stack(target_values[:-1], dim=0)

    return TDLambdaReturns(
        vs=target_values,
        advantages=target_values - values
    )
