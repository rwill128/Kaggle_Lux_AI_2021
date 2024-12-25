"""
Upgoing Policy Gradient (UPGO) Implementation

This module implements the Upgoing Policy Gradient algorithm, which provides an improved
advantage estimation method for policy gradient algorithms. UPGO combines the benefits of:
1. TD(λ) returns for stable value estimation
2. Q(λ) for better credit assignment
3. V-trace for off-policy correction

The key insight of UPGO is to use the maximum between the value estimate and a mixed
target that combines immediate rewards with bootstrapped values. This helps reduce
variance while maintaining a reasonable bias level in the advantage estimates.

References:
- "Policy Gradient Search: Online Planning and Expert Iteration without Search Trees"
  (Guo et al., 2014)
- IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
"""

import collections
import torch

UPGOReturns = collections.namedtuple("UPGOReturns", "vs advantages")
"""
Named tuple for storing UPGO return values.

Fields:
    vs (torch.Tensor): Target values computed using the UPGO estimator
    advantages (torch.Tensor): Advantage estimates (target_values - baseline_values)
"""


@torch.no_grad()
def upgo(
        rewards: torch.Tensor,
        values: torch.Tensor,
        bootstrap_value: torch.Tensor,
        discounts: torch.Tensor,
        lmb: float,
) -> UPGOReturns:
    """
    Computes Upgoing Policy Gradient returns and advantages.

    UPGO uses a hybrid target that combines immediate rewards with bootstrapped values:
    target[t] = r[t] + γ * max(V(s[t+1]), (1-λ)V(s[t+1]) + λ * target[t+1])

    This formulation:
    1. Reduces variance by using value estimates when they're higher than TD targets
    2. Maintains reasonable bias by using TD targets when they exceed value estimates
    3. Provides a smooth interpolation between V-trace and Q-trace via the λ parameter

    Args:
        rewards: Immediate rewards for each timestep [T, B, ...]
        values: Value function estimates V(s[t]) [T, B, ...]
        bootstrap_value: Value estimate for final state V(s[T]) [B, ...]
        discounts: Discount factors γ for each timestep [T, B, ...]
        lmb: Lambda parameter for interpolation between V-trace (λ=1) and Q-trace (λ=0)

    Returns:
        UPGOReturns containing:
            vs: Target values computed using UPGO estimator [T, B, ...]
            advantages: Advantage estimates (target_values - baseline_values) [T, B, ...]

    Notes:
        - Uses @torch.no_grad() for efficiency since this is used during acting
        - Implements dynamic programming by working backwards from bootstrap value
        - Handles arbitrary batch dimensions after time (first dimension)
    """
    # Append bootstrapped value to get [v1, ..., v_t+1]
    # This creates a tensor of next-state values aligned with current states
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
    )
    target_values = [bootstrap_value]
    # Compute targets backwards in time using dynamic programming
    for t in range(discounts.shape[0] - 1, -1, -1):
        # noinspection PyUnresolvedReferences
        target_values.append(
            rewards[t] + discounts[t] * torch.max(values_t_plus_1[t],
                                                  (1 - lmb) * values_t_plus_1[t] + lmb * target_values[-1])
        )
    # Reverse to get chronological order and convert to tensor
    target_values.reverse()
    # Remove bootstrap value from end of target_values list since it's not needed
    target_values = torch.stack(target_values[:-1], dim=0)

    # Return target values and advantages (target_values - baseline_values)
    # The advantages are used for policy gradient updates, while target values
    # can be used for value function training
    return UPGOReturns(
        vs=target_values,
        advantages=target_values - values
    )
