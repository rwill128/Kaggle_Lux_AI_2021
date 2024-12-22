"""Upgoing Policy Gradient (UPGO) implementation for improved credit assignment.

This module implements the UPGO algorithm, which modifies standard policy gradients
to better handle credit assignment in sparse reward settings. Key features:

1. Max Operator Usage:
   - Takes maximum between value estimates and mixed returns
   - Prevents policy updates when value prediction exceeds actual return
   - Reduces variance in policy gradient updates

2. Credit Assignment:
   - Better attributes rewards to responsible actions
   - Handles delayed rewards more effectively
   - Particularly useful for strategic games with sparse rewards

3. Policy Optimization:
   - Combines benefits of TD(λ) and policy gradients
   - More conservative updates than standard policy gradients
   - Helps prevent policy degradation during learning

Role in Lux AI Training:
1. Strategic Decision Making:
   - Better credit assignment for long-term strategic moves
   - Handles delayed rewards from resource management
   - Improves learning of complex multi-unit coordination

2. Stability Benefits:
   - Reduces variance in policy updates
   - More robust learning in competitive settings
   - Helps maintain performance during exploration

3. Reward Attribution:
   - Better handles sparse game outcomes (win/loss)
   - Improved credit assignment for early game decisions
   - More accurate advantage estimation
"""

import collections
import torch

UPGOReturns = collections.namedtuple("UPGOReturns", "vs advantages")


@torch.no_grad()
def upgo(
        rewards: torch.Tensor,
        values: torch.Tensor,
        bootstrap_value: torch.Tensor,
        discounts: torch.Tensor,
        lmb: float,
) -> UPGOReturns:
    """Compute UPGO returns and advantages for policy optimization.

    This function implements the UPGO algorithm which uses a max operator
    to combine value estimates with mixed returns, providing better credit
    assignment than standard policy gradients.

    Args:
        rewards: Immediate rewards for each timestep [T, B, ...]
        values: Value estimates from critic network [T, B, ...]
        bootstrap_value: Value estimate for terminal state [B, ...]
        discounts: Discount factors for future rewards [T, B, ...]
        lmb: Lambda parameter for mixing returns (0 ≤ λ ≤ 1)

    Returns:
        UPGOReturns containing:
        - vs: UPGO value targets [T, B, ...]
        - advantages: UPGO advantages [T, B, ...]

    Implementation Details:
    1. Return Computation:
       - Uses max operator between value and mixed return
       - Prevents updates when value prediction exceeds return
       - Maintains λ-weighted mixing of returns

    2. Credit Assignment:
       - Better attributes rewards to responsible actions
       - More conservative than standard policy gradients
       - Handles delayed rewards effectively

    3. Advantage Estimation:
       - Computes advantages using UPGO returns
       - Centers learning signal around value estimates
       - Reduces variance in policy updates

    Note:
        Uses @torch.no_grad() for efficiency since targets
        should not require gradients for optimization.
    """
    # Append bootstrapped value to get [v1, ..., v_t+1]
    # This creates a shifted sequence of values for computing returns
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
    )
    
    # Initialize target values with bootstrap value for terminal state
    target_values = [bootstrap_value]
    
    # Backward recursion to compute UPGO targets
    # This implements the key UPGO modification to standard returns
    for t in range(discounts.shape[0] - 1, -1, -1):
        # noinspection PyUnresolvedReferences
        # Compute max between value estimate and mixed return:
        # max(V(s_{t+1}), (1-λ)V(s_{t+1}) + λ * UPGO_{t+1})
        # This prevents policy updates when value prediction exceeds actual return
        target_values.append(
            rewards[t] + discounts[t] * torch.max(values_t_plus_1[t],
                                                  (1 - lmb) * values_t_plus_1[t] + lmb * target_values[-1])
        )
    
    # Reverse to get targets in forward time order
    target_values.reverse()
    
    # Remove bootstrap value from end of target_values list
    # Final shape is [T, B, ...] matching input tensors
    target_values = torch.stack(target_values[:-1], dim=0)

    return UPGOReturns(
        vs=target_values,
        advantages=target_values - values
    )
