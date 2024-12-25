import collections
import torch

TDLambdaReturns = collections.namedtuple("TDLambdaReturns", "vs advantages")
"""Named tuple containing TD(λ) return values.

Fields:
    vs (torch.Tensor): TD(λ) value estimates for each state [T, B]
    advantages (torch.Tensor): Advantage estimates (vs - baseline values) [T, B]
"""


@torch.no_grad()
def td_lambda(
        rewards: torch.Tensor,
        values: torch.Tensor,
        bootstrap_value: torch.Tensor,
        discounts: torch.Tensor,
        lmb: float,
) -> TDLambdaReturns:
    """Compute TD(λ) returns and advantages for a sequence of states.
    
    TD(λ) is a temporal difference learning method that combines multi-step returns
    using an exponentially weighted average controlled by λ. For each state s_t,
    the TD(λ) return is computed as:
    
    v_t = (1-λ)Σ_{n=1}^∞ λ^{n-1} v_t^(n)
    
    where v_t^(n) is the n-step return:
    v_t^(n) = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})
    
    Args:
        rewards (torch.Tensor): Immediate rewards for each step [T, B]
        values (torch.Tensor): Value estimates V(s_t) for each state [T, B]
        bootstrap_value (torch.Tensor): Value estimate V(s_T) for final state [B]
        discounts (torch.Tensor): Discount factors γ for each step [T, B]
        lmb (float): Lambda parameter λ ∈ [0,1] controlling the bias-variance tradeoff
            λ=0 gives one-step TD (high bias, low variance)
            λ=1 gives Monte Carlo returns (low bias, high variance)
    
    Returns:
        TDLambdaReturns: Named tuple containing:
            vs: TD(λ) value estimates [T, B]
            advantages: TD(λ) advantages [T, B]
            
    Note:
        Uses @torch.no_grad() for efficiency since this is typically used during acting
        T is time dimension, B is batch dimension
    """
    # Append bootstrapped value to get [V(s_1), ..., V(s_{t+1})]
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
    )
    
    # Initialize target values with bootstrap value for final state
    target_values = [bootstrap_value]
    
    # Compute TD(λ) targets recursively backward in time
    # v_t = r_t + γ_t[(1-λ)V(s_{t+1}) + λv_{t+1}]
    for t in range(discounts.shape[0] - 1, -1, -1):
        # Combine one-step return with bootstrapped value using λ-weighted average
        target_values.append(
            rewards[t] + discounts[t] * ((1 - lmb) * values_t_plus_1[t] + lmb * target_values[-1])
        )
    target_values.reverse()
    
    # Remove bootstrap value from end and stack into tensor
    target_values = torch.stack(target_values[:-1], dim=0)

    # Return TD(λ) targets and advantages (targets - baseline values)
    return TDLambdaReturns(
        vs=target_values,  # TD(λ) value estimates
        advantages=target_values - values  # Advantage estimates for policy gradient
    )
