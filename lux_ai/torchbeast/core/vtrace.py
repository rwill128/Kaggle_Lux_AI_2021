# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         cd66d00914d56c8ba2f0615d9cdeefcb169a8d70/vtrace.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import collections

import torch
import torch.nn.functional as F


VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)
"""Named tuple containing V-trace return values when working with policy logits.

Fields:
    vs (torch.Tensor): V-trace value estimates for each state
    pg_advantages (torch.Tensor): Policy gradient advantages for updating the target policy
    log_rhos (torch.Tensor): Log importance sampling ratios log(π_target/π_behavior)
    behavior_action_log_probs (torch.Tensor): Log probabilities of actions under behavior policy
    target_action_log_probs (torch.Tensor): Log probabilities of actions under target policy
"""

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")
"""Named tuple containing core V-trace return values.

Fields:
    vs (torch.Tensor): V-trace value estimates for each state
    pg_advantages (torch.Tensor): Policy gradient advantages for updating the target policy
"""


def action_log_probs(policy_logits, actions):
    """Compute action log probabilities from policy logits.
    
    Calculates log π(a|s) for given policy logits and actions by:
    1. Converting logits to log probabilities using log_softmax
    2. Computing negative log likelihood (NLL) loss without reduction
    3. Reshaping result to match action tensor shape
    
    Args:
        policy_logits (torch.Tensor): Policy network output logits [T, B, Action_Dim]
        actions (torch.Tensor): Selected actions [T, B]
        
    Returns:
        torch.Tensor: Log probabilities of selected actions under the policy [T, B]
    """
    return -F.nll_loss(
        F.log_softmax(policy_logits.view(-1, policy_logits.shape[-1]), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions)


def from_logits(
        behavior_policy_logits,
        target_policy_logits,
        actions,
        discounts,
        rewards,
        values,
        bootstrap_value,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
):
    """Compute V-trace returns from policy logits for softmax policies.
    
    This is the main entry point for V-trace when working with neural networks that
    output logits. It handles converting logits to log probabilities and computing
    importance sampling ratios before calling the core V-trace implementation.
    
    Args:
        behavior_policy_logits (torch.Tensor): Logits from behavior policy [T, B, Action_Dim]
        target_policy_logits (torch.Tensor): Logits from target policy [T, B, Action_Dim]
        actions (torch.Tensor): Actions taken by behavior policy [T, B]
        discounts (torch.Tensor): Discount factors for each step [T, B]
        rewards (torch.Tensor): Rewards received at each step [T, B]
        values (torch.Tensor): Value estimates for each state [T, B]
        bootstrap_value (torch.Tensor): Value estimate for final state [B]
        clip_rho_threshold (float): Clip threshold for importance sampling ratios (ρ)
        clip_pg_rho_threshold (float): Separate clip threshold for policy gradient ratios
        
    Returns:
        VTraceFromLogitsReturns: Named tuple containing V-trace returns and auxiliary data
        
    Note:
        T is time dimension, B is batch dimension
        Implementation follows Algorithm 1 in the IMPALA paper
    """

    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    return from_action_log_probs(
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold
    )


def from_action_log_probs(
        behavior_action_log_probs,
        target_action_log_probs,
        discounts,
        rewards,
        values,
        bootstrap_value,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
):
    """Compute V-trace returns from action log probabilities.
    
    This function implements V-trace using pre-computed log probabilities instead
    of policy logits. It's useful when working with policies that don't use
    the standard softmax parameterization.
    
    Args:
        behavior_action_log_probs (torch.Tensor): Log π_b(a|s) for behavior policy [T, B]
        target_action_log_probs (torch.Tensor): Log π_t(a|s) for target policy [T, B]
        discounts (torch.Tensor): Discount factors γ for each step [T, B]
        rewards (torch.Tensor): Rewards received at each step [T, B]
        values (torch.Tensor): Value estimates V(s_t) [T, B]
        bootstrap_value (torch.Tensor): Value estimate V(s_T) for final state [B]
        clip_rho_threshold (float): Clip threshold for importance sampling ratios (ρ)
        clip_pg_rho_threshold (float): Separate clip threshold for policy gradient ratios
        
    Returns:
        VTraceFromLogitsReturns: Named tuple containing V-trace returns and auxiliary data
        
    Note:
        Importance sampling ratios ρ are computed as exp(log π_t - log π_b)
    """
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
        log_rhos,
        discounts,
        rewards,
        values,
        bootstrap_value,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
):
    """Core V-trace implementation using importance sampling ratios.
    
    This is the main mathematical implementation of V-trace that computes:
    1. V-trace value estimates v_s using clipped importance sampling
    2. Policy gradient advantages for updating the target policy
    
    The v_s estimates are computed recursively as:
        v_s = V(s) + Σ_t γ^(t-s) c_t Π_{k=s}^{t-1} γ_k c_k δ_t
    where:
        δ_t = ρ_t (r_t + γ_t V(s_{t+1}) - V(s_t))  # TD error
        ρ_t = min(ρ_threshold, π_target/π_behavior)  # Clipped IS ratio
        c_t = min(1, π_target/π_behavior)           # Trace cutting coefficient
    
    Args:
        log_rhos (torch.Tensor): Log importance sampling ratios log(π_t/π_b) [T, B]
        discounts (torch.Tensor): Discount factors γ for each step [T, B]
        rewards (torch.Tensor): Rewards received at each step [T, B]
        values (torch.Tensor): Value estimates V(s_t) [T, B]
        bootstrap_value (torch.Tensor): Value estimate V(s_T) for final state [B]
        clip_rho_threshold (float): Clip threshold for importance sampling ratios (ρ)
        clip_pg_rho_threshold (float): Separate clip threshold for policy gradient ratios
        
    Returns:
        VTraceReturns: Named tuple containing:
            vs: V-trace value estimates [T, B]
            pg_advantages: Policy gradient advantages [T, B]
            
    Note:
        Uses @torch.no_grad() for efficiency since this is typically used during acting
        Implementation follows Equations (1-3) in the IMPALA paper
    """
    with torch.no_grad():
        # Convert log importance ratios to ratios and clip if needed
        rhos = torch.exp(log_rhos)  # ρ = π_target/π_behavior
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)  # ρ̄ = min(ρ_threshold, ρ)
        else:
            clipped_rhos = rhos

        # Compute trace cutting coefficients c = min(1, ρ)
        cs = torch.clamp(rhos, max=1.0)
        
        # Append bootstrapped value to get [V(s_1), ..., V(s_{t+1})]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        
        # Compute TD errors with clipped importance sampling ratios
        # δ_t = ρ̄_t (r_t + γ_t V(s_{t+1}) - V(s_t))
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        # Compute V-trace targets recursively backward in time
        # v_s = V(s) + Σ_t γ^(t-s) c_t Π_{k=s}^{t-1} γ_k c_k δ_t
        acc = torch.zeros_like(bootstrap_value)  # Initialize accumulator
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            # Update accumulator with discounted TD error and trace cutting
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)  # V-trace correction terms

        # Add base values V(s) to correction terms to get final V-trace targets v_s
        vs = torch.add(vs_minus_v_xs, values)

        # Compute advantages for policy gradient using potentially different clipping
        # A_t = ρ̄_t (r_t + γ_t v_{t+1} - V(s_t))
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)
