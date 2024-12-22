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
"""V-trace implementation for off-policy correction in distributed RL training.

This module implements the V-trace algorithm from the IMPALA paper:
"IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al. (https://arxiv.org/abs/1802.01561)

Role in Lux AI Training:
1. Off-Policy Correction:
   - Handles differences between behavior policy (actors) and target policy (learner)
   - Enables stable learning from old trajectories in replay buffer
   - Corrects for policy lag in distributed training setup

2. Importance Sampling:
   - Uses clipped importance weights to reduce variance
   - Balances bias-variance tradeoff for stable updates
   - Prevents excessive influence from outlier trajectories

3. Value Estimation:
   - Computes corrected value targets for critic training
   - Provides policy gradient advantages for actor training
   - Maintains temporal consistency through bootstrapping

Key Parameters:
- clip_rho_threshold: Controls maximum importance weight for value estimation
- clip_pg_rho_threshold: Controls maximum importance weight for policy gradient
- discounts: Time discounting for future rewards
- bootstrap_value: Value estimate for terminal states

Implementation adapted from DeepMind's scalable_agent repository.
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

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


def action_log_probs(policy_logits, actions):
    """Compute action log probabilities from policy logits.

    Args:
        policy_logits: Raw policy network outputs (batch_size, num_actions)
        actions: Selected actions to compute probabilities for

    Returns:
        torch.Tensor: Log probabilities of selected actions

    Implementation:
    1. Applies softmax to convert logits to probabilities
    2. Takes log of probabilities for selected actions
    3. Reshapes output to match action tensor shape

    Note:
        Uses negative NLL loss as an efficient way to compute
        log probabilities for specific actions from logits.
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
    """Compute V-trace targets from raw policy network outputs.

    Args:
        behavior_policy_logits: Logits from behavior policy (actors)
        target_policy_logits: Logits from target policy (learner)
        actions: Actually executed actions in trajectory
        discounts: Discount factors for future rewards
        rewards: Immediate rewards received
        values: Value estimates from critic network
        bootstrap_value: Value estimate for terminal state
        clip_rho_threshold: Max importance weight for value estimation
        clip_pg_rho_threshold: Max importance weight for policy gradient

    Returns:
        VTraceFromLogitsReturns containing:
        - vs: V-trace value targets
        - pg_advantages: Policy gradient advantages
        - log_rhos: Log importance weights
        - behavior/target_action_log_probs: Action probabilities

    Role in Training:
    1. Converts raw network outputs to action probabilities
    2. Computes importance weights between policies
    3. Applies V-trace algorithm for off-policy correction
    """
    """V-trace for softmax policies."""

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
    """Compute V-trace targets from action log probabilities.

    Args:
        behavior_action_log_probs: Log probs from behavior policy
        target_action_log_probs: Log probs from target policy
        discounts: Discount factors for future rewards
        rewards: Immediate rewards received
        values: Value estimates from critic network
        bootstrap_value: Value estimate for terminal state
        clip_rho_threshold: Max importance weight for value estimation
        clip_pg_rho_threshold: Max importance weight for policy gradient

    Returns:
        VTraceFromLogitsReturns containing:
        - vs: V-trace value targets
        - pg_advantages: Policy gradient advantages
        - log_rhos: Log importance weights
        - behavior/target_action_log_probs: Action probabilities

    Implementation:
    1. Computes log importance weights (target - behavior)
    2. Calls from_importance_weights for V-trace computation
    3. Packages results with additional probability info
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
    """Core V-trace implementation using importance sampling weights.

    This function implements the key V-trace algorithm equations from the
    IMPALA paper, computing corrected value targets and policy gradient
    advantages using clipped importance sampling.

    Args:
        log_rhos: Log importance weights (target/behavior ratio)
        discounts: Discount factors for future rewards
        rewards: Immediate rewards received
        values: Value estimates from critic network
        bootstrap_value: Value estimate for terminal state
        clip_rho_threshold: Max importance weight for value estimation
        clip_pg_rho_threshold: Max importance weight for policy gradient

    Returns:
        VTraceReturns containing:
        - vs: V-trace value targets
        - pg_advantages: Policy gradient advantages

    Implementation Details:
    1. Importance Weight Clipping:
       - Clamps rho_t for value estimation (clip_rho_threshold)
       - Clamps c_t always at 1.0 (bias-variance tradeoff)
       - Separate clipping for policy gradient (clip_pg_rho_threshold)

    2. V-trace Target Computation:
       - Computes temporal difference errors (deltas)
       - Recursively accumulates corrected returns
       - Adds baseline value estimates

    3. Advantage Estimation:
       - Uses clipped importance weights
       - Computes advantages for policy gradient
       - Ensures no gradient propagation through targets

    Note:
        Uses @torch.no_grad() for efficiency since targets
        should not require gradients for optimization.
    """
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)
