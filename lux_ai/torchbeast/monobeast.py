# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Distributed Actor-Critic Reinforcement Learning Training Infrastructure

This module implements a distributed training system for reinforcement learning using
the Actor-Critic architecture with V-trace importance weighting. The system consists of:

1. Multiple actor processes that interact with game environments to generate experience
2. A learner process that performs policy optimization using gathered experience
3. Optional teacher-student learning for knowledge distillation
4. Mixed precision training support for improved performance

Key Components:
- Actor processes: Run game simulations and collect experience
- Learner process: Optimizes policy and value networks using V-trace
- Experience replay: Manages experience buffers for batch learning
- Teacher model: Optional pre-trained model for knowledge distillation
- Checkpoint system: Saves and loads training progress

The training loop uses:
- V-trace for off-policy correction
- TD(λ) returns for value function learning
- UPGO (Upgoing Policy Gradient) for improved policy gradients
- Mixed precision training with gradient scaling
- Distributed experience collection with multiprocessing
- Wandb integration for experiment tracking

References:
- IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
- Student-Teacher RL: Learning Behaviors by Leveraging Training Data from Experts
"""
import logging
import math
import pickle

import numpy as np
from omegaconf import OmegaConf
import os
from pathlib import Path
import pprint
import threading
import time
import timeit
import traceback
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, Union
import wandb
import warnings

import torch
from torch import amp
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from .core import prof, td_lambda, upgo, vtrace
from .core.buffer_utils import Buffers, create_buffers, fill_buffers_inplace, stack_buffers, split_buffers, \
    buffers_apply
from ..lux_gym import create_env
from ..lux_gym.act_spaces import ACTION_MEANINGS
from ..nns import create_model
from ..utils import flags_to_namespace


KL_DIV_LOSS = nn.KLDivLoss(reduction="none")
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def combine_policy_logits_to_log_probs(
        behavior_policy_logits: torch.Tensor,
        actions: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Combines all policy_logits at a given step to get a single action_log_probs value for that step.

    * behavior_policy_logits: shape [time, batch, 1, players, x, y, n_actions]
        - These are the policy logits from the policy that was used to *collect* (or behave in) the data.
          It's often called the "behavior" policy because it's the policy that generated the actions seen in the
          replay buffer.

    * actions: shape [time, batch, 1, players, x, y] (holding the chosen actions)

    * actions_taken_mask: shape [time, batch, 1, players, x, y, n_actions]
        - A 0/1 mask that indicates if a particular action was actually selected or not.
          In a multi-discrete or multi-action scenario, certain partial actions might not be valid or used.
    Combines policy logits into log probabilities for taken actions, handling multiple action spaces.

    This function processes the policy logits to compute log probabilities of actions that were
    actually taken in the environment, accounting for the sequential nature of action selection
    (sampling without replacement). It handles the complexities of the Lux AI game's multiple
    action spaces and unit types.

    Args:
        behavior_policy_logits: Policy network outputs for each possible action.
            Shape: (time, batch, 1, players, x, y, n_actions)
        actions: Indices of actions that were taken.
            Shape: Same as behavior_policy_logits
        actions_taken_mask: Binary mask indicating which actions were actually taken.
            Shape: Same as behavior_policy_logits

    Returns:
        torch.Tensor: Combined log probabilities of taken actions.
            Shape: (time, batch, players)

    Notes:
        - Handles sampling without replacement by computing conditional probabilities
        - Avoids numerical issues with log(0) by masking invalid actions
        - Critical for proper importance sampling in V-trace
    Returned shape: [time, batch, players]
        This function returns, for each step, the *combined* log probability of the actions taken.

    This function effectively calculates the joint log probability of all selected actions (across x, y, etc.)
    by treating them as if they were sampled *without replacement*. This means once an action at a particular location
    is selected, the next action's probability is conditioned on that event having happened already.
    """
    # Convert the logits to probabilities across the last dimension (the n_actions dimension).
    probs = F.softmax(behavior_policy_logits, dim=-1)

    # Use the actions_taken_mask to keep only valid probabilities
    # (for example, if some action slots are not used or not valid at a certain location).
    probs = actions_taken_mask * probs

    # Now select the probabilities for the actual actions that were taken
    selected_probs = torch.gather(probs, -1, actions)

    # We sample actions "without replacement" in a sense, so for each subsequent action
    # we need to condition on previous actions that have already been chosen.
    # Here we compute the cumulative sum of previously selected probabilities.
    remaining_probability_density = 1. - torch.cat([
        torch.zeros(
            (*selected_probs.shape[:-1], 1),
            device=selected_probs.device,
            dtype=selected_probs.dtype
        ),
        selected_probs[..., :-1].cumsum(dim=-1)
    ], dim=-1)

    # Avoid division by zero (in cases where the cumsum sums to 1).
    remaining_probability_density = remaining_probability_density + torch.where(
        remaining_probability_density == 0,
        torch.ones_like(remaining_probability_density),
        torch.zeros_like(remaining_probability_density)
    )

    # Convert to conditional probabilities, so that each action probability is normalized
    # by the leftover mass after the previous picks.
    conditional_selected_probs = selected_probs / remaining_probability_density
    # Remove 0-valued conditional_selected_probs in order to eliminate neg-inf valued log_probs
    conditional_selected_probs = conditional_selected_probs + torch.where(
        conditional_selected_probs == 0,
        torch.ones_like(conditional_selected_probs),
        torch.zeros_like(conditional_selected_probs)
    )

    # Then we take the log of these conditional probabilities.
    log_probs = torch.log(conditional_selected_probs)

    # Finally, sum over actions, y, x to combine log_probs from different actions into a single scalar.
    # We flatten the last dimensions and sum them. This yields the total log probability for all actions in that step.
    return torch.flatten(log_probs, start_dim=-3, end_dim=-1).sum(dim=-1).squeeze(dim=-2)


def combine_policy_entropy(
        policy_logits: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes and combines policy entropy for a given step.

    * policy_logits: shape [time, batch, action_planes, players, x, y, n_actions]
    * actions_taken_mask: shape [time, batch, action_planes, players, x, y, n_actions]
    Computes and combines policy entropy across multiple action spaces and units.

    This function calculates the entropy of the policy distribution for each action space
    and combines them. The entropy serves as a regularization term to encourage exploration
    by penalizing overly deterministic policies.

    Returns: shape [time, batch, players]

    The function calculates the sum of individual entropies across the action planes, x, y coordinates,
    rather than the full joint entropy. The reason is that computing the exact joint entropy can be much
    more complicated when sampling multiple actions simultaneously.
    Args:
        policy_logits: Raw logits from the policy network.
            Shape: (time, batch, action_planes, players, x, y, n_actions)
        actions_taken_mask: Binary mask indicating valid actions.
            Shape: Same as policy_logits

    Returns:
        torch.Tensor: Combined entropy values.
            Shape: (time, batch, players)

    Notes:
        - Computes sum of individual entropies rather than joint entropy for efficiency
        - Handles numerical stability by masking out invalid actions
        - Higher entropy values indicate more exploratory policies
        - Used as a regularization term in the policy optimization objective
    """
    # Convert logits to probabilities
    policy = F.softmax(policy_logits, dim=-1)
    # Convert logits to log probabilities
    log_policy = F.log_softmax(policy_logits, dim=-1)

    # Replace -inf with 0 to avoid NaNs when multiplying policy * log_policy below.
    # (log_policy could be negative infinity when the probability is zero.)
    log_policy_masked_zeroed = torch.where(
        log_policy.isneginf(),
        torch.zeros_like(log_policy),
        log_policy
    )

    # Entropy for each valid action = p * log(p). Summing across actions.
    # Note that the sign is negative of p log p, so we sum p * log p, which is negative,
    # and we'll interpret it appropriately.
    entropies = (policy * log_policy_masked_zeroed).sum(dim=-1)

    # This is the same shape as actions_taken_mask, so we apply the mask
    # to ensure we only consider relevant action entries in each plane.
    assert actions_taken_mask.shape == entropies.shape
    entropies_masked = entropies * actions_taken_mask.float()

    # Now sum across the action_planes, x, y dimensions.
    return entropies_masked.sum(dim=-1).sum(dim=-1).squeeze(dim=-2)


def compute_teacher_kl_loss(
        learner_policy_logits: torch.Tensor,
        teacher_policy_logits: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes KL divergence loss between learner and teacher policies for knowledge distillation.

    This function implements the knowledge distillation component of the training, where a
    pre-trained teacher model guides the learning of the student model. It measures how much
    the learner's policy diverges from the teacher's policy using KL divergence.

    Args:
        learner_policy_logits: Raw logits from the learning model.
            Shape: (time, batch, action_planes, players, x, y, n_actions)
        teacher_policy_logits: Raw logits from the teacher model.
            Shape: Same as learner_policy_logits
        actions_taken_mask: Binary mask for valid actions.
            Shape: Same as learner_policy_logits

    Returns:
        torch.Tensor: KL divergence loss between learner and teacher policies.
            Shape: (time, batch, players)

    Notes:
        - Uses softmax to convert logits to probabilities
        - Detaches teacher policy to prevent gradient flow
        - Masks out KL divergence for invalid actions
        - Loss is used to guide learner towards teacher's behavior
    """
    """
    Computes a KL divergence loss between the learner's policy distribution
    and a teacher's policy distribution, over the valid actions.

    * learner_policy_logits: The policy logits from the learner model.
    * teacher_policy_logits: The policy logits from a teacher model (e.g., an expert or previously trained model).
    * actions_taken_mask: a boolean or 0/1 mask indicating which action entries are valid.

    Returns: shape [time, batch, players], containing the sum of KL divergences across x, y, and action_planes.
    """
    # Log probabilities for the learner
    learner_policy_log_probs = F.log_softmax(learner_policy_logits, dim=-1)

    # Teacher's probabilities
    teacher_policy = F.softmax(teacher_policy_logits, dim=-1)

    # kl_div() with log_target=False => normal KL(P||Q) but P=teacher, Q=learner in practice
    # In RL setups, "KL from teacher to learner" might be done in different ways, but here it's teacher -> learner.
    kl_div = F.kl_div(
        learner_policy_log_probs,
        teacher_policy.detach(),
        reduction="none",
        log_target=False
    ).sum(dim=-1)

    # Apply the mask
    assert actions_taken_mask.shape == kl_div.shape
    kl_div_masked = kl_div * actions_taken_mask.float()
    # Sum over y, x, and action_planes dimensions to combine kl divergences from different actions
    return kl_div_masked.sum(dim=-1).sum(dim=-1).squeeze(dim=-2)


def reduce(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    A small helper function to apply a 'mean' or 'sum' reduction to a tensor.
    """
    """
    Reduces a tensor of losses using mean or sum reduction.

    Args:
        losses: Tensor of loss values to reduce
        reduction: Reduction method, either 'mean' or 'sum'

    Returns:
        torch.Tensor: Reduced loss value

    Raises:
        ValueError: If reduction is not 'mean' or 'sum'
    """
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    else:
        raise ValueError(f"Reduction must be one of 'sum' or 'mean', was: {reduction}")


def compute_baseline_loss(values: torch.Tensor, value_targets: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Computes a 'baseline loss' for actor-critic style RL. Here using Smooth L1 loss between
    predicted values (values) and the value_targets (which come from V-trace, TD-lambda, or n-step returns).
    """
    """
    Computes the value function (baseline) loss using smooth L1 loss.

    This function computes the loss between predicted value function outputs and target values,
    using Huber loss for better stability compared to MSE loss.

    Args:
        values: Predicted value function outputs from the network
        value_targets: Target values (e.g., TD(λ) returns)
        reduction: Loss reduction method ('mean' or 'sum')

    Returns:
        torch.Tensor: Reduced baseline loss value
    """
    baseline_loss = F.smooth_l1_loss(values, value_targets.detach(), reduction="none")
    return reduce(baseline_loss, reduction=reduction)


def compute_policy_gradient_loss(
        action_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        reduction: str
) -> torch.Tensor:
    """
    Computes the basic policy gradient loss as -log_probs * advantages.

    * action_log_probs: log of the policy's probability of the chosen action
    * advantages: advantage function estimates (how much better an action is compared to a baseline)
    """
    """
    Computes the policy gradient loss using advantage estimates.

    This implements the policy gradient theorem, where the loss is the negative log probability
    of actions taken multiplied by their advantage estimates. The advantage estimates help
    determine which actions led to better-than-expected returns.

    Args:
        action_log_probs: Log probabilities of actions that were taken
        advantages: Advantage estimates (e.g., from V-trace)
        reduction: Loss reduction method ('mean' or 'sum')

    Returns:
        torch.Tensor: Reduced policy gradient loss value

    Notes:
        - Uses advantage detachment to prevent gradients flowing through value estimation
        - Critical for policy improvement in actor-critic methods
    """
    cross_entropy = -action_log_probs.view_as(advantages)
    return reduce(cross_entropy * advantages.detach(), reduction)


@torch.no_grad()
def act(
        flags: SimpleNamespace,
        teacher_flags: Optional[SimpleNamespace],
        actor_index: int,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        actor_model: torch.nn.Module,
        buffers: Buffers,
):
    """
    The function executed by each actor process (or thread).
    It runs an environment loop: picks actions from the actor_model, steps the environment,
    and writes experience data into the shared buffers.

    flags: overall config
    teacher_flags: extra config if there is a teacher model
    actor_index: the ID/index of this actor
    free_queue: queue of buffer indices that are 'free' to write fresh data into
    full_queue: queue of buffer indices that are 'full' and ready for learning
    actor_model: the policy model used to pick actions
    buffers: a shared-memory structure that stores experience

    This function runs until it receives None from free_queue (indicating it should shut down).
    """
    """
    Actor process that runs game simulations and collects experience.

    This function implements the actor part of the distributed training system. Each actor:
    1. Runs game episodes using the current policy
    2. Collects experience (observations, actions, rewards)
    3. Writes experience to shared buffers
    4. Coordinates with learner through queues

    Args:
        flags: Configuration namespace containing training parameters
        teacher_flags: Optional configuration for teacher model
        actor_index: Unique ID for this actor process
        free_queue: Queue of available buffer indices
        full_queue: Queue of filled buffer indices
        actor_model: Neural network model for action selection
        buffers: Shared experience buffers for collecting trajectories

    Notes:
        - Uses @torch.no_grad() for efficiency since no gradients needed
        - Handles environment resets and episode termination
        - Implements experience collection loop with fixed unroll length
        - Critical for parallel data collection in distributed training
    """
    if flags.debug:
        catch_me = AssertionError
    else:
        catch_me = Exception

    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()

        # Create a local environment. In multi-env setups, you might have multiple env instances here.
        env = create_env(flags, device=flags.actor_device, teacher_flags=teacher_flags)

        # Seed environment if applicable
        if flags.seed is not None:
            env.seed(flags.seed + actor_index * flags.n_actor_envs)
        else:
            env.seed()

        def save_obs(obs, filename="obs.pkl"):
            """
            Save the entire observation dict (with all array values printed) to a pickle file.
            Also turn off print truncation for debugging prints.
            """
            # Turn off the default truncation in printing:
            np.set_printoptions(threshold=np.inf)

            # (Optional) If you just want to print the entire array in the console right now:
            # print(obs)

            # Now pickle and store the full data
            with open(filename, "wb") as f:
                pickle.dump(obs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved obs to {filename}")


        # Initial environment reset
        env_output = env.reset(force=True)
        # Initial forward pass to get first actions
        agent_output = actor_model(env_output)



        while True:
            # Get a free buffer index from free_queue
            index = free_queue.get()
            if index is None:
                break  # Shutdown signal

            # Write initial data at time=0
            fill_buffers_inplace(buffers[index], dict(**env_output, **agent_output), 0)

            # Loop to run unroll_length steps
            for t in range(flags.unroll_length):
                timings.reset()

                # Evaluate the model on the current observation -> get actions, policy_logits, etc.
                agent_output = actor_model(env_output)
                timings.time("model")

                # save_obs(agent_output, "monobeat_train_loop_agent_output.pkl")

                # Step the environment with the chosen actions
                env_output = env.step(agent_output["actions"])

                # save_obs(env_output, "monobeat_train_loop_env_output.pkl")

                # logging.warning("reward: " +str(env_output["reward"][0]))
                #
                # if (max(env_output["reward"][0]) > .01):
                #     assert False

                if env_output["done"].any():
                    # Some episodes ended. We store the final transitions, then reset
                    cached_reward = env_output["reward"]
                    cached_done = env_output["done"]
                    cached_info_actions_taken = env_output["info"]["actions_taken"]
                    cached_info_logging = {
                        key: val for key, val in env_output["info"].items() if key.startswith("LOGGING_")
                    }

                    env_output = env.reset()
                    env_output["reward"] = cached_reward
                    env_output["done"] = cached_done
                    env_output["info"]["actions_taken"] = cached_info_actions_taken
                    env_output["info"].update(cached_info_logging)
                timings.time("step")

                # Write experience for time t+1
                fill_buffers_inplace(buffers[index], dict(**env_output, **agent_output), t + 1)
                timings.time("write")

            # Once we unroll for `unroll_length`, we place that buffer index into the full_queue
            # for the learner to consume.
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Graceful exit
    except catch_me as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
        flags: SimpleNamespace,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        buffers: Buffers,
        timings: prof.Timings,
        lock=threading.Lock(),
):
    """
    Retrieves a batch of data from the full_queue (i.e., from actors).

    This function:
      1. Dequeues indices from full_queue,
      2. Assembles them into a single batch (stacked along dimension=1),
      3. Moves them to the learner device,
      4. Puts the indices back on free_queue.

    flags: config object
    free_queue: queue to place used buffer indices back into
    full_queue: queue from which to retrieve full buffers
    buffers: the shared memory buffer
    timings: profiler object
    lock: concurrency lock
    """
    with lock:
        timings.time("lock")
        # We'll fetch enough indices to cover at least batch_size env transitions
        indices = [full_queue.get() for _ in range(max(flags.batch_size // flags.n_actor_envs, 1))]
        timings.time("dequeue")
    # Stack the buffer entries into a single batch
    batch = stack_buffers([buffers[m] for m in indices], dim=1)
    timings.time("batch")

    # Move the batch to the learner device (GPU or CPU)
    batch = buffers_apply(batch, lambda x: x.to(device=flags.learner_device, non_blocking=True))
    timings.time("device")

    # Return indices to the free queue so actors can refill them
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")

    return batch


def learn(
        flags: SimpleNamespace,
        actor_model: nn.Module,
        learner_model: nn.Module,
        teacher_model: Optional[nn.Module],
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_scaler: amp.grad_scaler,
        lr_scheduler: torch.optim.lr_scheduler,
        total_games_played: int,
        baseline_only: bool = False,
        lock=threading.Lock(),
) -> Tuple[Dict, int]:
    """
    Performs a learning (optimization) step. It:
      1. Computes forward pass on learner_model,
      2. (Optionally) obtains teacher outputs,
      3. Computes various RL losses (policy gradient, value function, teacher KL, etc.),
      4. Backpropagates and updates weights,
      5. Copies parameters back to actor_model,
      6. Returns training stats.

    * flags: config
    * actor_model: the model used by actors (for inference)
    * learner_model: the model being trained
    * teacher_model: optional teacher model for KL distillation
    * batch: the training batch from `get_batch`
    * optimizer: the optimizer instance
    * grad_scaler: gradient scaler for mixed precision
    * lr_scheduler: learning rate scheduler
    * total_games_played: tracking how many episodes have completed
    * baseline_only: if True, we only optimize the value/baseline loss (useful for warmup).
    """
    """
    Performs a learning step to update model parameters using collected experience.

    This is the core learning function that:
    1. Processes batches of experience
    2. Computes various losses (policy gradient, value function, teacher KL)
    3. Performs optimization step with mixed precision support
    4. Updates actor model with new parameters

    Args:
        flags: Configuration namespace containing training parameters
        actor_model: Model used by actor processes for collecting experience
        learner_model: Model being trained (parameters updated by optimizer)
        teacher_model: Optional pre-trained model for knowledge distillation
        batch: Dictionary containing experience data (obs, actions, rewards, etc.)
        optimizer: Optimizer for updating model parameters
        grad_scaler: Gradient scaler for mixed precision training
        lr_scheduler: Learning rate scheduler
        total_games_played: Counter for total completed games
        baseline_only: If True, only update value function
        lock: Threading lock for synchronization

    Returns:
        Tuple[Dict, int]: Statistics dictionary and updated games played counter

    Notes:
        - Implements V-trace for off-policy correction
        - Uses TD(λ) returns for value function training
        - Supports teacher-student learning through KL divergence
        - Handles mixed precision training with gradient scaling
        - Updates actor model with new parameters after optimization
    """
    with lock:
        with amp.autocast('cuda', enabled=flags.use_mixed_precision):
            # Flatten the batch's time and batch dims into a single dimension
            flattened_batch = buffers_apply(batch, lambda x: torch.flatten(x, start_dim=0, end_dim=1))

            # Forward pass on the learner model
            learner_outputs = learner_model(flattened_batch)
            learner_outputs = buffers_apply(learner_outputs, lambda x: x.view(flags.unroll_length + 1,
                                                                              flags.batch_size,
                                                                              *x.shape[1:]))

            # If a teacher model is used, do a forward pass on that as well
            if flags.use_teacher:
                with torch.no_grad():
                    teacher_outputs = teacher_model(flattened_batch)
                    teacher_outputs = buffers_apply(teacher_outputs, lambda x: x.view(flags.unroll_length + 1,
                                                                                      flags.batch_size,
                                                                                      *x.shape[1:]))
            else:
                teacher_outputs = None

            # The last value in the sequence is used as a 'bootstrap' for the value function
            bootstrap_value = learner_outputs["baseline"][-1]

            # Shift the batch from [0..unroll_length] to [1..unroll_length+1], so that
            # batch[t] aligns with the action chosen at t
            batch = buffers_apply(batch, lambda x: x[1:])
            learner_outputs = buffers_apply(learner_outputs, lambda x: x[:-1])
            if flags.use_teacher:
                teacher_outputs = buffers_apply(teacher_outputs, lambda x: x[:-1])

            # We'll combine the log probabilities across different action spaces
            # (like cart, worker, city_tile, etc.).
            combined_behavior_action_log_probs = torch.zeros(
                (flags.unroll_length, flags.batch_size, 2),
                device=flags.learner_device
            )
            combined_learner_action_log_probs = torch.zeros_like(combined_behavior_action_log_probs)
            combined_teacher_kl_loss = torch.zeros_like(combined_behavior_action_log_probs)
            teacher_kl_losses = {}
            combined_learner_entropy = torch.zeros_like(combined_behavior_action_log_probs)
            entropies = {}

            # For each action space (e.g., city_tile, worker, cart, etc.), we compute partial losses
            for act_space in batch["actions"].keys():
                actions = batch["actions"][act_space]
                actions_taken_mask = batch["info"]["actions_taken"][act_space]

                # Behavior policy logits come from the data (the policy that generated the data)
                behavior_policy_logits = batch["policy_logits"][act_space]
                behavior_action_log_probs = combine_policy_logits_to_log_probs(
                    behavior_policy_logits,
                    actions,
                    actions_taken_mask
                )
                combined_behavior_action_log_probs = combined_behavior_action_log_probs + behavior_action_log_probs

                # Learner's policy logits
                learner_policy_logits = learner_outputs["policy_logits"][act_space]
                learner_action_log_probs = combine_policy_logits_to_log_probs(
                    learner_policy_logits,
                    actions,
                    actions_taken_mask
                )
                combined_learner_action_log_probs = combined_learner_action_log_probs + learner_action_log_probs

                # Create a mask for any tile that has any actions taken
                any_actions_taken = actions_taken_mask.any(dim=-1)

                if flags.use_teacher:
                    teacher_kl_loss = compute_teacher_kl_loss(
                        learner_policy_logits,
                        teacher_outputs["policy_logits"][act_space],
                        any_actions_taken
                    )
                else:
                    teacher_kl_loss = torch.zeros_like(combined_teacher_kl_loss)
                combined_teacher_kl_loss = combined_teacher_kl_loss + teacher_kl_loss

                teacher_kl_losses[act_space] = (reduce(
                    teacher_kl_loss,
                    reduction="sum",
                ) / any_actions_taken.sum()).detach().cpu().item()

                # Entropy helps regularize the policy and keep it exploring.
                learner_policy_entropy = combine_policy_entropy(
                    learner_policy_logits,
                    any_actions_taken
                )
                combined_learner_entropy = combined_learner_entropy + learner_policy_entropy
                entropies[act_space] = -(reduce(
                    learner_policy_entropy,
                    reduction="sum"
                ) / any_actions_taken.sum()).detach().cpu().item()

            # Compute discount factors for each step
            discounts = (~batch["done"]).float() * flags.discounting
            # Expand to match [time, batch, 2]
            discounts = discounts.unsqueeze(-1).expand_as(combined_behavior_action_log_probs)

            # Compute the V-trace returns using the combined log-probs
            values = learner_outputs["baseline"]
            vtrace_returns = vtrace.from_action_log_probs(
                behavior_action_log_probs=combined_behavior_action_log_probs,
                target_action_log_probs=combined_learner_action_log_probs,
                discounts=discounts,
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value
            )

            # TD-lambda and UPGO returns for additional ways to compute value targets / advantages.
            td_lambda_returns = td_lambda.td_lambda(
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value,
                discounts=discounts,
                lmb=flags.lmb
            )
            upgo_returns = upgo.upgo(
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value,
                discounts=discounts,
                lmb=flags.lmb
            )

            # Policy gradient losses
            vtrace_pg_loss = compute_policy_gradient_loss(
                combined_learner_action_log_probs,
                vtrace_returns.pg_advantages,
                reduction=flags.reduction
            )

            # For UPGO, we also incorporate clipping on importance sampling
            upgo_clipped_importance = torch.minimum(
                vtrace_returns.log_rhos.exp(),
                torch.ones_like(vtrace_returns.log_rhos)
            ).detach()

            upgo_pg_loss = compute_policy_gradient_loss(
                combined_learner_action_log_probs,
                upgo_clipped_importance * upgo_returns.advantages,
                reduction=flags.reduction
            )

            # Baseline/value loss
            baseline_loss = compute_baseline_loss(
                values,
                td_lambda_returns.vs,
                reduction=flags.reduction
            )

            # Teacher KL loss
            teacher_kl_loss = flags.teacher_kl_cost * reduce(
                combined_teacher_kl_loss,
                reduction=flags.reduction
            )

            # Teacher baseline alignment, if desired
            if flags.use_teacher:
                teacher_baseline_loss = flags.teacher_baseline_cost * compute_baseline_loss(
                    values,
                    teacher_outputs["baseline"],
                    reduction=flags.reduction
                )
            else:
                teacher_baseline_loss = torch.zeros_like(baseline_loss)

            # Entropy loss for exploration
            entropy_loss = flags.entropy_cost * reduce(
                combined_learner_entropy,
                reduction=flags.reduction
            )

            # Optionally only optimize the baseline (value network) in early training
            if baseline_only:
                total_loss = baseline_loss + teacher_baseline_loss
                vtrace_pg_loss, upgo_pg_loss, teacher_kl_loss, entropy_loss = torch.zeros(4) + float("nan")
            else:
                total_loss = (vtrace_pg_loss +
                              upgo_pg_loss +
                              baseline_loss +
                              teacher_kl_loss +
                              teacher_baseline_loss +
                              entropy_loss)

            # Get current learning rate from the scheduler
            last_lr = lr_scheduler.get_last_lr()
            assert len(last_lr) == 1, 'Logging per-parameter LR still needs support'
            last_lr = last_lr[0]

            # Summarize some logging stats about how many times each action was taken
            action_distributions_flat = {
                key[16:]: val[batch["done"]][~val[batch["done"]].isnan()].sum().item()
                for key, val in batch["info"].items()
                if key.startswith("LOGGING_") and "ACTIONS_" in key
            }
            action_distributions = {space: {} for space in ACTION_MEANINGS.keys()}
            for flat_name, n in action_distributions_flat.items():
                space, meaning = flat_name.split(".")
                action_distributions[space][meaning] = n
            action_distributions_aggregated = {}
            for space, dist in action_distributions.items():
                if space == "city_tile":
                    action_distributions_aggregated[space] = dist
                elif space in ("cart", "worker"):
                    aggregated = {
                        a: n for a, n in dist.items() if "TRANSFER" not in a and "MOVE" not in a
                    }
                    aggregated["TRANSFER"] = sum({a: n for a, n in dist.items() if "TRANSFER" in a}.values())
                    aggregated["MOVE"] = sum({a: n for a, n in dist.items() if "MOVE" in a}.values())
                    action_distributions_aggregated[space] = aggregated
                else:
                    raise RuntimeError(f"Unrecognized action_space: {space}")
                n_actions = sum(action_distributions_aggregated[space].values())
                if n_actions == 0:
                    action_distributions_aggregated[space] = {
                        key: float("nan") for key in action_distributions_aggregated[space].keys()
                    }
                else:
                    action_distributions_aggregated[space] = {
                        key: val / n_actions for key, val in action_distributions_aggregated[space].items()
                    }

            # Increase total games played (for logging or scheduling)
            total_games_played += batch["done"].sum().item()

            # Collect stats
            stats = {
                "Env": {
                    key[8:]: val[batch["done"]][~val[batch["done"]].isnan()].mean().item()
                    for key, val in batch["info"].items()
                    if key.startswith("LOGGING_") and "ACTIONS_" not in key
                },
                "Actions": action_distributions_aggregated,
                "Loss": {
                    "vtrace_pg_loss": vtrace_pg_loss.detach().item(),
                    "upgo_pg_loss": upgo_pg_loss.detach().item(),
                    "baseline_loss": baseline_loss.detach().item(),
                    "teacher_kl_loss": teacher_kl_loss.detach().item(),
                    "teacher_baseline_loss": teacher_baseline_loss.detach().item(),
                    "entropy_loss": entropy_loss.detach().item(),
                    "total_loss": total_loss.detach().item(),
                },
                "Entropy": {
                    "overall": sum(e for e in entropies.values() if not math.isnan(e)),
                    **entropies
                },
                "Teacher_KL_Divergence": {
                    "overall": sum(tkld for tkld in teacher_kl_losses.values() if not math.isnan(tkld)),
                    **teacher_kl_losses
                },
                "Misc": {
                    "learning_rate": last_lr,
                    "total_games_played": total_games_played
                },
            }

            # Do the backward pass and optimizer step
            optimizer.zero_grad()
            if flags.use_mixed_precision:
                grad_scaler.scale(total_loss).backward()
                if flags.clip_grads is not None:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(learner_model.parameters(), flags.clip_grads)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                total_loss.backward()
                if flags.clip_grads is not None:
                    torch.nn.utils.clip_grad_norm_(learner_model.parameters(), flags.clip_grads)
                optimizer.step()

            if lr_scheduler is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    lr_scheduler.step()

        # Sync the updated parameters back to the actor_model
        # so that new rollouts use updated weights.
        actor_model.load_state_dict(learner_model.state_dict())
        return stats, total_games_played


def train(flags):
    """
    Main training loop that orchestrates the distributed training system.

    This function:
    1. Sets up the distributed training infrastructure
    2. Initializes actor processes and learner threads
    3. Manages experience collection and learning coordination
    4. Handles checkpointing and logging
    5. Monitors training progress

    Args:
        flags: Configuration namespace containing all training parameters

    Notes:
        - Creates multiple actor processes for parallel experience collection
        - Manages shared experience buffers and coordination queues
        - Handles model initialization and optional teacher model loading
        - Implements checkpointing for training state persistence
        - Integrates with wandb for experiment tracking
        - Uses mixed precision training for performance
        - Coordinates multiple learner threads for parallel optimization
    """
    # Necessary for multithreading and multiprocessing
    """
    Main training orchestration function:
      1. Validates some hyperparams,
      2. Creates buffers,
      3. Spawns actor processes,
      4. Sets up the learner model, optimizer, scheduler,
      5. Runs the main training loop until total_steps is reached,
      6. Periodically checkpoints.

    * flags: the main config / hyperparameters namespace
    """
    # Limit for threads used by some libraries
    os.environ["OMP_NUM_THREADS"] = "1"

    if flags.num_buffers < flags.num_actors:
        raise ValueError("num_buffers should >= num_actors")
    if flags.num_buffers < flags.batch_size // flags.n_actor_envs:
        raise ValueError("num_buffers should be larger than batch_size // n_actor_envs")

    t = flags.unroll_length
    b = flags.batch_size

    # If we have a teacher model, load teacher flags
    if flags.use_teacher:
        teacher_flags = OmegaConf.load(Path(flags.teacher_load_dir) / "config.yaml")
        teacher_flags = flags_to_namespace(OmegaConf.to_container(teacher_flags))
    else:
        teacher_flags = None

    # Create a sample environment to figure out observation/action shapes
    example_env = create_env(flags, torch.device("cpu"), teacher_flags=teacher_flags)
    buffers = create_buffers(
        flags,
        example_env.unwrapped[0].obs_space,
        example_env.reset(force=True)["info"]
    )
    del example_env

    # Possibly load from checkpoint
    if flags.load_dir:
        checkpoint_state = torch.load(Path(flags.load_dir) / flags.checkpoint_file, map_location=torch.device("cpu"))
    else:
        checkpoint_state = None

    # Create the actor model
    actor_model = create_model(flags, flags.actor_device, teacher_model_flags=teacher_flags, is_teacher_model=False)
    if checkpoint_state is not None:
        actor_model.load_state_dict(checkpoint_state["model_state_dict"])
    actor_model.eval()
    actor_model.share_memory()

    n_trainable_params = sum(p.numel() for p in actor_model.parameters() if p.requires_grad)
    logging.info(f'Training model with {n_trainable_params:,d} parameters.')

    # Create actor processes
    actor_processes = []
    free_queue = mp.SimpleQueue()
    full_queue = mp.SimpleQueue()

    for i in range(flags.num_actors):
        # In debug mode, use threads. Otherwise use processes for speedup.
        actor_start = threading.Thread if flags.debug else mp.Process
        actor = actor_start(
            target=act,
            args=(
                flags,
                teacher_flags,
                i,
                free_queue,
                full_queue,
                actor_model,
                buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)
        time.sleep(0.5)

    # Create the learner model
    learner_model = create_model(flags, flags.learner_device, teacher_model_flags=teacher_flags, is_teacher_model=False)
    if checkpoint_state is not None:
        learner_model.load_state_dict(checkpoint_state["model_state_dict"])
    learner_model.train()
    learner_model = learner_model.share_memory()

    # (Optionally) log model parameters to Weights & Biases
    if not flags.disable_wandb:
        wandb.watch(learner_model, flags.model_log_freq, log="all", log_graph=True)

    # Create optimizer
    optimizer = flags.optimizer_class(
        learner_model.parameters(),
        **flags.optimizer_kwargs
    )
    if checkpoint_state is not None and not flags.weights_only:
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    # If we use a teacher model, load it
    if flags.use_teacher:
        if flags.teacher_kl_cost <= 0. and flags.teacher_baseline_cost <= 0.:
            raise ValueError("It does not make sense to use teacher when teacher_kl_cost <= 0 "
                             "and teacher_baseline_cost <= 0")
        teacher_model = create_model(
            flags,
            flags.learner_device,
            teacher_model_flags=teacher_flags,
            is_teacher_model=True
        )
        teacher_model.load_state_dict(
            torch.load(
                Path(flags.teacher_load_dir) / flags.teacher_checkpoint_file,
                map_location=torch.device("cpu")
            )["model_state_dict"]
        )
        teacher_model.eval()
    else:
        teacher_model = None
        if flags.teacher_kl_cost > 0.:
            logging.warning(f"flags.teacher_kl_cost is {flags.teacher_kl_cost}, but use_teacher is False. "
                            f"Setting flags.teacher_kl_cost to 0.")
        if flags.teacher_baseline_cost > 0.:
            logging.warning(f"flags.teacher_baseline_cost is {flags.teacher_baseline_cost}, but use_teacher is False. "
                            f"Setting flags.teacher_baseline_cost to 0.")
        flags.teacher_kl_cost = 0.
        flags.teacher_baseline_cost = 0.

    # Learning rate schedule
    def lr_lambda(epoch):
        min_pct = flags.min_lr_mod
        pct_complete = min(epoch * t * b, flags.total_steps) / flags.total_steps
        scaled_pct_complete = pct_complete * (1. - min_pct)
        return 1. - scaled_pct_complete

    grad_scaler = amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if checkpoint_state is not None and not flags.weights_only:
        scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])

    step, total_games_played, stats = 0, 0, {}
    if checkpoint_state is not None and not flags.weights_only:
        if "step" in checkpoint_state.keys():
            step = checkpoint_state["step"]
        else:
            logging.warning("Loading old checkpoint_state without 'step' saved. Starting at step 0.")
        if "total_games_played" in checkpoint_state.keys():
            total_games_played = checkpoint_state["total_games_played"]
        else:
            logging.warning("Loading old checkpoint_state without 'total_games_played' saved. Starting at step 0.")

    def batch_and_learn(learner_idx, lock=threading.Lock()):
        """
        A thread target that repeatedly fetches a batch and calls 'learn'.
        Runs until we surpass flags.total_steps.
        """
        nonlocal step, total_games_played, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            # Get a full batch from the actors
            full_batch = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                timings,
            )

            # Possibly split the batch if we want to handle smaller batch sizes
            if flags.batch_size < flags.n_actor_envs:
                batches = split_buffers(full_batch, flags.batch_size, dim=1, contiguous=True)
            else:
                batches = [full_batch]

            for batch in batches:
                stats, total_games_played = learn(
                    flags=flags,
                    actor_model=actor_model,
                    learner_model=learner_model,
                    teacher_model=teacher_model,
                    batch=batch,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    lr_scheduler=scheduler,
                    total_games_played=total_games_played,
                    baseline_only=step / (t * b) < flags.n_value_warmup_batches,
                )
                with lock:
                    step += t * b
                    if not flags.disable_wandb:
                        wandb.log(stats, step=step)
            timings.time("learn")
        if learner_idx == 0:
            logging.info(f"Batch and learn timing statistics: {timings.summary()}")

    # Pre-load the free queue with all buffer indices
    for m in range(flags.num_buffers):
        free_queue.put(m)

    # Create learner threads
    learner_threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name=f"batch-and-learn-{i}", args=(i,)
        )
        thread.start()
        learner_threads.append(thread)

    def checkpoint(checkpoint_path: Union[str, Path]):
        """
        Saves a checkpoint: model weights, optimizer state, scheduler state, current training progress, etc.
        """
        logging.info(f"Saving checkpoint to {checkpoint_path}")

        cp_path_pt = checkpoint_path + ".pt"
        cp_path_weights = checkpoint_path + "_weights.pt"

        torch.save(
            {
                "model_state_dict": actor_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
                "total_games_played": total_games_played,
            },
            cp_path_pt,
            )
        torch.save(
            {
                "model_state_dict": actor_model.state_dict(),
            },
            cp_path_weights
        )

        # If wandb logging is enabled, log an artifact containing these checkpoints
        if not flags.disable_wandb:
            artifact = wandb.Artifact(name=f"model-{step}", type="model")
            artifact.add_file(cp_path_pt)
            artifact.add_file(cp_path_weights)
            wandb.log_artifact(artifact)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            # Checkpoint every X minutes
            if timer() - last_checkpoint_time > flags.checkpoint_freq * 60:
                cp_path = str(step).zfill(int(math.log10(flags.total_steps)) + 1)
                checkpoint(cp_path)
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            bps = (step - start_step) / (t * b) / (timer() - start_time)
            logging.info(f"Steps {step:d} @ {sps:.1f} SPS / {bps:.1f} BPS. Stats:\n{pprint.pformat(stats)}")
    except KeyboardInterrupt:
        # Attempt a graceful shutdown
        return
    else:
        # Wait for all learner threads
        for thread in learner_threads:
            thread.join()
        logging.info(f"Learning finished after {step:d} steps.")
    finally:
        # Stop actor processes
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

        # Final checkpoint
        cp_path = str(step).zfill(int(math.log10(flags.total_steps)) + 1)
        checkpoint(cp_path)
