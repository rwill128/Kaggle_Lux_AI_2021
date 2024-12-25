"""Utilities for managing experience replay buffers in distributed reinforcement learning.

This module provides functions for creating, manipulating, and managing shared memory
buffers that store experience data (observations, actions, rewards, etc.) during
distributed training. The buffers are designed to support efficient data transfer
between actors and learners in the IMPALA architecture.
"""

from copy import copy
import gym
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Tuple, Union

from ...lux_gym.act_spaces import MAX_OVERLAPPING_ACTIONS
from ...lux_gym.obs_spaces import BaseObsSpace

# Type alias for experience replay buffers
Buffers = List[Dict[str, Union[Dict, torch.Tensor]]]
"""Type definition for experience replay buffers.

A buffer is a list of dictionaries containing:
- obs: Observations from the environment
- reward: Rewards received from actions
- done: Episode termination flags
- policy_logits: Action probabilities from policy network
- baseline: Value estimates from value network
- actions: Actions taken by the agent
- info: Additional environment information

Each value can be either a tensor or a nested dictionary of tensors.
"""


def fill_buffers_inplace(buffers: Union[Dict, torch.Tensor], fill_vals: Union[Dict, torch.Tensor], step: int):
    """Fill experience buffers with new values at a specific timestep.

    Recursively fills nested buffer structures with new values, handling both
    dictionary and tensor data types. Modifies buffers in-place.

    Args:
        buffers: Target buffer structure to fill, either a dictionary of nested buffers
            or a tensor buffer
        fill_vals: Values to insert into buffers, matching the buffer structure
        step: Timestep index at which to insert the values

    Note:
        Uses recursive traversal to handle arbitrary nesting of dictionary buffers.
        For tensor buffers, copies all dimensions after the timestep dimension.
    """
    if isinstance(fill_vals, dict):
        for key, val in copy(fill_vals).items():
            fill_buffers_inplace(buffers[key], val, step)
    else:
        buffers[step, ...] = fill_vals[:]  # Copy all trailing dimensions


def stack_buffers(buffers: Buffers, dim: int) -> Dict[str, Union[Dict, torch.Tensor]]:
    """Stack multiple buffers along a specified dimension.

    Combines multiple experience buffers by concatenating their tensors along
    the specified dimension. Handles nested dictionary structures recursively.

    Args:
        buffers: List of buffer dictionaries to stack
        dim: Dimension along which to stack the tensors (e.g., time or batch dim)

    Returns:
        A single buffer dictionary with stacked tensors, preserving the original
        nested structure

    Note:
        Commonly used to combine experiences from multiple actors or episodes
        before sending to the learner for training updates.
    """
    stacked_buffers = {}
    for key, val in copy(buffers[0]).items():
        if isinstance(val, dict):
            stacked_buffers[key] = stack_buffers([b[key] for b in buffers], dim)
        else:
            stacked_buffers[key] = torch.cat([b[key] for b in buffers], dim=dim)
    return stacked_buffers


def split_buffers(
        buffers: Dict[str, Union[Dict, torch.Tensor]],
        split_size_or_sections: Union[int, List[int]],
        dim: int,
        contiguous: bool,
) -> List[Union[Dict, torch.Tensor]]:
    """Split a buffer into multiple smaller buffers along a specified dimension.

    Recursively splits nested buffer structures, maintaining the dictionary hierarchy
    while dividing the underlying tensors.

    Args:
        buffers: Buffer dictionary to split
        split_size_or_sections: Either the size of each split or a list of sizes
        dim: Dimension along which to split the tensors
        contiguous: If True, ensure resulting tensors are contiguous in memory

    Returns:
        List of buffer dictionaries, each containing a portion of the original data

    Note:
        Useful for:
        - Dividing large batches into mini-batches
        - Distributing experiences across multiple learners
        - Separating trajectories for parallel processing
    """
    buffers_split = None
    for key, val in copy(buffers).items():
        if isinstance(val, dict):
            bufs = split_buffers(val, split_size_or_sections, dim, contiguous)
        else:
            bufs = torch.split(val, split_size_or_sections, dim=dim)
            if contiguous:
                bufs = [b.contiguous() for b in bufs]  # Ensure memory efficiency

        if buffers_split is None:
            buffers_split = [{} for _ in range(len(bufs))]
        assert len(bufs) == len(buffers_split)
        # Merge split buffer with existing dictionary structure
        buffers_split = [dict(**{key: buf}, **d) for buf, d in zip(bufs, buffers_split)]
    return buffers_split


def buffers_apply(buffers: Union[Dict, torch.Tensor], func: Callable[[torch.Tensor], Any]) -> Union[Dict, torch.Tensor]:
    """Apply a function to all tensors in a buffer structure.

    Recursively traverses the buffer hierarchy, applying the given function
    to each tensor while preserving the dictionary structure.

    Args:
        buffers: Buffer structure to process
        func: Function to apply to each tensor, e.g., detach, cpu, cuda

    Returns:
        New buffer structure with function applied to all tensors

    Note:
        Common uses include:
        - Moving buffers between devices (CPU/GPU)
        - Detaching tensors from computation graph
        - Applying data transformations
    """
    if isinstance(buffers, dict):
        return {
            key: buffers_apply(val, func) for key, val in copy(buffers).items()
        }
    else:
        return func(buffers)


def _create_buffers_from_specs(specs: Dict[str, Union[Dict, Tuple, torch.dtype]]) -> Union[Dict, torch.Tensor]:
    """Create shared memory buffers from specifications.

    Internal helper function that creates tensor buffers in shared memory based on
    provided specifications for shape and dtype.

    Args:
        specs: Dictionary of tensor specifications, where each spec is either:
            - A nested dictionary of specs
            - A dict with 'size' and 'dtype' keys for tensor creation

    Returns:
        Buffer structure matching the spec hierarchy, with shared memory tensors

    Note:
        Uses torch.empty().share_memory_() to create tensors accessible across
        processes for distributed training.
    """
    if isinstance(specs, dict) and "dtype" not in specs.keys():
        new_buffers = {}
        for key, val in specs.items():
            new_buffers[key] = _create_buffers_from_specs(val)
        return new_buffers
    else:
        return torch.empty(**specs).share_memory_()  # Create shared memory tensor


def _create_buffers_like(buffers: Union[Dict, torch.Tensor], t_dim: int) -> Union[Dict, torch.Tensor]:
    """Create shared memory buffers with same structure and shapes as template.

    Internal helper function that creates new buffers matching the structure of
    existing ones, but with a specified time dimension.

    Args:
        buffers: Template buffer structure to match
        t_dim: Size of time dimension for new buffers

    Returns:
        New buffer structure in shared memory with specified time dimension

    Note:
        Preserves tensor shapes and dtypes while adding/modifying time dimension.
        Used primarily for creating info buffers that match environment output structure.
    """
    if isinstance(buffers, dict):
        torch_buffers = {}
        for key, val in buffers.items():
            torch_buffers[key] = _create_buffers_like(val, t_dim)
        return torch_buffers
    else:
        # Add time dimension and expand while keeping other dimensions
        buffers = buffers.unsqueeze(0).expand(t_dim, *[-1 for _ in range(len(buffers.shape))])
        return torch.empty_like(buffers).share_memory_()  # Create shared memory tensor


def create_buffers(
        flags,
        obs_space: BaseObsSpace,
        example_info: Dict[str, Union[Dict, np.ndarray, torch.Tensor]]
) -> Buffers:
    """Create a list of experience replay buffers for distributed training.

    Constructs shared memory buffer structures for storing trajectories from
    multiple environments. Handles complex observation and action spaces from
    the Lux game environment.

    Args:
        flags: Configuration object containing:
            - unroll_length: Number of timesteps per trajectory
            - n_actor_envs: Number of parallel environments
            - num_buffers: Number of separate replay buffers
        obs_space: Observation space specification from environment
        example_info: Template for additional info dictionary structure

    Returns: 
        List of buffer dictionaries, each containing:
        - obs: Environment observations with appropriate dtypes
        - reward: Reward signals for each player
        - done: Episode termination flags
        - policy_logits: Network policy outputs
        - baseline: Value function estimates
        - actions: Agent action selections
        - info: Additional environment information

    Note:
        - Uses shared memory for efficient multi-process communication
        - Handles both discrete and continuous observation spaces
        - Supports multi-agent setup with p=2 players
        - Limits maximum overlapping actions per unit
    """
    # Extract buffer dimensions
    t = flags.unroll_length  # Trajectory length
    n = flags.n_actor_envs   # Number of parallel envs
    p = 2                    # Number of players

    # Create observation specs based on environment space
    obs_specs = {}
    for key, spec in obs_space.get_obs_spec().spaces.items():
        if isinstance(spec, gym.spaces.MultiBinary):
            dtype = torch.int64
        elif isinstance(spec, gym.spaces.MultiDiscrete):
            dtype = torch.int64
        elif isinstance(spec, gym.spaces.Box):
            dtype = torch.float32
        else:
            raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")
        obs_specs[key] = dict(size=(t + 1, n, *spec.shape), dtype=dtype)

    # Define core buffer specifications
    specs = dict(
        obs=obs_specs,
        reward=dict(size=(t + 1, n, p), dtype=torch.float32),  # Rewards per player
        done=dict(size=(t + 1, n), dtype=torch.bool),          # Episode termination
        policy_logits={},                                       # Network outputs
        baseline=dict(size=(t + 1, n, p), dtype=torch.float32),# Value estimates
        actions={},                                             # Selected actions
    )

    # Add action-specific buffer specs
    act_space = flags.act_space()
    for key, expanded_shape in act_space.get_action_space_expanded_shape().items():
        # Policy network outputs (logits)
        specs["policy_logits"][key] = dict(size=(t + 1, n, *expanded_shape), dtype=torch.float32)
        # Actual actions (limited by MAX_OVERLAPPING_ACTIONS)
        final_actions_dim = min(expanded_shape[-1], MAX_OVERLAPPING_ACTIONS)
        specs["actions"][key] = dict(size=(t + 1, n, *expanded_shape[:-1], final_actions_dim), dtype=torch.int64)

    # Create specified number of buffer copies
    buffers: Buffers = []
    for _ in range(flags.num_buffers):
        new_buffer = _create_buffers_from_specs(specs)
        new_buffer["info"] = _create_buffers_like(example_info, t + 1)
        buffers.append(new_buffer)
    return buffers
