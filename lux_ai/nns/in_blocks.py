"""
Input processing layers for the Lux AI agent's neural network.

This module handles the conversion of raw game observations into feature maps:
1. Processes both continuous and discrete (MultiDiscrete/MultiBinary) inputs
2. Handles player-specific observations with symmetric processing
3. Combines embeddings and continuous features through convolutional layers
4. Supports counting-based features for unit stacking

Key Features:
- Efficient embedding lookup with optional index_select optimization
- Flexible player embedding merging (sum or concatenate)
- Handles multiple observation spaces with prefixing
- Maintains spatial structure through 1x1 convolutions
- Supports unit counting for stacked unit processing
"""

import gym.spaces
import numpy as np
import torch
from torch import nn
from typing import Callable, Dict, Optional, Tuple, Union


def _index_select(embedding_layer: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
    """
    Optimized embedding lookup using index_select.
    
    This method provides a potentially faster alternative to the default
    embedding forward pass, though it disables padding_idx functionality.
    
    Args:
        embedding_layer: The embedding layer to look up weights from
        x: Input tensor of indices
        
    Returns:
        Tensor of embeddings with shape (*x.shape, embedding_dim)
    """
    out = embedding_layer.weight.index_select(0, x.view(-1))
    return out.view(*x.shape, -1)


def _forward_select(embedding_layer: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
    """
    Standard embedding lookup using the layer's forward pass.
    
    This method uses the default PyTorch embedding lookup, which supports
    all embedding features including padding_idx.
    
    Args:
        embedding_layer: The embedding layer to look up from
        x: Input tensor of indices
        
    Returns:
        Tensor of embeddings
    """
    return embedding_layer(x)


def _get_select_func(use_index_select: bool) -> Callable:
    """
    Select the embedding lookup implementation based on performance vs. functionality.
    
    This function chooses between two embedding lookup implementations:
    1. index_select: Potentially faster but disables padding_idx functionality
    2. forward: Standard PyTorch implementation with full feature support
    
    Args:
        use_index_select: Whether to use the optimized index_select implementation
        
    Returns:
        Function that implements the chosen embedding lookup strategy
    
    Note:
        The index_select implementation may provide better performance but
        sacrifices padding_idx functionality, which can be important for
        handling missing or masked values in the input.
    """
    if use_index_select:
        return _index_select
    else:
        return _forward_select


def _player_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Sum embeddings across the player dimension.
    
    This merging strategy creates player-invariant features by adding
    embeddings from both players, useful when the order of players
    should not affect the network's behavior.
    
    Args:
        x: Input tensor with shape (batch, players, features, ...)
        
    Returns:
        Tensor with player dimension summed out
    """
    return x.sum(dim=1)


def _player_cat(x: torch.Tensor) -> torch.Tensor:
    """
    Concatenate embeddings across the player dimension.
    
    This merging strategy preserves player-specific information by
    keeping embeddings from different players separate, allowing the
    network to learn player-dependent patterns.
    
    Args:
        x: Input tensor with shape (batch, players, features, ...)
        
    Returns:
        Tensor with player and feature dimensions flattened together
    """
    return torch.flatten(x, start_dim=1, end_dim=2)


def _get_player_embedding_merge_func(sum_player_embeddings: bool) -> Callable:
    """
    Select the strategy for combining player embeddings.
    
    This function chooses between two approaches for handling player embeddings:
    1. Summation: Creates player-invariant features, reducing parameters
       and enforcing symmetry in the network's behavior
    2. Concatenation: Preserves player-specific information, allowing
       the network to learn different patterns for each player
    
    Args:
        sum_player_embeddings: If True, sum player embeddings; if False,
                             concatenate them
        
    Returns:
        Function implementing the chosen embedding merge strategy
        
    Note:
        The choice between summing and concatenating affects both the
        network's capacity and its ability to learn player-specific
        strategies vs. player-invariant patterns.
    """
    if sum_player_embeddings:
        return _player_sum
    else:
        return _player_cat


class DictInputLayer(nn.Module):
    """
    Basic input layer that processes dictionary observations.
    
    This layer extracts and organizes the key components from observation
    dictionaries:
    - Raw observations
    - Input masks for valid positions
    - Action masks for valid actions
    - Optional subtask embeddings
    """
    @staticmethod
    def forward(
            x: Dict[str, Union[Dict, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract key components from the observation dictionary.
        
        Args:
            x: Dictionary containing:
               - 'obs': Raw observations
               - 'info': Dictionary with masks and embeddings
        
        Returns:
            Tuple containing:
            - Dictionary of raw observations
            - Input mask tensor
            - Dictionary of available action masks
            - Optional subtask embedding tensor
            
        This method organizes the input data into a standardized format
        expected by the rest of the network, separating observations,
        masks, and embeddings.
        """
        return (x["obs"],
                x["info"]["input_mask"],
                x["info"]["available_actions_mask"],
                x["info"].get("subtask_embeddings", None))


class ConvEmbeddingInputLayer(nn.Module):
    """
    Sophisticated input processing layer that converts observations to feature maps.
    
    This layer handles:
    1. Processing different observation types:
       - Discrete observations through embeddings
       - Continuous observations through 1x1 convolutions
       - Count-based features for unit stacking
    2. Player perspective handling:
       - Duplicates observations for both players
       - Swaps player axes for symmetric processing
    3. Feature combination:
       - Merges embeddings through sum or concatenation
       - Processes continuous features separately
       - Combines all features through convolutional layers
    
    Args:
        obs_space: Dictionary observation space specification
        embedding_dim: Dimension of embedding vectors
        out_dim: Output feature dimension
        n_merge_layers: Number of layers for merging features
        sum_player_embeddings: Whether to sum (True) or concatenate (False) player embeddings
        use_index_select: Whether to use optimized embedding lookup
        activation: Activation function for convolutional layers
        obs_space_prefix: Optional prefix for observation space keys
    """
    def __init__(
            self,
            obs_space: gym.spaces.Dict,
            embedding_dim: int,
            out_dim: int,
            n_merge_layers: int = 1,
            sum_player_embeddings: bool = True,
            use_index_select: bool = True,
            activation: Callable = nn.LeakyReLU,
            obs_space_prefix: str = ""
    ):
        super(ConvEmbeddingInputLayer, self).__init__()

        embeddings = {}
        n_continuous_channels = 0
        n_embedding_channels = 0
        self.keys_to_op = {}
        self.obs_space_prefix = obs_space_prefix
        for orig_key, val in obs_space.spaces.items():
            assert val.shape[0] == 1
            # Used when performing inference with multiple models with different obs spaces on a single MultiObs env
            if self.obs_space_prefix:
                if orig_key.startswith(self.obs_space_prefix):
                    key = orig_key[len(obs_space_prefix):]
                else:
                    continue
            else:
                key = orig_key

            if key.endswith("_COUNT"):
                if orig_key[:-6] not in obs_space.spaces.keys():
                    raise ValueError(f"{key} was found in obs_space without an associated {key[:-6]}.")
                self.keys_to_op[key] = "count"
            elif isinstance(val, gym.spaces.MultiBinary) or isinstance(val, gym.spaces.MultiDiscrete):
                # assert embedding_dim % np.prod(val.shape[:2]) == 0, f"{key}: {embedding_dim}, {val.shape[:2]}"
                if isinstance(val, gym.spaces.MultiBinary):
                    n_embeddings = 2
                    padding_idx = 0
                elif isinstance(val, gym.spaces.MultiDiscrete):
                    if val.nvec.min() != val.nvec.max():
                        raise ValueError(
                            f"MultiDiscrete observation spaces must all have the same number of embeddings. "
                            f"Found: {np.unique(val.nvec)}")
                    n_embeddings = val.nvec.ravel()[0]
                    padding_idx = None
                else:
                    raise NotImplementedError(f"Got gym space: {type(val)}")
                n_players = val.shape[1]
                n_embeddings = n_players * (n_embeddings - 1) + 1
                embeddings[key] = nn.Embedding(n_embeddings, embedding_dim, padding_idx=padding_idx)
                if sum_player_embeddings:
                    n_embedding_channels += embedding_dim
                else:
                    n_embedding_channels += embedding_dim * n_players
                self.keys_to_op[key] = "embedding"
            elif isinstance(val, gym.spaces.Box):
                n_continuous_channels += np.prod(val.shape[:2])
                self.keys_to_op[key] = "continuous"
            else:
                raise NotImplementedError(f"{type(val)} is not an accepted observation space.")

        self.embeddings = nn.ModuleDict(embeddings)
        continuous_space_embedding_layers = []
        embedding_merger_layers = []
        merger_layers = []
        for i in range(n_merge_layers - 1):
            continuous_space_embedding_layers.extend([
                nn.Conv2d(n_continuous_channels, n_continuous_channels, (1, 1)),
                activation()
            ])
            embedding_merger_layers.extend([
                nn.Conv2d(n_embedding_channels, n_embedding_channels, (1, 1)),
                activation()
            ])
            merger_layers.extend([
                nn.Conv2d(out_dim * 2, out_dim * 2, (1, 1)),
                activation()
            ])
        continuous_space_embedding_layers.extend([
            nn.Conv2d(n_continuous_channels, out_dim, (1, 1)),
            activation()
        ])
        embedding_merger_layers.extend([
            nn.Conv2d(n_embedding_channels, out_dim, (1, 1)),
            activation()
        ])
        merger_layers.append(nn.Conv2d(out_dim * 2, out_dim, (1, 1)))
        self.continuous_space_embedding = nn.Sequential(*continuous_space_embedding_layers)
        self.embedding_merger = nn.Sequential(*embedding_merger_layers)
        self.merger = nn.Sequential(*merger_layers)
        self.player_embedding_merge_func = _get_player_embedding_merge_func(sum_player_embeddings)
        self.select = _get_select_func(use_index_select)

    def forward(self, x: Tuple[Dict[str, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process observations into feature maps suitable for the neural network.
        
        This method implements a sophisticated pipeline for converting raw
        observations into learned features:
        1. Handles different input types:
           - Discrete inputs through embeddings
           - Continuous inputs through convolutions
           - Count-based features for unit stacking
        2. Processes player perspectives:
           - Duplicates observations for both players
           - Swaps player axes for symmetric processing
        3. Combines features:
           - Merges player embeddings (sum or concatenate)
           - Processes continuous features
           - Combines all features through conv layers
        
        Args:
            x: Tuple containing:
               - Dictionary mapping observation keys to tensors of shape
                 (batch, channels, players|1, height, width) or
                 (batch, channels, players|1)
               - Input mask tensor indicating valid positions
        
        Returns:
            Tuple containing:
            - Feature maps of shape (batch*2, out_dim, height, width)
              where batch is duplicated for both player perspectives
            - Updated input mask
            
        The output feature maps maintain spatial structure while encoding
        both player-specific and player-invariant patterns, suitable for
        processing by subsequent network layers.
        """
        x, input_mask = x
        input_mask = torch.repeat_interleave(input_mask, 2, dim=0)
        continuous_outs = []
        embedding_outs = {}
        for key, op in self.keys_to_op.items():
            # Input should be of size (b, n, p|1, x, y) OR (b, n, p|1)
            in_tensor = x[self.obs_space_prefix + key]
            assert in_tensor.shape[2] <= 2
            # First we duplicate each batch entry and swap player axes when relevant
            in_tensor = in_tensor[
                        :,
                        :,
                        [np.arange(in_tensor.shape[2]), np.arange(in_tensor.shape[2])[::-1]],
                        ...
                        ]
            # Then we swap the new dims and channel dims so we can combine them with the batch dims
            in_tensor = torch.flatten(
                in_tensor.transpose(1, 2),
                start_dim=0,
                end_dim=1
            )
            # Finally, combine channel and player dims
            out = torch.flatten(in_tensor, start_dim=1, end_dim=2)
            # Size is now (b, n*p, x, y) or (b, n*p)
            if op == "count":
                embedding_expanded = embedding_outs[key[:-6]]
                embedding_expanded = embedding_expanded.view(-1, 2, embedding_expanded.shape[1] // 2,
                                                             *embedding_expanded.shape[-2:])
                embedding_outs[key[:-6]] = torch.flatten(
                    embedding_expanded * out.unsqueeze(2),
                    start_dim=1,
                    end_dim=2
                )
            elif op == "embedding":
                # Embedding out produces tensor of shape (b, p|1, ..., d)
                # This should be reshaped to size (b, d, ...)
                # First, we take all embeddings from the opponent and increment them
                if out.shape[1] == 2:
                    # noinspection PyTypeChecker
                    out[:, 1] = torch.where(
                        out[:, 1] != 0,
                        out[:, 1] + (self.embeddings[key].num_embeddings - 1) // 2,
                        out[:, 1]
                    )
                out = self.select(self.embeddings[key], out)
                # In case out is of size (b, p, d), expand it to (b, p, 1, 1, d)
                if len(out.shape) == 3:
                    out = out.unsqueeze(-2).unsqueeze(-2)
                assert len(out.shape) == 5
                embedding_outs[key] = self.player_embedding_merge_func(out.permute(0, 1, 4, 2, 3)) * input_mask
            elif op == "continuous":
                if len(out.shape) == 2:
                    out = out.unsqueeze(-1).unsqueeze(-1)
                assert len(out.shape) == 4
                continuous_outs.append(out * input_mask)
            else:
                raise RuntimeError(f"Unknown operation: {op}")
        continuous_out_combined = self.continuous_space_embedding(torch.cat(continuous_outs, dim=1))
        embedding_outs_combined = self.embedding_merger(torch.cat([v for v in embedding_outs.values()], dim=1))
        merged_outs = self.merger(torch.cat([continuous_out_combined, embedding_outs_combined], dim=1))
        return merged_outs, input_mask
