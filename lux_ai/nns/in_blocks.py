import gym.spaces
import numpy as np
import torch
from torch import nn
from typing import Callable, Dict, Optional, Tuple, Union


def _index_select(embedding_layer: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
    """
    Optimized embedding lookup using index_select.
    
    Args:
        embedding_layer: PyTorch embedding layer to select from
        x: Input tensor containing indices
        
    Returns:
        Embedded tensor with shape (*x.shape, embedding_dim)
        
    Note:
        Faster than standard embedding forward pass but
        disables padding_idx functionality
    """
    out = embedding_layer.weight.index_select(0, x.view(-1))
    return out.view(*x.shape, -1)


def _forward_select(embedding_layer: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
    """
    Standard embedding lookup using forward pass.
    
    Args:
        embedding_layer: PyTorch embedding layer
        x: Input tensor containing indices
        
    Returns:
        Embedded tensor with shape (*x.shape, embedding_dim)
        
    Note:
        Preserves padding_idx functionality but slower than index_select
    """
    return embedding_layer(x)


def _get_select_func(use_index_select: bool) -> Callable:
    """
    Use index select instead of default forward to possibly speed up embedding.
    NB: This disables padding_idx functionality
    """
    if use_index_select:
        return _index_select
    else:
        return _forward_select


def _player_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Sum embeddings across player dimension.
    
    Args:
        x: Input tensor of shape (batch, players, ...)
        
    Returns:
        Tensor with players dimension summed out
        
    Note:
        Used to combine player and opponent embeddings
        into a single representation
    """
    return x.sum(dim=1)


def _player_cat(x: torch.Tensor) -> torch.Tensor:
    """
    Concatenate embeddings across player dimension.
    
    Args:
        x: Input tensor of shape (batch, players, ...)
        
    Returns:
        Tensor with players dimension concatenated into channels
        
    Note:
        Alternative to summing that preserves separate 
        player and opponent representations
    """
    return torch.flatten(x, start_dim=1, end_dim=2)


def _get_player_embedding_merge_func(sum_player_embeddings: bool) -> Callable:
    """
    Whether to sum or concatenate player and opponent embeddings
    """
    if sum_player_embeddings:
        return _player_sum
    else:
        return _player_cat


class DictInputLayer(nn.Module):
    """
    Input layer for processing dictionary-structured observations.
    
    Extracts and organizes observation tensors, masks, and embeddings
    from a dictionary input format commonly used in RL environments.
    
    Features:
    - Handles nested dictionary observations
    - Extracts observation tensors
    - Processes input masks for valid positions
    - Handles action masks for valid moves
    - Optional subtask embeddings support
    """
    @staticmethod
    def forward(
            x: Dict[str, Union[Dict, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        return (x["obs"],
                x["info"]["input_mask"],
                x["info"]["available_actions_mask"],
                x["info"].get("subtask_embeddings", None))


class ConvEmbeddingInputLayer(nn.Module):
    """
    Neural network input layer combining embeddings and convolutions.
    
    This layer processes complex game state observations by:
    1. Converting discrete features to learned embeddings
    2. Processing continuous features with convolutions
    3. Merging both types into a unified representation
    
    Architecture:
    - Separate pathways for discrete and continuous features
    - Learned embeddings for categorical variables
    - Convolutional processing for spatial features
    - Flexible merging of player and opponent information
    - Multi-stage feature refinement through conv layers
    
    Features:
    - Handles both discrete (MultiBinary/MultiDiscrete) and continuous (Box) spaces
    - Supports counting-based feature augmentation
    - Flexible player embedding merging (sum or concatenate)
    - Optional prefix filtering for multi-model setups
    - Optimized embedding lookups with index_select
    
    Note:
    Designed specifically for processing Lux AI game states
    with both spatial and categorical features while maintaining
    player/opponent symmetry in the representation.
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
        """
        Initialize the ConvEmbeddingInputLayer.
        
        Args:
            obs_space: Dictionary observation space containing game state features
            embedding_dim: Dimension of embeddings for discrete features
            out_dim: Output channel dimension after merging all features
            n_merge_layers: Number of convolutional layers for feature merging
            sum_player_embeddings: Whether to sum or concatenate player embeddings
            use_index_select: Whether to use optimized embedding lookup
            activation: Activation function for convolutional layers
            obs_space_prefix: Optional prefix for filtering observation keys
            
        Architecture Details:
        1. Embedding Creation:
           - Creates embeddings for discrete features (MultiBinary/MultiDiscrete)
           - Handles player-specific features with appropriate embedding counts
           - Supports optional counting-based feature augmentation
           
        2. Processing Pathways:
           - Continuous features: Conv layers with activation
           - Discrete features: Embeddings followed by conv layers
           - Both pathways use n_merge_layers for feature refinement
           
        3. Feature Merging:
           - Separate pathways for continuous and embedding features
           - Progressive refinement through multiple conv layers
           - Final merging into unified representation
           
        Note:
        - Handles both spatial (H×W) and non-spatial features
        - Maintains player/opponent symmetry in processing
        - Supports flexible observation space filtering
        """
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
        Process input observations through embedding and convolutional pathways.
        
        Args:
            x: Tuple of (observation_dict, input_mask) where:
               - observation_dict: Dictionary of tensors with shape (b, n, p|1, x, y) or (b, n, p|1)
                 where b=batch, n=channels, p=players, x,y=spatial dimensions
               - input_mask: Binary mask for valid positions
               
        Returns:
            Tuple of (processed_features, input_mask) where:
            - processed_features: Tensor of shape (b*2, out_dim, x, y) containing
              processed observations with duplicated and swapped player axes
            - input_mask: Updated mask matching processed features shape
            
        Processing Steps:
        1. Input Processing:
           - Duplicates batch entries for player/opponent views
           - Swaps player axes in duplicated entries
           - Flattens channel dimensions appropriately
           
        2. Feature Type Handling:
           - count: Multiplies embeddings by count values
           - embedding: Processes discrete features through embeddings
           - continuous: Applies spatial broadcasting if needed
           
        3. Feature Combination:
           - Merges continuous features through conv layers
           - Processes embeddings through separate pathway
           - Combines both pathways into final representation
           
        Note:
        - Maintains symmetry between player and opponent views
        - Handles both spatial and non-spatial inputs
        - Applies masking throughout processing
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
