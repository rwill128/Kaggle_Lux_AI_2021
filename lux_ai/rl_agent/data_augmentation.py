from abc import ABC, abstractmethod
import torch
from typing import Dict

from ..lux.constants import Constants
from ..lux.game import Game
from ..lux_gym.act_spaces import ACTION_MEANINGS_TO_IDX
from ..utils import DEBUG_MESSAGE


class DataAugmenter(ABC):
    """
    Abstract base class for data augmentation operations in the Lux AI environment.
    
    This class provides a framework for implementing various spatial transformations
    (rotations, flips) on game state tensors and corresponding action spaces.
    
    Key Features:
    1. Bidirectional Transformations:
       - Forward transformation of observations and actions
       - Inverse transformation to maintain consistency
       
    2. Policy Handling:
       - Transforms both state observations and policy outputs
       - Maintains action space validity after transformation
       
    3. Direction Mapping:
       - Maps movement directions (N,S,E,W) during transformations
       - Ensures action semantics are preserved
       
    Implementation Notes:
    - Subclasses must implement get_directions_mapped() and op()
    - Handles both policy tensors (6D) and observation tensors (5D)
    - Automatically updates action indices based on direction mapping
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the data augmenter with direction mappings.
        
        Process:
        1. Get direction mappings (e.g., N->S for vertical flip)
        2. Create inverse mappings
        3. Build action index transformation tables for:
           - Forward transformation
           - Inverse transformation
        
        The transformation tables ensure that action semantics
        (e.g., "MOVE_NORTH") are correctly mapped during augmentation.
        """
        directions_mapped_forward = self.get_directions_mapped()
        direction_mapped_inverse = {val: key for key, val in directions_mapped_forward.items()}
        assert len(directions_mapped_forward) == len(direction_mapped_inverse)

        self.transformed_action_idxs_forward = {}
        self.transformed_action_idxs_inverse = {}
        for space, meanings_to_idx in ACTION_MEANINGS_TO_IDX.items():
            transformed_space_idxs_forward = []
            for action, idx in meanings_to_idx.items():
                for d, d_mapped in directions_mapped_forward.items():
                    if action.endswith(f"_{d_mapped}"):
                        transformed_space_idxs_forward.append(meanings_to_idx[action[:-1] + d])
                        break
                else:
                    transformed_space_idxs_forward.append(idx)
            self.transformed_action_idxs_forward[space] = transformed_space_idxs_forward

            transformed_space_idxs_inverse = []
            for action, idx in meanings_to_idx.items():
                for d, d_mapped in direction_mapped_inverse.items():
                    if action.endswith(f"_{d_mapped}"):
                        transformed_space_idxs_inverse.append(meanings_to_idx[action[:-1] + d])
                        break
                else:
                    transformed_space_idxs_inverse.append(idx)
            self.transformed_action_idxs_inverse[space] = transformed_space_idxs_inverse

    def apply(self, x: Dict[str, torch.Tensor], inverse: bool, is_policy: bool) -> Dict[str, torch.Tensor]:
        """
        Apply the augmentation transformation to a batch of tensors.
        
        Args:
            x: Dictionary of tensors representing observations or policy outputs
            inverse: If True, apply inverse transformation
            is_policy: If True, handle as policy tensor (6D), else as observation (5D)
            
        Returns:
            Dict[str, torch.Tensor]: Transformed tensors
            
        Notes:
            - Policy tensors have shape (batch, time, height, width, features, actions)
            - Observation tensors have shape (batch, time, height, width, features)
            - Only transforms tensors matching expected dimensionality
            - Automatically handles action space transformations for policies
        """
        n_dims = 6 if is_policy else 5
        for tensor in x.values():
            if tensor.dim() != n_dims:
                continue
            if is_policy:
                assert tensor.shape[-3] == tensor.shape[-2]
            else:
                assert tensor.shape[-2] == tensor.shape[-1]
        x_transformed = {
            key: self.op(val, inverse=inverse, is_policy=is_policy) if val.dim() == n_dims else val
            for key, val in x.items()
        }
        if is_policy:
            return self._transform_policy(x_transformed, inverse=inverse)
        return x_transformed

    def _apply_and_apply_inverse(self, x: Dict[str, torch.Tensor], is_policy: bool) -> Dict[str, torch.Tensor]:
        """
        This method is for debugging only.
        If everything is working correctly, it should leave the input unchanged.
        """
        x_transformed = self.apply(x, inverse=False, is_policy=is_policy)
        return self.apply(x_transformed, inverse=True, is_policy=is_policy)

    @abstractmethod
    def get_directions_mapped(self) -> Dict[str, str]:
        """
        Define the mapping of directions for this transformation.
        
        Returns:
            Dict[str, str]: Mapping from original to transformed directions
                           e.g., {'NORTH': 'SOUTH', 'EAST': 'WEST', ...}
                           
        Note:
            Must be implemented by concrete augmentation classes to define
            how cardinal directions are transformed (e.g., for rotations/flips)
        """
        pass

    @abstractmethod
    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        """
        Apply the core transformation operation to a tensor.
        
        Args:
            t: Input tensor to transform
            inverse: If True, apply inverse transformation
            is_policy: If True, handle as policy tensor (different dims)
            
        Returns:
            torch.Tensor: Transformed tensor
            
        Note:
            Must be implemented by concrete augmentation classes to define
            the actual transformation (e.g., torch.flip, torch.rot90)
        """
        pass

    def __repr__(self):
        return self.__class__.__name__

    def _transform_policy(self, policy: Dict[str, torch.Tensor], inverse: bool) -> Dict[str, torch.Tensor]:
        if inverse:
            return {
                space: p[..., self.transformed_action_idxs_inverse[space]]
                for space, p in policy.items()
            }
        else:
            return {
                space: p[..., self.transformed_action_idxs_forward[space]]
                for space, p in policy.items()
            }


class VerticalFlip(DataAugmenter):
    """
    Implements vertical (North-South) flip augmentation.
    
    
    This transformation:
    - Flips the game board vertically
    - Swaps NORTH and SOUTH actions
    - Preserves EAST and WEST actions
    """
    def get_directions_mapped(self) -> Dict[str, str]:
        # Switch all N/S actions
        return {
            Constants.DIRECTIONS.NORTH: Constants.DIRECTIONS.SOUTH,
            Constants.DIRECTIONS.EAST: Constants.DIRECTIONS.EAST,
            Constants.DIRECTIONS.SOUTH: Constants.DIRECTIONS.NORTH,
            Constants.DIRECTIONS.WEST: Constants.DIRECTIONS.WEST
        }

    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        dims = (-2,) if is_policy else (-1,)
        return torch.flip(t, dims=dims)


class HorizontalFlip(DataAugmenter):
    """
    Implements horizontal (East-West) flip augmentation.
    
    This transformation:
    - Flips the game board horizontally
    - Swaps EAST and WEST actions
    - Preserves NORTH and SOUTH actions
    """
    def get_directions_mapped(self) -> Dict[str, str]:
        # Switch all E/W actions
        return {
            Constants.DIRECTIONS.NORTH: Constants.DIRECTIONS.NORTH,
            Constants.DIRECTIONS.EAST: Constants.DIRECTIONS.WEST,
            Constants.DIRECTIONS.SOUTH: Constants.DIRECTIONS.SOUTH,
            Constants.DIRECTIONS.WEST: Constants.DIRECTIONS.EAST
        }

    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        dims = (-3,) if is_policy else (-2,)
        return torch.flip(t, dims=dims)


class Rot90(DataAugmenter):
    """
    Implements 90-degree rotation augmentation.
    
    This transformation:
    - Rotates the game board 90 degrees clockwise
    - Maps directions: N->E->S->W->N
    - Handles inverse rotation (-90 degrees) when needed
    """
    def get_directions_mapped(self) -> Dict[str, str]:
        # Rotate all actions 90 degrees
        return {
            Constants.DIRECTIONS.NORTH: Constants.DIRECTIONS.EAST,
            Constants.DIRECTIONS.EAST: Constants.DIRECTIONS.SOUTH,
            Constants.DIRECTIONS.SOUTH: Constants.DIRECTIONS.WEST,
            Constants.DIRECTIONS.WEST: Constants.DIRECTIONS.NORTH
        }

    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        k = -1 if inverse else 1
        dims = (-3, -2) if is_policy else (-2, -1)
        return torch.rot90(t, k=k, dims=dims)


class Rot180(DataAugmenter):
    """
    Implements 180-degree rotation augmentation.
    
    This transformation:
    - Rotates the game board 180 degrees
    - Swaps N<->S and E<->W
    - Is its own inverse operation
    """
    def get_directions_mapped(self) -> Dict[str, str]:
        # Rotate all actions 180 degrees
        return {
            Constants.DIRECTIONS.NORTH: Constants.DIRECTIONS.SOUTH,
            Constants.DIRECTIONS.EAST: Constants.DIRECTIONS.WEST,
            Constants.DIRECTIONS.SOUTH: Constants.DIRECTIONS.NORTH,
            Constants.DIRECTIONS.WEST: Constants.DIRECTIONS.EAST
        }

    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        dims = (-3, -2) if is_policy else (-2, -1)
        return torch.rot90(t, k=2, dims=dims)


class Rot270(DataAugmenter):
    """
    Implements 270-degree rotation augmentation.
    
    This transformation:
    - Rotates the game board 270 degrees clockwise (90 counter-clockwise)
    - Maps directions: N->W->S->E->N
    - Handles inverse rotation (90 degrees) when needed
    """
    def get_directions_mapped(self) -> Dict[str, str]:
        # Rotate all actions 270 degrees
        return {
            Constants.DIRECTIONS.NORTH: Constants.DIRECTIONS.WEST,
            Constants.DIRECTIONS.EAST: Constants.DIRECTIONS.NORTH,
            Constants.DIRECTIONS.SOUTH: Constants.DIRECTIONS.EAST,
            Constants.DIRECTIONS.WEST: Constants.DIRECTIONS.SOUTH
        }

    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        k = 1 if inverse else -1
        dims = (-3, -2) if is_policy else (-2, -1)
        return torch.rot90(t, k=k, dims=dims)


def player_relative_reflection(game_state: Game) -> DataAugmenter:
    """
    Choose reflection augmentation based on initial player city positions.
    
    This function analyzes the starting city positions and returns the
    appropriate reflection augmentation to normalize player positions:
    - VerticalFlip: When cities are vertically aligned (same x)
    - HorizontalFlip: When cities are horizontally aligned (same y)
    
    Args:
        game_state: Current game state with player positions
        
    Returns:
        DataAugmenter: Appropriate flip augmentation for the game layout
        
    Note:
        Used to standardize game states relative to player positions,
        making the learning task more consistent
    """
    p1_city_pos, p2_city_pos = [p.city_tiles[0].pos for p in game_state.players]
    if p1_city_pos.x == p2_city_pos.x:
        DEBUG_MESSAGE("Reflection mode: vertical")
        return VerticalFlip()
    else:
        DEBUG_MESSAGE("Reflection mode: horizontal")
        return HorizontalFlip()
