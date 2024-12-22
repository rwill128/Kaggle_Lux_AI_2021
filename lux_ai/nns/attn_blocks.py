import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Tuple

from .weight_init import trunc_normal_


class RelPosSelfAttention(nn.Module):
    """
    Relative Position Self Attention module for processing spatial game state information.
    This attention mechanism is crucial for the RL agent to understand spatial relationships
    between game elements like units, resources, and cities on the game map.
    
    The relative positional encoding allows the model to better understand:
    1. Unit positioning and movement possibilities
    2. Resource proximity and accessibility
    3. City tile connections and expansion opportunities
    4. Enemy unit threat ranges and strategic positions
    
    Original implementation from: https://gist.github.com/ShoufaChen/ec7b70038a6fdb488da4b34355380569
    
    Key RL applications:
    - Enables the agent to learn spatial patterns in the game state
    - Helps in planning unit movements considering relative positions
    - Assists in resource gathering by understanding proximity relationships
    - Supports strategic city placement through spatial understanding
    """

    def __init__(self, h: int, w: int, dim: int, relative=True, fold_heads=False):
        super(RelPosSelfAttention, self).__init__()
        self.relative = relative
        self.fold_heads = fold_heads
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits += self.relative_logits(q)
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = weights + torch.where(
            attn_mask.view(-1, 1, 1, 1, h * w),
            torch.zeros_like(weights),
            torch.zeros_like(weights) + float('-inf')
        )
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        """
        Compute relative position logits for both width and height dimensions.
        This is crucial for the RL agent to understand spatial relationships on the game board.

        Args:
            q: Query tensor of shape (batch_size, n_heads, height, width, dim)
               Represents features for attention queries

        Returns:
            torch.Tensor: Combined relative position logits for both dimensions
        
        Process:
        1. Compute relative logits in width dimension using rel_emb_w
        2. Compute relative logits in height dimension using rel_emb_h
        3. Combine both dimensions to capture full 2D spatial relationships

        Strategic importance:
        - Enables understanding of unit positions relative to resources
        - Helps in planning city expansions in optimal directions
        - Supports tactical positioning in combat scenarios
        - Facilitates efficient resource gathering paths
        """
        # Relative logits in width dimension.
        rel_logits_w = self.relative_logits_1d(
            q,
            self.rel_emb_w,
            transpose_mask=[0, 1, 2, 4, 3, 5]
        )
        # Relative logits in height dimension
        rel_logits_h = self.relative_logits_1d(
            q.permute(0, 1, 3, 2, 4),
            self.rel_emb_h,
            transpose_mask=[0, 1, 4, 2, 5, 3]
        )
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        """
        Compute relative position logits for a single dimension (width or height).
        This method processes spatial relationships along one axis of the game board.

        Args:
            q: Query tensor of shape (batch_size, n_heads, h, w, dim)
            rel_k: Relative position embeddings
            transpose_mask: List defining the permutation for proper tensor alignment

        Returns:
            torch.Tensor: Relative position logits for the specified dimension

        Process:
        1. Compute attention logits between queries and relative position keys
        2. Reshape for efficient relative-to-absolute conversion
        3. Convert relative positions to absolute positions
        4. Reshape and align dimensions for the full attention computation

        Used for:
        - Processing directional relationships (horizontal/vertical)
        - Understanding unit movement possibilities
        - Analyzing resource accessibility
        - Planning expansion directions
        """
        bs, heads, h, w, dim = q.shape
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits = rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative position indexing to absolute position indexing.
        This conversion is essential for the RL agent to properly process
        spatial relationships in the game state.

        Args:
            x: Tensor with relative indices [bs, heads, length, 2*length - 1]
               Contains relative position information between game board positions

        Returns:
            torch.Tensor: Tensor with absolute indices [bs, heads, length, length]
                         Represents relationships between absolute positions

        Strategic importance:
        - Enables precise unit positioning calculations
        - Supports accurate distance-based decision making
        - Facilitates efficient pathfinding
        - Helps in analyzing strategic position control

        Implementation note:
        The conversion process maintains the spatial structure of the game board
        while transforming relative position information into a format suitable
        for attention computations.
        """
        bs, heads, length, _ = x.shape
        col_pad = torch.zeros((bs, heads, length, 1), dtype=x.dtype, device=x.device)
        x = torch.cat([x, col_pad], dim=3)
        flat_x = torch.reshape(x, [bs, heads, -1])
        flat_pad = torch.zeros((bs, heads, length - 1), dtype=x.dtype, device=x.device)
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        final_x = torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x = final_x[:, :, :length, length - 1:]
        return final_x


class GroupPointWise(nn.Module):
    """
    Group-wise point-wise convolution layer for efficient feature processing in the RL agent.
    This layer helps in processing multi-head features while maintaining computational efficiency.
    
    In the context of Lux AI:
    - Processes features from different aspects of the game state (units, resources, cities)
    - Enables parallel processing of different strategic considerations
    - Helps in combining different types of game information efficiently
    
    The grouped processing allows the model to:
    1. Learn specialized features for different game aspects
    2. Process multiple strategic considerations in parallel
    3. Maintain computational efficiency with grouped convolutions
    4. Scale feature processing based on input complexity
    """
    def __init__(self, in_channels, heads=4, proj_factor=1, target_dimension=None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels = target_dimension // proj_factor
        else:
            proj_channels = in_channels // proj_factor
        self.w = nn.Parameter(
            torch.Tensor(in_channels, heads, proj_channels // heads)
        )

        nn.init.normal_(self.w, std=0.01)

    def forward(self, x):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC
        x = x.permute(0, 2, 3, 1)
        """
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        out = torch.einsum('bhwc,cnp->bnhwp', x, self.w)
        return out


class RPSA(nn.Module):
    """
    Relative Position Self-Attention module specifically designed for the Lux AI agent.
    This module combines relative positional encoding with self-attention to process
    the game state while maintaining spatial relationships.
    
    Key components for RL:
    1. Query/Key/Value projections for attention computation
    2. Relative position encoding for spatial awareness
    3. Multi-head attention for parallel feature processing
    
    Strategic benefits:
    - Enables long-range dependencies in strategic planning
    - Maintains spatial relationships for tactical decisions
    - Processes multiple game aspects simultaneously
    - Helps in understanding complex board states
    
    The module is crucial for:
    - Unit coordination and movement planning
    - Resource allocation strategies
    - City expansion decisions
    - Enemy threat assessment
    """
    def __init__(self, in_channels, heads, height, width, pos_enc_type='relative'):
        super(RPSA, self).__init__()
        self.q_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.k_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.v_proj = GroupPointWise(in_channels, heads, proj_factor=1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type == 'relative':
            self.self_attention = RelPosSelfAttention(height, width, in_channels // heads, fold_heads=True)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        o = self.self_attention(q=q, k=k, v=v, attn_mask=input_mask).permute(0, 3, 1, 2)
        return o


class GPSA(nn.Module):
    """
    Gated Positional Self-Attention module adapted for the Lux AI environment.
    Originally from: https://github.com/facebookresearch/convit/blob/main/convit.py
    
    This advanced attention mechanism combines gating with positional awareness,
    allowing the RL agent to dynamically focus on relevant spatial regions of the game state.
    
    Key RL features:
    1. Gating mechanism to control attention flow
    2. Learnable positional encoding
    3. Local-global attention balance
    4. Multi-head processing for parallel strategy evaluation
    
    Strategic applications:
    - Dynamic focus on critical game areas
    - Balanced local-global strategic planning
    - Adaptive attention based on game phase
    - Efficient processing of complex game states
    
    Implementation details:
    - Uses locality strength to control attention spread
    - Combines patch-based and position-based attention
    - Implements efficient relative position computation
    - Supports masked attention for valid action selection
    """
    def __init__(self, dim, height, width, n_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True):
        super().__init__()
        self.num_heads = n_heads
        self.dim = dim
        self.height = height
        self.width = width
        head_dim = dim // n_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, n_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.apply(self._init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)
        self.register_buffer("rel_indices", self.get_rel_indices())

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=-2, end_dim=-1).permute(0, 2, 1)
        input_mask = torch.flatten(input_mask, start_dim=-2, end_dim=-1)
        B, N, C = x.shape

        attn = self.get_attention(x, input_mask)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.permute(0, 2, 1)
        x = x.view(B, C, self.height, self.width)
        return x

    def get_attention(self, x, input_mask):
        """
        Compute gated positional attention weights for the game state.

        Args:
            x: Input tensor of shape (batch_size, n_patches, dim)
               Flattened game state features
            input_mask: Mask tensor of shape (batch_size, n_patches)
                       Indicates valid positions in the game state

        Returns:
            torch.Tensor: Attention weights combining positional and content-based attention,
                         shape (batch_size, n_heads, n_patches, n_patches)
        
        The attention computation:
        1. Projects input to queries and keys
        2. Computes content-based attention (patch_score)
        3. Computes position-based attention (pos_score)
        4. Combines both using learnable gating
        5. Applies masking and normalization
        """
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1, -1)
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        attn = attn * input_mask.float().unsqueeze(-2)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        return attn

    def get_attention_map(self, x, input_mask, return_map=False):
        attn_map = self.get_attention(x, input_mask).mean(0)  # average over batch
        distances = self.rel_indices.squeeze()[:, :, -1] ** .5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self, locality_strength=1.):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1  # max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads ** .5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self):
        """
        Compute relative position indices for the game board.
        Creates a tensor encoding relative positions between all pairs of cells.

        Returns:
            torch.Tensor: Relative position indices of shape (1, n_patches, n_patches, 3)
                         Each position contains (dx, dy, d^2) where:
                         - dx: relative x-coordinate difference
                         - dy: relative y-coordinate difference
                         - d^2: squared Euclidean distance

        Used for:
        - Unit movement planning
        - Resource distance evaluation
        - City placement strategies
        - Combat positioning
        """
        assert self.height == self.width
        img_size = self.height
        num_patches = self.height * self.width
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        return rel_indices.to(device)


class ViTBlock(nn.Module):
    """
    Vision Transformer Block adapted for the Lux AI reinforcement learning agent.
    This block combines multi-head self-attention with MLP layers to process
    spatial game information and extract strategic features.
    
    Architecture components:
    1. Layer normalization for stable training
    2. Multi-head self-attention for parallel feature processing
    3. MLP with GELU activation for non-linear transformations
    4. Residual connections for gradient flow
    
    RL-specific features:
    - Processes game state as a spatial grid
    - Maintains input masking for valid action selection
    - Combines local and global game information
    - Supports stable policy and value prediction
    
    The block is crucial for:
    - Strategic feature extraction
    - Long-range dependency modeling
    - Action space understanding
    - Value function approximation
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            height: int,
            width: int,
            mhsa_layer: nn.Module,
            normalize: bool = True,
            activation: Callable = nn.GELU
    ):
        super(ViTBlock, self).__init__()

        self.norm1 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.mhsa = mhsa_layer

        self.norm2 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1)
            ),
            activation(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1)
            )
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process game state through the Vision Transformer block.

        Args:
            x: Tuple of (features, mask) where:
               - features: Game state tensor of shape (batch_size, channels, height, width)
               - mask: Valid position mask of shape (batch_size, 1, height, width)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed (features, mask) with:
            - features: Updated game state incorporating global dependencies
            - mask: Unchanged input mask for maintaining valid positions

        Processing steps:
        1. Layer normalization for training stability
        2. Multi-head self-attention for global context
        3. Residual connection to preserve spatial information
        4. MLP processing for feature transformation
        5. Final residual connection and masking
        """
        x, input_mask = x
        identity = x
        x = self.mhsa(self.norm1(x), input_mask)
        x = x + identity
        return (self.mlp(self.norm2(x)) + x) * input_mask, input_mask
