from typing import Tuple
import torch
from torch import nn

from .conv_blocks import ResidualBlock


class UNET(nn.Module):
    """
    U-Net architecture for processing spatial game state information.
    
    This implementation follows the classic U-Net design with:
    - Downsampling path (encoder) with residual blocks and average pooling
    - Bottleneck layer with maximum feature channels
    - Upsampling path (decoder) with transposed convolutions and skip connections
    
    Architecture Details:
    - Three resolution levels with progressive channel doubling
    - Residual blocks at each resolution for feature extraction
    - Skip connections between encoder and decoder paths
    - Input mask propagation for valid position tracking
    
    Features:
    - Maintains spatial information through skip connections
    - Processes multi-scale features efficiently
    - Handles variable input sizes through dynamic padding
    - Preserves valid position information via mask propagation
    
    Game-Specific Considerations:
    - Designed for processing game board states
    - Maintains spatial relationships between game elements
    - Supports both local and global feature extraction
    - Handles masked inputs for valid game positions
    """
    def __init__(
            self,
            n_blocks_per_reduction: int,
            in_out_channels: int,
            height: int,
            width: int,
            **residual_block_kwargs
    ):
        """
        Initialize the U-Net architecture.
        
        Args:
            n_blocks_per_reduction: Number of residual blocks at each resolution level
            in_out_channels: Number of input and output channels
            height: Height of the input feature maps
            width: Width of the input feature maps
            **residual_block_kwargs: Additional arguments passed to ResidualBlock
            
        Architecture Layout:
        1. Encoder Path (Downsampling):
           - block1_down: n residual blocks at original resolution
           - reduction1: 2x downsampling
           - block2_down: n residual blocks at half resolution
           - reduction2: 2x downsampling
           
        2. Bottleneck:
           - block3: n residual blocks at quarter resolution
           
        3. Decoder Path (Upsampling):
           - expansion2: 2x upsampling
           - block2_up: n residual blocks with skip connection from block2
           - expansion1: 2x upsampling
           - block1_up: n residual blocks with skip connection from block1
           
        Channel Progression:
        - Level 1: in_out_channels
        - Level 2: 2 * in_out_channels
        - Level 3: 4 * in_out_channels
        
        Note:
        - Features are processed at multiple scales
        - Skip connections preserve spatial details
        - Input masks are propagated through the network
        """
        super(UNET, self).__init__()

        block1_channels = in_out_channels
        block1_h = height
        block1_w = width

        block2_channels = 2 * in_out_channels
        block2_h = height // 2
        block2_w = width // 2

        block3_channels = 4 * in_out_channels
        block3_h = height // 4
        block3_w = width // 4
        
        self.block1_down = nn.Sequential(
            *[ResidualBlock(
                in_channels=block1_channels,
                out_channels=block1_channels,
                height=block1_h,
                width=block1_w,
                **residual_block_kwargs
            ) for _ in range(n_blocks_per_reduction)]
        )
        self.reduction1 = nn.AvgPool2d(kernel_size=2)
        self.expansion1 = nn.ConvTranspose2d(block2_channels, block1_channels, kernel_size=(2, 2), stride=(2, 2))
        self.block1_up = nn.Sequential(
            ResidualBlock(
                in_channels=block2_channels,
                out_channels=block1_channels,
                height=block1_h,
                width=block1_w,
                **residual_block_kwargs
            ),
            *[ResidualBlock(
                in_channels=block1_channels,
                out_channels=block1_channels,
                height=block1_h,
                width=block1_w,
                **residual_block_kwargs
            ) for _ in range(n_blocks_per_reduction - 1)]
        )

        self.block2_down = nn.Sequential(
            ResidualBlock(
                in_channels=block1_channels,
                out_channels=block2_channels,
                height=block2_h,
                width=block2_w,
                **residual_block_kwargs
            ),
            *[ResidualBlock(
                in_channels=block2_channels,
                out_channels=block2_channels,
                height=block2_h,
                width=block2_w,
                **residual_block_kwargs
            ) for _ in range(n_blocks_per_reduction - 1)]
        )
        self.reduction2 = nn.AvgPool2d(kernel_size=2)
        self.expansion2 = nn.ConvTranspose2d(block3_channels, block2_channels, kernel_size=(2, 2), stride=(2, 2))
        self.block2_up = nn.Sequential(
            ResidualBlock(
                in_channels=block3_channels,
                out_channels=block2_channels,
                height=block2_h,
                width=block2_w,
                **residual_block_kwargs
            ),
            *[ResidualBlock(
                in_channels=block2_channels,
                out_channels=block2_channels,
                height=block2_h,
                width=block2_w,
                **residual_block_kwargs
            ) for _ in range(n_blocks_per_reduction - 1)]
        )

        self.block3 = nn.Sequential(
            ResidualBlock(
                in_channels=block2_channels,
                out_channels=block3_channels,
                height=block3_h,
                width=block3_w,
                **residual_block_kwargs
            ),
            *[ResidualBlock(
                in_channels=block3_channels,
                out_channels=block3_channels,
                height=block3_h,
                width=block3_w,
                **residual_block_kwargs
            ) for _ in range(n_blocks_per_reduction - 1)]
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through U-Net architecture.
        
        Args:
            x: Tuple of (features, mask) where:
               - features: Input tensor of shape (batch, channels, height, width)
               - mask: Binary mask indicating valid positions
               
        Returns:
            Tuple of (processed_features, mask) where:
            - processed_features: Output tensor with same shape as input
            - mask: Original input mask
            
        Processing Flow:
        1. Encoder Path:
           - Process through block1_down at original resolution
           - Downsample and process through block2_down
           - Further downsample and process through block3
           
        2. Decoder Path:
           - Upsample block3 output and concatenate with block2 features
           - Process through block2_up
           - Upsample and concatenate with block1 features
           - Process through block1_up
           
        Note:
        - Skip connections preserve spatial information
        - Masks are used to focus on valid game positions
        - Feature maps are concatenated channel-wise
        """
        x_orig, input_mask_orig = x

        x1, input_mask1 = self.block1_down((x_orig, input_mask_orig.float()))
        x2, input_mask2 = self.block2_down((self.reduction1(x1), self.reduction1(input_mask1)))
        x3, _ = self.block3((self.reduction2(x2), self.reduction2(input_mask2)))

        x_out = torch.cat([x2, self.expansion2(x3)], dim=-3) * input_mask2
        x_out, _ = self.block2_up((x_out, input_mask2))
        x_out = torch.cat([x1, self.expansion1(x_out)], dim=-3) * input_mask1
        x_out, _ = self.block1_up((x_out, input_mask_orig))

        return x_out, input_mask_orig
