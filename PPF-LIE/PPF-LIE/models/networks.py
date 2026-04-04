"""
Network architectures for PPF-LIE

Includes UNet backbone, time embedding, and conditional encoder.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List


class TimeEmbedding(nn.Module):
    """
    Sinusoidal Time Embedding for diffusion timesteps.

    Follows the original DDPM implementation using sinusoidal position encoding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps into continuous vector.

        Args:
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Embedded timestep of shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        # Compute embeddings
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]

        # Concatenate sin and cos
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        return embeddings


class ConditionalEncoder(nn.Module):
    """
    Conditional feature encoder for extracting multi-scale features from low-light input.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        # Multi-scale feature extraction
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else base_channels * (2 ** (i-1)),
                         base_channels * (2 ** i), 3, stride=2 if i > 0 else 1, padding=1),
                nn.BatchNorm2d(base_channels * (2 ** i)),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for i in range(4)
        ])

        # Channel reduction for decoder
        self.channel_reduce = nn.ModuleList([
            nn.Conv2d(base_channels * (2 ** i), base_channels, 1)
            for i in range(4)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale conditional features.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Multi-scale features
        """
        features = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            # Reduce channels for efficient fusion
            reduced = self.channel_reduce[i](x)
            features.append(reduced)

        return features[0]  # Return primary feature for simplicity


class ResidualBlock(nn.Module):
    """
    Residual block with group normalization and SiLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU(inplace=True)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels * 2) if time_channels else None,
        )

        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional time conditioning.

        Args:
            x: Input tensor (B, C, H, W)
            t_emb: Time embedding (B, time_dim)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        # Add time embedding
        if t_emb is not None and self.time_mlp[1] is not None:
            t_proj = self.time_mlp(t_emb)
            # Split into scale and shift
            scale, shift = t_proj.chunk(2, dim=-1)
            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]
            h = h * (1 + scale) + shift

        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block for feature aggregation.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(8, channels)

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Attended tensor (B, C, H, W)
        """
        B, C, H, W = x.shape
        h = self.norm(x)

        # Reshape for attention
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        # Compute QKV
        qkv = self.qkv(h).reshape(B, H*W, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        h = (attn @ v).reshape(B, H*W, C)

        # Project back
        h = self.proj(h)

        # Reshape back
        h = h.permute(0, 2, 1).reshape(B, C, H, W)

        return x + h


class DownBlock(nn.Module):
    """Downsampling block with residual and attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool = False,
    ):
        super().__init__()

        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else None
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downsampling with skip connection."""
        x = self.res(x, t_emb)
        skip = x
        if self.attn is not None:
            x = self.attn(x)
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with residual and attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool = False,
    ):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Upsampling with skip connection."""
        x = self.up(x)

        # Handle size mismatch
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for denoising with time conditioning.

    Predicts the continuous photon rate field from noisy observations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 6,
        time_channels: int = 128,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, time_channels * 4),
            nn.SiLU(),
            nn.Linear(time_channels * 4, time_channels),
        )

        # Encoder
        self.encoder = nn.ModuleList()
        ch = base_channels
        for i in range(num_layers):
            self.encoder.append(
                DownBlock(ch, ch * 2, time_channels, has_attn=(i % 2 == 1))
            )
            ch *= 2

        # Middle
        self.middle = nn.ModuleList([
            ResidualBlock(ch, ch, time_channels),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_channels),
        ])

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_layers):
            self.decoder.append(
                UpBlock(ch * 2, ch // 2, time_channels, has_attn=((num_layers - i) % 2 == 1))
            )
            ch //= 2

        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with time conditioning.

        Args:
            x: Input tensor (B, C, H, W)
            t_emb: Time embedding (B, time_channels)
            cond_features: Optional conditional features

        Returns:
            Predicted photon rate field (B, C, H, W)
        """
        # Project time embedding
        t_emb = self.time_mlp(t_emb)

        # Initial convolution
        h = self.input_conv(x)

        # Encoder with skip connections
        skips = []
        for down in self.encoder:
            h, skip = down(h, t_emb)
            skips.append(skip)

        # Middle block
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        # Decoder
        for up in self.decoder:
            skip = skips.pop()
            h = up(h, skip, t_emb)

        # Output
        h = self.output_conv(h)

        return h
