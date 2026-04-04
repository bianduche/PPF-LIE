"""
Color Consistency Module (CCM) for PPF-LIE

This module transforms the recovered continuous photon rate field into
standard sRGB color space while preserving color fidelity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ColorConsistencyModule(nn.Module):
    """
    Color Consistency Module for spatial color adaptation.

    Predicts local color affine transformations to generate the final
    enhanced image that conforms to standard sRGB color space.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        num_affine: int = 3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_affine = num_affine

        # Feature extraction from radiance field
        self.radiance_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Affine transformation predictor
        self.affine_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, num_affine * in_channels * 2, 1),
        )

    def forward(
        self,
        radiance_field: torch.Tensor,
        conditional_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply color consistency transformation.

        Args:
            radiance_field: Continuous photon rate field lambda_0
            conditional_features: Conditional features from low-light input

        Returns:
            Color-corrected enhanced image
        """
        batch_size, channels, height, width = radiance_field.shape

        # Extract radiance features
        radiance_feat = self.radiance_encoder(radiance_field)

        # Concatenate with conditional features
        combined_feat = torch.cat([radiance_feat, conditional_features], dim=1)

        # Predict per-pixel affine parameters
        # Output: (batch, num_affine * channels * 2, H, W)
        # 2 for scale and bias in affine transform
        affine_params = self.affine_predictor(combined_feat)

        # Reshape to get scale and bias
        num_params = self.num_affine * channels
        scale = affine_params[:, :num_params, :, :]
        bias = affine_params[:, num_params:num_params*2, :, :]

        # Normalize scale to be close to 1
        scale = torch.sigmoid(scale) * 2.0 + 0.5  # Range [0.5, 2.5]

        # Apply affine transformation
        output = radiance_field * scale + bias

        return output


class AdaptiveColorCorrection(nn.Module):
    """
    Adaptive color correction with learnable color transfer.
    """

    def __init__(self, num_segments: int = 4):
        super().__init__()
        self.num_segments = num_segments

        # Learnable color transformation matrix
        self.color_transform = nn.Parameter(torch.eye(3))

        # Segment-wise correction
        self.segment_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, num_segments * 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive color correction.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Color-corrected tensor
        """
        batch_size = x.shape[0]

        # Global color transformation
        x_flat = x.view(batch_size, 3, -1)
        transformed = torch.matmul(self.color_transform, x_flat)
        x = transformed.view_as(x)

        # Segment-wise adjustment
        weights = self.segment_weights(x)
        weights = weights.view(batch_size, self.num_segments, 3, 1, 1)

        # Apply segment-wise correction
        x = x.unsqueeze(1) * weights
        x = x.sum(dim=1)

        return x
