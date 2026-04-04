"""
Loss functions for PPF-LIE training.

Includes Poisson NLL loss, perceptual loss, and total variation loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PoissonNLLLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood Loss.

    This loss is derived from the Poisson distribution that governs photon counting
    in low-light imaging. It is the core loss for training PPF-LIE.

    Mathematically:
        L = sum(lambda_0 - N_0 * log(lambda_0))
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Poisson NLL loss.

        Args:
            pred: Predicted photon rate field lambda_0
            target: Ground truth discrete photon counts

        Returns:
            Scalar loss
        """
        # Clamp predictions to prevent log(0)
        pred = torch.clamp(pred, min=self.eps)

        # Poisson NLL: lambda - N * log(lambda)
        loss = pred - target * torch.log(pred)

        return loss.mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.

    Measures perceptual similarity in deep feature space.
    """

    def __init__(
        self,
        network: str = 'vgg19',
        layers: list = None,
        weights: list = None,
    ):
        super().__init__()

        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']

        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.layers = layers
        self.weights = weights

        # Load VGG network
        vgg = models.vgg19(pretrained=True).features

        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def extract_features(self, x: torch.Tensor) -> list:
        """Extract features from VGG layers."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted image
            target: Ground truth image

        Returns:
            Perceptual loss
        """
        # Normalize to ImageNet stats
        mean = self.mean.to(pred.device)
        std = self.std.to(pred.device)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        # Extract features
        pred_features = self.extract_features(pred_norm)
        target_features = self.extract_features(target_norm)

        # Compute weighted L1 loss
        loss = 0
        for pred_feat, target_feat, weight in zip(
            pred_features, target_features, self.weights
        ):
            loss += weight * F.l1_loss(pred_feat, target_feat)

        return loss


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for smooth outputs.

    Encourages spatial smoothness in predictions.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            TV loss
        """
        batch_size, channels, height, width = x.shape

        # Compute differences in horizontal and vertical directions
        diff_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        diff_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

        loss = diff_x.sum() + diff_y.sum()

        # Normalize by number of elements
        num_elements = batch_size * channels * (height - 1) * (width - 1)
        loss = loss / num_elements

        return self.weight * loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for PPF-LIE training.

    Total loss = diffusion_loss + gamma1 * perceptual_loss
                + gamma2 * color_loss + gamma3 * tv_loss
    """

    def __init__(
        self,
        gamma_perceptual: float = 0.1,
        gamma_color: float = 0.05,
        gamma_tv: float = 0.01,
    ):
        super().__init__()

        self.poisson_loss = PoissonNLLLoss()
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TotalVariationLoss()
        self.color_loss = nn.L1Loss()

        self.gamma_perceptual = gamma_perceptual
        self.gamma_color = gamma_color
        self.gamma_tv = gamma_tv

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_dict: bool = False,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            pred: Predicted enhanced image
            target: Ground truth image
            return_dict: Whether to return individual losses

        Returns:
            Combined loss or dictionary of losses
        """
        # Poisson NLL loss (main diffusion loss)
        loss_diffusion = self.poisson_loss(pred, target)

        # Perceptual loss
        loss_perceptual = self.perceptual_loss(pred, target)

        # Color consistency loss
        loss_color = self.color_loss(pred, target)

        # Total variation loss
        loss_tv = self.tv_loss(pred)

        # Combined loss
        total_loss = (
            loss_diffusion
            + self.gamma_perceptual * loss_perceptual
            + self.gamma_color * loss_color
            + self.gamma_tv * loss_tv
        )

        if return_dict:
            return {
                'total': total_loss,
                'diffusion': loss_diffusion,
                'perceptual': loss_perceptual,
                'color': loss_color,
                'tv': loss_tv,
            }

        return total_loss
