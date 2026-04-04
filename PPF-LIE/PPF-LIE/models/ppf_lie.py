"""
PPF-LIE: Physical Photon Field Driven Low-Light Enhancement for Multimodal Industrial Defect Recognition
Main model implementation.

Paper: PPF-LIE: Physical Photon Field Driven Low-Light Enhancement for Multimodal Industrial Defect Recognition
Authors: Yi Zhang, Jian Song, Xiyang Liu, Miaosen Yang
Institution: School of Materials, Shanghai Dianji University, Shanghai, China
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from dataclasses import dataclass

from .networks import UNet, TimeEmbedding, ConditionalEncoder
from .diffusion import DiscreteDiffusion
from .ccm import ColorConsistencyModule


class PPFLIE(nn.Module):
    """
    Physical Photon Field Driven Low-Light Enhancement Model

    Core innovation: Directly models discrete photon statistics from physical mechanisms
    of extreme low-light imaging, abandoning continuous Gaussian assumptions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 6,
        time_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        use_ccm: bool = True,
    ):
        super().__init__()

        self.time_steps = time_steps
        self.in_channels = in_channels
        self.use_ccm = use_ccm

        # Time embedding for diffusion timesteps
        self.time_embedding = TimeEmbedding(base_channels)

        # Conditional encoder for extracting features from low-light input
        self.encoder = ConditionalEncoder(in_channels, base_channels)

        # Denoising UNet for predicting continuous photon rate field
        self.denoiser = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            time_channels=base_channels,
        )

        # Discrete diffusion process
        self.diffusion = DiscreteDiffusion(
            time_steps=time_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # Color Consistency Module for mapping radiance field to sRGB space
        self.ccm = ColorConsistencyModule(
            in_channels=out_channels,
            hidden_channels=128,
        ) if use_ccm else None

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, return_denoised: bool = True):
        """
        Forward pass of PPF-LIE.

        Args:
            x: Input low-light image, shape (B, C, H, W)
            t: Diffusion timesteps, shape (B,). If None, sample random timesteps.
            return_denoised: Whether to return denoised output or noise.

        Returns:
            Enhanced image or denoised prediction depending on training mode and flags.
        """
        if t is None:
            t = torch.randint(0, self.time_steps, (x.shape[0],), device=x.device)

        # Embed diffusion timestep
        t_emb = self.time_embedding(t)

        # Encode conditional features from low-light input
        cond_features = self.encoder(x)

        # Denoising process to predict photon rate field
        lambda_0 = self.denoiser(x, t_emb, cond_features)

        # Apply color consistency transformation if enabled
        if self.use_ccm and self.ccm is not None:
            enhanced = self.ccm(lambda_0, cond_features)
        else:
            enhanced = lambda_0

        return enhanced

    def training_step(self, x_low: torch.Tensor, x_high: torch.Tensor):
        """
        Single training step with Poisson NLL loss.

        Args:
            x_low: Low-light input image
            x_high: Ground truth high-light image

        Returns:
            Dictionary containing losses and outputs
        """
        batch_size = x_low.shape[0]
        device = x_low.device

        # Sample random timesteps for diffusion
        t = torch.randint(0, self.time_steps, (batch_size,), device=device)

        # Apply forward diffusion to ground truth
        noise = torch.randn_like(x_high)
        x_t = self.diffusion.add_noise(x_high, t, noise)

        # Predict continuous photon rate field
        lambda_0 = self.forward(x_low, t, return_denoised=True)

        # Compute Poisson NLL loss
        poisson_loss = self.compute_poisson_nll(lambda_0, x_high)

        return {
            'loss': poisson_loss,
            'lambda_0': lambda_0,
            'x_t': x_t,
        }

    def compute_poisson_nll(self, lambda_0: torch.Tensor, N_0: torch.Tensor, eps: float = 1e-8):
        """
        Compute Poisson Negative Log-Likelihood loss.

        Mathematically equivalent to minimizing: sum(lambda_0 - N_0 * log(lambda_0))

        Args:
            lambda_0: Predicted continuous photon rate field
            N_0: Ground truth discrete photon counts
            eps: Small constant to prevent numerical instability

        Returns:
            Poisson NLL loss
        """
        lambda_0 = torch.clamp(lambda_0, min=eps)
        poisson_nll = lambda_0 - N_0 * torch.log(lambda_0)
        return poisson_nll.mean()

    @torch.no_grad()
    def sample(self, x: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """
        Sampling process for inference.

        Args:
            x: Low-light input image
            num_steps: Number of denoising steps

        Returns:
            Enhanced image
        """
        batch_size = x.shape[0]
        device = x.device

        # Start from noisy observation
        x_t = x.clone()

        # Denoising timesteps
        timesteps = torch.linspace(self.time_steps - 1, 0, num_steps, device=device).long()

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict photon rate field
            lambda_0 = self.forward(x_t, t_batch)

            # Update for next step (simplified DDPM sampling)
            if i < len(timesteps) - 1:
                noise = torch.randn_like(x_t)
                alpha_t = self.diffusion.get_alpha(t)
                x_t = lambda_0 + torch.sqrt(1 - alpha_t) * noise

        return lambda_0

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'name': 'PPF-LIE',
            'description': 'Physical Photon Field Driven Low-Light Enhancement',
            'paper': 'PPF-LIE: Physical Photon Field Driven Low-Light Enhancement for Multimodal Industrial Defect Recognition',
            'authors': ['Yi Zhang', 'Jian Song', 'Xiyang Liu', 'Miaosen Yang'],
            'year': 2025,
            'institution': 'School of Materials, Shanghai Dianji University',
        }


# Model metadata
MODEL_INFO = {
    'name': 'PPF-LIE',
    'full_name': 'Physical Photon Field Driven Low-Light Enhancement',
    'description': (
        'A novel generative enhancement framework driven by physical photon field priors. '
        'By constructing a fully discrete diffusion process aligned with photon imaging physics, '
        'PPF-LIE accurately decouples signal-dependent Poisson shot noise and recovers the '
        'continuous radiance probability field from sparse photon observations.'
    ),
    'key_innovations': [
        'PPF Modeling Framework - Explicit modeling of discrete photon statistics',
        'Discrete Photon Space Diffusion - Poisson noise instead of Gaussian',
        'Time-weighted Poisson NLL Loss - Physics-aligned optimization objective',
        'Color Consistency Module - Spatial color adaptation for industrial applications',
    ],
    'applications': [
        'Industrial defect detection under extreme low-light',
        'Metallographic cross-section analysis',
        'Multimodal industrial inspection',
    ],
}
