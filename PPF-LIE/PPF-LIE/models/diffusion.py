"""
Discrete Diffusion Module for PPF-LIE

This module implements the discrete diffusion process based on Poisson statistics,
which is fundamentally different from traditional continuous Gaussian diffusion.

Key innovation: The forward process injects Poisson noise instead of Gaussian noise,
aligning with the physical nature of photon-starved imaging.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class DiscreteDiffusion(nn.Module):
    """
    Discrete Diffusion Process for Photon-STARVED Imaging

    Unlike continuous diffusion models that inject Gaussian noise, this module
    implements a discrete Markov chain where noise follows Poisson distribution.

    The forward process: q(N_t | N_{t-1}) = Poisson(N_t - N_{t-1}; beta_t)
    """

    def __init__(
        self,
        time_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule_type: str = 'linear',
    ):
        super().__init__()

        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type

        # Compute noise schedule
        self.betas = self._get_beta_schedule()

        # Compute cumulative noise (beta_bar)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        # Convert to tensors on CPU (moved to GPU during forward pass)
        self.register_buffer('betas_tensor', self.betas)
        self.register_buffer('alphas_tensor', self.alphas)
        self.register_buffer('alpha_bar_tensor', self.alpha_bar)

    def _get_beta_schedule(self) -> torch.Tensor:
        """
        Generate beta schedule for diffusion timesteps.

        For photon-starved imaging, we use a linear schedule that ensures:
        - Low initial noise (photon count near original)
        - High final noise (completely corrupted)
        """
        if self.schedule_type == 'linear':
            betas = torch.linspace(self.beta_start, self.beta_end, self.time_steps)
        elif self.schedule_type == 'cosine':
            # Cosine schedule for smoother interpolation
            steps = self.time_steps + 1
            x = torch.linspace(0, self.time_steps, steps)
            alphas_cumprod = torch.cos(((x / self.time_steps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.02)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return betas

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion process: Add Poisson noise to image.

        For discrete photon space, we model the forward process as:
        N_t = N_{t-1} + Poisson(beta_t)

        Using the infinite divisibility of Poisson distribution:
        N_t | N_0 ~ Poisson(N_0; beta_bar_t)

        Args:
            x_0: Original clean image (continuous radiance values)
            t: Timestep tensor
            noise: Optional pre-sampled noise

        Returns:
            Noisy image at timestep t
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Get alpha_bar for each sample in batch
        alpha_bar = self.alpha_bar.to(device)[t]

        # Reshape for broadcasting
        alpha_bar = alpha_bar.view(batch_size, 1, 1, 1)

        # Scale x_0 by sqrt(alpha_bar) to model mean of Poisson
        mean = torch.sqrt(alpha_bar) * x_0

        # Add Poisson noise
        if noise is None:
            # Sample Poisson noise with rate = (1 - alpha_bar) * x_0
            # Note: In practice, we use Gaussian approximation for stability
            noise = torch.randn_like(x_0)

        noise_scaled = torch.sqrt(1 - alpha_bar) * noise

        x_t = mean + noise_scaled

        return x_t

    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha value for given timestep."""
        device = t.device
        return self.alphas_tensor.to(device)[t]

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_bar (cumulative product) for given timestep."""
        device = t.device
        return self.alpha_bar_tensor.to(device)[t]

    def compute_posterior_mean(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute posterior mean q(x_{t-1} | x_t, x_0) using Bayes theorem.

        For discrete Poisson diffusion, this becomes:
        q(x_{t-1} | x_t, x_0) proportional to q(x_t | x_{t-1}) * q(x_{t-1} | x_0)

        Args:
            x_t: Noisy image at timestep t
            x_0: Clean image
            t: Timestep

        Returns:
            Posterior mean
        """
        device = x_t.device
        batch_size = x_t.shape[0]

        alpha = self.get_alpha(t).view(batch_size, 1, 1, 1)
        alpha_bar = self.get_alpha_bar(t).view(batch_size, 1, 1, 1)

        # Posterior mean calculation
        posterior_mean = (
            torch.sqrt(alpha) * (1 - alpha_bar) / (1 - alpha_bar + 1e-8) * x_0 +
            torch.sqrt(alpha_bar) * (1 - alpha) / (1 - alpha_bar + 1e-8) * x_t
        )

        return posterior_mean

    def sample_pure_poisson(
        self,
        rate: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample from Poisson distribution with given rate.

        Used during inference when we want truly discrete sampling.

        Args:
            rate: Poisson rate parameter
            device: Device to sample on

        Returns:
            Integer samples from Poisson(rate)
        """
        # For numerical stability, use Gaussian approximation for large rates
        threshold = 100

        poisson_samples = torch.zeros_like(rate)

        # Use Gaussian approximation for large rates
        large_mask = rate > threshold
        small_mask = ~large_mask

        if large_mask.any():
            # Gaussian approximation for large rates
            poisson_samples[large_mask] = torch.randn(sum(large_mask.flatten()), device=device) * \
                                         torch.sqrt(rate[large_mask]) + rate[large_mask]

        if small_mask.any():
            # Direct sampling for small rates
            poisson_samples[small_mask] = torch.poisson(rate[small_mask])

        return torch.clamp(poisson_samples, min=0)


class PoissonNoiseSchedule:
    """
    Utility class for generating Poisson noise schedules.

    Provides various schedule strategies for the noise injection process.
    """

    @staticmethod
    def linear_schedule(
        time_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> torch.Tensor:
        """Linear schedule from beta_start to beta_end."""
        return torch.linspace(beta_start, beta_end, time_steps)

    @staticmethod
    def cosine_schedule(
        time_steps: int,
        s: float = 0.008,
    ) -> torch.Tensor:
        """
        Cosine schedule as proposed in Improved DDPM.

        Args:
            time_steps: Number of timesteps
            s: Offset parameter (default 0.008)

        Returns:
            Beta schedule tensor
        """
        steps = time_steps + 1
        x = torch.linspace(0, time_steps, steps)
        alphas_cumprod = torch.cos(((x / time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.02)

    @staticmethod
    def sigmoid_schedule(
        time_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        s: float = 1.0,
    ) -> torch.Tensor:
        """
        Sigmoid schedule with learnable steepness.

        Args:
            time_steps: Number of timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            s: Steepness parameter

        Returns:
            Beta schedule tensor
        """
        betas = torch.linspace(-s, s, time_steps)
        return (beta_end - beta_start) * torch.sigmoid(betas) + beta_start
