import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    """
    Advanced embedding layer for node features.

    Structure:
    Input Indices -> Lookup -> Linear -> Norm -> Activation -> Dropout -> Output
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        out_dim: int = 128,
        dropout: float = 0.0,
    ):
        """
        Args:
            num_embeddings: Vocabulary size (e.g., number of unique node types).
            embedding_dim: Size of the initial lookup vector.
            out_dim: Size of the output vector (if None, typically matches
                embedding_dim).
            dropout: Probability of dropout.
        """
        super().__init__()

        # Explicit None keeps the input dimension instead of the GEOM output size.
        if out_dim is None:
            out_dim = embedding_dim

        # 1. The Base Lookup
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

        # 2. The Feature Projection Block (MLP)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Optional: Initialize weights specifically for embeddings if needed
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization often works better than default for projections
        nn.init.xavier_uniform_(self.emb.weight)
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        """Project categorical tokens with independent dropout per occurrence."""
        # Molecular vocabularies are tiny and contain many repeated tokens.
        # Project each category once. Training's dense lookup avoids Inductor's
        # expensive repeated-index embedding backward in deterministic mode.
        # Keep independent dropout draws per occurrence during training, and
        # avoid a large one-hot allocation for non-molecular vocabularies.
        stochastic = self.training and any(
            isinstance(layer, nn.Dropout) and layer.p > 0 for layer in self.projection
        )
        if (
            stochastic
            or self.emb.num_embeddings > 64
            or x.dtype not in (torch.int32, torch.int64)
        ):
            return self.projection(self.emb(x))
        table = self.projection(self.emb.weight)
        if not torch.is_grad_enabled():
            # Match the dense lookup's autocast output (normalization may have
            # promoted the projected table back to FP32).
            if (
                torch.is_autocast_enabled(table.device.type)
                and table.dtype != torch.float64
            ):
                table = table.to(torch.get_autocast_dtype(table.device.type))
            return F.embedding(x, table)
        selector = F.one_hot(x.long(), num_classes=self.emb.num_embeddings)
        return selector.to(table.dtype) @ table


class SinusoidalEncoding(nn.Module):
    """
    The mathematical engine for sinusoidal embeddings.
    Computes: [sin(x*w0), cos(x*w0), sin(x*w1), cos(x*w1)...]
    """

    def __init__(self, embedding_dim: int, max_period: float = 5000.0):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dimension {embedding_dim} must be even.")

        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2

        # Calculate frequencies (DDPM style / Vaswani style)
        # We want frequencies ranging from 1 to 1/max_period
        # The formula creates a geometric progression of frequencies
        lambda_max = max_period

        # Derived from: exp(-2 * i / d * ln(10000))
        # This matches the JAX 'half_dim - 1' logic for exact boundary alignment
        exponent = -math.log(lambda_max) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32
        )
        exponent = exponent / (half_dim - 1)

        freqs = torch.exp(exponent)

        # Register as fixed buffer (not a learnable parameter)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [Batch, 1] (values can be float or int)

        Returns:
            Tensor of shape [Batch, embedding_dim]
        """
        # x: [Batch, 1], freqs: [Half_Dim]
        # args: [Batch, Half_Dim]
        args = x.float() * self.freqs.unsqueeze(0)

        # Concatenate sin and cos -> [Batch, Dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


class ExtrapolatableCountEmbedding(nn.Module):
    """Count embedding designed to extrapolate slightly outside the training range.

    A naive sinusoidal count embedding whose shortest period is ``2π`` (a
    geometric series of frequencies starting at ``freq=1``) produces aliased,
    non-monotonic features for small integer counts (e.g. QM9 with counts in
    ``[3, 29]``): neighbouring counts map to unrelated sin/cos patterns, and
    counts just outside the training support land on patterns the downstream
    linear projection has never seen — so it cannot extrapolate.

    This module fixes both issues:

    * It uses a **low-frequency** Fourier basis whose shortest period is larger
      than the expected training support (``min_period``), so every dimension
      varies smoothly over the training range and its immediate neighbourhood.
    * It concatenates a **raw normalised scalar** ``count / max_count``, giving
      the MLP a strictly linear (monotonic, unbounded) axis to extrapolate
      along — the Fourier basis only contributes local precision within range.

    Defaults follow GEOM-Drugs: ``min_period=128``, ``max_period=1024``, and
    ``max_count=96``. The QM9 recipe explicitly selects its smaller count range.

    Input:  Long / Float Tensor ``[Batch]`` or ``[Batch, 1]``.
    Output: Float Tensor ``[Batch, out_dim]``.
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int = 128,
        min_period: float = 128.0,
        max_period: float = 1024.0,
        max_count: float = 96.0,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = embedding_dim
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"embedding_dim {embedding_dim} must be even (sin/cos pairs)."
            )
        if min_period <= 0 or max_period < min_period:
            raise ValueError(
                f"Expected 0 < min_period <= max_period, got "
                f"min_period={min_period}, max_period={max_period}."
            )
        if max_count <= 0:
            raise ValueError(f"max_count must be > 0, got {max_count}.")

        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        self.max_count = float(max_count)

        # Log-spaced periods in [min_period, max_period].  Highest frequency
        # has the *shortest* period (= min_period), ensuring every component
        # stays in a monotonic half-cycle across [0, max_count].
        half_dim = embedding_dim // 2
        if half_dim == 1:
            periods = torch.tensor([min_period], dtype=torch.float32)
        else:
            log_pmin = math.log(min_period)
            log_pmax = math.log(max_period)
            periods = torch.exp(torch.linspace(log_pmin, log_pmax, half_dim))
        self.register_buffer("freqs", 2 * math.pi / periods)

        # MLP takes [fourier_features, raw_scalar] and mixes them.  The raw
        # scalar is the path responsible for linear extrapolation.
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim + 1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """Embed counts using Fourier features and a scaled scalar count."""
        if counts.ndim == 1:
            counts = counts.unsqueeze(-1)

        x = counts.float()
        args = x * self.freqs.unsqueeze(0)
        fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        scalar = x / self.max_count
        return self.projection(torch.cat([fourier, scalar], dim=-1))


class TimeEmbedding(nn.Module):
    """
    Embeds continuous time/noise levels.
    Input:  Float Tensor [Batch, 1] (e.g., values 0.0 to 1.0)
    Output: Float Tensor [Batch, Dim]
    """

    def __init__(self, embedding_dim: int, out_dim: int = 128):
        super().__init__()
        if out_dim is None:
            out_dim = embedding_dim
        # For time in [0,1], max_period=5000 ensures high precision
        # for even very small time steps.
        self.encoder = SinusoidalEncoding(embedding_dim, max_period=5000.0)

        # Standard MLP block for Diffusion/Flow models
        # Projects the fixed sinusoids into the learnable semantic space
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),  # SiLU is standard for time
            nn.Linear(embedding_dim, out_dim),
        )

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Encode graph times after scaling the unit interval by 1000."""
        # Ensure input is [Batch, 1]
        if times.ndim == 1:
            times = times.unsqueeze(-1)

        times = times * 1000.0
        t_enc = self.encoder(times)
        return self.mlp(t_enc)


class RBFEncoding(nn.Module):
    """
    Radial Basis Function (RBF) encoding with GEOM's fixed distance basis.
    Set trainable=True to learn Gaussian centers and widths (sigmas).
    """

    def __init__(
        self, num_rbf: int = 128, rbf_dmax: float = 3.5, trainable: bool = False
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.rbf_dmax = rbf_dmax

        # 1. Initialize centers evenly spaced (same as before)
        initial_centers = torch.linspace(0, rbf_dmax, num_rbf)

        # 2. Initialize sigmas (widths)
        # The gap between centers is dmax / num_rbf.
        # A good default sigma is often the gap size itself or slightly smaller.
        initial_sigma = (rbf_dmax / num_rbf) * torch.ones(num_rbf)

        if trainable:
            # Wrap in nn.Parameter to enable gradient descent
            self.rbf_centers = nn.Parameter(initial_centers)
            self.rbf_sigma = nn.Parameter(initial_sigma)
        else:
            # Keep them fixed if desired
            self.register_buffer("rbf_centers", initial_centers)
            self.register_buffer("rbf_sigma", initial_sigma)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Returns: [N, num_rbf]"""
        if distances.ndim > 1:
            distances = distances.squeeze(-1)

        # Broadcasting logic:
        # distances:       [N, 1]
        # centers/sigmas:  [num_rbf]

        # We calculate gamma dynamically based on the current learnable sigma
        # Gamma = 1 / sigma^2
        # We use abs() or clamp() on sigma to ensure widths stay positive and non-zero
        sigmas = torch.abs(self.rbf_sigma) + 1e-5  # Stability trick
        gamma = 1.0 / (sigmas**2)

        # Formula: exp( -gamma * (x - mu)^2 )
        return torch.exp(-gamma * (distances.unsqueeze(-1) - self.rbf_centers) ** 2)


class RBFEmbedding(nn.Module):
    """
    Radial Basis Function (RBF) embedding for distances with projection layer.
    Wraps RBFEncoding with a learnable projection MLP.

    Structure:
    Input Distances -> RBFEncoding -> Linear -> Norm -> Activation -> Dropout -> Output

    Input:  Float Tensor [N] (distances)
    Output: Float Tensor [N, out_dim]
    """

    def __init__(
        self,
        num_rbf: int = 128,
        rbf_dmax: float = 3.5,
        out_dim: int = 64,
        dropout: float = 0.0,
        trainable: bool = False,
    ):
        """
        Args:
            num_rbf: Number of RBF basis functions
            rbf_dmax: Maximum distance for RBF
            out_dim: Output dimension (if None, uses num_rbf)
            dropout: Dropout probability
            trainable: Whether RBF centers and widths receive gradients.
        """
        super().__init__()

        # Explicit None uses one output per basis function.
        if out_dim is None:
            out_dim = num_rbf

        # Store out_dim for easy access
        self.out_dim = out_dim

        # RBF encoding layer
        self.rbf_encoding = RBFEncoding(
            num_rbf=num_rbf, rbf_dmax=rbf_dmax, trainable=trainable
        )

        # Simple projection block (matching Embedding class style)
        self.projection = nn.Sequential(
            nn.Linear(num_rbf, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for projection layers."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF embeddings for distances.

        Args:
            distances: Tensor of shape [N] or [N, 1] containing distances

        Returns:
            Tensor of shape [N, out_dim] containing RBF embeddings
        """
        # Get RBF encoding
        rbf_enc = self.rbf_encoding(distances)  # [N, num_rbf]

        # Apply projection
        return self.projection(rbf_enc)  # [N, out_dim]
