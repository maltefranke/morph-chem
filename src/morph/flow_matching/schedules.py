from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import beta


class KappaSchedule(nn.Module, ABC):
    """
    Abstract base class for flow-matching time schedules (kappa schedules).

    A schedule maps time t in [0, 1] to kappa(t) (CDF) and provides the rate
    kappa'(t) / (1 - kappa(t)) used in rate-based flow matching.
    """

    @abstractmethod
    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        """Maps t to kappa(t) (CDF)."""
        ...

    @abstractmethod
    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Derivative of kappa w.r.t. t (PDF)."""
        ...

    def rate(self, t: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """Rate: kappa_t_dot(t) / (1 - kappa_t(t)).

        Default implementation derived from ``kappa_t`` / ``kappa_t_dot``.
        Subclasses with a cheaper closed form (e.g. ``FastPowerSchedule``) may
        override. The clamp guards the division as t -> 1.
        """
        return self.kappa_and_rate(t, epsilon)[1]

    def kappa_and_rate(
        self, t: torch.Tensor, epsilon: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the CDF and hazard without computing the CDF twice."""
        k_t = self.kappa_t(t)
        k_t_dot = self.kappa_t_dot(t)
        denominator = torch.clamp(1 - k_t, min=epsilon)
        return k_t, k_t_dot / denominator

    def integrated_rate(
        self, t: torch.Tensor, dt, epsilon: float = 1e-6
    ) -> torch.Tensor:
        """Return the exact schedule hazard integrated over ``[t, t + dt]``.

        Since ``rate = kappa' / (1 - kappa)``, its integral is the log ratio
        of the survival probabilities. This avoids the left-endpoint
        ``rate(t) * dt`` approximation without requiring schedule-specific code.
        """
        survival = torch.clamp(1.0 - self.kappa_t(t), min=epsilon)
        t_next = torch.clamp(t + dt, max=1.0)
        survival_next = torch.clamp(1.0 - self.kappa_t(t_next), min=epsilon)
        return torch.log(survival) - torch.log(survival_next)


class BetaSchedule(KappaSchedule):
    """Beta-CDF schedule evaluated through SciPy for numerical stability.

    Evaluation detaches time from autograd and transfers CUDA inputs to the CPU.
    Prefer an analytic schedule for sampling workloads sensitive to these copies.
    """

    def __init__(self, k_alpha: float, k_beta: float):
        """Initialize the Beta distribution shape parameters."""
        super().__init__()
        self.k_alpha = k_alpha
        self.k_beta = k_beta

    def _to_numpy(self, t: torch.Tensor):
        """Helper to convert tensor to numpy array safely."""
        # Detach from graph, move to CPU, convert to numpy
        return t.detach().cpu().numpy()

    def _to_tensor(self, x: np.ndarray, t_template: torch.Tensor):
        """Helper to convert numpy result back to tensor on correct device."""
        return torch.tensor(x, device=t_template.device, dtype=t_template.dtype)

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the Beta CDF."""
        t_np = self._to_numpy(t)

        # scipy.stats.beta.cdf handles the math
        cdf_val = beta.cdf(t_np, self.k_alpha, self.k_beta)

        return self._to_tensor(cdf_val, t)

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the Beta PDF."""
        t_np = self._to_numpy(t)

        # scipy.stats.beta.pdf handles the math
        pdf_val = beta.pdf(t_np, self.k_alpha, self.k_beta)

        return self._to_tensor(pdf_val, t)


class CubicSchedule(KappaSchedule):
    """Cubic schedule ``kappa(t) = t^3``."""

    def __init__(self):
        super().__init__()

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the cubic CDF."""
        return t**3

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the cubic CDF derivative."""
        return 3 * t**2


class LinearSchedule(KappaSchedule):
    """Linear schedule ``kappa(t) = t``."""

    def __init__(self):
        super().__init__()

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the linear CDF."""
        return t

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the constant linear CDF derivative."""
        return torch.ones_like(t)


class FastPowerSchedule(KappaSchedule):
    """Optimized Beta schedule for alpha equal to one."""

    def __init__(self, beta: float):  # e.g. 1.5
        super().__init__()
        self.register_buffer("beta", torch.tensor(float(beta)))

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the analytic power-schedule CDF."""
        # Analytical CDF for Alpha=1: 1 - (1-t)^beta
        return 1 - torch.pow(1 - t, self.beta)

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the analytic power-schedule PDF."""
        # Analytical PDF for Alpha=1: beta * (1-t)^(beta-1)
        return self.beta * torch.pow(1 - t, self.beta - 1)

    def rate(self, t: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """Evaluate the analytic power-schedule hazard."""
        # Analytical Rate for Alpha=1: beta / (1-t)
        denom = torch.clamp(1 - t, min=epsilon)
        return self.beta / denom

    def kappa_and_rate(
        self, t: torch.Tensor, epsilon: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the analytic CDF and hazard together."""
        return self.kappa_t(t), self.rate(t, epsilon)


class SmoothstepSchedule(KappaSchedule):
    """
    A simple polynomial schedule that concentrates edits in a bell shape.
    Defaults to GEOM's insertion/deletion shift of 0.8; substitutions use 1.5.
    - shift = 1.0 : Edits peak perfectly in the middle.
    - shift < 1.0 : Edits happen earlier (pushes the peak towards t=0).
    - shift > 1.0 : Edits happen later (pushes the peak towards t=1).
    """

    def __init__(self, shift: float = 0.8):
        super().__init__()
        self.shift = shift

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the time-warped smoothstep CDF."""
        # Standard smoothstep is 3t^2 - 2t^3.
        # By raising t to the power of 'shift' (p), we warp time to move the peak.
        p = self.shift
        # Clamped because the two terms nearly cancel as t -> 1: in fp32 the
        # difference can round to ~1 + 1e-7, and a CDF above 1 turns downstream
        # ``(1 - kappa)`` mixing weights negative.
        return (3 * t ** (2 * p) - 2 * t ** (3 * p)).clamp(0.0, 1.0)

    def kappa_t_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the time-warped smoothstep CDF derivative."""
        p = self.shift
        # We clamp t slightly above 0 to prevent NaN errors when shift < 0.5
        t_safe = torch.clamp(t, min=1e-7)

        # The exact mathematical derivative of kappa_t
        return 6 * p * (t_safe ** (2 * p - 1) - t_safe ** (3 * p - 1))
