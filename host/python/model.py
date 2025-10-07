import torch as th
from torch import nn
import math


class ConvolutionNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.SiLU(),
            nn.InstanceNorm2d(8),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.SiLU(),
            nn.InstanceNorm2d(16),
            nn.Conv2d(16, 24, 3, 2, 1),
            nn.SiLU(),
            nn.InstanceNorm2d(24),
            nn.Conv2d(24, 32, 3, 2, 1),
            nn.SiLU(),
            nn.InstanceNorm2d(32),
            nn.Conv2d(32, 40, 3, 2, 1),
            nn.SiLU(),
            nn.InstanceNorm2d(40),
            nn.Conv2d(40, 48, 3, 2, 1),
            nn.SiLU(),
            nn.InstanceNorm2d(48),
            nn.Flatten(1, -1),
        )

    def forward(self, vision: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.cnn(vision)
        return out


class SacActor(nn.Module):
    def __init__(
            self,
            nb_sensors: int,
            nb_actions: int,
            hidden_size_sensors: int,
            hidden_size: int,
    ) -> None:
        super().__init__()

        self.vision_encoder = ConvolutionNetwork()
        self.sensors_encoder = nn.Sequential(
            nn.Linear(nb_sensors, hidden_size_sensors),
            nn.SiLU(),
            nn.LayerNorm(hidden_size_sensors),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size_sensors + 2 * 2 * 48, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size)
        )

        self.mu = nn.Sequential(
            nn.Linear(hidden_size, nb_actions),
            nn.Tanh(),
        )
        self.sigma = nn.Sequential(
            nn.Linear(hidden_size, nb_actions),
            nn.Softplus(),
        )

    def forward(self, vision: th.Tensor, sensors: th.Tensor) -> th.Tensor:
        encoded_vision = self.vision_encoder(vision)
        encoded_sensors = self.sensors_encoder(sensors)

        encoded_latent = self.head(th.cat([encoded_vision, encoded_sensors], dim=1))

        mu, sigma = self.mu(encoded_latent), self.sigma(encoded_latent)
        return SacActor.__truncated_normal_sample(mu, sigma)

    @staticmethod
    def __truncated_normal_sample(mu: th.Tensor, sigma: th.Tensor) -> th.Tensor:
        min_value = -1.0
        max_value = 1.0

        sigma_lower_bound = 1e-6
        sigma_upper_bound = 1e6
        alpha_beta_bound = 5.0

        safe_sigma = th.clamp(sigma, sigma_lower_bound, sigma_upper_bound)

        alpha = th.clamp((min_value - mu) / safe_sigma, -alpha_beta_bound, alpha_beta_bound)
        beta = th.clamp((max_value - mu) / safe_sigma, -alpha_beta_bound, alpha_beta_bound)

        cdf = th.clamp(SacActor.__theta(alpha) + th.rand(alpha.size(), device=mu.device) * (SacActor.__theta(beta) - SacActor.__theta(alpha)), 0.0, 1.0)

        return th.clamp(SacActor.__theta_inv(cdf) * safe_sigma + mu, min_value, max_value)


    @staticmethod
    def __phi(z: th.Tensor) -> th.Tensor:
        return th.exp(-0.5 * th.pow(z, 2.0)) / math.sqrt(2.0 * th.pi)

    @staticmethod
    def __theta(x: th.Tensor) -> th.Tensor:
        return 0.5 * (1.0 + th.erf(x / math.sqrt(2.0)))

    @staticmethod
    def __theta_inv(theta: th.Tensor) -> th.Tensor:
        return math.sqrt(2.0) * SacActor.__erf_inv(2.0 * theta - 1.0)

    @staticmethod
    def __erf_inv(x: th.Tensor, newton_steps: int = 3) -> th.Tensor:
        pos_one = (x == 1)
        neg_one = (x == -1)

        finfo = th.finfo(x.dtype)
        eps = 10 * finfo.eps
        xc = x.clamp(min=-1 + eps, max=1 - eps)

        a = 0.147
        s = th.sign(xc)

        ln = th.log1p(-xc * xc)
        t = 2 / (math.pi * a) + 0.5 * ln

        y = s * th.sqrt(th.sqrt(t * t - ln / a) - t)

        c = 2.0 / math.sqrt(math.pi)
        for _ in range(max(0, int(newton_steps))):
            ey = th.erf(y)
            dy = c * th.exp(-(y * y))
            y = y - (ey - xc) / dy

        inf = th.tensor(float('inf'), dtype=x.dtype, device=x.device)
        y = th.where(pos_one, inf, y)
        y = th.where(neg_one, -inf, y)

        return y
