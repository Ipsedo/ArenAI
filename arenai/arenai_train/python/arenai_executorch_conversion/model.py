from sndhdr import test_aifc

import torch as th
from torch import nn

from .constants import ENEMY_VISION_HEIGHT, ENEMY_VISION_WIDTH


class ConvolutionNetwork(nn.Module):
    def __init__(
        self, channels: list[tuple[int, int]], group_norm_nums: list[int]
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential()

        padding = 1
        stride = 2
        kernel = 3

        w = ENEMY_VISION_WIDTH
        h = ENEMY_VISION_HEIGHT

        for i, ((c_i, c_o), g) in enumerate(zip(channels, group_norm_nums)):
            self.cnn.append(nn.Conv2d(c_i, c_o, kernel, stride, padding))
            self.cnn.append(nn.GroupNorm(g, c_o, affine=True))
            self.cnn.append(nn.SiLU())

            w = (w - kernel + 2 * padding) // stride + 1
            h = (h - kernel + 2 * padding) // stride + 1

        self.__output_size = w * h * channels[-1][1]

        self.cnn.append(nn.Flatten(1, -1))

    def forward(self, vision: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.cnn(vision)
        return out

    @property
    def output_size(self) -> int:
        return self.__output_size


class GumbelSoftmax(nn.Module):
    def __init__(self, dim: int, tau: float = 1.0, epsilon: float = 1e-20) -> None:
        super().__init__()

        self.__dim = dim
        self.__tau = tau
        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        u = th.clamp(th.rand(x.size(), device=x.device), self.__epsilon, 1.0 - self.__epsilon)
        gumbel_noise = -th.log(-th.log(u))

        y = (x + gumbel_noise) / self.__tau
        return th.softmax(y, self.__dim)


class SacActor(nn.Module):
    def __init__(
        self,
        nb_sensors: int,
        nb_continuous_actions: int,
        nb_discrete_actions: int,
        hidden_size_sensors: int,
        hidden_size: int,
        channels: list[tuple[int, int]],
        group_norm_nums: list[int],
    ) -> None:
        super().__init__()

        self.vision_encoder = ConvolutionNetwork(channels, group_norm_nums)

        self.sensors_encoder = nn.Sequential(
            nn.Linear(nb_sensors, hidden_size_sensors),
            nn.LayerNorm(hidden_size_sensors),
            nn.SiLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_size_sensors + self.vision_encoder.output_size,
                hidden_size,
            ),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        self.mu = nn.Sequential(
            nn.Linear(hidden_size, nb_continuous_actions),
            nn.Tanh(),
        )
        self.sigma = nn.Sequential(
            nn.Linear(hidden_size, nb_continuous_actions),
            nn.Softplus(),
        )
        self.discrete = nn.Sequential(
            nn.Linear(hidden_size, nb_discrete_actions),
            GumbelSoftmax(-1),
        )

    def forward(
        self, vision: th.Tensor, sensors: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        encoded_vision = self.vision_encoder(vision)
        encoded_sensors = self.sensors_encoder(sensors)

        encoded_latent = self.head(
            th.cat([encoded_vision, encoded_sensors], dim=1)
        )

        return self.mu(encoded_latent), self.sigma(encoded_latent), self.discrete(encoded_latent)
