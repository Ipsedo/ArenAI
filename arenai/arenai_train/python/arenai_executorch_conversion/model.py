import torch as th
from torch import nn

from .constants import ENEMY_VISION_SIZE


class ConvolutionNetwork(nn.Module):
    def __init__(self, channels: list[tuple[int, int]]) -> None:
        super().__init__()

        self.cnn = nn.Sequential()

        padding = 1
        stride = 2
        kernel = 3

        w = ENEMY_VISION_SIZE
        h = ENEMY_VISION_SIZE

        for i, (c_i, c_o) in enumerate(channels):
            self.cnn.append(nn.Conv2d(c_i, c_o, kernel, stride, padding))

            if i < len(channels) - 1:
                self.cnn.append(nn.SiLU())

            w = (w - kernel + 2 * padding) // stride + 1
            h = (h - kernel + 2 * padding) // stride + 1

        self.__output_size = w * h * channels[-1][1]

        self.cnn.extend(
            nn.Sequential(
                nn.Flatten(1, -1),
                nn.LayerNorm(self.__output_size),
                nn.SiLU(),
            )
        )

    def forward(self, vision: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.cnn(vision)
        return out

    @property
    def output_size(self) -> int:
        return self.__output_size


class SacActor(nn.Module):
    def __init__(
        self,
        nb_sensors: int,
        nb_actions: int,
        hidden_size_sensors: int,
        hidden_size: int,
        channels: list[tuple[int, int]],
    ) -> None:
        super().__init__()

        self.vision_encoder = ConvolutionNetwork(channels)

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
            nn.Linear(hidden_size, nb_actions),
            nn.Tanh(),
        )
        self.sigma = nn.Sequential(
            nn.Linear(hidden_size, nb_actions),
            nn.Softplus(),
        )

    def forward(
        self, vision: th.Tensor, sensors: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        encoded_vision = self.vision_encoder(vision)
        encoded_sensors = self.sensors_encoder(sensors)

        encoded_latent = self.head(
            th.cat([encoded_vision, encoded_sensors], dim=1)
        )

        return self.mu(encoded_latent), self.sigma(encoded_latent)
