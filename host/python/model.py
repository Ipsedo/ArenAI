import torch as th
from torch import nn

class ConvolutionNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.Mish(),
            nn.InstanceNorm2d(8),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.Mish(),
            nn.InstanceNorm2d(16),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.Mish(),
            nn.InstanceNorm2d(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.Mish(),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.Mish(),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.Mish(),
            nn.InstanceNorm2d(256),
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
            nn.Mish(),
            nn.LayerNorm(hidden_size_sensors),
            nn.Linear(hidden_size_sensors, hidden_size_sensors),
            nn.Mish(),
            nn.LayerNorm(hidden_size_sensors),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size_sensors + 2 * 2 * 256, hidden_size),
            nn.Mish(),
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

    def forward(self, vision: th.Tensor, sensors: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        encoded_vision = self.vision_encoder(vision)
        encoded_sensors = self.sensors_encoder(sensors)

        encoded_latent = self.head(th.cat([encoded_vision, encoded_sensors], dim=1))

        return self.mu(encoded_latent), self.sigma(encoded_latent)
