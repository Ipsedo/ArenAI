from typing import Final

ENEMY_VISION_HEIGHT: Final[int] = 128
ENEMY_VISION_WIDTH: Final[int] = 256
ENEMY_PROPRIOCEPTION_SIZE: Final[int] = (3 + 3 + 3 + 3 + 3) * (6 + 3) - 3
ENEMY_NB_CONTINUOUS_ACTIONS: Final[int] = 2 + 2
ENEMY_NB_DISCRETE_ACTIONS: Final[int] = 2
