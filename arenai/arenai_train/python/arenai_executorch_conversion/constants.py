from typing import Final

ENEMY_VISION_HEIGHT: Final[int] = 256
ENEMY_VISION_WIDTH: Final[int] = 512
ENEMY_PROPRIOCEPTION_SIZE: Final[int] = (3 + 3 + 3 + 3 + 3) * (6 + 3) - 3
ENEMY_NB_ACTIONS: Final[int] = 2 + 2 + 1
