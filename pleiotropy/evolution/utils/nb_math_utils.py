import math
from typing import Tuple

import numba as nb
import numpy as np


@nb.njit()
def distance_between_points(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """
    Calculate the distance between the two given coordinates: (x1, y1, z1) and (x2, y2, z2).
    """
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2) + ((z1 - z2) ** 2))


def distance_between_coords(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the distance between the two given coordinates.
    """
    return np.linalg.norm(np.asarray(coord1) - np.asarray(coord2))


def distance_from_origin(coord: Tuple[float, float, float]) -> float:
    """
    Calculate the distance between the given coordinate and origin.
    """
    return np.linalg.norm(np.asarray(coord))
