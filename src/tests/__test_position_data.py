import math
from src.quadratic_spline_surface.position_data import *
from src.core.common import *
import numpy as np
import pytest


def test_gradients_find_constant() -> None:
    V = np.array([[1.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0]])

    F = np.array(
        [[0, 1, 2],
         [0, 2, 3],
         [0, 3, 1]])

    vertex_index: int = 0
    vertex_one_ring: list[int] = [1, 2, 3]
    face_one_ring: list[int] = [0, 1, 2]

    one_ring_uv_positions: np.ndarray = np.array(
        [[1.0, 0.0],
         [(-math.sqrt(3) / 2.0), 0.5],
         [(-math.sqrt(3) / 2.0), -0.5]])

    assert V.shape == (4, 3)
    assert F.shape == (3, 3)
    assert one_ring_uv_positions.shape == (3, 2)

    todo("There are no assert statements in the ASOC code for this test")
