"""
File for methods validating specific objects related to contours
"""

import numpy as np

from src.core.common import COLS, Matrix3x3f, Matrix6x3f, float_equal
from src.core.rational_function import RationalFunction


def is_valid_spatial_mapping(spatial_mapping_coeffs: Matrix6x3f) -> bool:
    """
    Checks that spatial mapping maps to R^3
    """
    # Must map to R3
    if spatial_mapping_coeffs.shape[COLS] != 3:
        return False

    return True


def is_valid_frame(frame: Matrix3x3f) -> bool:
    """
    Checks if frame has determinant 1
    """
    # Must be a 3x3 matrix
    # if frame.rows() != 3) return False
    # if frame.cols() != 3) return False

    # Must have determinant 1
    if not float_equal(np.linalg.det(frame), 1.0):
        return False

    return True


def are_valid_contour_segments(contour_segments: list[RationalFunction]  # <4, 3>
                               ) -> bool:
    """
    Checks if domain of contour segments are both finite and compact.
    """
    for i, _ in enumerate(contour_segments):
        if not contour_segments[i].domain.is_finite():
            return False
        if not contour_segments[i].domain.is_compact():
            return False
    return True
