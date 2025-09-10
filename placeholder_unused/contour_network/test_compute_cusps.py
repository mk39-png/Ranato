"""
Methods for testing compute_cusps.py
"""

import os

import numpy as np
import numpy.testing as npt

from src.contour_network.compute_intersections import (
    IntersectionParameters, compute_planar_curve_intersections)
from src.contour_network.intersection_heuristics import IntersectionStats
from src.core.common import Matrix3x2f, Matrix6x3f, float_equal
from src.core.rational_function import RationalFunction
from src.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface
from src.quadratic_spline_surface.quadratic_spline_surface_patch import \
    QuadraticSplineSurfacePatch


def test_compute_intersections_simple_linear_functions():
    """
    Simple Linear Functions.
    """
    first_P_coeffs: Matrix3x2f = np.array([[0, 0],
                                           [1, 0],
                                           [0, 0]])
    first_Q_coeffs: np.ndarray = np.array([1])
    second_P_coeffs: Matrix3x2f = np.array([[0, 0],
                                            [0, 1],
                                            [0, 0]])
    second_Q_coeffs: np.ndarray = np.array([1])
    intersections: list[float] = []
    first_curve_intersections: list[float] = []
    second_curve_intersections: list[float] = []
    intersection_stats: IntersectionStats = IntersectionStats()
    intersection_params: IntersectionParameters = IntersectionParameters()

    first_image_segment = RationalFunction(4, 2, first_P_coeffs, first_Q_coeffs)
    second_image_segment = RationalFunction(4, 2, second_P_coeffs, second_Q_coeffs)
    compute_planar_curve_intersections(first_image_segment, second_image_segment,
                                       intersection_params,
                                       first_curve_intersections, second_curve_intersections,
                                       intersection_stats)
    assert len(intersections) == 1
    float_equal(first_curve_intersections[0], 0.0)
    float_equal(second_curve_intersections[0], 0.0)

# def test_tangent_x():
#     """
#     Attempting to demystify tangent x code.
#     """
#     # Need to initialize QuadraticSplineSurface with our test file
#     filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
#     filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")
#     spline_surface: QuadraticSplineSurface = QuadraticSplineSurface.from_file(filepath)
#     patch_0: QuadraticSplineSurfacePatch = spline_surface.get_patch(0)
#     normalized_surface_mapping_coeffs: Matrix6x3f = patch_0.get_normalized_surface_mapping()

#     # With this, we can not test implementations.
#     px = normalized_surface_mapping_coeffs[:, 0]
#     py = normalized_surface_mapping_coeffs[:, 1]
#     pz = normalized_surface_mapping_coeffs[:, 2]
#     assert px.shape == (6, )
#     assert py.shape == (6, )

#     dx_test, dy_test = _tangent_/(px, py)
#     tx_test = np.polyval(dx_test, 1)
#     ty_test = np.polyval(dy_test, 1)

#     tx_control = np.zeros(shape=(6, ))
#     t10: float
#     t11: float
#     t13: float
#     t14: float
#     t17: float
#     t20: float
#     t24: float
#     t27: float
#     t3: float
#     t35: float
#     t4: float
#     t41: float
#     t48: float
#     t52: float
#     t7: float
#     t8: float
#     t3 = px[1]
#     t4 = py[5]
#     t7 = px[2]
#     t8 = py[3]
#     t10 = px[3]
#     t11 = py[2]
#     t13 = px[5]
#     t14 = py[1]
#     t17 = -t11 * t10 + 0.2e1 * t14 * t13 - 0.2e1 * t3 * t4 + t7 * t8
#     t20 = py[4]
#     t24 = px[4]
#     t27 = -t14 * t10 + 0.2e1 * t11 * t24 - 0.2e1 * t20 * t7 + t3 * t8
#     t35 = 0.4e1 * t13 * t20 - 0.4e1 * t24 * t4
#     t41 = -0.4e1 * t10 * t20 + 0.4e1 * t24 * t8
#     t48 = -0.4e1 * t10 * t4 + 0.4e1 * t13 * t8
#     t52 = -t35
#     tx_control[0] = t17 * t3 + t27 * t7
#     tx_control[1] = t10 * t27 + 0.2e1 * t17 * t24 + t3 * t35 + t41 * t7
#     tx_control[2] = t10 * t17 + 0.2e1 * t13 * t27 + t3 * t48 + t52 * t7
#     tx_control[3] = t10 * t35 + t10 * t52 + 0.2e1 * t13 * t41 + 0.2e1 * t24 * t48
#     tx_control[4] = t10 * t41 + 0.2e1 * t24 * t35
#     tx_control[5] = t10 * t48 + 0.2e1 * t13 * t52

#     npt.assert_allclose(tx_test, tx_control)
