from src.core.common import *
from src.utils.generate_shapes import *
from src.quadratic_spline_surface.optimize_spline_surface import *
from src.quadratic_spline_surface.twelve_split_spline import *
from src.utils.generate_position_data import *
from src.core.bivariate_quadratic_function import evaluate_quadratic_mapping

import pytest
import numpy as np


def twelve_split_quadratic_reproduction(
    uv_coeff: float,
    uu_coeff: float,
    vv_coeff: float
) -> bool:
    """
    Test that a quadratic surface can be reproduced from analytic corner and midpoint data

    This is more of a test that goes through the process and makes sure that everything operates normally.
    """

    V: np.ndarray = np.array([
        [1.0,  0.0],
        [0.0,  1.0],
        [0.0,  0.0]
    ], dtype=float)  # shape (3, 2)
    assert V.shape == (3, 2)
    F: np.ndarray = np.array([
        [0, 1, 2]
    ], dtype=int)  # shape (1, 3)
    assert F.shape == (1, 3)
    parametric_affine_manifold = ParametricAffineManifold(F, V)
    position_func = QuadraticPositionFunction(uv_coeff, uu_coeff, vv_coeff)
    gradient_func = QuadraticGradientFunction(uv_coeff, uu_coeff, vv_coeff)

    # Generate function data
    corner_data: list[list[TriangleCornerData]] = generate_parametric_affine_manifold_corner_data(
        position_func,
        gradient_func,
        parametric_affine_manifold)

    midpoint_data: list[list[TriangleMidpointData]] = generate_parametric_affine_manifold_midpoint_data(
        gradient_func,
        parametric_affine_manifold)

    surface_mappings: list[Matrix6x3r] = generate_twelve_split_spline_patch_surface_mapping(
        corner_data[0],
        midpoint_data[0])  # length 12 list
    assert len(surface_mappings) == 12

    domain_point: PlanarPoint = np.array([
        [0.2],
        [0.3]
    ])
    assert domain_point.shape == (1, 2)
    q: SpatialVector = evaluate_quadratic_mapping(3, surface_mappings[0], domain_point)
    assert q.shape == (1, 3)

    if len(surface_mappings) != 12:
        return False

    if not vector_equal(q, position_func(0.2, 0.3)):
        return False

    return True


def test_twelve_split_spline_constant_surface():
    # Build constant function triangle data
    p: SpatialVector = np.array([
        [1.0],
        [2.0],
        [3.0]])

    zero: SpatialVector = np.array([
        [0.0],
        [0.0],
        [0.0]
    ])
    corner_data: list[TriangleCornerData] = [
        TriangleCornerData(p, zero, zero),
        TriangleCornerData(p, zero, zero),
        TriangleCornerData(p, zero, zero)
    ]

    midpoint_data: list[TriangleMidpointData] = [
        TriangleMidpointData(zero),
        TriangleMidpointData(zero),
        TriangleMidpointData(zero)
    ]

    surface_mappings: list[Matrix6x3r]  # length 12 array with matrices shape (6, 3)
    surface_mappings = generate_twelve_split_spline_patch_surface_mapping(
        corner_data,
        midpoint_data)

    domain_point: PlanarPoint = np.array([[0.25], [0.25]])
    q: SpatialVector = evaluate_quadratic_mapping(3, surface_mappings[0], domain_point)

    assert len(surface_mappings) == 12
    assert vector_equal(q, p)


def test_twelve_split_spline_linear_surface():
    """
    Build linear "quadratic" functionals
    """
    assert twelve_split_quadratic_reproduction(0.0, 0.0, 0.0)


def test_twelve_split_spline_quadratic_surface():
    """
    Test linear "quadratic" functionals
    """
    assert twelve_split_quadratic_reproduction(1.0, 0.0, 0.0)
    assert twelve_split_quadratic_reproduction(0.0, 1.0, 0.0)
    assert twelve_split_quadratic_reproduction(0.0, 0.0, 1.0)
    assert twelve_split_quadratic_reproduction(1.0, 2.0, -1.0)
