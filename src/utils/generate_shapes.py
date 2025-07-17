
from src.core.common import *
from src.core.conic import Conic
import numpy as np
import math


def generate_circle(radius: float) -> Conic:
    """
    Generate circle of given radius

    :param radius: radius of the circle
    :return: parametrized circle (missing point at bottom)
    """
    unimplemented()


def generate_torus_point(major_radius: float, minor_radius: float, i: int, j: int, resolution: int, angle_offset: float):
    theta = generate_angle(i, resolution, angle_offset)
    phi = generate_angle(j, resolution, angle_offset)

    return np.array([(major_radius + minor_radius * math.cos(theta)) * math.cos(phi),
                     (major_radius + minor_radius *
                      math.cos(theta)) * math.sin(phi),
                     minor_radius * math.sin(theta)])


def generate_angle(i: float, resolution: int, angle_offset: float) -> float:
    return angle_offset + 2 * math.pi * i / resolution


def generate_angle_derivative(resolution: int) -> float:
    unimplemented()


def generate_elliptic_contour_quadratic_surface():
    """
    Generate a quadratic surface with an ellipse as the parametric contour.
    :return: surface_mapping_coeffs: Coefficients for the quadratic surface
    :return: normal_mapping_coeffs: Coefficients for the quadratic surface normal
    """
    surface_mapping_coeffs: Matrix6x3r
    normal_mapping_coeffs: Matrix6x3r

    unimplemented("na")

    return surface_mapping_coeffs, normal_mapping_coeffs

# ***************
# VF construction
# ***************


def generate_equilateral_triangle_VF(length: float = 1) -> tuple[np.ndarray, np.ndarray]:
    V: np.ndarray
    F: np.ndarray
    unimplemented()
    return V, F


def generate_right_triangle_VF(width: float = 1.0, height: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    V: np.ndarray
    F: np.ndarray
    unimplemented()
    return V, F


def generate_rectangle_VF(width: float = 1.0, height: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    V: np.ndarray
    F: np.ndarray
    unimplemented()
    return V, F


def generate_square_VF(length: float = 1) -> tuple[np.ndarray, np.ndarray]:
    V: np.ndarray
    F: np.ndarray
    unimplemented()
    return V, F


def generate_global_layout_grid(resolution: int) -> list[list[PlanarPoint]]:
    """
    Used in quadratic_spline_surface and optimize_spline_surface test cases.
    """
    assert resolution != 0

    center: float = resolution / 2.0
    layout_grid: list[list[PlanarPoint]] = []
    # TODO: have planarpoint check that shape is (1, 2)

    for i in range(resolution):
        layout_grid.append([])
        for j in range(resolution):
            layout_grid[i].append(np.array([[i - center], [j - center]]))

    return layout_grid
#


def generate_tetrahedron_VF() -> tuple[np.ndarray, np.ndarray]:
    # TODO: how to include typing in np.ndarray? like, int64 and whatnot?
    # np.ndarray[np.dtype[np.float64]]?
    """
    Generate simple tetrahedron mesh.

    :return: tuple of (tetrahedron vertices (V), tetrahedron faces (F))
    :rtype: tuple[np.ndarray[dtype=np.float64], np.ndarray[dtype=np.int64]]
    """

    V = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)

    F = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3]
    ], dtype=int)

    return V, F


def generate_minimal_torus_VF(major_radius: float = 3.0, minor_radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate simple torus mesh.

    :param major_radius: 
    :type major_radius: float    
    :param minor_radius:
    :type minor_radius: float

    :return: tuple of ( V: Torus vertices, F: Torus facets )
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    V = np.ndarray(shape=(9, 3))
    F = np.array([[0, 1, 3],
                  [1, 4, 3],
                  [1, 2, 4],
                  [2, 5, 4],
                  [2, 0, 5],
                  [0, 3, 5],
                  [3, 4, 6],
                  [4, 7, 6],
                  [4, 5, 7],
                  [5, 8, 7],
                  [5, 3, 8],
                  [3, 6, 8],
                  [6, 7, 0],
                  [7, 1, 0],
                  [7, 8, 1],
                  [8, 2, 1],
                  [8, 6, 2],
                  [6, 0, 2]])
    resolution: int = 3

    V[0, :] = generate_torus_point(
        major_radius, minor_radius, 0, 0, resolution, 0.1)
    V[1, :] = generate_torus_point(
        major_radius, minor_radius, 0, 1, resolution, 0.1)
    V[2, :] = generate_torus_point(
        major_radius, minor_radius, 0, 2, resolution, 0.1)
    V[3, :] = generate_torus_point(
        major_radius, minor_radius, 1, 0, resolution, 0.1)
    V[4, :] = generate_torus_point(
        major_radius, minor_radius, 1, 1, resolution, 0.1)
    V[5, :] = generate_torus_point(
        major_radius, minor_radius, 1, 2, resolution, 0.1)
    V[6, :] = generate_torus_point(
        major_radius, minor_radius, 2, 0, resolution, 0.1)
    V[7, :] = generate_torus_point(
        major_radius, minor_radius, 2, 1, resolution, 0.1)
    V[8, :] = generate_torus_point(
        major_radius, minor_radius, 2, 2, resolution, 0.1)

    return V, F

# ********************
# Polygon construction
# ********************
# def generate_rectangle(float: x0, )
