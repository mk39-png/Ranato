import numpy as np
import math


def generate_angle(i: float, resolution: int, angle_offset: float) -> float:
    return angle_offset + 2 * math.pi * i / resolution


def generate_torus_point(major_radius: float, minor_radius: float, i: int, j: int, resolution: int, angle_offset: float):
    theta = generate_angle(i, resolution, angle_offset)
    phi = generate_angle(j, resolution, angle_offset)

    return np.array([(major_radius + minor_radius * math.cos(theta)) * math.cos(phi),
                     (major_radius + minor_radius *
                      math.cos(theta)) * math.sin(phi),
                     minor_radius * math.sin(theta)])


def generate_tetrahedron_VF():
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

    Args: 
        major_radius.
        minor_radius.

    Returns:
        V: Torus vertices.
        F: Torus facets.
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
