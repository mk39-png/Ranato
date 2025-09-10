"""
test_compute_ray_intersections_pencil_method.py

Testing various methods for calculating intersections.
"""


from src.contour_network.compute_ray_intersections_pencil_method import \
    solve_quadratic_quadratic_equation_pencil_method
from src.core.common import (PlanarPoint1d, Vector6f,
                             compare_eigen_numpy_matrix,
                             compare_list_list_varying_lengths_float,
                             deserialize_eigen_matrix_csv_to_numpy)


def test_solve_quadratic_quadratic_equation_pencil_method_spot_mesh() -> None:
    """
    Testing with values from the default spot control mesh.
    """
    # TODO: deserialize
    filepath: str = "spot_control\\contour_network\\compute_ray_intersections_pencil_method\\solve_quadratic_equation_pencil_method\\"
    a: Vector6f = deserialize_eigen_matrix_csv_to_numpy(filepath+"a.csv")
    b: Vector6f = deserialize_eigen_matrix_csv_to_numpy(filepath+"b.csv")
    num_intersections: int
    intersection_points: list[PlanarPoint1d]
    num_intersections, intersection_points = solve_quadratic_quadratic_equation_pencil_method(a, b)

    # TODO: test the intersection points nad whatnot...
    compare_eigen_numpy_matrix(filepath+"intersection_points.csv", intersection_points)


def test_pencil_first_part_spot_mesh() -> None:
    """
    Testing with values from the default spot control mesh.
    """
