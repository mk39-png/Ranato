from src.core.common import *
from src.core.apply_transformation import *
from src.core.convex_polygon import ConvexPolygon
from src.core.generate_transformation import *
from src.core.compute_boundaries import *
from src.contour_network.contour_network import *
from src.quadratic_spline_surface.quadratic_spline_surface_patch import QuadraticSplineSurfacePatch
from src.utils.generate_shapes import *
from src.quadratic_spline_surface.quadratic_spline_surface import *
from src.quadratic_spline_surface.optimize_spline_surface import *
from src.quadratic_spline_surface.twelve_split_spline import *
from igl import readOBJ, writeOBJ

import logging
import sys
import argparse
import os
import pytest
import numpy as np
import numpy.testing as npt


def test_read_write_spline_surface_serialization() -> None:
    """
    # test by deserializing control file, then serialize numpy code and compare to see if get the same file back
    Though, we would have to read through the files and compare the inputs...
    Deserialize, serialize, then check if true values..

    Utilizes write_spline and read_spline
    """
    filename_control: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    filepath_control: str = os.path.abspath(f"src\\tests\\spot_control\\{filename_control}")
    filename_test: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_NUMPY.txt"
    filepath_test: str = os.path.abspath(f"src\\tests\\spot_control\\{filename_test}")

    # NOTE: need a placeholder to call write_spline() and deserialize()...
    # TODO: better to separate deserialize() and write_spline() as separate from, QuadraticSplineSurface class and take QuadraticSplineSurface as the parameter or whatnot.
    spline_surface_placeholder = QuadraticSplineSurface(filename=filename_control)
    # Write the saved spline data to an external file. So, converting Eigen TXT -> Numpy Implementation -> NumPy TXT
    spline_surface_placeholder.write_spline(filename_test)

    # FIXME: make deserialize() independent of QudaraticSplineSurface...
    # TODO: now compare the files to see that they ghave the same contents
    control_patches: list[QuadraticSplineSurfacePatch]
    test_patches: list[QuadraticSplineSurfacePatch]

    # First open files to convert into list[QuadraticSplineSurfacePatch]
    with open(filepath_control, "r", encoding="utf-8") as file_control:
        control_patches: list[QuadraticSplineSurfacePatch] = spline_surface_placeholder.deserialize(file_control)
        file_control.close()

    with open(filepath_test, "r", encoding="utf-8") as file_test:
        test_patches: list[QuadraticSplineSurfacePatch] = spline_surface_placeholder.deserialize(file_test)
        file_test.close()

    assert len(control_patches) == len(test_patches)
    num_patches: int = len(control_patches)

    # Now, checking the values that have been saved
    for i in range(num_patches):
        surface_mapping_coeffs_control: Matrix6x3r = control_patches[i].get_surface_mapping()  # cx, cy, cz
        domain_control: ConvexPolygon = control_patches[i].get_domain
        vertices_control: Matrix3x2r = domain_control.get_vertices  # p1, p2, p3

        surface_mapping_coeffs_test: Matrix6x3r = test_patches[i].get_surface_mapping()  # cx, cy, cz
        domain_test: ConvexPolygon = test_patches[i].get_domain
        vertices_test: Matrix3x2r = domain_test.get_vertices  # p1, p2, p3

        npt.assert_allclose(surface_mapping_coeffs_control, surface_mapping_coeffs_test, atol=FLOAT_EQUAL_PRECISION)
        npt.assert_allclose(vertices_control, vertices_test, atol=FLOAT_EQUAL_PRECISION)


def test_read_view_spline_surface_deserialization() -> None:
    """
    Testing to see that the file can be deserialized and the surface displayed properly.
    """
    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    spline_surface = QuadraticSplineSurface(filename=filename)
    spline_surface.view()  # FIXME surface mesh displayed properly, but patch_boundaries is not working.


def test_discretize_patch_boundaries() -> None:
    """
    NOTE: also includes convert_nested_vector_to_matrix() and convert_polylines_to_edges() test.
    """
    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    spline_surface = QuadraticSplineSurface(filename=filename)
    boundary_points: list[SpatialVector]
    boundary_polylines: list[list[int]]
    boundary_points, boundary_polylines = spline_surface.discretize_patch_boundaries()

    # Discretize patch boundaries
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_points.csv", np.array(boundary_points).squeeze())
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_polylines.csv", np.array(boundary_polylines))

    # View contour curve network
    boundary_points_matrix: np.ndarray = convert_nested_vector_to_matrix(boundary_points)

    # FIXME: boundary polylines are not made properly...
    boundary_edges: list[list[int]] = convert_polylines_to_edges(boundary_polylines)

    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_points_mat.csv", boundary_points_matrix)
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_edges.csv", np.array(boundary_edges))


def test_discretize() -> None:
    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    surface_disc_params: SurfaceDiscretizationParameters = SurfaceDiscretizationParameters()
    spline_surface = QuadraticSplineSurface(filename=filename)
    V: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    F: np.ndarray[tuple[int, int], np.dtype[np.int64]]
    N: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    V, F, N = spline_surface.discretize(surface_disc_params)

    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\F_discretized_2_subdiv.csv", F)
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\V_discretized_2_subdiv.csv", V)
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\N_discretized_2_subdiv.csv", N)
