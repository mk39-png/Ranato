# This needs various things to work...


import logging
import os

import numpy as np

from ranato.algebraic_contours.contour_network.compute_intersections import \
    IntersectionParameters
from ranato.algebraic_contours.contour_network.contour_network import (
    ContourNetwork, InvisibilityMethod, InvisibilityParameters,
    _build_contour_labels)
from ranato.algebraic_contours.core.affine_manifold import AffineManifold
from ranato.algebraic_contours.core.apply_transformation import \
    apply_camera_frame_transformation_to_vertices
from ranato.algebraic_contours.core.common import (
    LOGGER, Matrix3x3f, MatrixNx3f, deserialize_eigen_matrix_csv_to_numpy,
    initialize_spot_control_mesh)
from ranato.algebraic_contours.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from ranato.algebraic_contours.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface
from ranato.algebraic_contours.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)
from ranato.algebraic_contours.tests.test_compute_rational_bezier_curve_intersections import \
    ROOT_FOLDER
from ranato.algebraic_contours.utils.projected_curve_networks_utils import (
    SVGOutputMode, compare_list_segment_geometry, compare_segment_labels)

# TODO: the deserialization of rational functions and then printing of rational functions should be the same as the whole rational functions.txt file


def test_build_contour_labels_spot_control() -> None:
    """

    """
    filepath: str = f"{ROOT_FOLDER}\\contour_network\\contour_network\\build_contour_labels\\"

    contour_patch_indices: list[int] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"contour_patch_indices.csv").tolist()
    contour_is_boundary: list[bool] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath+"contour_is_boundary.csv"), dtype=bool).tolist()

    contour_segment_labels_test: list[dict[str, int]] = _build_contour_labels(
        contour_patch_indices,
        contour_is_boundary)

    compare_segment_labels(filepath+"contour_segment_labels.json",
                           contour_segment_labels_test)


def test_contour_network() -> None:
    """
    Testing contour network creation with control spot mesh
    """
    svg_output_mode = SVGOutputMode.UNIFORM_VISIBLE_CURVES
    optimization_params = OptimizationParameters()
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()
    weight: float = optimization_params.position_difference_factor
    trim: float = intersect_params.trim_amount
    pad: float = invisibility_params.pad_amount
    invisibility_method: InvisibilityMethod = invisibility_params.invisibility_method
    show_nodes: bool = False
    LOGGER.setLevel(logging.INFO)

    # Set up the camera
    frame: Matrix3x3f = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
    LOGGER.info("Projecting onto frame:\n%s", frame)
    V:  np.ndarray
    uv: np.ndarray
    F:  np.ndarray
    FT: np.ndarray
    V, uv, F, FT = initialize_spot_control_mesh()
    V_transformed: MatrixNx3f = apply_camera_frame_transformation_to_vertices(V, frame)

    # Generate quadratic spline
    LOGGER.info("Computing spline surface")
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V_transformed,
                                                                        affine_manifold,
                                                                        optimization_params)

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]]
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices))

    # Build the contours
    LOGGER.info("Computing contours")
    contour_network = ContourNetwork(
        spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges
    )

    # Save the contours to file
    LOGGER.info("Saving contours")
    contour_network_file: str = "contours.svg"
    contour_network_path: str = os.path.abspath(f"src\\tests\\spot_control\\{contour_network_file}")
    contour_network.write(contour_network_path, svg_output_mode, show_nodes)
