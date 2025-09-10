# This needs various things to work...


import logging
import os

import numpy as np

from src.contour_network.compute_intersections import IntersectionParameters
from src.contour_network.contour_network import (ContourNetwork,
                                                 InvisibilityMethod,
                                                 InvisibilityParameters)
from src.core.affine_manifold import AffineManifold
from src.core.apply_transformation import \
    apply_camera_frame_transformation_to_vertices
from src.core.common import (Matrix3x3f, MatrixNx3f,
                             initialize_spot_control_mesh, logger)
from src.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from src.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface
from src.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)
from src.utils.projected_curve_networks_utils import SVGOutputMode

# TODO: the deserialization of rational functions and then printing of rational functions should be the same as the whole rational functions.txt file


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
    logger.setLevel(logging.INFO)

    # Set up the camera
    frame: Matrix3x3f = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
    logger.info("Projecting onto frame:\n%s", frame)
    V:  np.ndarray
    uv: np.ndarray
    F:  np.ndarray
    FT: np.ndarray
    V, uv, F, FT = initialize_spot_control_mesh()
    V_transformed: MatrixNx3f = apply_camera_frame_transformation_to_vertices(V, frame)

    # Generate quadratic spline
    logger.info("Comnputing spline surface")
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V_transformed,
                                                                        affine_manifold,
                                                                        optimization_params)

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]]
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices))

    # Build the contours
    logger.info("Computing contours")
    contour_network = ContourNetwork(
        spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges
    )

    # Save the contours to file
    logger.info("Saving contours")
    contour_network_file: str = "contours.svg"
    contour_network_path: str = os.path.abspath(f"src\\tests\\spot_control\\{contour_network_file}")
    contour_network.write(contour_network_path, svg_output_mode, show_nodes)
