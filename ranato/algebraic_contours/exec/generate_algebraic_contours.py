"""
This is the user-facing interface to input a mesh and generate the algebraic contours.
"""
import os
import sys

import igl
import numpy as np
from mathutils import Matrix

from ranato.algebraic_contours.contour_network.compute_intersections import \
    IntersectionParameters
from ranato.algebraic_contours.contour_network.contour_network import (
    ContourNetwork, InvisibilityParameters)
from ranato.algebraic_contours.core.affine_manifold import AffineManifold
from ranato.algebraic_contours.core.apply_transformation import \
    apply_transformation_to_vertices
from ranato.algebraic_contours.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from ranato.algebraic_contours.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)
from ranato.algebraic_contours.utils.projected_curve_networks_utils import \
    SVGOutputMode


def generate_algebraic_contours(projection_matrix: Matrix) -> None:
    """
    Testing contour network creation with control spot mesh.
    Reads from the temporary file.

    :param: projection matrix
    """
    svg_output_mode = SVGOutputMode.UNIFORM_VISIBLE_CURVES

    # TODO: implement some ability for the user to change these parameters
    optimization_params = OptimizationParameters()
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()
    show_nodes: bool = False

    # Retrieve the uv unwrapped mesh
    base_directory: str = os.path.dirname(__file__)

    # HACK: janky solution to get file read, prone to bugs
    relative_path: str = "..\\..\\temp\\temp_out.obj"
    absolute_path: str = os.path.abspath(os.path.join(base_directory, relative_path))
    # temp_directory: str = os.path.join("..\\..\\", "temp", "temp_out.obj")
    print("LOOK HERE", sys.prefix)
    print(absolute_path)
    V, uv, N, F, FT, FN = igl.readOBJ(absolute_path)

    # Preparing mesh data for use in contours calculation
    # TODO: will this work? The whole conversion of the MathUtils matrix to NumPy matrix?
    V_transformed = apply_transformation_to_vertices(V, np.array(projection_matrix))

    print(V_transformed)
    # # Generate quadratic spline
    # logger.info("Computing spline surface")
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)

    # # TODO: should cache this result somewhere
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V_transformed,
                                                                        affine_manifold,
                                                                        optimization_params)

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]]
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices))

    # # Build the contours
    # logger.info("Computing contours")
    contour_network: ContourNetwork = ContourNetwork(
        spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges
    )

    # # Save the contours to file
    # logger.info("Saving contours")
    contour_network_file: str = "contours.svg"
    contour_network_path: str = os.path.abspath(f"src\\tests\\spot_control\\{contour_network_file}")
    contour_network.write(contour_network_path, svg_output_mode, show_nodes)
