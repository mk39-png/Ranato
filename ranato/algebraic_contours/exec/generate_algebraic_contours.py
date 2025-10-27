"""
This is the user-facing interface to input a mesh and generate the algebraic contours.
"""
import os
import sys
from asyncio import base_events

import igl
import numpy as np
from mathutils import Matrix
from numpy.typing import ArrayLike

from ranato.algebraic_contours.contour_network.compute_intersections import \
    IntersectionParameters
from ranato.algebraic_contours.contour_network.contour_network import (
    ContourNetwork, InvisibilityParameters)
from ranato.algebraic_contours.core.affine_manifold import AffineManifold
from ranato.algebraic_contours.core.apply_transformation import \
    apply_transformation_to_vertices
from ranato.algebraic_contours.core.common import MatrixNx3f
from ranato.algebraic_contours.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from ranato.algebraic_contours.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)
from ranato.algebraic_contours.utils.projected_curve_networks_utils import \
    SVGOutputMode
from ranato.common import DIRECTORY_TEMP


# NOTE: this should also take some file...
# To preserve state or something?
def generate_algebraic_contours(projection_matrix: Matrix) -> None:
    """
    Testing contour network creation with control spot mesh.
    Reads from the temporary file.

    :param: projection matrix
    """
    # TODO: implement some ability for the user to change these parameters
    svg_output_mode = SVGOutputMode.UNIFORM_VISIBLE_CURVES
    optimization_params = OptimizationParameters()
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()
    show_nodes: bool = False

    # Retrieve the uv unwrapped mesh
    V: ArrayLike
    uv: ArrayLike
    N: ArrayLike
    F: ArrayLike
    FT: ArrayLike
    FN: ArrayLike
    V, uv, N, F, FT, FN = igl.readOBJ(os.path.join(DIRECTORY_TEMP, "temp_out.obj"))
    print(projection_matrix)

    # Preparing mesh data for use in contours calculation
    # TODO: will this work? The whole conversion of the MathUtils matrix to NumPy matrix?
    # TODO: maybe filter out non-numbers like 8.22e-16 and set to 0
    V_transformed: MatrixNx3f = apply_transformation_to_vertices(V, np.array(projection_matrix))
    print(V_transformed)
    print(V_transformed.shape)

    # Generate quadratic spline
    print("Computing spline surface...")
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    # TODO: should cache this result somewhere since the camera can be
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V_transformed,
                                                                        affine_manifold,
                                                                        optimization_params)

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices))

    # Build the contours
    print("Computing contours...")
    contour_network: ContourNetwork = ContourNetwork(
        spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges
    )

    # # # Save the contours to file
    # # logger.info("Saving contours")
    # contour_network_file: str = "contours.svg"
    # print(base_directory)
    # contour_network_path: str = os.path.abspath(
    #     f"{base_directory}\\tests\\spot_control\\{contour_network_file}")
    # contour_network.write(contour_network_path, svg_output_mode, show_nodes)
