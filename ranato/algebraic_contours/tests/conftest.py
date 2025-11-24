"""
File holding all utility methods related to testing. 
"""


import os

import igl
import numpy as np
import pytest

from ..contour_network.compute_intersections import IntersectionParameters
from ..contour_network.contour_network import (ContourNetwork,
                                               InvisibilityMethod,
                                               InvisibilityParameters)
from ..core.affine_manifold import AffineManifold
from ..core.apply_transformation import apply_transformation_to_vertices
from ..core.common import MatrixNx3f
from ..quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from ..quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)

FILE_BASE = "algebraic_contours"


#
# TESTING PARAMETERS
#
@pytest.fixture
def obj_filepaths() -> list[str]:
    print(os.path.abspath(f'{FILE_BASE}'))
    return [
        # (os.path.abspath(
        #     f'{FILE_BASE}\\tests\\spot_control\\spot_control_mesh-cleaned_conf_simplified_with_uv.obj')),
        (os.path.abspath("default_cube\\temp_out.obj"))
    ]
# @pytest.mark.parametrize(
#     "filepath",
#     [
#         # (os.path.abspath(
#         #     f'{FILE_BASE}\\tests\\spot_control\\spot_control_mesh-cleaned_conf_simplified_with_uv.obj')),
#         (os.path.abspath(
#             f'{FILE_BASE}\\tests\\default_cube\\temp_out.obj')),
#     ]
# )

# XXX: change to proper mark


@pytest.fixture
def projection_matrices() -> list[np.ndarray]:
    return [np.array([
        [2.777777671813965, 0.0, 0.0, 0.0],
        [0.0, 4.938271522521973, 0.0, 0.0],
        [0.0, 0.0, -1.0020020008087158, -0.20020020008087158],
        [0.0, 0.0, -1.0, 0.0]], dtype=np.float64)]
# @pytest.mark.parametrize(
#     "projection_matrices",
#     [
#         # TODO: this may not correspond to the Blender's actual projection matrix.
#         # That I need to test.
#         np.array([
#             [2.777777671813965, 0.0, 0.0, 0.0],
#             [0.0, 4.938271522521973, 0.0, 0.0],
#             [0.0, 0.0, -1.0020020008087158, -0.20020020008087158],
#             [0.0, 0.0, -1.0, 0.0]], dtype=np.float64)
#     ]
# )
#
# TESTING FIXTURES
#


@pytest.fixture(name="load_mesh_testing")
def load_mesh_testing(obj_filepaths) -> tuple[np.typing.ArrayLike,
                                              np.typing.ArrayLike,
                                              np.typing.ArrayLike,
                                              np.typing.ArrayLike,
                                              np.typing.ArrayLike,
                                              np.typing.ArrayLike]:
    """
    Returns deserialized .obj file.
    """
    return igl.readOBJ(obj_filepaths[0])


@pytest.fixture
def initialize_affine_manifold(load_mesh_testing) -> AffineManifold:
    """
    Fixture to calculate the AffineManifold from the load_mesh_testing fixture.
    """
    V_raw, uv, N, F, FT, FN = load_mesh_testing
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    return affine_manifold


@pytest.fixture(name="projection_on_vertices")
def projection_on_vertices(load_mesh_testing, projection_matrices) -> np.ndarray:
    """
    Returns vertices under projection matrix.
    """
    V_raw, uv, N, F, FT, FN = load_mesh_testing
    projection_matrix: np.ndarray = projection_matrices[0]

    V_transformed: MatrixNx3f = apply_transformation_to_vertices(V_raw, projection_matrix)
    return V_transformed


@pytest.fixture
def initialize_twelve_split_spline_transformed(initialize_affine_manifold,
                                               projection_on_vertices) -> TwelveSplitSplineSurface:
    """
    This is used to test the member variables of TwevleSplitSplineSurface().
    Helper function that initialized TwelveSplitSplineSurface from the spot_control mesh.
    Also returns the affine_manifold used to build the TwelveSplitSplineSurface object.
    Also returns vertices used to initialize TwelveSplitSplineSurface
    (i.e. vertices of the spot_control mesh)

    NOTE: initializes 12-split spline for use in contour network.
    """
    affine_manifold: AffineManifold = initialize_affine_manifold
    V_transformed: np.ndarray = projection_on_vertices
    optimization_params: OptimizationParameters = OptimizationParameters()

    # Generate quadratic spline
    spline_surface_transformed: TwelveSplitSplineSurface = TwelveSplitSplineSurface(
        V_transformed,
        affine_manifold,
        optimization_params)
    return spline_surface_transformed


@pytest.fixture
def initialize_contour_network(
        load_mesh_testing,
        initialize_twelve_split_spline_transformed) -> ContourNetwork:
    """
    Used for testing of contour network generat
    """
    # Retrieve parameters
    V_raw, uv, N, F, FT, FN = load_mesh_testing
    spline_surface: TwelveSplitSplineSurface = initialize_twelve_split_spline_transformed
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices))

    # Build the contours
    contour_network = ContourNetwork(
        spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges
    )

    return contour_network
