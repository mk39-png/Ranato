import os
import logging
import pytest
import igl
import numpy as np
import numpy.testing as npt
from collections import defaultdict

from src.core.common import (
    compare_eigen_numpy_matrix,
    SKY_BLUE,
    DISCRETIZATION_LEVEL,
    logger,
    PlanarPoint,
    SpatialVector,
    MatrixXf,
    MatrixXi,
    Matrix6x3r,
)
from src.core.affine_manifold import (
    AffineManifold,
    ParametricAffineManifold,
)
from src.core.bivariate_quadratic_function import (
    evaluate_quadratic_mapping
)

from src.core.convex_polygon import ConvexPolygon

from src.quadratic_spline_surface.quadratic_spline_surface_patch import QuadraticSplineSurfacePatch
from src.quadratic_spline_surface.quadratic_spline_surface import QuadraticSplineSurface
from src.quadratic_spline_surface.optimize_spline_surface import (
    OptimizationParameters,
    build_twelve_split_spline_energy_system,
    generate_optimized_twelve_split_position_data,
)

from src.quadratic_spline_surface.twelve_split_spline import (
    TriangleCornerData,
    TriangleMidpointData,
    TwelveSplitSplineSurface,
    # generate_affine_manifold_corner_data,
    # generate_affine_manifold_midpoint_data,
    generate_twelve_split_spline_patch_surface_mapping,
    generate_twelve_split_spline_patch_patch_boundaries,
    generate_twelve_split_spline_patch_patch_to_corner_map,
)

from src.utils.generate_position_data import (
    QuadraticGradientFunction,
    QuadraticPositionFunction,
    generate_parametric_affine_manifold_corner_data,
    generate_parametric_affine_manifold_midpoint_data,
)


# ****************
# Helper Methods
# ****************

def initialize_twelve_split_spline_from_spot_mesh() -> tuple[TwelveSplitSplineSurface, AffineManifold, MatrixXf]:
    """
    This is used to test the member variables of TwevleSplitSplineSurface().
    Helper function that initialized TwelveSplitSplineSurface from the spot_control mesh.
    Also returns the affine_manifold used to build the TwelveSplitSplineSurface object.
    Also returns vertices used to initialize TwelveSplitSplineSurface
    (i.e. vertices of the spot_control mesh)
    """
    log_level = logging.DEBUG
    color: tuple[float, float, float] = SKY_BLUE
    num_subdivisions: int = DISCRETIZATION_LEVEL
    optimization_params: OptimizationParameters = OptimizationParameters()

    # Set logger level
    logger.setLevel(log_level)

    # Get input mesh
    V: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    uv: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    N: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    F: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)
    FT: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)
    FN: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)

    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv.obj"
    filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")
    V, uv, N, F, FT, FN = igl.readOBJ(filepath)

    # Generate quadratic spline
    logger.info("Computing spline surface")
    # NOTE: must input a mesh that is already UV unwrapped....
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V,
                                                                        affine_manifold,
                                                                        optimization_params)

    return spline_surface, affine_manifold, V


# ****************
# Test Methods
# ****************

def test_init_twelve_split_patches_from_spot_mesh() -> None:
    """
    Tests init_twelve_split_patches using spot_control mesh.
    NOTE: relies on generate_twelve_split_spline_patch_patch_boundaries,
    ConvexPolygon.init_from_boundary_segments_coeffs(patch_boundaries[i]),
    generate_twelve_split_spline_patch_patch_to_corner_map, and
    generate_twelve_split_spline_patch_surface_mapping
    on working.

    Relies on AffineManifold compute_cone_corners() on working.
    """
    # Get input mesh
    spline_surface: TwelveSplitSplineSurface
    affine_manifold: AffineManifold
    V: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    spline_surface, affine_manifold, V = initialize_twelve_split_spline_from_spot_mesh()

    corner_data: dict[int, dict[int, TriangleCornerData]] = defaultdict(dict)
    midpoint_data: dict[int, dict[int, TriangleMidpointData]] = defaultdict(dict)
    face_to_patch_indices: list[list[int]]
    patch_to_face_indices: list[int]
    is_cone_corner: list[list[bool]] = affine_manifold.compute_cone_corners()

    face_to_patch_indices, patch_to_face_indices = spline_surface.init_twelve_split_patches(corner_data,
                                                                                            midpoint_data,
                                                                                            is_cone_corner)

    compare_eigen_numpy_matrix("spot_control\\12_split_spline\\init_twelve_split_patches\\face_to_patch_indices.csv",
                               np.array(face_to_patch_indices))
    compare_eigen_numpy_matrix("spot_control\\12_split_spline\\init_twelve_split_patches\\patch_to_face_indices.csv",
                               np.array(patch_to_face_indices))


def test_generate_twelve_split_spline_patch_patch_to_corner_map_from_spot_mesh() -> None:
    """
    Tests generate_twelve_split_spline_patch_patch_to_corner_map().
    """
    patch_to_corner_map: list[tuple[int, int]]  # list of length 12
    patch_to_corner_map = generate_twelve_split_spline_patch_patch_to_corner_map()
    assert len(patch_to_corner_map) == 12
    compare_eigen_numpy_matrix("spot_control\\12_split_spline\\init_twelve_split_patches\\patch_to_corner_map.csv",
                               np.array(patch_to_corner_map))


def test_generate_twelve_split_spline_patch_patch_boundaries_from_spot_mesh() -> None:
    """
    Tests generate_twelve_split_spline_patch_patch_boundaries().
    """
    patch_boundaries: list[list[np.ndarray]] = generate_twelve_split_spline_patch_patch_boundaries()
    assert len(patch_boundaries) == 12
    assert len(patch_boundaries[0]) == 3
    assert patch_boundaries[0][0].shape == (3, 1)
    compare_eigen_numpy_matrix("spot_control\\12_split_spline\\init_twelve_split_patches\\patch_boundaries.csv",
                               np.array(patch_boundaries).squeeze(), make_3d=True)


def test_generate_face_normals_from_spot_mesh() -> None:
    """
    This tests generate_face_normals
    """
    # Get input mesh
    spline_surface: TwelveSplitSplineSurface
    affine_manifold: AffineManifold
    V: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    spline_surface, affine_manifold, V = initialize_twelve_split_spline_from_spot_mesh()
    compare_eigen_numpy_matrix("spot_control\\12_split_spline\\generate_face_normals\\V.csv", V)
    # compare_eigen_numpy_matrix("spot_control\\12_split_spline\\generate_face_normals\\F.csv", F)

    N_test: MatrixXf = spline_surface.generate_face_normals(V, affine_manifold)
    # XXX: N was fixed since there was a typo with angle_from_positions() in common.py
    compare_eigen_numpy_matrix("spot_control\\12_split_spline\\generate_face_normals\\N.csv", N_test)


def test_twelve_split_spline_from_spot_model() -> None:
    """
    This is used to test and view the spot model.
    """
    log_level = logging.DEBUG
    color: tuple[float, float, float] = SKY_BLUE
    num_subdivisions: int = DISCRETIZATION_LEVEL
    optimization_params: OptimizationParameters = OptimizationParameters()

    # Set logger level
    logger.setLevel(log_level)

    # Get input mesh
    V: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    uv: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    N: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    F: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)
    FT: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)
    FN: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)

    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv.obj"
    filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")
    V, uv, N, F, FT, FN = igl.readOBJ(filepath)

    # Generate quadratic spline
    logger.info("Computing spline surface")
    # NOTE: must input a mesh that is already UV unwrapped....
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V, affine_manifold,  optimization_params)

    # View the mesh
    spline_surface.view(color, num_subdivisions)


def test_twelve_split_spline_patches_with_spot_control() -> None:
    """
    Testing to see if parent class QuadraticSplineSurface is 
    utilized properly by TwelveSplitSpline.
    """
    log_level = logging.DEBUG
    color: tuple[float, float, float] = SKY_BLUE
    num_subdivisions: int = DISCRETIZATION_LEVEL
    optimization_params: OptimizationParameters = OptimizationParameters()

    # Set logger level
    logger.setLevel(log_level)

    # Get input mesh
    V: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    uv: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    N: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    F: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)
    FT: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)
    FN: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)

    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv.obj"
    filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")
    V, uv, N, F, FT, FN = igl.readOBJ(filepath)

    # Generate quadratic spline
    logger.info("Computing spline surface")
    # NOTE: must input a mesh that is already UV unwrapped....
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V, affine_manifold,  optimization_params)

    # TODO: Making sure that the 12 split spline patches are the same as quadratic patch...
    # First open files to convert into list[QuadraticSplineSurfacePatch]
    filename_control: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    filepath_control: str = os.path.abspath(f"src\\tests\\spot_control\\{filename_control}")

    # NOTE: need the placeholder to utilize its deserialize() method
    spline_surface_placeholder = QuadraticSplineSurface(filepath=filepath_control)
    with open(filepath_control, "r", encoding="utf-8") as file_control:
        control_patches: list[QuadraticSplineSurfacePatch] = spline_surface_placeholder.deserialize(file_control)
        file_control.close()

    # Now, grabbing the patches made from twelve_split_spline (i.e. our patches to test)
    test_patches: list[QuadraticSplineSurfacePatch] = spline_surface.m_patches
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

        # lower precision because serialization loses precision
        npt.assert_allclose(vertices_control, vertices_test, atol=1e-5)
        npt.assert_allclose(surface_mapping_coeffs_control, surface_mapping_coeffs_test, atol=1e-3)

    # View the mesh
    spline_surface.view(color, num_subdivisions)


# *******************
# Original Test Cases
# *******************

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

    domain_point: PlanarPoint = np.array([[0.2, 0.3]])
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
    p: SpatialVector = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    zero: SpatialVector = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    assert p.shape == (1, 3)
    assert zero.shape == (1, 3)

    corner_data: dict[int, TriangleCornerData] = {
        0: TriangleCornerData(p, zero, zero),
        1: TriangleCornerData(p, zero, zero),
        2: TriangleCornerData(p, zero, zero)
    }

    midpoint_data: dict[int, TriangleMidpointData] = {
        0: TriangleMidpointData(zero),
        1: TriangleMidpointData(zero),
        2: TriangleMidpointData(zero)
    }

    surface_mappings: list[Matrix6x3r]  # length 12 array with matrices shape (6, 3)
    surface_mappings = generate_twelve_split_spline_patch_surface_mapping(
        corner_data,
        midpoint_data)

    domain_point: PlanarPoint = np.array([[0.25, 0.25]], dtype=np.float64)
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
