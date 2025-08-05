from src.core.common import *
from src.utils.generate_shapes import *
from src.quadratic_spline_surface.optimize_spline_surface import *
from src.quadratic_spline_surface.twelve_split_spline import *
from src.utils.generate_position_data import *
from src.core.bivariate_quadratic_function import evaluate_quadratic_mapping

import pytest
import numpy as np
import os


def test_spot_model():
    # Build maps from strings to enums
    log_level_map: dict[str, int] = {
        "off": logging.NOTSET,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARN,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    # Get command line arguments
    # parser = argparse.ArgumentParser(
    #     prog="ASOC",
    #     description="Generate smooth occluding contours for a mesh.",
    # )
    input_filename: str = "spot_control\\spot_control_mesh-cleaned_conf_simplified_with_uv.obj"
    # input_filename: str = "spot_quadrangulated_tri_clean_conf_simplified_with_uv.obj"
    output_dir: str = "./"
    # log_level = logging.NOTSET
    log_level = logging.DEBUG
    color: Matrix3x1r = SKY_BLUE
    num_subdivisions: int = DISCRETIZATION_LEVEL
    optimization_params: OptimizationParameters = OptimizationParameters()
    weight: float = optimization_params.position_difference_factor

    # parser.add_argument('-i', '--input', required=True, help="Mesh filepath", type=str)
    # parser.add_argument('--log_level', help="Level of logging", action='store_const', type=int)
    # parser.add_argument("--num_subdivisions", help="Number of subdivisions", action='store_const', type=int)
    # parser.add_argument(
    #     '-w', '--weight', help="Fitting weight for the quadratic surface approximation", action='store_const', type=float)
    # args = parser.parse_args()

    # Set logger level
    logger.setLevel(log_level)

    # Set optimization parameters
    weight: float = optimization_params.position_difference_factor

    # Get input mesh
    V: np.ndarray = np.ndarray(shape=(0, 0), dtype=np.float64)
    uv = np.ndarray(shape=(0, 0), dtype=np.float64)
    N = np.ndarray(shape=(0, 0), dtype=np.float64)
    F = np.ndarray(shape=(0, 0), dtype=np.int64)
    FT = np.ndarray(shape=(0, 0), dtype=np.int64)
    FN = np.ndarray(shape=(0, 0), dtype=np.int64)
    root_folder = os.getcwd()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    obj_dir = os.path.join(script_dir, input_filename)
    V, uv, N, F, FT, FN = igl.readOBJ(obj_dir)

    # Generate quadratic spline
    logger.info("Computing spline surface")

    # NOTE: must input a mesh that is already UV unwrapped....
    # TODO: but the specific UV unwrapping algorithm is special?
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    # FIXME: there actually might be something wrong with the affine_manifold... which would make sense because I have yet to check said function.

    # eigen_array = deserialize_eigen_matrix_csv_to_numpy("1_fit_corner_to_corner_uv_positions_CONTROL_LAST_ITER.csv")
    # print(eigen_array)

    # NOTE: m_F of affine manifold seems off... maybe. TODO: TwelveSplitSpline surface calc takes approxs 2 min-ish
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V, affine_manifold,  optimization_params)

    # TODO: Move the tests below to a separate case

    # TODO: Making sure that the 12 split spline patches are the same as quadratic patch...
    # First open files to convert into list[QuadraticSplineSurfacePatch]
    # filename_control: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    # filepath_control: str = os.path.abspath(f"src\\tests\\{filename_control}")
    # # NOTE: need the placeholder to utilize its deserialize() method
    # spline_surface_placeholder = QuadraticSplineSurface(filename=filename_control)
    # with open(filepath_control, "r", encoding="utf-8") as file_control:
    #     control_patches: list[QuadraticSplineSurfacePatch] = spline_surface_placeholder.deserialize(file_control)
    #     file_control.close()

    # Now, grabbing the patches made from twelve_split_spline (i.e. our patches to test)
    # test_patches: list[QuadraticSplineSurfacePatch] = spline_surface.m_patches
    # assert len(control_patches) == len(test_patches)
    # num_patches: int = len(control_patches)
    # # Now, checking the values that have been saved
    # # tests domain
    # for i in range(num_patches):
    #     surface_mapping_coeffs_control: Matrix6x3r = control_patches[i].get_surface_mapping()  # cx, cy, cz
    #     domain_control: ConvexPolygon = control_patches[i].get_domain
    #     vertices_control: Matrix3x2r = domain_control.get_vertices  # p1, p2, p3

    #     surface_mapping_coeffs_test: Matrix6x3r = test_patches[i].get_surface_mapping()  # cx, cy, cz
    #     domain_test: ConvexPolygon = test_patches[i].get_domain
    #     vertices_test: Matrix3x2r = domain_test.get_vertices  # p1, p2, p3

    #     npt.assert_allclose(vertices_control, vertices_test, atol=FLOAT_EQUAL_PRECISION)

    # Now, checking the values that have been saved
    # tests surface_mapping_coeffs...
    # for i in range(num_patches):
    #     surface_mapping_coeffs_control: Matrix6x3r = control_patches[i].get_surface_mapping()  # cx, cy, cz
    #     domain_control: ConvexPolygon = control_patches[i].get_domain
    #     vertices_control: Matrix3x2r = domain_control.get_vertices  # p1, p2, p3

    #     surface_mapping_coeffs_test: Matrix6x3r = test_patches[i].get_surface_mapping()  # cx, cy, cz
    #     domain_test: ConvexPolygon = test_patches[i].get_domain
    #     vertices_test: Matrix3x2r = domain_test.get_vertices  # p1, p2, p3

    #     # tests domain
    #     # FIXME: surface_mapping_coeffs still bad...
    #     npt.assert_allclose(vertices_control, vertices_test, atol=FLOAT_EQUAL_PRECISION)
    #     npt.assert_allclose(surface_mapping_coeffs_control, surface_mapping_coeffs_test, atol=FLOAT_EQUAL_PRECISION)

    # View the mesh
    spline_surface.view(color, num_subdivisions)


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
