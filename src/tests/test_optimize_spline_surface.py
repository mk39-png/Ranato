import os
import pytest
import numpy as np
import numpy.testing as npt
import igl
from scipy.sparse import coo_matrix
from cholespy import CholeskySolverD


from src.core.common import (
    Vector1D,
    MatrixXf,
    MatrixXi,
    compare_eigen_numpy_matrix,
    initialize_spot_control_mesh,
    deserialize_eigen_matrix_csv_to_numpy,
    float_equal,
)

from src.quadratic_spline_surface.optimize_spline_surface import (
    shift_array,
    generate_optimized_twelve_split_position_data,
    build_twelve_split_spline_energy_system,
    OptimizationParameters,
)

from src.core.affine_manifold import (
    AffineManifold
)

from src.tests.test_affine_manifold import (
    initialize_affine_manifold_from_spot_control,
)


def test_generate_optimized_twelve_split_position_data_from_spot_mesh() -> None:
    """
    Testing generate_optimized_twelve_split_position_data() from the TwelveSplitSplineSurface()
    constructor on the spot_control mesh.

    NOTE: this method has the following dependencies to work properly:
    * build_twelve_split_spline_energy_system() for fit case
    * build_twelve_split_spline_energy_system() for full case
    """


def test_generate_zero_vertex_gradients_from_spot_mesh() -> None:


def test_generate_zero_edge_gradients_from_spot_mesh() -> None:
    """
    Tests generate_zero_edge_gradients() from spot_mesh.
    Used in TwelveSplitSplineSurface generation.


    """


def test_compute_twelve_split_energy_quadratic_from_spot_mesh_fit() -> None:
    """
    Tests compute_twelve_split_energy_quadratic.
    Generates fit_energy, fit_derivatives, fit_hessian, fit_hessian_inverse.
    Used in TwelveSplitSplineSurface generation.
    """


def test_compute_twelve_split_energy_quadratic_from_spot_mesh_full() -> None:
    """
    Tests compute_twelve_split_energy_quadratic.
    Generates energy, derivatives, energy_hessian, energy_hessian_inverse
    Used in TwelveSplitSplineSurface generation.
    """


def test_build_twelve_split_spline_energy_system_from_spot_mesh() -> None:
    """
    Used in TwelveSplitSplineSurface generation.

    NOTE: this method has the following dependencies:
    * AffineManifold.he_to_corner.
    * index_vector_complement
    * generate_zero_vertex_gradients
    * generate_zero_edge_gradients
    * build_variable_vertex_indices_map
    * build_variable_edge_indices_map
    * compute_twelve_split_energy_quadratic
    NOTE: most of the above methods will result in the same values for fit 
    and full cases of the TwelveSplitSplineSurface constructor.
    But, compute_twelve_split_energy_quadratic will result in different values.

    Tests entirity of the build_twelve_split_spline_energy_system function.
    Retrieving proper he_to_corner from affine_manifold for spot_control mesh.
    Testing for both fit and full cases.
    """


def test_shift_array() -> None:
    """

    """
    # NOTE: I checked this behavior with the ASOC code.
    # In list[np.ndarray] = [0, 1, 2]
    # change such that it shifts to [2, 0, 1]
    # or like [10, 20, 30] becomes [30, 10, 20]
    # Basically, moving elements to the left by "shift" amount.

    # Simple array
    int_list: list[int] = [0, 1, 2]

    shift_array(int_list, 2)
    assert int_list[0] == 2
    assert int_list[1] == 0
    assert int_list[2] == 1

    # list of NumPy array
    numpy_list: list[np.ndarray[tuple[int, int], np.dtype[np.int64]]] = [
        np.full(shape=(2, 2), fill_value=0, dtype=np.int64),
        np.full(shape=(2, 2), fill_value=1, dtype=np.int64),
        np.full(shape=(2, 2), fill_value=2, dtype=np.int64)]

    shift_array(numpy_list, 1)
    npt.assert_array_equal(numpy_list[0], np.full(shape=(2, 2), fill_value=1))
    npt.assert_array_equal(numpy_list[1], np.full(shape=(2, 2), fill_value=2))
    npt.assert_array_equal(numpy_list[2], np.full(shape=(2, 2), fill_value=0))


def test_build_twelve_split_spline_energy_system_from_spot_mesh_fit() -> None:
    """
    This tests build_twelve_split_spline_energy() with 
    optimization_params_fit.parametrized_quadratic_surface_mapping_factor = 0.0
    NOTE: relies on generate_face_normals() to work.

    Used in TwelveSplitSplineSurface generation.

    """
    # Get input mesh
    V, uv, F, FT = initialize_spot_control_mesh()

    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)

    # Get input mesh
    optimization_params_fit = OptimizationParameters(parametrized_quadratic_surface_mapping_factor=0.0)

    # NOTE: use this file because it is what I tested it with.
    N_from_generate_face_normals = "spot_control\\12_split_spline\\generate_face_normals\\N.csv"
    N_test: MatrixXf = deserialize_eigen_matrix_csv_to_numpy(N_from_generate_face_normals, False)

    fit_energy: float
    fit_derivatives: Vector1D
    fit_matrix: coo_matrix
    fit_matrix_inverse: CholeskySolverD
    fit_energy, fit_derivatives, fit_matrix, fit_matrix_inverse = build_twelve_split_spline_energy_system(
        V, N_test, affine_manifold, optimization_params_fit)

    assert float_equal(fit_energy, 0.0)  # NOTE: magic number from ASOC code output for fit_energy
    # TODO: move these to different test case.
    compare_eigen_numpy_matrix(
        "spot_control\\12_split_spline\\fit_derivatives.csv", fit_derivatives.flatten())
    # NOTE: below works! But, takes quite a bit of time to run.
    compare_eigen_numpy_matrix(
        "spot_control\\12_split_spline\\fit_matrix_dense.csv", fit_matrix.todense())


def test_build_twelve_split_spline_energy_system_from_spot_mesh_full() -> None:
    """
    This tests build_twelve_split_spline_energy() with 
    optimization_params_fit.parametrized_quadratic_surface_mapping_factor = 1.0
    (i.e. the full energy hessian system is built)
    NOTE: build_twelve_split_spline_energy_system() has the following dependencies:
    * generate_face_normals() for generating the correct values as the original spot_control mesh.
    * AffineManifold.he_to_corner.
    * index_vector_complement
    * generate_zero_vertex_gradients
    * generate_zero_edge_gradients
    * build_variable_vertex_indices_map
    * build_variable_edge_indices_map
    * compute_twelve_split_energy_quadratic
    """
    # Get input mesh
    V, uv, F, FT = initialize_spot_control_mesh()

    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)

    # Get input mesh
    optimization_params = OptimizationParameters(parametrized_quadratic_surface_mapping_factor=1.0)

    # NOTE: use this file because it is what I tested it with.
    N_from_generate_face_normals = "spot_control\\12_split_spline\\generate_face_normals\\N.csv"
    N_test: MatrixXf = deserialize_eigen_matrix_csv_to_numpy(N_from_generate_face_normals, False)

    energy: float
    derivatives: Vector1D
    energy_hessian: coo_matrix
    energy_hessian_inverse: CholeskySolverD
    energy, derivatives, energy_hessian, energy_hessian_inverse = build_twelve_split_spline_energy_system(V,
                                                                                                          N_test,
                                                                                                          affine_manifold,
                                                                                                          optimization_params)
    # NOTE: magic number from ASOC code's output for energy
    # TODO: move to different test case.
    assert float_equal(energy, 1269.9805595159069)
    compare_eigen_numpy_matrix("spot_control\\12_split_spline\\derivatives.csv", derivatives.flatten())
    # TODO: precision for below is quite small.
    compare_eigen_numpy_matrix(
        "spot_control\\12_split_spline\\energy_hessian_dense.csv", energy_hessian.todense(), atol=1e-4)
