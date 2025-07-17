from src.core.common import *
from PS12_patch_coeffs import *

from src.core.affine_manifold import AffineManifold
# from src.core.differentiable_variable
# from src.core.halfedge import *
# from src.core.polynomial_function import *
from src.quadratic_spline_surface.position_data import TriangleCornerData, TriangleMidpointData


class OptimizationParameters:
    # NOTE: It's OK having these member variables "global" throughout all OptimizationParameter objects
    # TODO: wait, it might not be depending on how this all works...
    # Meaning, I may have to make a constructor method for this class.

    # -- Main optimization weight options --
    position_difference_factor: float = 1.0
    parametrized_quadratic_surface_mapping_factor: float = 1.0

    # -- Weights for cone positions, normals, and gradients --
    # fitting weight for cone vertices
    cone_position_difference_factor: float = 1.0

    # fitting weight for cone vertex gradients
    cone_vertex_gradient_difference_factor: float = 1e6

    # fitting weight for vertices collapsed to a cone
    cone_adjacent_position_difference_factor: float = 1.0

    # fitting weight for vertex gradients collapsing to a cone
    cone_adjacent_vertex_gradient_difference_factor: float = 0.0

    # fitting weight for vertex edge gradients collapsing to a cone
    cone_adjacent_edge_gradient_difference_factor: float = 0.0

    # weight for encouraging orthogonality with a normal at a cone
    cone_normal_orthogonality_factor: float = 0.0

    # TODO
    # Perform one more energy computation for the final value
    compute_final_energy = False
    # Perform final optimization with fixed vertices and flatten cone constraints
    flatten_cones = False
    hessian_builder: int = 1  # 1 for assemble, 0 for autodiff, otherwise assemble


def build_twelve_split_spline_energy_system(initial_V: np.ndarray, initial_face_normals: np.ndarray, affine_manifold: AffineManifold, optimization_params: OptimizationParameters) -> tuple[float, VectorX, np.ndarray, np.ndarray]:
    """
    Build the quadratic energy system for the twelve-split spline with thin
    plate, fitting, and planarity energies.

    @param[in] initial_V: initial vertex positions
    @param[in] initial_face_normals: initial vertex normals
    @param[in] affine_manifold: mesh topology and affine manifold structure
    @param[in] optimization_params: parameters for the spline optimization

    @param[out] energy: energy value (i.e., constant term)
    @param[out] derivatives: energy gradient (i.e., linear term)
    @param[out] hessian: energy Hessian (i.e., quadratic term)
    @param[out] hessian_inverse: solver for inverting the Hessian
    """
    # TODO: change hessian to sparse matrix
    # TODO: change hessian_inverse to cholmad sparse matrix
    todo()


def generate_optimized_twelve_split_position_data(V: np.ndarray, affine_manifold: AffineManifold, fit_matrix: np.ndarray, hessian_inverse: np.ndarray) -> tuple[list[list[TriangleCornerData]], list[list[TriangleMidpointData]]]:
    """
    Compute the optimal per triangle position data for given vertex positions.

    @param[in] V: vertex positions
    @param[in] affine_manifold: mesh topology and affine manifold structure
    @param[in] fit_matrix: quadratic fit energy Hessian matrix
    @param[in] hessian_inverse: solver for inverting the energy Hessian

    @param[out] corner_data: quadratic vertex position and derivative data
    @param[out] midpoint_data: quadratic edge midpoint derivative data
    """
    # TODO: change fit_matrix to sparse
    assert len(corner_data[0]) == 3
    assert len(midpoint_data[0]) == 3
    todo()


def generate_zero_vertex_gradients(num_vertices: int) -> list[Matrix2x3r]:
    """
    Generate zero value gradients for a given number of vertices.

    @param[in] num_vertices: number of vertices |V|
    @param[out] gradients: |V| trivial vertex gradient matrices
    """
    return gradients
    todo()


def generate_zero_edge_gradients(num_faces: int) -> list[list[SpatialVector]]:
    """
    Generate zero value gradients for a given number of halfedges.

    @param[in] num_faces: number of faces |F|
    @param[out] gradients: 3|F| trivial edge gradient matrices
    """
    return edge_gradients

    todo()


def convert_full_edge_gradients_to_reduced(edge_gradients: list[list[Matrix2x3r]]) -> list[list[SpatialVector]]:
    """
    Given edge and opposite corner direction gradients at triangle edge midpoints,
    extract just the opposite corner direction gradient
    @param[in] edge_gradients: edge and corner directed gradients per edge midpoints
    @param[out] reduced_edge_gradients: opposite corner directed gradients per edge midpoints
    """
    unimplemented()


def convert_reduced_edge_gradients_to_full(reduced_edge_gradients: list[list[SpatialVector]], corner_data: list[list[TriangleCornerData]], affine_manifold: AffineManifold) -> list[list[Matrix2x3r]]:
    """
    Given edge direction gradients at triangle edge midpoints, append the gradients in the
    direction of the opposite triangle corners, which are determined by gradients and
    position data at the corners.

    @param[in] reduced_edge_gradients: opposite corner directed gradients per edge midpoints
    @param[in] corner_data: quadratic vertex position and derivative data
    @param[in] affine_manifold: mesh topology and affine manifold structure
    @param[out] edge_gradients: edge and corner directed gradients per edge midpoints
    """
    todo()
