from src.core.common import *
from PS12_patch_coeffs import *

from src.core.affine_manifold import AffineManifold
from src.core.differentiable_variable import *
from src.core.halfedge import *
from src.core.polynomial_function import *

from src.quadratic_spline_surface.position_data import TriangleCornerData, TriangleMidpointData
from src.quadratic_spline_surface.powell_sabin_local_to_global_indexing import *
from src.quadratic_spline_surface.compute_local_twelve_split_hessian import *

from src.quadratic_spline_surface.planarH import planarHfun

import copy  # used for shifting array


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


class LocalHessianData:
    """
    Structure for the energy quadratic Hessian data
    """
    # NOTE: I added the default values below to be more Pythonic

    def __init__(self) -> None:
        self.H_f: np.ndarray = np.zeros(shape=(12, 12), dtype=float)  # fitting term hessian
        self.H_s: np.ndarray = np.zeros(shape=(12, 12), dtype=float)  # smoothness term hessian
        self.H_p: np.ndarray = np.zeros(shape=(36, 36), dtype=float)  # planarity term hessian
        self.w_f: float = 0.0  # fitting term weight
        self.w_s: float = 0.0  # smoothness term weight
        self.w_p: float = 0.0  # planarity term weight


class LocalDOFData:
    """
    Structure for the energy quadratic local degree of freedom data
    """

    def __init__(self) -> None:
        # NOTE: default types added to be more Pythonic.
        self.r_alpha_0: np.ndarray = np.zeros(shape=(12, 3), dtype=float)  # initial local DOF
        self.r_alpha: np.ndarray = np.zeros(shape=(12, 3), dtype=float)  # local DOF
        self.r_alpha_flat: np.ndarray = np.zeros(shape=(36, 1), dtype=float)  # flattened local DOF


def generate_local_hessian_data(face_vertex_uv_positions: list[PlanarPoint],
                                corner_to_corner_uv_positions: list[Matrix2x2r],
                                reverse_edge_orientations: list[bool],
                                is_cone: list[bool],
                                is_cone_adjacent: list[bool],
                                face_normal: SpatialVector,
                                optimization_params: OptimizationParameters) -> LocalHessianData:
    todo("This could be placed inside its class and serve as a constructor of some sort.")
    # Build uv from global positions

    # Each face_vertex_uv_position element is shape (1, 2)
    # TODO: shaping may be off
    uv: np.ndarray = np.array([face_vertex_uv_positions[0],
                               face_vertex_uv_positions[1],
                               face_vertex_uv_positions[2]])
    assert uv.shape == (3, 2)

    # TODO: the more Python way of creating the class would be to pass the values into the constructor of local_hessian_data, right?
    local_hessian_data = LocalHessianData()
    # H_s: local smoothness hessian
    local_hessian_data.H_s = build_local_smoothness_hessian(
        uv, corner_to_corner_uv_positions, reverse_edge_orientations)

    # H_f: position fit hessian
    local_hessian_data.H_f = build_local_fit_hessian(is_cone, is_cone_adjacent, optimization_params)

    # H_p: planar fitting term
    # DEPRECATED
    if (optimization_params.cone_normal_orthogonality_factor != 0.0):
        local_hessian_data.H_p = build_planar_constraint_hessian(uv,
                                                                 corner_to_corner_uv_positions,
                                                                 reverse_edge_orientations,
                                                                 face_normal)

    # w_s: smoothing weight
    local_hessian_data.w_s = optimization_params.parametrized_quadratic_surface_mapping_factor

    # w_f: fitting weight
    local_hessian_data.w_f = optimization_params.position_difference_factor

    # w_p: cone planar constraint weight
    local_hessian_data.w_p = optimization_params.cone_normal_orthogonality_factor

    return local_hessian_data


def generate_local_dof_data(initial_vertex_positions_T: list[SpatialVector],
                            vertex_positions_T: list[SpatialVector],
                            vertex_gradients_T: list[Matrix2x3r],
                            edge_gradients_T: list[SpatialVector]) -> LocalDOFData:
    todo("This could be placed as a constructor inside the class itself...")
    """
    Assemble the local degree of freedom data.
    """
    assert len(initial_vertex_positions_T) == 3
    assert len(vertex_positions_T) == 3
    assert len(vertex_gradients_T) == 3
    assert len(edge_gradients_T) == 3

    # r_alpha_0: fitting values
    # WARNING: Only fitting to zero implemented for gradients
    # TODO: Implement fitting for creases
    local_dof_data = LocalDOFData()
    local_dof_data.r_alpha_0[0, :] = initial_vertex_positions_T[0].flatten()
    local_dof_data.r_alpha_0[3, :] = initial_vertex_positions_T[1].flatten()
    local_dof_data.r_alpha_0[6, :] = initial_vertex_positions_T[2].flatten()

    # r_alpha: input values
    local_dof_data.r_alpha[0, :] = vertex_positions_T[0].flatten()
    local_dof_data.r_alpha[1, :] = vertex_gradients_T[0][0, :]  # row
    local_dof_data.r_alpha[2, :] = vertex_gradients_T[0][1, :]  # row
    local_dof_data.r_alpha[3, :] = vertex_positions_T[1].flatten()
    local_dof_data.r_alpha[4, :] = vertex_gradients_T[1][0, :]  # row
    local_dof_data.r_alpha[5, :] = vertex_gradients_T[1][1, :]  # row
    local_dof_data.r_alpha[6, :] = vertex_positions_T[2].flatten()
    local_dof_data.r_alpha[7, :] = vertex_gradients_T[2][0, :]  # row
    local_dof_data.r_alpha[8, :] = vertex_gradients_T[2][1, :]  # row
    local_dof_data.r_alpha[9, :] = edge_gradients_T[0].flatten()
    local_dof_data.r_alpha[10, :] = edge_gradients_T[1].flatten()
    local_dof_data.r_alpha[11, :] = edge_gradients_T[2].flatten()
    logger.info("Input values:\n%s", local_dof_data.r_alpha)

    for i in range(12):
        for j in range(3):
            # NOTE: r_alpha_flat is shape (36, 1)
            local_dof_data.r_alpha_flat[3 * i + j][0] = local_dof_data.r_alpha[i, j]

    return local_dof_data


def compute_local_twelve_split_energy_quadratic(local_hessian_data: LocalHessianData,
                                                local_dof_data: LocalDOFData) -> tuple[float, TwelveSplitGradient, TwelveSplitHessian]:
    todo("So, there is probably some issue with the matrix multiplication here")

    """
    Compute the local twelve split energy quadratic from Hessian and local
    degree of freedom data


    out: local_energy
    out: local_derivatives
    out: local_hessian
    """
    local_derivatives: TwelveSplitGradient = np.ndarray(shape=(36, 1))
    local_hessian: TwelveSplitHessian = np.ndarray(shape=(36, 36))

    assert local_derivatives.shape == (36, 1)
    assert local_hessian.shape == (36, 36)

    # Extract local hessian data
    H_f: np.ndarray = local_hessian_data.H_f  # shape (12, 12)
    H_s: np.ndarray = local_hessian_data.H_s  # shape (12, 12)
    H_p: np.ndarray = local_hessian_data.H_p  # shape (36, 36)
    w_f: float = local_hessian_data.w_f
    w_s: float = local_hessian_data.w_s
    w_p: float = local_hessian_data.w_p

    # Extract local degrees of freedom data
    r_alpha_0: np.ndarray = local_dof_data.r_alpha_0  # shape (12, 3)
    r_alpha: np.ndarray = local_dof_data.r_alpha  # shape (12, 3)
    r_alpha_flat: np.ndarray = local_dof_data.r_alpha_flat  # shape (36, 1)

    # full local 12x12 hessian (only smoothness and fitting terms)
    local_hessian_12x12: np.ndarray = np.full(shape=(12, 12),
                                              fill_value=2 * (w_s * H_s + w_f * H_f))

    # Add smoothness and fitting term blocks to the full local hessian per coordinate
    local_hessian[:, :] = 0
    for i in range(12):
        for j in range(12):
            for k in range(3):
                local_hessian[3 * i + k, 3 * j + k] = local_hessian_12x12[i, j]

    # Add 36x36 planar constraint term to the local hessian
    local_hessian += 2.0 * w_p * H_p

    # build per coordinate gradients for smoothness and fit terms
    # TODO: below, does it make a copy by values of r_alpha or no?
    g_alpha: np.ndarray = 2 * (w_s * H_s) * r_alpha
    logger.info("Block gradient after adding smoothness term:\n%s", g_alpha)
    g_alpha += 2 * (w_f * H_f) * (r_alpha - r_alpha_0)
    logger.info("Block gradient after adding fit term:\n%s", g_alpha)

    # Combine per coordinate gradients into the local
    for i in range(12):
        local_derivatives[3 * i, 0] = g_alpha[i, 0]
        local_derivatives[3 * i + 1, 0] = g_alpha[i, 1]
        local_derivatives[3 * i + 2, 0] = g_alpha[i, 2]

    # Add planar constraint term
    local_derivatives += 2.0 * w_p * H_p * r_alpha_flat

    # Add smoothness term
    smoothness_term: float = 0.0
    for i in range(3):
        # r_alpha shape = (1, 12) * (w_s * H_s) * (12, 1)
        # TODO: matmul @ or regular *????
        smoothness_term += (r_alpha[:, [i]].T * (w_s * H_s) * r_alpha[:, [i]])[0, 0]
    logger.info("Smoothness term is %s", smoothness_term)

    # Add fit term
    fit_term: float = 0.0
    for i in range(3):
        # r_alpha_diff shape = (12, 1)
        r_alpha_diff: np.ndarray = r_alpha[:, [i]] - r_alpha_0[:, [i]]  # gets columns

        fit_term += (r_alpha_diff.T * (w_f * H_f) * r_alpha_diff)[0, 0]
    logger.info("Fit term is %s", fit_term)

    # Add planar fitting term
    planar_term: float = 0.0
    planar_term += r_alpha_flat.T * (w_p * H_p) * r_alpha_flat
    logger.info("Planar orthogonality term is %s", planar_term)

    # Compute final energy
    local_energy: float = smoothness_term + fit_term + planar_term
    return local_energy, local_derivatives, local_hessian


def shift_array(__ref_arr: list, shift: int) -> list:
    """
    Helper function to cyclically shift an array of three elements.

    NOTE: performs DEEP copy and then shifts. So good for basic datatypes but not something like list[np.ndarray] for performance reasons.

    TODO: check that this modifies by reference
    """
    arr_copy = copy.deepcopy(__ref_arr)
    for i in range(3):
        __ref_arr[i] = arr_copy[(i + shift) % 3]

    # return arr


def shift_local_energy_quadratic_vertices(
        vertex_positions_T: list[SpatialVector],
        vertex_gradients_T: list[Matrix2x3r],
        edge_gradients_T: list[SpatialVector],
        initial_vertex_positions_T: list[SpatialVector],
        face_vertex_uv_positions: list[PlanarPoint],
        corner_to_corner_uv_positions: list[Matrix2x2r],
        reverse_edge_orientations: list[bool],
        is_cone: list[bool],
        is_cone_adjacent: list[bool],
        face_global_vertex_indices: list[int],
        face_global_edge_indices: list[int],
        shift: int) -> None:
    """
    Method to cyclically shift the indexing of all energy quadratic data arrays
    for triangle vertex values.
    NOTE: all array arguments/parameters for this function should be size 3.
    NOTE: should modifies parameters by reference.
    """
    shift_array(vertex_positions_T, shift)
    shift_array(vertex_gradients_T, shift)
    shift_array(edge_gradients_T, shift)
    shift_array(initial_vertex_positions_T, shift)
    shift_array(face_vertex_uv_positions, shift)
    shift_array(corner_to_corner_uv_positions, shift)
    shift_array(reverse_edge_orientations, shift)
    shift_array(is_cone, shift)
    shift_array(is_cone_adjacent, shift)
    shift_array(face_global_vertex_indices, shift)
    shift_array(face_global_edge_indices, shift)


def compute_twelve_split_energy_quadratic():
    """Compute the energy system for a twelve-split spline."""
    todo()

# ******************************************
# methods translated from .h are below
# ******************************************


def build_local_fit_hessian(is_cone: list[bool], is_cone_adjacent: list[bool], optimization_params: OptimizationParameters) -> np.ndarray:
    """
    Build the hessian for the local fitting energy
    This is a diagonal matrix with special weights for cones and cone adjacent
    vertices.
    Shape returned = (12,12)

    """
    H_f: np.ndarray = np.zeros(shape=(12, 12))

    # Check for cone collapsing vertices
    for i in range(3):
        vi: int = generate_local_vertex_position_variable_index(i, 0, 1)  # local vertex index

        # Weight for cone vertices
        if is_cone[i]:
            # Add increased weight to the cone position fit
            logger.info("Weighting cone vertices by %s",
                        optimization_params.cone_position_difference_factor)
            H_f[vi, vi] = optimization_params.cone_position_difference_factor

            # Add cone gradient fitting term
            logger.info("Weighting cone gradients by %s",
                        optimization_params.cone_vertex_gradient_difference_factor)
            g1i: int = generate_local_vertex_gradient_variable_index(i, 0, 0, 1)  # local first gradient index
            g2i: int = generate_local_vertex_gradient_variable_index(i, 1, 0, 1)  # local second gradient index
            H_f[g1i, g1i] = optimization_params.cone_vertex_gradient_difference_factor
            H_f[g2i, g2i] = optimization_params.cone_vertex_gradient_difference_factor
        # Weights for cone adjacent vertices (which can be collapsed to the cone)
        elif is_cone_adjacent[i]:
            # Add increased weight to the cone adjacent position fit
            logger.info(
                "Weighting cone adjacent vertices by %s",
                optimization_params.cone_adjacent_position_difference_factor)
            H_f[vi, vi] = optimization_params.cone_adjacent_position_difference_factor

            # Add cone adjacent vertex gradient fitting term
            logger.info("Weighting cone adjacent gradients by %s",
                        optimization_params.cone_adjacent_vertex_gradient_difference_factor)
            g1i = generate_local_vertex_gradient_variable_index(i, 0, 0, 1)  # local first gradient index
            g2i = generate_local_vertex_gradient_variable_index(i, 1, 0, 1)  # local second gradient index
            H_f[g1i, g1i] = optimization_params.cone_adjacent_vertex_gradient_difference_factor
            H_f[g2i, g2i] = optimization_params.cone_adjacent_vertex_gradient_difference_factor
        # Default fitting weight is 1.0
        else:
            H_f[vi, vi] = 1.0

    # Check for edges collapsing to a cone and add weight
    for i in range(3):
        vj: int = (i + 1) % 3  # next local vertex index
        vk: int = (i + 2) % 3  # prev local vertex index

        if ((is_cone_adjacent[vj]) and (is_cone_adjacent[vk])):
            # Add cone adjacent adjacent edge gradient fit
            logger.info(
                "Weighting cone edge gradients by %s",
                optimization_params.cone_adjacent_edge_gradient_difference_factor)
            gjk: int = generate_local_edge_gradient_variable_index(i, 0, 1)  # local first gradient index
            H_f[gjk, gjk] = optimization_params.cone_adjacent_edge_gradient_difference_factor

    return H_f


def build_planar_constraint_hessian(uv: np.ndarray,
                                    corner_to_corner_uv_positions: list[Matrix2x2r],
                                    reverse_edge_orientations: list[bool],
                                    normal: SpatialVector) -> np.ndarray:
    """
    Build the hessian for the planar normal constraint term for cone adjacent
    vertices
    WARNING: Unlike the other hessians, which are 12x12 matrices assembled per
    x,y,z coordinate and combined into a 36x3 block matrix, this Hessian has
    mixed coordinate terms and is thus directly 36x36

    shape returned = (36, 36)
    """
    # Asserting shape checking.
    assert uv.shape == (3, 2)
    assert len(corner_to_corner_uv_positions) == 3
    assert len(reverse_edge_orientations) == 3
    assert normal.shape == (1, 3)

    # Build planar hessian array for derived derivative quantities (shape = (36, 36)
    planarH: np.ndarray = planarHfun(normal[0], normal[1], normal[2])

    # Build C_gl matrix (shape = (12, 12))
    C_gl: np.ndarray = get_C_gl(uv, corner_to_corner_uv_positions, reverse_edge_orientations)

    # Make block diagonal C_gl matrix
    C_gl_diag: np.ndarray = np.ndarray(shape=(36, 36), dtype=float)
    C_gl_diag[0:12, 0:12] = C_gl
    # TODO: is the below slicing correct with ASOC code?
    C_gl_diag[12:24, 12:24] = C_gl
    C_gl_diag[24:36, 24:36] = C_gl

    # Build the planar constraint Hessian with indexing so that DoF per
    # coordinate are contiguous

    # Below shape (36, 36) = (36, 36).T @ (36, 36) @ (36, 36)
    H_p_permuted: np.ndarray = 0.5 * C_gl_diag.T @ planarH @ C_gl_diag

    # Reindex so that coordinates per DoF are contiguous
    # TODO: maybe below can be made with NumPy indexing??? Probably not.
    H_p: np.ndarray = np.zeros(shape=(36, 36))
    for ri in range(12):
        for rj in range(3):
            for ci in range(12):
                for cj in range(3):
                    H_p[3 * ri + rj, 3 * ci + cj] = H_p_permuted[12 * rj + ri, 12 * cj + ci]

    return H_p


def build_twelve_split_spline_energy_system(initial_V: np.ndarray,
                                            initial_face_normals: np.ndarray,
                                            affine_manifold: AffineManifold,
                                            optimization_params: OptimizationParameters) -> tuple[float, VectorX, np.ndarray, np.ndarray]:
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

    num_vertices: int = initial_V.shape[0]  # rows
    num_faces = affine_
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
