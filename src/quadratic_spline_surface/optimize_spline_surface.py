from src.core.common import *
from src.quadratic_spline_surface.PS12_patch_coeffs import *

from src.core.affine_manifold import AffineManifold, EdgeManifoldChart
from src.core.differentiable_variable import *
from src.core.halfedge import *
from src.core.halfedge import Halfedge
from src.core.polynomial_function import *

from src.quadratic_spline_surface.position_data import TriangleCornerData, TriangleMidpointData
from src.quadratic_spline_surface.powell_sabin_local_to_global_indexing import *
from src.quadratic_spline_surface.compute_local_twelve_split_hessian import *

from src.quadratic_spline_surface.planarH import planarHfun

import copy  # used for shifting array
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from cholespy import CholeskySolverD, MatrixType

from dataclasses import dataclass


@dataclass
class OptimizationParameters:
    """
    Parameters for surface spline optimization.
    """

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
        self.H_f: np.ndarray = np.zeros(shape=(12, 12), dtype=np.float64)  # fitting term hessian
        self.H_s: np.ndarray = np.zeros(shape=(12, 12), dtype=np.float64)  # smoothness term hessian
        self.H_p: np.ndarray = np.zeros(shape=(36, 36), dtype=np.float64)  # planarity term hessian
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
    # TODO This could be placed inside its class and serve as a constructor of some sort
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
    # DEPRECATED (according to ASOC code)
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
    """
    Assemble the local degree of freedom data.
    """
    # TODO: "This could be placed as a constructor inside the class itself...")

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
    """
    Compute the local twelve split energy quadratic from Hessian and local
    degree of freedom data
    TODO: finish docstring

    out: local_energy
    out: local_derivatives (shape 36, 1)
    out: local_hessian (shape 36, 36)
    """
    todo("So, there is probably some issue with the matrix multiplication here")

    local_derivatives: TwelveSplitGradient = np.ndarray(shape=(36, 1))
    local_hessian: TwelveSplitHessian = np.zeros(shape=(36, 36))

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
    for i in range(12):
        for j in range(12):
            for k in range(3):
                local_hessian[3 * i + k, 3 * j + k] = local_hessian_12x12[i, j]

    # Add 36x36 planar constraint term to the local hessian
    local_hessian += 2.0 * w_p * H_p

    # Build per coordinate gradients for smoothness and fit terms
    g_alpha: np.ndarray = 2 * (w_s * H_s) * r_alpha
    assert g_alpha.shape == (12, 3)
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
        # r_alpha shape sliced = (12, 1)
        smoothness_term += (w_s * H_s) * np.dot(r_alpha[:, [i]].T, r_alpha[:, [i]])[0, 0]

    logger.info("Smoothness term is %s", smoothness_term)

    # Add fit term
    fit_term: float = 0.0
    for i in range(3):
        # r_alpha_diff shape sliced = (12, 1)
        r_alpha_diff: np.ndarray = r_alpha[:, [i]] - r_alpha_0[:, [i]]  # gets columns
        fit_term += (w_f * H_f) * np.dot(r_alpha_diff.T, r_alpha_diff)[0, 0]
    logger.info("Fit term is %s", fit_term)

    # Add planar fitting term
    planar_term: float = 0.0
    planar_term += (w_p * H_p) * np.dot(r_alpha_flat.T, r_alpha_flat)
    logger.info("Planar orthogonality term is %s", planar_term)

    # Compute final energy
    local_energy: float = smoothness_term + fit_term + planar_term
    return local_energy, local_derivatives, local_hessian


def shift_array(arr_ref: list, shift: int) -> None:
    """
    Helper function to cyclically shift an array of three elements.

    NOTE: performs DEEP copy and then shifts. So good for basic datatypes but not something like list[np.ndarray] for performance reasons.

    TODO: check that this modifies by reference
    """
    arr_copy: list = copy.deepcopy(arr_ref)
    for i in range(3):
        arr_ref[i] = arr_copy[(i + shift) % 3]

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
    NOTE: should modify all parameters by reference.
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


def compute_twelve_split_energy_quadratic(
        vertex_positions: list[SpatialVector],
        vertex_gradients: list[SpatialVector],
        edge_gradients: list[list[SpatialVector]],  # list[SpatialVector] of length 3
        global_vertex_indices: list[int],
        global_edge_indices: list[list[int]],  # list[int] of length 3
        initial_vertex_positions: list[SpatialVector],
        initial_face_normals: np.ndarray,  # matrix
        manifold: AffineManifold,
        optimization_params: OptimizationParameters,
        num_variable_vertices: int,
        num_variable_edges: int
) -> tuple[float, Vector2D, csr_matrix, CholeskySolverD]:
    """
    Compute the energy system for a twelve-split spline.

    NEW BEHAVIOR: also calculates the hessian_inverse because that is needed.
    NOTE: derivatives shape (num_independent_variables, 1) because trying to be similar to TwelveSplitGradient shape of (36, 1)

    :param vertex_positions:         /
    :param vertex_gradients:         /
    :param edge_gradients:           /
    :param global_vertex_indices:    /
    :param global_edge_indices:      /
    :param initial_vertex_positions: /
    :param initial_face_normals:     /
    :param manifold:                 /
    :param optimization_params:      /
    :param num_variable_vertices:    /
    :param num_variable_edges:       /

    :return: (energy, derivatives, hessian, hessian_inverse)
    :rtype: tuple[float, Vector2D, csr_matrix, CholeskySolverD]
    """
    num_independent_variables: int = 9 * num_variable_vertices + 3 * num_variable_edges
    energy: float = 0.0
    derivatives: Vector2D = np.zeros(shape=(num_independent_variables, 1), dtype=np.float64)
    hessian_entries: list[tuple[float, float, float]] = []

    for face_index in range(manifold.num_faces):
        # Get face vertices
        F: np.ndarray = manifold.get_faces  # F has int dtype
        i: int = F[face_index, 0]
        j: int = F[face_index, 1]
        k: int = F[face_index, 2]

        # Bundle relevant global variables into per face local vectors (all list of length 3)
        initial_vertex_positions_T: list[SpatialVector] = build_face_variable_vector(
            initial_vertex_positions, i, j, k)
        vertex_positions_T: list[SpatialVector] = build_face_variable_vector(vertex_positions, i, j, k)
        edge_gradients_T: list[SpatialVector] = edge_gradients[face_index]
        vertex_gradients_T: list[Matrix2x3r] = build_face_variable_vector(vertex_gradients, i, j, k)

        # GeGet the global uv values for the face vertices
        face_vertex_uv_positions: list[PlanarPoint] = manifold.get_face_global_uv(face_index)  # length 3

        # Get corner uv positions for the given face corners.
        # NOTE: These may differ from the edge difference vectors computed from the global
        # uv by a rotation per vertex due to the local layouts performed at each vertex.
        # Since vertex gradients are defined in terms of these local vertex charts, we must
        # use these directions when computing edge direction gradients from the vertex uv
        # gradients.
        corner_to_corner_uv_positions: list[Matrix2x3r] = manifold.get_face_corner_charts(face_index)  # length 3

        # Get edge orientations
        # NOTE: The edge frame is oriented so that one basis vector points along the edge
        # counterclockwise and the other points perpendicular into the interior of the
        # triangle. If the given face is the bottom face in the edge chart, the sign of
        # the midpoint gradient needs to be reversed.
        reverse_edge_orientations: list[bool] = []  # length 3
        for i in range(3):
            chart: EdgeManifoldChart = manifold.get_edge_chart(face_index, i)
            reverse_edge_orientations.append(chart.top_face_index != face_index)

        # Mark cone vertices
        is_cone: list[bool] = []  # length 3
        for i in range(3):
            vi: int = F[face_index, i]
            is_cone.append(manifold.get_vertex_chart(vi).is_cone)

        # Mark cone adjacent vertices
        is_cone_adjacent: list[bool] = []  # length 3
        for i in range(3):
            vi = F[face_index, i]
            is_cone_adjacent.append(manifold.get_vertex_chart(vi).is_cone_adjacent)

        # Get global indices of the local vertex and edge DOFs
        face_global_vertex_indices: list[int] = build_face_variable_vector(global_vertex_indices, i, j, k)  # length 3
        face_global_edge_indices: list[int] = global_edge_indices[face_index]  # length 3

        # Check if an edge is collapsing and make sure any collapsing edges have
        # local vertex indices 0 and 1
        # WARNING: This is a somewhat fragile operation that must occur after all
        # of these arrays are build and before the local to global map is built
        # and is not necessary in the current framework used in the paper but is for
        # some deprecated experimental methods
        is_cone_adjacent_face: bool = False
        for i in range(3):
            if is_cone[(i + 2) % 3]:
                shift_local_energy_quadratic_vertices(vertex_positions_T,
                                                      vertex_gradients_T,
                                                      edge_gradients_T,
                                                      initial_vertex_positions_T,
                                                      face_vertex_uv_positions,
                                                      corner_to_corner_uv_positions,
                                                      reverse_edge_orientations,
                                                      is_cone,
                                                      is_cone_adjacent,
                                                      face_global_vertex_indices,
                                                      face_global_edge_indices,
                                                      i)
                is_cone_adjacent_face = True
                break

        # Get normal for the face
        normal: SpatialVector = np.zeros(shape=(1, 3))
        if is_cone_adjacent_face:
            normal = initial_face_normals[[face_index], :]
            logger.info("Weighting by normal %s", normal.T)

        # Get local to global map
        local_to_global_map: list[int] = generate_twelve_split_local_to_global_map(
            face_global_vertex_indices,
            face_global_edge_indices,
            num_variable_vertices)  # length = 36

        # Compute local hessian data
        local_hessian_data: LocalHessianData = generate_local_hessian_data(
            face_vertex_uv_positions,
            corner_to_corner_uv_positions,
            reverse_edge_orientations,
            is_cone,
            is_cone_adjacent,
            normal,
            optimization_params)

        # Compute local degree of freedom data
        local_dof_data: LocalDOFData = generate_local_dof_data(
            initial_vertex_positions_T,
            vertex_positions_T,
            vertex_gradients_T,
            edge_gradients_T)

        # Compute the local energy quadratic system for the face
        local_energy: float
        local_derivatives: TwelveSplitGradient
        local_hessian: TwelveSplitHessian
        local_energy, local_derivatives, local_hessian = compute_local_twelve_split_energy_quadratic(
            local_hessian_data,
            local_dof_data
        )

        # Update the energy quadratic with the new face energy
        # NOTE: update_energy_quadratic is only used here.
        energy = update_energy_quadratic(local_energy,
                                         local_derivatives,
                                         local_hessian,
                                         local_to_global_map,
                                         energy,
                                         derivatives,
                                         hessian_entries)
    # Set hessian from the triplets
    # https://stackoverflow.com/questions/65126682/create-sparse-matrix-from-list-of-tuples
    # TODO: might be more efficient to just modify hessian by reference and resize when needed... which then overrides the elements in hessian as we setFromTriplets

    global_indices_i: tuple[float]  # equivalent to rows
    global_indices_j: tuple[float]  # equivalent to cols
    hessian_value: tuple[float]  # equivalent to data
    global_indices_i, global_indices_j, hessian_value = zip(*hessian_entries)

    rows: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(global_indices_i, dtype=np.float64)
    cols: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(global_indices_j, dtype=np.float64)
    data: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(hessian_value, dtype=np.float64)

    hessian: csr_matrix = csr_matrix((data, (rows, cols)),
                                     shape=(num_independent_variables, num_independent_variables),
                                     dtype=float)
    num_rows: int = hessian.get_shape()[ROWS]

    # TODO: I don't know the shape of derivatives...
    assert derivatives.shape == (num_independent_variables, 1)

    # Build the inverse... kind of.
    # Just have the solver available is all.
    # TODO: This is very finicky with CSR sparse matrices...
    # NOTE: -1 to num_rows or else it acts up with CSR matrices.
    hessian_inverse: CholeskySolverD = CholeskySolverD(num_rows - 1,
                                                       rows,
                                                       cols,
                                                       data,
                                                       MatrixType.CSR)

    return energy, derivatives, hessian, hessian_inverse


# ******************************************
# methods translated from .h are below
# ******************************************

def build_local_fit_hessian(is_cone: list[bool], is_cone_adjacent: list[bool], optimization_params: OptimizationParameters) -> np.ndarray:
    """
    Build the hessian for the local fitting energy
    This is a diagonal matrix with special weights for cones and cone adjacent
    vertices.
    Shape returned = (12, 12)
    """
    H_f: np.ndarray = np.zeros(shape=(12, 12), dtype=np.float64)

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
    assert planarH.shape == (36, 36)

    # Build C_gl matrix (shape = (12, 12))
    C_gl: np.ndarray = get_C_gl(uv, corner_to_corner_uv_positions, reverse_edge_orientations)
    assert C_gl.shape == (12, 12)

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
                                            optimization_params: OptimizationParameters
                                            ) -> tuple[float, VectorX, csr_matrix, CholeskySolverD]:
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

    num_vertices: int = initial_V.shape[ROWS]
    num_faces: int = affine_manifold.num_faces

    # Build halfedge
    he_to_corner: list[tuple[Index, Index]] = affine_manifold.get_he_to_corner
    halfedge: Halfedge = affine_manifold.get_halfedge
    num_edges: int = halfedge.num_edges

    # Assume all vertices and edges are variable
    variable_vertices: list[int] = index_vector_complement([], num_vertices)
    variable_edges: list[int] = index_vector_complement([], num_edges)

    # Get variable counts
    num_variable_vertices: int = len(variable_vertices)
    num_variable_edges: int = len(variable_edges)

    # Initialize variables to optimize
    vertex_positions: list[SpatialVector] = []
    initial_vertex_positions: list[SpatialVector] = []
    for i in range(num_vertices):
        assert initial_V[[i], :].shape == (1, 3)
        vertex_positions.append(initial_V[[i], :])  # shape (1, 3) for SpatialVectors
        initial_vertex_positions.append(initial_V[[i], :])

    vertex_gradients: list[Matrix2x3r] = generate_zero_vertex_gradients(num_vertices)
    edge_gradients: list[list[SpatialVector]] = generate_zero_edge_gradients(num_faces)
    # NOTE: assertion == 3 below sicne ASOC code originally has array<int, 3> as elements of edge_gradients
    assert len(edge_gradients[0]) == 3

    # Build edge variable indices
    global_vertex_indices: list[int] = build_variable_vertex_indices_map(num_vertices, variable_vertices)

    # Build edge variable indices
    global_edge_indices: list[list[int]] = build_variable_edge_indices_map(
        num_faces, variable_edges, halfedge, he_to_corner)
    assert len(global_edge_indices[0]) == 3

    # Build energy for the affine manifold (all below)
    energy: float
    derivatives: Vector2D  # shape (n, 1)
    hessian: csr_matrix
    hessian_inverse: CholeskySolverD

    # Build the inverse as well...
    energy, derivatives, hessian, hessian_inverse = compute_twelve_split_energy_quadratic(
        vertex_positions,
        vertex_gradients,
        edge_gradients,
        global_vertex_indices,
        global_edge_indices,
        initial_vertex_positions,
        initial_face_normals,
        affine_manifold,
        optimization_params,
        num_variable_vertices,
        num_variable_edges)

    return energy, derivatives, hessian, hessian_inverse


def optimize_twelve_split_spline_surface(
        initial_V: MatrixNx3,
        affine_manifold: AffineManifold,
        halfedge: Halfedge,
        he_to_corner: list[tuple[Index, Index]],
        variable_vertices: list[int],
        variable_edges: list[int],
        fit_matrix: csr_matrix,
        hessian_inverse: CholeskySolverD
) -> tuple[MatrixNx3, list[Matrix2x3r], list[list[SpatialVector]]]:
    """
    Helper function for generate_optimized_twelve_split_position_data()

    NOTE: Returns new optimized V, vertex_gradients, and edge_gradients rather than modifying 
    parameters by reference since it is only called in generate_optimized_twelve_split_position_data() 
    and is only used internally within that method.

    :param initial_V: vertices, probably with shape (n, 3)
    :param affine_manifold: affine manifold.
    :param halfedge: halfedge
    :param he_to_corner: halfedge to corner index
    :param variable_vertices: variable vertices
    :param variable_edges: variable edges
    :param fit_matrix: fit matrix
    :param hessian_inverse: Cholespy package Cholesky solver

    :return: (optimized_V, optimized_vertex_gradients, optimized_edge_gradients) with optimized_spatial_vector with list[SpatialVector] of length 3
    :rtype: tuple[MatrixNx3, list[Matrix2x3r], list[list[SpatialVector]]]
    """
    # Get variable coutns
    assert initial_V.shape[COLS] == 3
    num_vertices: int = initial_V.shape[ROWS]
    num_faces: int = affine_manifold.num_faces

    # Initialize variables to optimize
    vertex_positions: list[SpatialVector] = []
    initial_vertex_positions: list[SpatialVector] = []
    for i in range(num_vertices):
        assert initial_V[[i], :].shape == (1, 3)
        vertex_positions.append(initial_V[[i], :])
        initial_vertex_positions.append(initial_V[[i], :])
    assert len(vertex_positions) == num_vertices
    assert len(initial_vertex_positions) == num_vertices
    vertex_gradients: list[Matrix2x3r] = generate_zero_vertex_gradients(num_vertices)
    edge_gradients: list[list[SpatialVector]] = generate_zero_edge_gradients(num_faces)
    assert len(edge_gradients[0]) == 3

    # Build variable values gradient as H
    initial_variable_values: Vector1D = generate_twelve_split_variable_value_vector(vertex_positions,
                                                                                    vertex_gradients,
                                                                                    edge_gradients,
                                                                                    variable_vertices,
                                                                                    variable_edges,
                                                                                    halfedge,
                                                                                    he_to_corner)
    assert initial_variable_values.ndim == 1
    logger.info("Initial variable value vector:\n%s", initial_variable_values)

    # Solve hessian system to get optimized values
    # TODO: double check that behavior below is like that of Eigen
    right_hand_side: Vector1D = fit_matrix * initial_variable_values
    optimized_variable_values: Vector1D = np.ndarray(shape=right_hand_side.shape)
    hessian_inverse.solve(right_hand_side, optimized_variable_values)

    # Update variables
    # NOTE: Below are simply references to the original lists, but renamed for readability.
    optimized_vertex_positions_ref: list[SpatialVector] = vertex_positions
    optimized_vertex_gradients_ref: list[Matrix2x3r] = vertex_gradients
    optimized_edge_gradients_ref: list[list[SpatialVector]] = edge_gradients
    update_position_variables(
        optimized_variable_values, variable_vertices, optimized_vertex_positions_ref)
    update_vertex_gradient_variables(
        optimized_variable_values, variable_vertices, optimized_vertex_gradients_ref)
    update_edge_gradient_variables(optimized_variable_values,
                                   variable_vertices,
                                   variable_edges,
                                   halfedge,
                                   he_to_corner,
                                   optimized_edge_gradients_ref)

    # Copy variable values to constants... which gets overridden anyways.
    optimized_V: MatrixNx3 = np.ndarray(shape=(num_vertices, 3))
    for i in range(num_vertices):
        # Flattens so that broadcasting works.
        optimized_V[i, :] = optimized_vertex_positions_ref[i].flatten()

    return optimized_V, optimized_vertex_gradients_ref, optimized_edge_gradients_ref


def generate_optimized_twelve_split_position_data(V: np.ndarray,  # matrix
                                                  affine_manifold: AffineManifold,
                                                  fit_matrix: csr_matrix,
                                                  hessian_inverse: CholeskySolverD,
                                                  corner_data_ref: list[list[TriangleCornerData]],
                                                  midpoint_data_ref: list[list[TriangleMidpointData]]
                                                  ) -> None:
    """
    Compute the optimal per triangle position data for given vertex positions.
    NOTE: used by twelve_split_spline.py
    NOTE: used by update_positions() in TwelveSplitSplineSurface, which is reliant on corner_data and midpoint_data being modified by reference rather than returned as new lists.

    @param[in] V: vertex positions
    @param[in] affine_manifold: mesh topology and affine manifold structure
    @param[in] fit_matrix: quadratic fit energy Hessian matrix
    @param[in] hessian_inverse: solver for inverting the energy Hessian

    @param[out] corner_data: quadratic vertex position and derivative data with list[TriangleCornerData] of length 3
    @param[out] midpoint_data: quadratic edge midpoint derivative data with list[TriangleMidpointData] of length 3
    """
    assert V.ndim == 2
    num_vertices: int = V.shape[ROWS]

    # Build halfedge
    he_to_corner: list[tuple[int, int]] = affine_manifold.get_he_to_corner
    halfedge: Halfedge = affine_manifold.get_halfedge
    num_edges: int = halfedge.num_edges

    # Assume all vertices and edges are variable
    fixed_vertices: list[int] = []
    fixed_edges: list[int] = []
    variable_vertices: list[int] = index_vector_complement(fixed_vertices, num_vertices)
    variable_edges: list[int] = index_vector_complement(fixed_edges, num_edges)

    # Run optimization
    optimized_V: np.ndarray  # matrix
    optimized_vertex_gradients: list[Matrix2x3r]
    optimized_reduced_edge_gradients: list[list[SpatialVector]]  # list[SpatialVector] of length 3
    optimized_V, optimized_vertex_gradients, optimized_reduced_edge_gradients = optimize_twelve_split_spline_surface(
        V,
        affine_manifold,
        halfedge,
        he_to_corner,
        variable_vertices,
        variable_edges,
        fit_matrix,
        hessian_inverse)
    assert len(optimized_reduced_edge_gradients[0]) == 3

    # Build corner position data from the optimized gradients
    generate_affine_manifold_corner_data(optimized_V,
                                         affine_manifold,
                                         optimized_vertex_gradients,
                                         corner_data_ref)

    # Build the full edge gradients with first gradient determined by the corner position data
    optimized_edge_gradients: list[list[Matrix2x3r]] = convert_reduced_edge_gradients_to_full(
        optimized_reduced_edge_gradients,
        corner_data_ref,
        affine_manifold)  # list[Matrix2x3r] of length 3
    assert len(optimized_edge_gradients[0]) == 3

    # Build midpoint position data from the optimized gradients
    generate_affine_manifold_midpoint_data(
        affine_manifold,
        optimized_edge_gradients,
        midpoint_data_ref)

    # Making sure list with sublists of length 3
    assert len(corner_data_ref[0]) == 3
    assert len(midpoint_data_ref[0]) == 3


def generate_zero_vertex_gradients(num_vertices: int) -> list[Matrix2x3r]:
    """
    Generate zero value gradients for a given number of vertices.
    Helper function.

    @param[in] num_vertices: number of vertices |V|
    @param[out] gradients: |V| trivial vertex gradient matrices
    """
    gradients: list[Matrix2x3r] = []

    # Set the zero gradient for each vertex
    for _ in range(num_vertices):
        gradients.append(np.zeros(shape=(2, 3)))

    return gradients


def generate_zero_edge_gradients(num_faces: int) -> list[list[SpatialVector]]:
    """
    Generate zero value gradients for a given number of halfedges.
    Helper function.

    @param[in] num_faces: number of faces |F|
    @param[out] gradients: 3|F| trivial edge gradient matrices
    """

    # Set the zero gradient for each vertex
    edge_gradients: list[list[SpatialVector]] = []  # list of list of 3 SpatialVector elements

    for _ in range(num_faces):
        edge_gradients.append([np.zeros(shape=(1, 3)),
                               np.zeros(shape=(1, 3)),
                               np.zeros(shape=(1, 3))])
    return edge_gradients


def convert_full_edge_gradients_to_reduced(edge_gradients: list[list[Matrix2x3r]]) -> None:
    """
    Given edge and opposite corner direction gradients at triangle edge midpoints,
    extract just the opposite corner direction gradient
    @param[in] edge_gradients: edge and corner directed gradients per edge midpoints
    @param[out] reduced_edge_gradients: opposite corner directed gradients per edge midpoints
    :type reduced_edge_gradients: list[list[SpatialVector]]
    """
    unimplemented("Method is not used anywhere in original ASOC code.")


def convert_reduced_edge_gradients_to_full(reduced_edge_gradients: list[list[SpatialVector]],
                                           corner_data: list[list[TriangleCornerData]],
                                           affine_manifold: AffineManifold
                                           ) -> list[list[Matrix2x3r]]:
    """
    Given edge direction gradients at triangle edge midpoints, append the gradients in the
    direction of the opposite triangle corners, which are determined by gradients and
    position data at the corners.

    NOTE: returns edge_gradients rather than modifying it by reference since this method is only used internally within generate_optimized_twelve_split_position_data() and does not need to modify by reference.

    @param[in] reduced_edge_gradients: opposite corner directed gradients per edge midpoints with list[SpatialVector] of length 3
    @param[in] corner_data: quadratic vertex position and derivative data with list[TriangleCornerData] of length 3
    @param[in] affine_manifold: mesh topology and affine manifold structure
    @param[out] edge_gradients: edge and corner directed gradients per edge midpoints with list[Matrix2x3r] of length 3
    """
    F: np.ndarray = affine_manifold.get_faces
    num_faces: int = len(reduced_edge_gradients)

    # Compute the first gradient and copy the second for each edge
    edge_gradients: list[list[Matrix2x3r]] = []

    for i in range(num_faces):
        edge_gradients.append([np.ndarray(shape=(2, 3), dtype=np.float64),
                               np.ndarray(shape=(2, 3), dtype=np.float64),
                               np.ndarray(shape=(2, 3), dtype=np.float64)])
        for j in range(3):
            chart: EdgeManifoldChart = affine_manifold.get_edge_chart(i, j)
            f_top: int = chart.top_face_index
            if f_top != i:
                continue  # Only process top faces of edge charts to prevent redundancy

            # Get midpoint position and derivative along the edge
            # TODO: maybe rename midpoint to "_" to show that it is not used?
            midpoint: SpatialVector
            midpoint_edge_gradient: SpatialVector
            midpoint, midpoint_edge_gradient = compute_edge_midpoint_with_gradient(
                corner_data[i][(j + 1) % 3],
                corner_data[i][(j + 2) % 3])

            # Copy the gradients
            edge_gradients[i][j][0, :] = midpoint_edge_gradient.flatten()  # row(0)
            edge_gradients[i][j][1, :] = reduced_edge_gradients[i][j].flatten()  # row(1)

            # If the edge isn't on the boundary, set the other face corner corresponding to it
            if not chart.is_boundary:
                f_bottom: int = chart.bottom_face_index
                v_bottom: int = chart.bottom_vertex_index
                j_bottom: int = find_face_vertex_index(F[f_bottom, :].flatten(), v_bottom)
                edge_gradients[f_bottom][j_bottom] = edge_gradients[i][j]

    return edge_gradients
