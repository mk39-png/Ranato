"""
position_data.py

Methods to generate a per triangle local degrees of freedom from the global
degrees of freedom
"""

from src.core.affine_manifold import *
from src.core.common import *
from src.core.line_segment import *
from src.core.polynomial_function import *
from polyscope import polyscope, surface_mesh
from src.core.rational_function import *
from src.core.vertex_circulator import *

import numpy as np

# NOTE: VectorX also encompasses SpatialVectors as well.
VectorX = np.ndarray


@dataclass
class TriangleCornerData:
    """
    Position and derivative data at the corner of a triangle.
    """

    def __init__(self,
                 input_function_value: VectorX,
                 input_first_edge_derivative: VectorX,
                 input_second_edge_derivative: VectorX) -> None:
        """
        Default constructor

        :param function_value: position value vector at the corner
        :param first_edge_derivative: derivative in the counter-clockwise edge direction
        :param second_edge_derivative: derivative in the clockwise edge direction
        """
        assert input_function_value.shape[0] == 1
        assert input_first_edge_derivative.shape[0] == 1
        assert input_second_edge_derivative.shape[0] == 1

        self.function_value: VectorX = input_function_value
        self.first_edge_derivative: VectorX = input_first_edge_derivative
        self.second_edge_derivative: VectorX = input_second_edge_derivative


@dataclass
class TriangleMidpointData:
    """
    Derivative data at the midpoint of a triangle
    """

    def __init__(self, input_normal_derivative: VectorX) -> None:
        """
        Default constructor.

        :param input_normal_derivative: derivative in the direction of the opposite corner
        """
        assert input_normal_derivative.shape[0] == 1

        # derivative in the direction of the opposite corner
        self.normal_derivative: VectorX = input_normal_derivative


def view_triangle_corner_data():
    unimplemented()


def generate_affine_manifold_chart_corner_data(V: np.ndarray, F: np.ndarray,
                                               chart: VertexManifoldChart,
                                               gradient: Matrix2x3r) -> list[list[TriangleCornerData, TriangleCornerData, TriangleCornerData]]:
    """
    Helper function to add corner data for a given chart
    """
    corner_data: list[list[TriangleCornerData]] = []

    #  Build position data for the corners adjacent to the given vertex
    for face_index, f in enumerate(chart.face_one_ring):
        # Get the face and index of the vertex in it
        # NOTE: find_face_vertex_index expects ndarray ndim = 1 where shape = (n, )
        face_vertex_index: int = find_face_vertex_index(
            F[f, :], chart.vertex_index)

        # Compute position and edge derivatives from layout positions
        pi: SpatialVector = V[[chart.vertex_index], :]
        assert pi.shape == (1, 3)
        # TODO: is OneFormXr supposed to be ndim == 1 or what?
        # FIXME: check if OneFormXr is supposed to be ndim == 1
        uj: OneFormXr = chart.one_ring_uv_positions[face_index]
        uk: OneFormXr = chart.one_ring_uv_positions[face_index + 1]

        # shapes: (1, 3) @ (2, 3) = (1, 3)... which doesn't seem right
        # TODO: fix shaping
        todo("Fix shaping")
        dij: SpatialVector = uj @ gradient
        dik: SpatialVector = uk @ gradient
        logger.info("Vertex position: %s", pi)
        logger.info("Vertex gradient:\n%s", gradient)
        logger.info("First edge layout: %s", uj)
        logger.info("Second edge layout: %s", uk)
        logger.info("First edge derivative: %s", dij)
        logger.info("Second edge derivative: %s", dik)

        todo("finish implementation")
        # corner[]


def generate_affine_manifold_corner_data(V: np.ndarray,
                                         affine_manifold: AffineManifold,
                                         gradients: list[Matrix2x3r]
                                         ) -> list[tuple[TriangleCornerData, TriangleCornerData, TriangleCornerData]]:
    """
    Generate corner position data for a mesh with an affine manifold structure and per-vertex position and gradients.

    :param V: mesh vertex embedding
    :type V: np.ndarray

    :param affine_manifold: mesh topology and affine manifold structure
    :type affine_manifold: AffineManifold

    :param gradients: per vertex uv gradients in the local charts
    :type gradients: list[Matrix2x3r]

    :return: quadratic vertex position and derivative data for each triangle corner
    :rtype: list[tuple[TriangleCornerData, TriangleCornerData, TriangleCornerData]]
    """
    corner_data: list[tuple[TriangleCornerData,
                            TriangleCornerData, TriangleCornerData]]
    raise NotImplementedError()
    # Used in optimize_spline_surface.cpp


def generate_affine_manifold_midpoint_data(affine_manifold: AffineManifold,
                                           edge_gradients: list[tuple[Matrix2x3r, Matrix2x3r, Matrix2x3r]],
                                           midpoint_data: list[tuple[TriangleMidpointData, TriangleMidpointData, TriangleMidpointData]]) -> list[tuple[TriangleMidpointData, TriangleMidpointData, TriangleMidpointData]]:
    """
    Generate midpoint position data for a mesh with an affine manifold structure and
    per edge gradients.

    :param affine_manifold: mesh topology and affine manifold structure
    :type affine_manifold: AffineManifold

    :param edge_gradients: per edge gradients in the local charts
    :type edge_gradients: list[tuple[Matrix2x3r, Matrix2x3r, Matrix2x3r]]

    :return: quadratic edge midpoint derivative data
    :rtype: list[tuple[TriangleMidpointData, TriangleMidpointData, TriangleMidpointData]]
    """
    # Used in optimize_spline_surface.cpp
    midpoint_data: list[tuple[TriangleMidpointData,
                              TriangleMidpointData, TriangleMidpointData]]
    todo()
    return midpoint_data


def compute_edge_midpoint_with_gradient(edge_origin_corner_data: TriangleCornerData,
                                        edge_dest_corner_data: TriangleCornerData) -> tuple[SpatialVector, SpatialVector]:
    """
    Given corner data for the endpoints of the edge, compute the midpoint
    and the edge aligned midpoint gradient of the corresponding Powell-Sabin
    quadratic spline patch

    :param edge_origin_corner_data: corner data at the origin corner of the
        oriented edge
    :type edge_origin_corner_data: TriangleCornerData

    :param edge_dest_corner_data: corner data at the destination corner of
        the oriented edge
    :type edge_dest_corner_data: TriangleCornerData

    :return:
        - midpoint: quadratic edge function midpoint

        - midpoint_edge_gradient: edge aligned gradient
    :rtype: tuple[SpatialVector, SpatialVector]
    """
    # Used in optimize_spline_surface.cpp

    fi: SpatialVector = edge_origin_corner_data.function_value
    fj: SpatialVector = edge_dest_corner_data.function_value
    dij: SpatialVector = edge_origin_corner_data.first_edge_derivative
    dji: SpatialVector = edge_dest_corner_data.second_edge_derivative
    midpoint = 0.5 * (fi + fj) + 0.125 * (dij + dji)
    midpoint_edge_gradient = 2.0 * (fj - fi) + 0.5 * (dji - dij)
    assert midpoint.shape == (1, 3)
    assert midpoint_edge_gradient.shape == (1, 3)

    return midpoint, midpoint_edge_gradient


def generate_corner_data_matrices(corner_data: list[tuple[TriangleCornerData, TriangleCornerData, TriangleCornerData]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given per corner position data, rearrange it into matrices with row i
    corresponding to the data for vertex i.

    :param corner_data: quadratic vertex position and derivative data
    :return:
        - position_matrix: matrix with position data as rows
        - first_derivative_matrix: matrix with first derivative data as rows
        - second_derivative_matrix: matrix with second derivative data as rows
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # function Used in twelve_split_spline.cpp
    num_faces: int = len(corner_data)

    # Corner data matrices
    position_matrix: np.ndarray = np.ndarray(shape=(3 * num_faces, 3))
    first_derivative_matrix: np.ndarray = np.ndarray(shape=(3 * num_faces, 3))
    second_derivative_matrix: np.ndarray = np.ndarray(shape=(3 * num_faces, 3))

    # Organize position data into matrices
    # TODO: could try to use NumPy indexing magic
    for i in range(num_faces):
        for j in range(3):
            # TODO: is slicing done correctly?
            # TODO: may be error with Vectors not necessarily being vectors...
            position_matrix[3 * i + j] = corner_data[i][j].function_value
            first_derivative_matrix[3 * i +
                                    j] = corner_data[i][j].first_edge_derivative
            second_derivative_matrix[3 * i +
                                     j] = corner_data[i][j].second_edge_derivative

    return position_matrix, first_derivative_matrix, second_derivative_matrix


def generate_midpoint_data_matrices(corner_data: list[tuple[TriangleCornerData, TriangleCornerData, TriangleCornerData]],
                                    midpoint_data: list[tuple[TriangleMidpointData, TriangleMidpointData, TriangleMidpointData]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given position data, rearrange the edge midpoint data into matrices with row i
    corresponding to the data for edge i.

    Note that the normal derivative matrix is extracted directly from the midpoint data
    and the position and tangent derivative matrices are inferred from the corner data.

    :param corner_data: quadratic vertex position and derivative data
    :type corner_data: ist[tuple[TriangleCornerData, TriangleCornerData, TriangleCornerData]]

    :param midpoint_data: quadratic edge midpoint derivative data
    :type midpoint_data: list[tuple[TriangleMidpointData, TriangleMidpointData, TriangleMidpointData]]

    :return:
        - position_matrix: matrix with midpoint position data as rows
        - tangent_derivative_matrix: matrix with edge tangent derivative data as rows
        - normal_derivative_matrix: matrix with normal derivative data as rows
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # function Used in twelve_split_spline.cpp
    num_faces: int = len(corner_data)
    position_matrix: np.ndarray = np.ndarray(shape=(3 * num_faces, 3))
    tangent_derivative_matrix: np.ndarray = np.ndarray(
        shape=(3 * num_faces, 3))
    normal_derivative_matrix: np.ndarray = np.ndarray(shape=(3 * num_faces, 3))

    # Organize position data into matrices
    # TODO: could try to use NumPy indexing magic
    for i in range(num_faces):
        for j in range(3):
            # TODO: is slicing done correctly?
            # TODO: may be error with Vectors not necessarily being vectors...
            midpoint, midpoint_edge_gradient = compute_edge_midpoint_with_gradient(
                corner_data[i][(j + 1) % 3], corner_data[i][(j + 2) % 3])
            position_matrix[3 * i + j, :] = midpoint
            tangent_derivative_matrix[3 * i + j, :] = midpoint_edge_gradient
            normal_derivative_matrix[3 * i + j,
                                     :] = midpoint_data[i][j].normal_derivative

    return position_matrix, tangent_derivative_matrix, normal_derivative_matrix
