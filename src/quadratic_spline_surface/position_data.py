"""
position_data.py

Methods to generate a per triangle local degrees of freedom from the global
degrees of freedom

NOTE: a lot of the methods in this file modify by reference.
"""

from src.core.affine_manifold import *
from src.core.common import *
from src.core.line_segment import *
from src.core.polynomial_function import *
from polyscope import polyscope, surface_mesh
from src.core.rational_function import *
from src.core.vertex_circulator import *

import numpy as np


@dataclass
class TriangleCornerData:
    """
    Position and derivative data at the corner of a triangle.
    """

    def __init__(self,
                 input_function_value: VectorX | None = None,
                 input_first_edge_derivative: VectorX | None = None,
                 input_second_edge_derivative: VectorX | None = None) -> None:
        """
        Default constructor
        NOTE: made arguments optional to have some sort of default constructor.
        Because there is behavior in position_data.py with midpoint_data that I do not completely understand and that seems to not create a new midpoint_data but rather modifies the original. Hence, this having default values so that TriangleCornerData() can be called and to implement the code as is.

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

    def __init__(self, input_normal_derivative: VectorX | None = None) -> None:
        """
        Default constructor.
        NOTE: made arguments optional to have some sort of default constructor.

        :param input_normal_derivative: derivative in the direction of the opposite corner
        """
        assert input_normal_derivative.shape[0] == 1

        # derivative in the direction of the opposite corner
        self.normal_derivative: VectorX = input_normal_derivative


def view_triangle_corner_data() -> None:
    """
    This method is not called anywhere in the original ASOC code.
    """
    unimplemented()


def generate_affine_manifold_chart_corner_data(V: np.ndarray,
                                               F: np.ndarray,
                                               chart: VertexManifoldChart,
                                               gradient: Matrix2x3r,
                                               corner_data_ref: list[list[TriangleCornerData]]) -> None:
    """
    Helper function to add corner data for a given chart.
    Used by genenerate_affine_manifold_corner_data().

    NOTE: Modifies corner_data by reference since this method is indirectly used in update_positions() in twelve_split_spline.py to update m_corner_data of the TwelveSplitSplineSurface class.
    Which means that since this updating the pre-existing corner_data, we do not want to overwrite all of pre-existing data inside corner_data by creating a new list[list[TriangleCornerData]]

    :param V: vertices
    :type V: np.ndarray of float64
    :param F: faces
    :type F: np.ndarray of int64
    :param chart: VertexManifoldChart object
    :type chart: VertexManifoldChart
    :param gradient: matrix of the gradient
    :type gradient: Matrix2x3r
    :param corner_data_ref: (out) corner data for given chart with list[TriangleCornerData] of length 3.
    :type corner_data_ref: list[list[TriangleCornerData]]
    """

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
        # FIXME: check if OneFormXr is supposed to be ndim == 1... probably not
        uj: OneFormXr = chart.one_ring_uv_positions[[face_index], :]
        uk: OneFormXr = chart.one_ring_uv_positions[[face_index + 1], :]

        # Presumably shaped (1, 2) since accessing UV position (which are typically (1, 2))
        assert uj.shape == (1, 2)
        assert uk.shape == (1, 2)

        # shapes: (1, 2) @ (2, 3) = (1, 3)
        dij: SpatialVector = uj @ gradient
        dik: SpatialVector = uk @ gradient
        assert dij.shape == (1, 3)
        assert dik.shape == (1, 3)

        logger.info("Vertex position: %s", pi)
        logger.info("Vertex gradient:\n%s", gradient)
        logger.info("First edge layout: %s", uj)
        logger.info("Second edge layout: %s", uk)
        logger.info("First edge derivative: %s", dij)
        logger.info("Second edge derivative: %s", dik)

        # Build position data
        corner_data_ref[f][face_vertex_index] = TriangleCornerData(pi, dij, dik)

    # Quickly checking that the element list is of size 3.
    assert len(corner_data_ref[-1]) == 3


def generate_affine_manifold_corner_data(V: np.ndarray,
                                         affine_manifold: AffineManifold,
                                         gradients: list[Matrix2x3r],
                                         corner_data_ref: list[list[TriangleCornerData]]) -> None:
    """
    Generate corner position data for a mesh with an affine manifold structure and per-vertex position and gradients.
    NOTE: Used in optimize_spline_surface.py.
    NOTE: must modify corner_data_ref by reference since this method is indirectly called in update_positions() via generate_optimized_twelve_split_position_data().

    :param V: mesh vertex embedding
    :type V: np.ndarray
    :param affine_manifold: mesh topology and affine manifold structure
    :type affine_manifold: AffineManifold
    :param gradients: per vertex uv gradients in the local charts
    :type gradients: list[Matrix2x3r]
    :param corner_data_ref: (out) quadratic vertex position and derivative data for each triangle corner with list[TriangleCornerData] of length 3.
    :type corner_data_ref: list[list[TriangleCornerData]]
    """
    # Resize the gradient and position data
    list_resize(corner_data_ref, affine_manifold.num_faces, [
                TriangleCornerData(), TriangleCornerData(), TriangleCornerData()])

    # Compute the gradient and position data per vertesx
    for i in range(V.shape[ROWS]):  # equivalent to Eigen V.rows()
        generate_affine_manifold_chart_corner_data(
            V,
            affine_manifold.get_faces,
            affine_manifold.get_vertex_chart(i),
            gradients[i],
            corner_data_ref)


def generate_affine_manifold_midpoint_data(affine_manifold: AffineManifold,
                                           edge_gradients: list[list[Matrix2x3r]],
                                           midpoint_data_ref: list[list[TriangleMidpointData]]) -> None:
    """
    Generate midpoint position data for a mesh with an affine manifold structure and
    per edge gradients.
    NOTE: modifies midpoint_data by reference
    NOTE: Used in optimize_spline_surface.cpp
    NOTE: need this method to modify midpoint_data by reference since it is used in generate_optimized_twelve_split_position_data(), which is then used in TwelveSplitSplineSurface.update_positions(). update_positions() is then called multiple times to update midpoint_data_ref, which means that we must modify the original midpoint_data rather than create a new one.

    :param affine_manifold: mesh topology and affine manifold structure
    :type affine_manifold: AffineManifold

    :param edge_gradients: per edge gradients in the local charts with list[Matrix2x3r] of length 3
    :type edge_gradients: list[list[Matrix2x3r]]

    :return: quadratic edge midpoint derivative data with list[TriangleMidpointData] of length 3
    :rtype: list[list[TriangleMidpointData]]
    """
    F: np.ndarray = affine_manifold.get_faces

    # Set midpoint data per face corner
    list_resize(midpoint_data_ref, affine_manifold.num_faces, [
                TriangleMidpointData(), TriangleMidpointData(), TriangleMidpointData()])
    assert len(edge_gradients[0]) == 3
    assert len(midpoint_data_ref[0]) == 3

    for i in range(affine_manifold.num_faces):
        for j in range(3):
            # Get local edge chart
            chart: EdgeManifoldChart = affine_manifold.get_edge_chart(i, j)

            # Get the corner index for the top face
            f_top: int = chart.top_face_index
            v_top: int = chart.top_vertex_index
            j_top: int = find_face_vertex_index(F[f_top, :], v_top)

            # Only process top faces of edge charts to prevent redundancy
            if (f_top != i):
                continue
            assert j_top == j

            # Compute midpoint to opposite corner derivative for the top face
            uv_top: PlanarPoint = chart.top_vertex_uv_position
            assert uv_top.shape == (1, 2)
            # uv_top @ edge_gradients = (1, 2) @ (2, 3) == (1, 3)
            midpoint_data_ref[f_top][j_top].normal_derivative = uv_top @ edge_gradients[i][j]
            assert midpoint_data_ref[f_top][j_top].normal_derivative.shape == (1, 3)
            logger.info("Midpoint data for corner (%s, %s) is %s = %s\n%s", f_top, j_top,
                        midpoint_data_ref[f_top][j_top].normal_derivative.T,
                        uv_top,
                        edge_gradients[i][j])

            # Only set the bottom vertex if the edge is not on the boundary
            if not chart.is_boundary:
                # Get the corner index for the bottom face
                f_bottom: int = chart.bottom_face_index
                v_bottom: int = chart.bottom_vertex_index
                j_bottom: int = find_face_vertex_index(F[f_bottom, :], v_bottom)

                # Compute midpoint to opposite corner derivative for the bottom face
                uv_bottom: PlanarPoint = chart.bottom_vertex_uv_position
                assert uv_bottom.shape == (1, 2)
                # uv_bottom @ edge_gradients = (1, 2) @ (2, 3) == (1, 3)
                midpoint_data_ref[f_bottom][j_bottom].normal_derivative = uv_bottom @ edge_gradients[i][j]
                assert midpoint_data_ref[f_bottom][j_bottom].normal_derivative.shape == (1, 3)
                logger.info("Midpoint data for corner (%s, %s) is %s",
                            f_bottom,
                            j_bottom,
                            midpoint_data_ref[f_bottom][j_bottom].normal_derivative)


def compute_edge_midpoint_with_gradient(edge_origin_corner_data: TriangleCornerData,
                                        edge_dest_corner_data: TriangleCornerData
                                        ) -> tuple[SpatialVector, SpatialVector]:
    """
    Given corner data for the endpoints of the edge, compute the midpoint
    and the edge aligned midpoint gradient of the corresponding Powell-Sabin
    quadratic spline patch

    NOTE: method used in optimize_spline_surface.py

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

    fi: SpatialVector = edge_origin_corner_data.function_value
    fj: SpatialVector = edge_dest_corner_data.function_value
    dij: SpatialVector = edge_origin_corner_data.first_edge_derivative
    dji: SpatialVector = edge_dest_corner_data.second_edge_derivative
    midpoint: SpatialVector = 0.5 * (fi + fj) + 0.125 * (dij + dji)
    midpoint_edge_gradient: SpatialVector = 2.0 * (fj - fi) + 0.5 * (dji - dij)
    assert midpoint.shape == (1, 3)
    assert midpoint_edge_gradient.shape == (1, 3)

    return midpoint, midpoint_edge_gradient


def generate_corner_data_matrices(corner_data: list[list[TriangleCornerData]],
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given per corner position data, rearrange it into matrices with row i
    corresponding to the data for vertex i.
    NOTE: Used in twelve_split_spline.py

    :param corner_data: quadratic vertex position and derivative data with elements list[TriangleCornerData, TriangleCornerData, TriangleCornerData]
    :return:
        - position_matrix: matrix with position data as rows
        - first_derivative_matrix: matrix with first derivative data as rows
        - second_derivative_matrix: matrix with second derivative data as rows
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    num_faces: int = len(corner_data)

    # Resize the corner data matrices
    position_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    first_derivative_matrix:  np.ndarray[tuple[int, int], np.dtype[np.float64]]
    second_derivative_matrix:  np.ndarray[tuple[int, int], np.dtype[np.float64]]
    position_matrix = np.zeros(shape=(3 * num_faces, 3), dtype=np.float64)
    first_derivative_matrix = np.zeros(shape=(3 * num_faces, 3), dtype=np.float64)
    second_derivative_matrix = np.zeros(shape=(3 * num_faces, 3), dtype=np.float64)

    # Organize position data into matrices
    for i in range(num_faces):
        for j in range(3):
            # NOTE: flatten() for NumPy broadcasting
            position_matrix[3 * i + j, :] = corner_data[i][j].function_value.flatten()
            first_derivative_matrix[3 * i + j, :] = corner_data[i][j].first_edge_derivative.flatten()
            second_derivative_matrix[3 * i + j, :] = corner_data[i][j].second_edge_derivative.flatten()

    return position_matrix, first_derivative_matrix, second_derivative_matrix


def generate_midpoint_data_matrices(corner_data: list[list[TriangleCornerData]],
                                    midpoint_data: list[list[TriangleMidpointData]],
                                    # position_matrix_ref: np.ndarray,
                                    # tangent_derivative_matrix_ref: np.ndarray,
                                    # normal_derivative_matrix_ref: np.ndarray
                                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given position data, rearrange the edge midpoint data into matrices with row i
    corresponding to the data for edge i.

    Note that the normal derivative matrix is extracted directly from the midpoint data
    and the position and tangent derivative matrices are inferred from the corner data.

    NOTE: function used in twelve_split_spline.cpp

    :param corner_data: quadratic vertex position and derivative data
    :type corner_data: list[list[TriangleCornerData, TriangleCornerData, TriangleCornerData]]

    :param midpoint_data: quadratic edge midpoint derivative data
    :type midpoint_data: list[list[TriangleMidpointData, TriangleMidpointData, TriangleMidpointData]]

    :return:
        - position_matrix: matrix with midpoint position data as rows
        - tangent_derivative_matrix: matrix with edge tangent derivative data as rows
        - normal_derivative_matrix: matrix with normal derivative data as rows
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    assert len(corner_data[0]) == 3
    assert len(midpoint_data[0]) == 3
    num_faces: int = len(corner_data)

    # Resize the corner data matrices
    position_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    tangent_derivative_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    normal_derivative_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    position_matrix = np.zeros(shape=(3 * num_faces, 3), dtype=np.float64)
    tangent_derivative_matrix = np.zeros(shape=(3 * num_faces, 3), dtype=np.float64)
    normal_derivative_matrix = np.zeros(shape=(3 * num_faces, 3), dtype=np.float64)

    # Organize position data into matrices
    # TODO: could try to use NumPy indexing magic
    for i in range(num_faces):
        for j in range(3):
            midpoint: SpatialVector
            midpoint_edge_gradient: SpatialVector
            midpoint, midpoint_edge_gradient = compute_edge_midpoint_with_gradient(
                corner_data[i][(j + 1) % 3],
                corner_data[i][(j + 2) % 3])
            assert midpoint.shape == (1, 3)
            assert midpoint_edge_gradient.shape == (1, 3)
            position_matrix[3 * i + j, :] = midpoint.flatten()
            tangent_derivative_matrix[3 * i + j, :] = midpoint_edge_gradient.flatten()
            normal_derivative_matrix[3 * i + j, :] = midpoint_data[i][j].normal_derivative.flatten()

    return position_matrix, tangent_derivative_matrix, normal_derivative_matrix
