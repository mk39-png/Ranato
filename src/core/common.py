import numpy as np
import numpy.linalg as LA
import numpy.typing as npt
from typing import Annotated, Literal, TypeVar
import scipy as sp

# from typing import NewType
import logging
import mathutils
import math

from scipy.sparse import linalg as splinalg, csr_matrix
# import scipy.sparse as s
# from cvxopt import spmatrix
import sys

import igl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
# UserId = NewType('UserId', int)

# *******
# GLOBALS
# *******
# Epsilon for default float
FLOAT_EQUAL_PRECISION: float = 1e-10
# Epsilon for chaining contours
ADJACENT_CONTOUR_PRECISION: float = 1e-6
# Epsilon for curve-curve bounding box padding
PLANAR_BOUNDING_BOX_PRECISION: float = 0
# Epsilon for Bezier clipping intersections
FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION: float = 1e-7
# Spline surface discretization level
DISCRETIZATION_LEVEL: int = 2
# Size of spline surface hash table
HASH_TABLE_SIZE: int = 70

# Real number representations

# Including typing here for better code.
# https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype
OneFormXr = np.ndarray  # TODO: what shape is this... I forget
PlanarPoint = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (1, 2)
PlanarPoint1d = np.ndarray[tuple[int], np.dtype[np.float64]]  # shape (2, )
SpatialVector = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (1, 3)
# SpatialVector = Annotated[npt.NDArray[], Literal[1, 3]]
PatchIndex = int
# TODO: combine VectorX and VectorXr into one?
VectorX = np.ndarray  # shape (n, )... sometimes. Oftentimes it's shape (n, 1) or (1, n)...
VectorXr = np.ndarray  # TODO: maybe change name to Vector1D to denote that it's shape (n, )
Vector2D = np.ndarray  # shape (1, n) (or sometimes shape (n, 1) as in the case of optimize_spline_surface)
Vector1D = np.ndarray  # shape (n, )
MatrixXr = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (n, m)
MatrixNx3 = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Matrix6x6r = np.ndarray
Matrix3x6r = np.ndarray
Matrix12x12r = np.ndarray
Matrix12x3f = np.ndarray
Index = int
FaceIndex = int
VertexIndex = int
Edge = list[int]  # length 2

# NOTE: VectorX also encompasses SpatialVectors as well.
# VectorX = np.ndarray

# Used for accessing numpy shape for clarity sake
ROWS = 0
COLS = 1

# PlanarPoint = NewType('PlanarPoint', npt.NDArray(shape=[(1, 2)], dtype=float))
# SpatialVector = np.ndarray(shape=(1, 3), dtype=float)
# Edge = list[int, int]

Matrix3x1r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (3, 1)
Matrix2x3r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (2, 3)
Matrix2x2r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (2, 2)
Matrix3x2r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (3, 2)
Matrix3x3r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (3, 3)
Matrix6x3r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (6, 3)

TwelveSplitGradient = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (36, 1)
TwelveSplitHessian = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (36, 36)


# **********************
# Math-itensive Methods
# **********************

def sparse_cholesky(A):
    """
    https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d#file-sparsecholesky-md
    I can't seem to install scikit-sparse since it relies on CHOLMOD from SuiteSparse and Cython, which does not really help for my purposes of this project being a Blender addon and also scikit-sparse being a process to install on Windows.

    :param A: input sparse matrix that must be sparse symmetric positive-definite
    """
    deprecated()
    sparse_matrix = A.T @ A
    sparse_matrix += 1e-6 * sparse.identity(sparse_matrix.shape[0])  # force the sparse matrix is positive definite
    n = sparse_matrix.shape[0]
    LU = sparse.linalg.splu(sparse_matrix, diag_pivot_thresh=0.0, permc_spec="NATURAL")  # sparse LU decomposition

    L = LU.L @ sparse.diags(LU.U.diagonal()**0.5)

    return L  # return L (lower triangular matrix)

# https://stackoverflow.com/questions/25314067/scipy-sparse-matrix-to-cvxopt-spmatrix
# def scipy_sparse_to_spmatrix(A: csr_matrix) -> spmatrix:
#     coo = A.tocoo()
#     SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
#     return SP

# **********************
# Debug/Helper Methods
# **********************


def deprecated(msg: str = "Method no longer in use"):
    raise Exception(msg)


def unimplemented(msg: str = "Method yet to be implemented"):

    raise Exception(msg)


def todo(msg: str = "Method needs some work"):
    raise Exception(msg)


def unreachable(msg: str = "Method should never reach this part"):
    raise Exception(msg)


# class PlanarPoint(np.ndarray):
#     def __new__(cls, input_array):
#         arr = np.asarray(input_array, dtype=float)

#         # Flatten all allowed forms to shape (2,)
#         if arr.shape == (2,):
#             flat = arr
#         elif arr.shape == (1, 2) or arr.shape == (2, 1):
#             flat = arr.reshape(2,)
#         else:
#             raise ValueError(
#                 "PlanarPoint must be shape (2,), (1, 2), or (2, 1)")

#         obj = flat.view(cls)
#         return obj


# class Index(int):
#     # Basically, this is unsigned stuff only. that's it.
#     todo()


class Matrix:
    def __init__(self, arr: np.ndarray):
        assert arr.ndim != 1
        self.arr = arr

    def row(self, i):
        todo()


# Colors
MINT_GREEN = np.array([[0.170], [0.673], [0.292]])
SKY_BLUE = np.array([[0.297], [0.586], [0.758]])
OFF_WHITE = np.array([[0.896], [0.932], [0.997]])
GOLD_YELLOW = np.array([[0.670], [0.673], [0.292]])


# *******************************************
# New functionality for Python implementation
# *******************************************
# https://stackoverflow.com/questions/8849833/python-list-reserving-space-resizing
def list_resize(l: list, newsize: int, filling=None) -> None:
    if newsize > len(l):
        l.extend([filling for x in range(len(l), newsize)])
    else:
        del l[newsize:]


# -----------------------------------------


def float_equal_zero(x: float, eps=FLOAT_EQUAL_PRECISION):
    # TODO: I don't think this is used since anyone could just input 0.0 into "y" of float_equal....
    # So maybe get rid of this method or at least note that it is out of use
    """
    /// @brief  Check if some floating point value is numerically zero.
    ///
    /// @param[in] x: value to compare with zero
    /// @param[in] eps: threshold for equality
    /// @return true iff x is below 1e-10
    """

    # NOTE: Use absolute tolerance! Relative tolerance is not suited for our purpose.
    return math.isclose(x, 0.0, abs_tol=eps)


# TODO: just use the Math library for this?

def float_equal(x: float, y: float, eps=FLOAT_EQUAL_PRECISION) -> bool:
    """
    @brief Check if two floating point values are numerically equal

    @param[in] x: first value to compare
    @param[in] y: second value to compare
    @param[in] eps: threshold for equality
    @return true iff x - y is numerically zero
    """

    # NOTE: Use absolute tolerance! Relative tolerance is not suited for our purpose.
    return math.isclose(x, y, abs_tol=eps)


def vector_equal(v: np.ndarray, w: np.ndarray, eps: float = FLOAT_EQUAL_PRECISION):
    """
    @brief Check if two row vectors of floating point values are numerically
    equal

    @param[in] v: first vector of values to compare
    @param[in] w: second vector of values to compare
    @param[in] eps: threshold for equality
    @return true iff v - w is numerically the zero vector
    """

    # Just using numpy comparison.
    # TODO :compare with ASOC code and if atol is the way to go.
    return np.allclose(v, w, atol=eps)


def column_vector_equal():
    todo()


def matrix_equal():
    todo()


def view_mesh():
    todo()


def view_parameterized_mesh():
    todo()


def screenshot_mesh():
    todo()

# ****************
# Basic arithmetic
# ****************


def sgn():
    todo()


def power():
    todo()


def compute_discriminant():
    todo()


def dot_product():
    todo()


def cross_product(v_ref: Matrix3x1r, w_ref: Matrix3x1r) -> Matrix3x1r:
    """
    @brief  Compute the cross product of two vectors of arbitrary scalars.
    @tparam Scalar: scalar field (must support addition and multiplication)
    @param[in] v: first vector to cross product
    @param[in] w: second vector to cross product
    @return cross product v x w in shape (3, 1)
    """
    # Flatten (1, 3) or (3, 1) matrices into (3, ) arrays
    v: Vector1D = v_ref.flatten()
    w: Vector1D = w_ref.flatten()
    assert v.size == 3
    assert w.size == 3

    # TODO: make these 2D matrices act like Eigen matrices with shape (n, 1) where they can be accessed like vectors and whatnot
    # TODO: use NumPy's version of cross products
    n: Matrix3x1r = np.array([
        [v[1] * w[2] - v[2] * w[1]],
        [-(v[0] * w[2] - v[2] * w[0])],
        [v[0] * w[1] - v[1] * w[0]]],
        dtype=np.float64)

    # TODO: decide on flattening n to shape (3, ) or keep as shape (3, 1)
    assert n.shape == (3, 1)
    return n


def triple_product():
    todo()


def normalize():
    todo()


def elementary_basis_vector():
    todo()


def reflect_across_x_axis(vector: PlanarPoint) -> PlanarPoint:
    """
    @brief  Reflect a vector in the plane across the x-axis.

    @param[in] vector: vector to reflect
    @return reflected vector of shape (1, 2)
    """
    reflected_vector = PlanarPoint(shape=(1, 2))
    # FIXME maybe problem with index accessing
    reflected_vector[0][0] = vector[0][0]
    reflected_vector[0][1] = -vector[0][1]
    return reflected_vector


# this is a void.
def rotate_vector():
    todo()

# this returns a SpatialVector class
# def rotate_vector():
#     todo()


def project_vector_to_plane():
    todo()


def vector_min():
    todo()


def vector_max():
    todo()


def column_vector_min():
    todo()


def column_vector_max():
    todo()


def vector_contains(vec: list, item) -> bool:
    return item in vec


def convert_index_vector_to_boolean_array(index_vector: list[int], num_indices: int) -> list[bool]:
    boolean_array: list[bool] = [False for _ in range(num_indices)]

    for i, _ in enumerate(index_vector):
        boolean_array[index_vector[i]] = True

    return boolean_array


def convert_boolean_array_to_index_vector(boolean_array: list[bool]) -> list[int]:
    """
    @brief From a boolean array, build a vector of the indices that are true.
    @param[in] boolean_array: array of boolean values
    @param[out] index_vector: indices where the array is true
    """
    num_indices: int = len(boolean_array)
    index_vector: list[int] = []
    for i in range(num_indices):
        if (boolean_array[i]):
            index_vector.append(i)

    return index_vector


def index_vector_complement(index_vector: list[int], num_indices: int) -> list[int]:
    """
    Returns the complement of the index_vector as a list of int.

    :param index_vector: vector to take the complement of
    :type index_vector: list[int]

    :param num_indices: determines the size of the complement_vector
    :type num_indices: int

    :return: complement_vector
    :rtype: list[int]
    """
    # TODO: test this function to see if it's working correctly

    # Build index boolean array
    boolean_array: list[bool] = convert_index_vector_to_boolean_array(
        index_vector, num_indices)

    # Build complement
    complement_vector: list[int] = []
    for i in range(num_indices):
        if not boolean_array[i]:
            complement_vector.append(i)

    return complement_vector


def convert_signed_vector_to_unsigned():
    todo()


def convert_unsigned_vector_to_signed(unsigned_vector: list[int]):
    # Pretty sure we don't need this function
    todo()


def remove_vector_values(indices_to_remove: list[Index], vec: list) -> list:
    """
    Removes elements from vec with indices specified in indices_to_remove.

    :param indices_to_remove: indices to remove
    :type indices_to_remove: list[Index]

    :param vec: vector to remove from
    :type vec: list

    :return: vector with indices removed
    :rtype: list
    """
    # Removes indices from vev

    # Remove faces adjacent to cones
    indices_to_keep: list[Index] = index_vector_complement(indices_to_remove, len(vec))
    subvec: list = []

    # TODO: double check logic here with ASOC code
    for _, index_to_keep in enumerate(indices_to_keep):
        subvec.append(vec[index_to_keep])

    assert len(subvec) == len(indices_to_keep)

    return subvec


def copy_to_planar_point():
    todo()


def copy_to_spatial_vector():
    todo()


# TODO: don't think we need this since Python prints out vectors just fine.... maybe
# Unless there's an extra fancy vector type in the C++ code like vector<RationalFunction> or something like that.
def formatted_vector(vec: list[np.float64], delim: str = "\n") -> str:
    # raise Exception(
    # "formatted_vector() is not implmemented. Print out object as-is instead.")
    vector_string: str = ""
    for i, _ in enumerate(vec):
        vector_string += (str(vec[i]) + delim)

    return vector_string


def write_vector():
    todo()


def write_float_vector():
    todo()


def append():
    todo()


def nested_vector_size():
    todo()


def convert_nested_vector_to_matrix(vec: list[Vector2D]) -> np.ndarray:
    """
    WARNING: Do not use this method, implementation does not generalize to a list types.
    Especially with list[np.ndarray] where ndarray is some shape (n, 1) or (1, n)

    """
    # n: int = len(vec)
    # if (n <= 0):
    #     return np.ndarray(shape=(0, 0))

    # # TODO: problem may arise when vec is list of np.ndarray ndim >= 2
    # # TODO: below is supposed to be size3...
    # matrix = np.ndarray(shape=(len(vec), vec[0].size))
    # for i in range(n):
    #     # inner_vec_size = len(vec[i])
    #     inner_vec: np.ndarray = vec[i].flatten()
    #     for j in range(inner_vec.size):
    #         matrix[i, j] = vec[i].flatten()[j]
    matrix = np.array(vec).squeeze()
    # assert matrix.shape == (len(vec))

    return matrix


def append_matrix():
    todo()


def flatten_matrix_by_row():
    todo()


# TODO: this seems like something that would interact with the Blender API.
#       Move this over to a different file to separate the parts that interact with the Blender API
def read_camera_matrix():
    todo()


def generate_linspace(t_0: float, t_1: float, num_points: int) -> np.ndarray:
    """
    Originally under "Pythonic methods" in ASOC code.
    """
    # TODO: compare NumPy linspace with ASOC linspace
    return np.linspace(t_0, t_1, num_points)


def arrange():
    todo()

#  *******************
#  Basic mesh topology
#  *******************


def contains_vertex(face: np.ndarray[tuple[int], np.dtype[np.int_]], vertex_index: int) -> bool:
    """
    Returns true iff the face contains the given vertex.

    :param face: 1D NumPy array of integers. Shape is (n, )
    :type face: np.ndarray

    :param vertex_index: the index to check for inside face
    :type vertex_index: int

    :return: boolean if vertex_index is in face
    """
    return vertex_index in face


def find_face_vertex_index(face: np.ndarray, vertex_index: int) -> int:
    """
    :param face: np.ndarray of shape (n, ) of ndim = 1
    :type face: np.ndarray

    :param vertex_index:
    :type vertex_index: int

    :return:  face vertex index
    :rtype: int
    """
    # TODO: test this numpy-esque implementation with the ASOC version...
    # NOTE: we want to check that face is a vector rather than a matrix
    assert face.ndim == 1

    vertex_indices = np.argwhere(face == vertex_index)
    if vertex_indices.size > 0:
        return vertex_indices[0][0]

    return -1


def is_manifold(F: np.ndarray) -> bool:
    """
    @brief Check if F describes a manifold mesh with a single component

    @param[in] F: mesh faces
    @return true iff the mesh is manifold
    """

    # Check edge manifold condition
    # Checks the tuple of elements that are returned. first element tells us if all edges are manifold or not.
    if not igl.is_edge_manifold(F)[0]:
        logger.error("Mesh is not edge manifold")
        return False

    # Check vertex manifold condition

    invalid_vertices: np.ndarray = igl.is_vertex_manifold(F)  # array of bool values
    if not invalid_vertices.any():
        logger.error("Mesh is not vertex manifold")
        return False

    # Check single component
    # TODO: check datatype on component_ids and if it's a numpy array
    component_ids: np.ndarray = igl.vertex_components(F)

    if (component_ids.max() - component_ids.min()) > 0:
        logger.error("Mesh has multiple components")
        return False

    # Manifold otherwise
    return True


#  *******************
#  Basic mesh geometry
#  *******************
def area_from_length(l0: float, l1: float, l2: float) -> float:
    """
    @brief Compute the area of a triangle from the edge lengths.

    @param[in] l0: first edge length
    @param[in] l1: second edge length
    @param[in] l2: third edge length
    @return area of the triangle
    """
    # Return the area (or zero if there is a triangle inequality violation)
    s: float = 0.5 * (l0 + l1 + l2)  # semi-perimeter
    area: float = math.sqrt(max(s * (s - l0) * (s - l1) * (s - l2), 0.0))
    assert not math.isnan(area)
    return area


def area_from_positions(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    assert p0.shape[0] == 1  # making sure that p0 is shape (1, n)
    assert p0.shape == p1.shape
    assert p1.shape == p2.shape

    # TODO: double check that numpy norm is doing what we want
    l0: float = LA.norm(p2 - p1)
    l1: float = LA.norm(p0 - p2)
    l2: float = LA.norm(p1 - p0)

    assert isinstance(l0, float)

    return area_from_length(l0, l1, l2)


def angle_from_length(edge_length_opposite_corner: float,
                      first_adjacent_edge_length: float,
                      second_adjacent_edge_length: float
                      ) -> float:
    """
    @brief Compute the angle of a triangle corner with given edge lengths

    @param[in] edge_length_opposite_corner: length of the edge opposite the
    corner
    @param[in] first_adjacent_edge_length: length of one of the edges adjacent
    to the corner
    @param[in] first_adjacent_edge_length: length of the other edge adjacent to
    the corner
    @return angle of the corner
    """
    # Rename variables for readability
    l0: float = edge_length_opposite_corner
    l1: float = first_adjacent_edge_length
    l2: float = second_adjacent_edge_length

    # Compute the angle
    # FIXME Avoid potential division by 0
    Ijk: float = (-l0 * l0 + l1 * l1 + l2 * l2)
    return math.acos(min(max(Ijk / (2.0 * l1 * l2), -1.0), 1.0))


def angle_from_positions(dimension: int, angle_corner_position: np.ndarray, second_corner_position: np.ndarray, third_corner_position: np.ndarray) -> float:
    """
    @brief Compute the angle of a triangle corner with given positions

    @param[in] angle_corner_position: position of the corner to compute the
    angle for
    @param[in] second_corner_position: position of one of the other two corners
    of the triangle
    @param[in] third_corner_position: position of the final corner of the
    triangle
    @return angle of the corner
    """
    assert angle_corner_position.shape == (1, dimension)
    assert second_corner_position.shape == (1, dimension)
    assert third_corner_position.shape == (1, dimension)

    # TODO: double check that the below are going to be floats...
    l0: float = LA.norm(third_corner_position - second_corner_position)
    l1: float = LA.norm(third_corner_position - second_corner_position)
    l2: float = LA.norm(third_corner_position - second_corner_position)

    return angle_from_length(l0, l1, l2)


def interval_lerp():
    todo()


def compute_point_cloud_bounding_box(points: np.ndarray) -> tuple[SpatialVector, SpatialVector]:
    """ Compute the bounding box for a matrix of points in R^n.
    The points are assumed to be the rows of the points matrix.

    :param points: points to compute the bounding box for.
    :type points: np.ndarray

    :return (min_point, max_point): tuple of (point with minimum coordinates for the bounding box, point with maximum coordinates for the bounding box).
    :rtype: tuple[Vector, Vector]
    """

    # TODO: cahnge type of points to matrix
    num_points = points.shape[0]
    dimension = points.shape[1]

    if (num_points == 0):
        todo("deal with 0 case")
    if (dimension == 0):
        todo("deal with 0 case")

    # Get minimum and maximum coordinates for the points
    # TODO: test this with NumPy, and also maybe mathutils
    # Get minimum and maximum coordinates for the points
    min_point = points[[0], :]
    max_point = points[[0], :]
    assert min_point.shape == (1, 3)
    assert max_point.shape == (1, 3)

    for pi in range(num_points):
        for j in range(dimension):
            min_point[0][j] = min(min_point[0][j], points[pi, j])
            max_point[0][j] = max(max_point[0][j], points[pi, j])

    # NOTE: returning "Vector", which is just shape (1, 3) iirc
    return min_point, max_point


def remove_mesh_faces(V: np.ndarray,
                      F: np.ndarray,
                      faces_to_remove: list[FaceIndex]) -> tuple[np.ndarray, np.ndarray]:
    """
    Using igl to remove unreferenced vertices from V using faces_to_remove and updating F accordingly.

    :param V: vertices to remove unreferenced vertices from. np.ndarray of float
    :type V: np.ndarray
    :param F: faces with np.ndarray of int
    :type F: np.ndarray
    :param faces_to_remove: index of faces to remove
    :type faces_to_remove: list[int]

    :return: tuple of V and F submeshes (V, F)
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    faces_to_keep: list[FaceIndex] = index_vector_complement(faces_to_remove, F.shape[0])  # rows

    # TODO: in the ASOC code, this F_unsimplified_submesh was initialized to shape (faces_to_keep.size(), F.cols()) and then immediately resized.
    F_unsimplified_submesh: np.ndarray = np.ndarray(shape=(len(faces_to_keep), 3), dtype=int)

    for i, _ in enumerate(faces_to_keep):
        F_unsimplified_submesh[i, :] = F[faces_to_keep[i], :]

    # Remove unreferenced vertices and update face indices
    # FIXME: not sure if the below is doing what it needs to do
    F_submesh: np.ndarray
    V_submesh: np.ndarray
    # TODO: Pylint shows error with igl not having member, but I'm sure it is fine.
    F_submesh, V_submesh, __placeholder1, __placeholder2 = igl.remove_unreferenced(F, V)

    logger.info("Final mesh has %s faces and %s vertices",
                F_submesh.shape[0], V_submesh.shape[0])  # rows

    return V_submesh, F_submesh


def remove_mesh_vertices(V: np.ndarray,
                         F: np.ndarray[tuple[int], np.dtype[np.int_]],
                         vertices_to_remove: list[VertexIndex]) -> tuple[np.ndarray, np.ndarray, list[FaceIndex]]:
    """
    Removes mesh vertices from V based on the indices inside vertices_to_remove and updates F accordingly.

    :param V: vertices matrix of floats
    :type V: np.ndarray
    :param F: faces matrix of integers
    :type F: np.ndarray
    :param vertices_to_remove: list of indices of vertices to remove
    :type vertices_to_remove: list[int]

    :return: tuple of vertex matrix with vertices removed, updated faces, and list of face indices that were removed
    :rtype: tuple[np.ndarray, np.ndarray, list[FaceIndex]]
    """
    logger.info("Removing %s vertices from mesh with %s faces and %s vertices", len(
        vertices_to_remove), F.shape[0], V.shape[0])

    # Tag faces adjacent to the vertices to remove
    # TODO: implement some numpy version of of finding a vertex in a row of F
    faces_to_remove: list[FaceIndex] = []
    faces_to_remove.clear()
    for face_index in range(F.shape[0]):
        for i, _ in enumerate(vertices_to_remove):
            # NOTE: contains_vertex expects NumPy array of 1 dimension (i.e. shape (n , ))
            if contains_vertex(F[face_index, :], vertices_to_remove[i]):
                faces_to_remove.append(face_index)
                break
    logger.info("Remove %s faces", len(faces_to_remove))

    # Remove faces adjacent to cones
    V_submesh: np.ndarray
    F_submesh: np.ndarray
    V_submesh, F_submesh = remove_mesh_faces(V, F, faces_to_remove)

    return V_submesh, F_submesh, faces_to_remove


def join_path():
    todo()


def matrix_contains_nan(mat: np.ndarray) -> bool:
    assert mat.ndim > 1

    # TODO: add test case to check this function with ASOC code version
    return np.isnan(mat).any()


def vector_contains_nan(vec: np.ndarray) -> bool:
    assert vec.shape[0] == 1

    # TODO: add test case to check this function with ASOC code version
    return np.isnan(vec).any()


def convert_polylines_to_edges(polylines: list[list[int]]) -> list[Edge]:
    """
    TODO: is this really needed? check if polylines are just list of edges in Python cuz the ASOC code may just be some vector to array conversion.
    NOTE: returning list[list[Edge]], which is really jsut list[list[int, int]]... but the change is the switch to NumPy arrays since Polyscope used in add_surface_to_viewer() in quadratic_spline_surface.py need NumPy arrays rather than Python list[list[int]].... though...
    """

    # TODO: check to see if functionality of NumPy version is the same as the old one.
    edges: list[Edge] = []
    for i, _ in enumerate(polylines):
        for j, _ in enumerate(polylines[i]):
            edge: Edge = [polylines[i][j - 1], polylines[i][j]]
            edges.append(edge)

    return edges
