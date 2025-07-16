import numpy as np
import numpy.linalg as LA
import numpy.typing as npt

# from typing import NewType
import logging
import mathutils
import math

# TODO: there's something wrong with the IGL import statement here that's causing everything to error...
# And that was because I accidentally uninstall SciPy...
import igl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
# UserId = NewType('UserId', int)

# *******
# GLOBALS
# *******
FLOAT_EQUAL_PRECISION: float = 1e-10  # Epsilon for default float
ADJACENT_CONTOUR_PRECISION: float = 1e-6  # Epsilon for chaining contours
# Epsilon for curve-curve bounding box padding
PLANAR_BOUNDING_BOX_PRECISION: float = 0
# Epsilon for Bezier clipping intersections
FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION: float = 1e-7
DISCRETIZATION_LEVEL: int = 2  # Spline surface discretization level

# Real number representations
PlanarPoint = np.ndarray
SpatialVector = np.ndarray

# PlanarPoint = NewType('PlanarPoint', npt.NDArray(shape=[(1, 2)], dtype=float))
# SpatialVector = np.ndarray(shape=(1, 3), dtype=float)

# Matrix2x3r = np.ndarray(shape=(2, 3), dtype=float)
# Matrix2x2r = np.ndarray(shape=(2, 2), dtype=float)
# Matrix3x2r = np.ndarray(shape=(3, 2), dtype=float)
# Matrix3x3r = np.ndarray(shape=(3, 3), dtype=float)
Matrix6x3r = np.ndarray
# Edge = list[int, int]


# **********************
# Debug/Helper Methods
# **********************
def unimplemented(msg: str):
    raise Exception(msg)


def todo(msg: str | None = None):
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


class Index(int):
    # Basically, this is unsigned stuff only. that's it.
    pass


class Matrix:
    def __init__(self, arr: np.ndarray):
        assert arr.ndim != 1
        self.arr = arr

    def row(self, i):
        pass


# Colors
MINT_GREEN = np.array([[0.170], [0.673], [0.292]])
# <double, 3, 1> SKY_BLUE(0.297, 0.586, 0.758);
# <double, 3, 1> OFF_WHITE(0.896, 0.932, 0.997);
GOLD_YELLOW = np.array([[0.670], [0.673], [0.292]])


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
    pass


def matrix_equal():
    pass


def view_mesh():
    pass


def view_parameterized_mesh():
    pass


def screenshot_mesh():
    pass

# ****************
# Basic arithmetic
# ****************


def sgn():
    pass


def power():
    pass


def compute_discriminant():
    pass


def dot_product():
    pass


def cross_product(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    @brief  Compute the cross product of two vectors of arbitrary scalars.
    @tparam Scalar: scalar field (must support addition and multiplication)
    @param[in] v: first vector to cross product
    @param[in] w: second vector to cross product
    @return cross product v x w
    """
    assert v.shape == (3, 1)
    assert w.shape == (3, 1)

    # TODO: make these 2D matrices act like Eigen matrices with shape (n, 1) where they can be accessed like vectors and whatnot
    # TODO: use NumPy's version of cross products
    n = np.array([
        [v[1][0] * w[2][0] - v[2][0] * w[1][0]],
        [-(v[0][0] * w[2][0] - v[2][0] * w[0][0])],
        [v[0][0] * w[1][0] - v[1][0] * w[0][0]]])

    assert n.shape == (3, 1)
    return n


def triple_product():
    pass


def normalize():
    pass


def elementary_basis_vector():
    pass


def reflect_across_x_axis(vector: PlanarPoint) -> PlanarPoint:
    """
    @brief  Reflect a vector in the plane across the x-axis.

    @param[in] vector: vector to reflect
    @return reflected vector    
    """
    reflected_vector = PlanarPoint(shape=(1, 2))
    # FIXME maybe problem with index accessing
    reflected_vector[0][0] = vector[0][0]
    reflected_vector[0][1] = -vector[0][1]
    return reflected_vector


# this is a void.
def rotate_vector():
    pass

# this returns a SpatialVector class
# def rotate_vector():
#     pass


def project_vector_to_plane():
    pass


def vector_min():
    pass


def vector_max():
    pass


def column_vector_min():
    pass


def column_vector_max():
    pass


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
    pass


def convert_unsigned_vector_to_signed(unsigned_vector: list[int]):
    # Pretty sure we don't need this function
    pass


def remove_vector_values():
    pass


def copy_to_planar_point():
    pass


def copy_to_spatial_vector():
    pass


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
    pass


def write_float_vector():
    pass


def append():
    pass


def nested_vector_size():
    pass


def convert_nested_vector_to_matrix():
    pass


def append_matrix():
    pass


def flatten_matrix_by_row():
    pass


# TODO: this seems like something that would interact with the Blender API.
#       Move this over to a different file to separate the parts that interact with the Blender API
def read_camera_matrix():
    pass


def generate_linspace(t_0: float, t_1: float, num_points: int):
    """
    Originally under "Pythonic methods" in ASOC code.
    """
    # TODO: compare NumPy linspace with ASOC linspace
    return np.linspace(t_0, t_1, num_points)


def arrange():
    pass

#  *******************
#  Basic mesh topology
#  *******************


def contains_vertex(face: np.ndarray[tuple[int], np.dtype[np.int_]], vertex_index: int) -> bool:
    return vertex_index in face


def find_face_vertex_index(face: np.ndarray, vertex_index: int) -> int:
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
    if (not igl.is_edge_manifold(F)):
        logger.error("Mesh is not edge manifold")
        return False

    # Check vertex manifold condition
    invalid_vertices = igl.is_vertex_manifold(F)
    if not igl.is_vertex_manifold(F).any():
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
    pass


def compute_point_cloud_bounding_box():
    pass


def remove_mesh_faces(V: np.ndarray, F: np.ndarray, faces_to_remove: list[int], V_submesh: np.ndarray, F_submesh: np.ndarray):
    faces_to_keep: list[int]
    index_vector_complement(faces_to_remove, F.shape[0], faces_to_keep)
    F_unsimplified_submesh = np.ndarray(shape=(len(faces_to_keep), F.shape[1]))
    for i, _ in enumerate(faces_to_keep):
        F_unsimplified_submesh[i, :] = F[faces_to_keep[i], :]

    # Remove unreferenced vertices and update face indices
    # FIXME: not sure if the below is doing what it needs to do
    F_submesh, V_submesh = igl.remove_unreferenced(F, V)
    logger.info("Final mesh has %s faces and %s vertices",
                F_submesh.shape[0], V_submesh.shape[0])


def remove_mesh_vertices(V: np.ndarray, F: np.ndarray, vertices_to_remove: list[int], V_submesh: np.ndarray, F_submesh: np.ndarray, faces_to_remove: list[int]):
    logger.info("Removing %s vertices from mesh with %s faces and %s vertices", len(
        vertices_to_remove), F.shape[0], V.shape[0])

    # Tag faces adjacent to the vertices to remove
    # TODO: implement some numpy version of of finding a vertex in a row of F
    faces_to_remove.clear()
    for face_index in range(F.shape[0]):
        for i, _ in enumerate(vertices_to_remove):
            if contains_vertex(F[face_index, :], vertices_to_remove[i]):
                faces_to_remove.append(face_index)
                break
    logger.info("Remove %s faces", len(faces_to_remove))

    # Remove faces adjacent to cones
    remove_mesh_faces(V, F, faces_to_remove, V_submesh, F_submesh)


def join_path():
    pass


def matrix_contains_nan(mat: np.ndarray) -> bool:
    assert mat.ndim > 1

    # TODO: add test case to check this function with ASOC code version
    return np.isnan(mat).any()


def vector_contains_nan(vec: np.ndarray) -> bool:
    assert vec.shape[0] == 1

    # TODO: add test case to check this function with ASOC code version
    return np.isnan(vec).any()


def convert_polylines_to_edges():
    pass
