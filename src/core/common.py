# TODO: include logging here?
import logging
import math
import numpy as np

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


def float_equal_zero(x: float, eps=FLOAT_EQUAL_PRECISION):
    """
    /// @brief  Check if some floating point value is numerically zero.
    ///
    /// @param[in] x: value to compare with zero
    /// @param[in] eps: threshold for equality
    /// @return true iff x is below 1e-10
    """
    return math.isclose(x, 0.0, rel_tol=eps)


# TODO: just use the Math library for this?

def float_equal(x: float, y: float, eps=FLOAT_EQUAL_PRECISION) -> bool:
    """
    /// @brief Check if two floating point values are numerically equal
    ///
    /// @param[in] x: first value to compare
    /// @param[in] y: second value to compare
    /// @param[in] eps: threshold for equality
    /// @return true iff x - y is numerically zero
    """
    return math.isclose(x, y, rel_tol=eps)


def vector_equal():
    pass


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


def cross_product():
    pass


def triple_product():
    pass


def normalize():
    pass


def elementary_basis_vector():
    pass


def reflect_across_x_axis():
    pass


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


def convert_index_vector_to_boolean_array():
    pass


def convert_boolean_array_to_index_vector():
    pass


def index_vector_complement():
    pass


def convert_signed_vector_to_unsigned():
    pass


def convert_unsigned_vector_to_signed():
    pass


def remove_vector_values():
    pass


def copy_to_planar_point():
    pass


def copy_to_spatial_vector():
    pass


# TODO: don't think we need this since Python prints out vectors just fine.... maybe
# Unless there's an extra fancy vector type in the C++ code like vector<RationalFunction> or something like that.
def formatted_vector():
    pass


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


def generate_linspace():
    pass


def arrange():
    pass

#  *******************
#  Basic mesh topology
#  *******************


def contains_vertex(face: np.ndarray[tuple[int], np.dtype[np.int_]], vertex_index: int) -> bool:
    return vertex_index in face


def find_face_vertex_index():
    pass


def is_manifold():
    pass


def area_from_length():
    pass


def area_from_positions():
    pass


def angle_from_length():
    pass


def angle_from_positions():
    pass


def interval_lerp():
    pass


def compute_point_cloud_bounding_box():
    pass


def remove_mesh_faces():
    pass


def remove_mesh_vertices():
    pass


def join_path():
    pass


def matrix_contains_nan():
    pass


def vector_contains_nan():
    pass


def convert_polylines_to_edges():
    pass
