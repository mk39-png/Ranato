"""
generate_transformations.py
Methods to generate projective transformation matrices.
"""
from ..core.common import *


def get_frame():
    # Skip


def origin_to_infinity_projective_matrix():
    """
    Generate the projective matrix that sends the origin to infinity while
    fixing the plane z = plane_distance

    @param[in] plane_distance: distance from the origin to the plane
    @return 4x4 projective matrix for the transformation
    """
    pass


def infinity_to_origin_projective_matrix():
    """
    Generate the projective matrix that sends a point at infinity to the origin
    while fixing the plane z = plane_distance.

    This is the inverse of the map sending the origin to infinity.

    @param[in] plane_distance: distance from the origin to the plane
    @return 4x4 projective matrix for the transformation
    """
    pass


def rotate_frame_projective_matrix():
    """
    Generate the rotation matrix that sends the given frame to the standard
    frame.

    @param[in] frame: 3x3 frame matrix to align with the standard frame
    @return 4x4 projective matrix for the transformation
    """
    pass


def translation_projective_matrix():
    """
    Generate the projective matrix representing translation by the given
    translation vector.

    @param[in] translation: 1x3 translation vector
    @return 4x4 projective matrix for the transformation    
    """
    pass


def scaling_projective_matrix():
    # Skip
    pass


def x_axis_rotation_projective_matrix():
    """
    Generate the projective matrix for rotation around the x axis.

    @param[in] degree: degree of rotation around the x axis
    @return 4x4 projective matrix for the transformation
    """
    pass


def y_axis_rotation_projective_matrix():
    """
    Generate the projective matrix for rotation around the y axis.

    @param[in] degree: degree of rotation around the y axis
    @return 4x4 projective matrix for the transformation
    """
    pass


def z_axis_rotation_projective_matrix():
    """
    Generate the projective matrix for rotation around the z axis.

    @param[in] degree: degree of rotation around the z axis
    @return 4x4 projective matrix for the transformation
    """
    pass


def axis_rotation_projective_matrix():
    """
    Generate the projective matrix for chained rotation around the standard
    axes.

    The order of rotation is z axis -> y axis -> x axis

    @param[in] x_degree: degree of rotation around the x axis
    @param[in] y_degree: degree of rotation around the y axis
    @param[in] z_degree: degree of rotation around the z axis
    @return 4x4 projective matrix for the transformation
    """
    pass
