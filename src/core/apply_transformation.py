"""
Methods to apply projective transformation matrices to various point data
types.

TODO: try and utilize Blender's Python API for this to make life easier for us.
TODO: but this involves mathutils...
But anyways, could convert to MathUtils and back...
"""
from src.core.common import *
import src.core.generate_transformation


def convert_point_to_homogeneous_coords():
    todo()


def convert_homogeneous_coords_to_point():
    todo()


def apply_transformation_to_point():
    todo()


def apply_transformation_to_points():
    todo()


def apply_transformation_to_points_in_place():
    todo()


def apply_transformation_to_control_points():
    todo("Used internally")


def apply_transformation_to_control_points_in_place():
    unimplemented("Not used")


def apply_transformation_to_vertices():
    """
    Used in generate_algebraic_contours.py
    """
    todo()


def apply_transformation_to_vertices_in_place():
    """
    Used in generate_algebraic_contours.py
    """
    todo()


def generate_projective_transformation():
    unimplemented("Not used ")


def initialize_control_points():
    unimplemented("Not used")


def initialize_vertices():
    """
    Used in generate_perspective_figure
    """
    todo()


def apply_camera_frame_transformation_to_vertices():
    """
    Used in generate_algebraic_contours
    """
