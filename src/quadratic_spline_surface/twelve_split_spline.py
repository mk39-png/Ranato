"""
Methods to generate a quadratic surface with twelve split Powell-Sabin basis
coefficients
"""

import src.quadratic_spline_surface.PS12_patch_coeffs
import src.quadratic_spline_surface.PS12tri_bounds_coeffs
import src.core.common

import src.core.line_segment
import src.core.polynomial_function
import src.core.rational_function

from src.quadratic_spline_surface.quadratic_spline_surface import *
from src.quadratic_spline_surface.position_data import *

from igl import per_vertex_normals
