"""
Original ASOC code utilizes the AutoDiff library by Wenzel Jakob. Based on code by Jon Kaldor
and Eitan Grinspun.
So, this file focuses on utilizing automatic differentiation via JAX. 

NOTE: really only need this for later on for implementing quadratic_spline_surface() folder
"""

import jax.numpy as jnp


def generate_local_variable_matrix_index(row: int, col: int, dimension=3) -> int:
    # TODO: used in testing.
    return dimension * row + col


def generate_independent_variable():
    # This is only used by build_independent_variable_vector() and build_independent_variable_matrix()


def build_independent_variable_vector():
    # This is only used in testing... no where else.


def generate_constant_variable():
    # This is used by build_constant_variable_vector() and build_constant_variable_matrix()


def build_constant_variable_vector():
    # This isn't used anywhere...


def build_constant_variable_matrix():
    # This isn't used anywhere...
    pass


def compute_variable_value():
    # TODO: implement this. quite import for tests l8r


def compute_variable_gradient():
    # TODO: implement. useful l8r


def compute_variable_hessian():
    # TODO: useful l8r


def extract_variable_vector_values():
    # used by extract_variable_v.... what?


def extract_variable_matrix_values():
    # TODO: useful


def vector_contains_nan():
    # Overriding something, not needed.


def matrix_contains_nan():
    # n/a


def variable_square():
    # somewhat useful...


def variable_square_norm():
    # not used anywhere...
