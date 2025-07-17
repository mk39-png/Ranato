"""
Original ASOC code utilizes the AutoDiff library by Wenzel Jakob. Based on code by Jon Kaldor
and Eitan Grinspun.
So, this file focuses on utilizing automatic differentiation via JAX. 

NOTE: really only need this for later on for implementing quadratic_spline_surface() folder
"""

import jax.numpy as jnp
from src.core.common import *


def generate_local_variable_matrix_index(row: int, col: int, dimension=3) -> int:
    # TODO: used in testing.
    return dimension * row + col


def generate_independent_variable():
    # This is only used by build_independent_variable_vector() and build_independent_variable_matrix()
    """
     Build an independent variable with a given value and variable index.
    ///
     @param[in] value: initial value of the variable
     @param[in] variable_index: global index of the variable in the full system
     @param[in] total_independent_variables: total number of variables in the
     system
     @return constructed differentiable variable
    """

    todo()


def build_independent_variable_vector():
    # This is only used in testing... no where else.
    """
    Build a vector of independent variables with a given initial value and
    contiguous variable indices from some starting index.
    @param[in] value_vector: initial values of the variables
    @param[out] variable_vector: constructed differentiable variables
    @param[in] start_variable_index: global index of the first variable in the
    full system
    @param[in] total_independent_variables: total number of variables in the
    system
    """
    todo()


def build_independent_variable_matrix():
    """
     Build a matrix of independent variables with a given initial value and
    contiguous row-major variable indices from some starting index.
    ///
    @param[in] value_matrix: initial values of the variables
    @param[out] value_matrix: constructed differentiable variables
    @param[in] start_variable_index: global index of the first variable in the
    full system
    @param[in] total_independent_variables: total number of variables in the
    system
    """


def generate_constant_variable():
    """
    Build a differentiable constant with a given value.
    @param[in] value: value of the constant
    @return constructed differentiable variable
    """
    # This is used by build_constant_variable_vector() and build_constant_variable_matrix()
    todo()


def build_constant_variable_vector():
    """
    Build a vector of differentiable constants with given values.
    @param[in] value_vector: values of the constants
    @param[out] constant_variable_vector: vector of constant variables
    """
    # This isn't used anywhere...
    todo()


def build_constant_variable_matrix():
    # This isn't used anywhere...
    """
    Build a matrix of differentiable constants with given values.
    @param[in] value_matrix: values of the constants
    @param[out] constant_variable_matrix: matrix of constant variables
    """
    todo()


def compute_variable_value():
    """
    Extract the value of a differentiable variable.
    @param[in] variable: differentiable variable
    @return value of the variable
    """
    # TODO: implement this. quite import for tests l8r
    todo()


def compute_variable_gradient():
    # TODO: implement. useful l8r
    """
    Extract the gradient of a differentiable variable with respect to the
    independent variables.
    @param[in] variable: differentiable variable
    @return gradient of the variable
    """
    todo()


def compute_variable_hessian():
    # TODO: useful l8r
    """
    Extract the hessian of a differentiable variable with respect to the
    independent variables.
    @param[in] variable: differentiable variable
    @return hessian of the variable
    """
    todo()


def extract_variable_vector_values():
    # used by extract_variable_v.... what?
    """
    Extract the values of a vector of differentiable variables.
    @param[in] variable_vector: vector of differentiable variables
    @param[out] values_vector: vector of the values of the variables
    """
    todo()


def extract_variable_matrix_values():
    # TODO: useful
    """
    Extract the values of a matrix of differentiable variables.
    @param[in] variable_matrix: matrix of differentiable variables
    @param[out] values_matrix: matrix of the values of the variables
    """
    todo()


def vector_contains_nan():
    # Overriding something, not needed.
    """
     Determine if a vector of differentiable variables contains NaN.
    @param[in] variable_vector: vector of differentiable variables
    @return true iff the vector contains NaN
    """
    unimplemented()


def matrix_contains_nan():
    # n/a
    unimplemented()


def variable_square():
    """
     Compute the square of a variable
    @param[in] x: differentiable variables
    @return square of the variable
    """
    # somewhat useful...


def variable_square_norm():
    # not used anywhere...
    """
     Compute the square norm of a variable vector.
///
 @param[in] variable_vector: vector of differentiable variables
 @return square norm of the variable
    """
    unimplemented()
