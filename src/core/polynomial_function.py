
"""
Methods to construct and evaluate polynomials in one variable. We represent
polynomials as column vectors of coefficients with basis order 1, t,
t^2,.... For vector valued polynomial functions (i.e. functions f: R -> R^n
where each coordinate projection is a polynomial), we use matrices with the
column vectors as columns. We refer to these vector valued polynomial
functions as polynomial mappings to distinguish them from scalar valued
polynomials. Evaluation of the polynomials and polynomial mappings is then
simply the left product of the row vector [1 t t^2 ... t^n] with the column
vector or matrix.


NOTE: This should be left as is according to how they were implementing in ASOC
as to not break anything.
Meaning, vectors treated as (1, n) and (n, 1) rather than (n, )
"""

import math
from venv import logger

import numpy as np

from src.core.common import (COLS, ROWS, MatrixNx3f, MatrixXf, Vector1D,
                             Vector2D, Vector2f, Vector3f, float_equal,
                             float_equal_zero, unimplemented)


def remove_polynomial_trailing_coefficients(A_coeffs_ref: Vector1D) -> Vector1D:
    """ 
    Remove any zeros at the end of the polynomial coefficient vector

    :param A_coeffs: [in] polynomal coefficient vector
    :return reduced_coeffs: copy of A_coeffs with trailing zeros removed
    """
    assert A_coeffs_ref.ndim == 1

    # Make a copy of A_coeffs_ref as to not modify the original array when trimming zeros
    reduced_coeffs: Vector1D = np.copy(A_coeffs_ref)
    np.trim_zeros(reduced_coeffs, 'b')

    return reduced_coeffs


def generate_monomials(degree: int, t: float) -> Vector2D:
    """ 
    Generate the row vector T of n + 1 monomials 1, t, ... , t^n.

    :param int degree: [in] maximum monomial degree.
    :param float t: [in] evaluation point for the monomials.

    :return T: row vector of monomials of shape (1, degree + 1).
    :rtype: np.ndarray
    """
    # NOTE: Keep below as 2D vector for use in evaluate_polynomial() where dimension is 1
    # and we want the behavior to be the same for any number of dimension.
    # Because if dimension == 1, we want to create a 2D (n, 1) shape rather than a 1D (n, ) shape
    T: Vector2D = np.ndarray(shape=(1, degree + 1), dtype=np.float64)

    T[0][0] = 1.0
    for i in range(1, degree + 1):
        T[0][i] = T[0][i - 1] * t

    return T


def evaluate_polynomial(degree: int,
                        dimension: int,
                        polynomial_coeffs: Vector2D,
                        t: float) -> Vector1D:
    """
    Evaluate the polynomial with given coefficients at t. 
    NOTE: this has been modified from the ASOC code to support any dimension.


    :param int degree: [in] maximum monomial degree.
    :param int dimension: [in] polynomial dimension.
    :param polynomial_coeffs: [in] coefficients of the polynomial.
    :param t: [in] evaluation point for the polynomial.

    :return polynomial_evaluation: evaluation of the polynomial of shape (dimension, )
    """
    assert polynomial_coeffs.shape == (degree + 1, dimension), ("polynomial_coeffs supposed to be "
                                                                "matrix (i.e. ndim > 1 OR shape != "
                                                                "(degree+1,))")

    T: Vector2D = generate_monomials(degree, t)
    assert T.shape == (1, degree + 1)

    # shape (1, dimension) = (1, degree + 1) @ (degree + 1, dimension)
    polynomial_evaluation = T @ polynomial_coeffs

    assert polynomial_evaluation.shape == (1, dimension)
    return polynomial_evaluation.flatten()


def evaluate_polynomial_mapping(degree: int,
                                dimension: int,
                                polynomial_coeffs: np.ndarray,
                                t: float,
                                polynomial_evaluation: np.ndarray) -> None:
    """
    # NOTE: This function is just here because the C++ has it.
    # NOTE: But, this should be replaced with the more Pythonic version evaluate_polynomial()
    """
    raise unimplemented("Deprecated. Use evalute_polynomial() instead.")


def compute_polynomial_mapping_product(first_degree: int,
                                       second_degree: int,
                                       dimension: int,
                                       first_polynomial_coeffs_ref: np.ndarray,
                                       second_polynomial_coeffs_ref: np.ndarray) -> np.ndarray:
    """ 
    Generate the polynomial coefficients for the kronecker product of two
    polynomials of the same dimension.

    :param first_degree:  [in] maximum monomial degree of the first polynomial.
    :param second_degree: [in] maximum monomial degree of the second polynomial.
    :param dimension:     [in] polynomial dimension.
    :param first_polynomial_coeffs:  [in] coefficients of the first polynomial.
    :param second_polynomial_coeffs: [in] coefficients of the second polynomial.

    :return product_polynomial_coeffs: product polynomial coefficients. 
        shape = (first_degree + second_degree + 1, dimension)
    """
    # TODO: ask about the shape b/c originally I had something like the pseudocode below
    # assert first_polynomial_coeffs.shape == (first_degree + 1, dimension) OR (first_degree + 1,)
    first_polynomial_coeffs = first_polynomial_coeffs_ref
    second_polynomial_coeffs = second_polynomial_coeffs_ref

    if dimension == 1:
        first_polynomial_coeffs = first_polynomial_coeffs_ref.reshape((first_degree + 1, dimension))
        second_polynomial_coeffs = second_polynomial_coeffs_ref.reshape((second_degree + 1, dimension))

    assert first_polynomial_coeffs.shape == (first_degree + 1, dimension)
    assert second_polynomial_coeffs.shape == (second_degree + 1, dimension)

    # Compute the new polynomial coefficients by convolution.
    # product_polynomial_coeffs = np.convolve(
    #     first_polynomial_coeffs.flatten(), second_polynomial_coeffs.flatten())
    product_polynomial_coeffs = np.zeros(
        shape=(first_degree + second_degree + 1, dimension))

    for k in range(dimension):
        product_polynomial_coeffs[:, k] = np.convolve(
            first_polynomial_coeffs[:, k],
            second_polynomial_coeffs[:, k]
        )

    # TODO: Double checking that the shape is indeed what we expect.
    # XXX: Doesn't work with shape (n,)
    # assert product_polynomial_coeffs.shape == (
    #     first_degree + second_degree + 1, dimension)

    return product_polynomial_coeffs


def compute_polynomial_mapping_scalar_product(first_degree: int, second_degree: int, dimension: int,
                                              scalar_polynomial_coeffs: np.ndarray,
                                              polynomial_coeffs: np.ndarray) -> np.ndarray:
    """ 
    Generate the polynomial coefficients for the product of a
    scalar polynomial and a vector valued polynomial mapping.

    Args:
        first_degree: maximum monomial degree of the first polynomial.
        second_degree: maximum monomial degree of the second polynomial.
        dimension: polynomial mapping dimension.
        scalar_polynomial_coeffs: [in] coefficients of the scalar polynomial.
        polynomial_coeffs: [in] coefficients of the vector valued polynomial.

    Returns:
        product_polynomial_coeffs: [out] product vector valued polynomial mapping coefficients.

    """
    assert np.shape(scalar_polynomial_coeffs) == (first_degree + 1, 1)
    assert np.shape(polynomial_coeffs) == (second_degree + 1, dimension)
    # assert np.shape(product_polynomial_coeffs) == (
    # first_degree + second_degree + 1, dimension)

    # Compute the new polynomial mapping coefficients by convolution
    product_polynomial_coeffs = np.ndarray(
        shape=(first_degree + second_degree + 1, dimension))
    product_polynomial_coeffs[:, :] = 0

    # TODO: perhaps use NumPy vectorization to fix the issue of shape and whatnot...
    for i in range(first_degree + 1):
        for j in range(second_degree + 1):
            for k in range(dimension):
                # XXX: there may be problem with scalar_polynomial_coeffs accessing b/c of shape
                product_polynomial_coeffs[[i + j], k] += (
                    scalar_polynomial_coeffs[i] * polynomial_coeffs[j, k])

    return product_polynomial_coeffs


def compute_polynomial_mapping_cross_product(first_degree: int, second_degree: int,
                                             first_polynomial_coeffs: np.ndarray,
                                             second_polynomial_coeffs: np.ndarray) -> np.ndarray:
    """
    Generate the polynomial coefficients for the cross product of two
    vector valued polynomial mappings with range R^3.

    :param first_degree: [in] maximum monomial degree of the first polynomial.
    :param second_degree: [in] maximum monomial degree of the second polynomial.
    :param first_polynomial_coeffs: [in] coefficients of the first polynomial.
    :param second_polynomial_coeffs: [in] coefficients of the second polynomial.

    :return product_polynomial_coeffs: [out] product vector valued polynomial
        mapping coefficients. Shape = (first_degree + second_degree + 1, 3)
    """
    assert np.shape(first_polynomial_coeffs) == (first_degree + 1, 3)
    assert np.shape(second_polynomial_coeffs) == (second_degree + 1, 3)

    # Below lines of code retrieving particular columns of first_polynomial_coeffs and
    # second_polynomial_coeffs.
    # Column retrieval wrapped in [] to preserve shape of Matrices as (degree, dimension) in case #
    # where dimension is 1 (i.e. shape is (degree, 1))
    A0B1 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:, [0]],
                                              second_polynomial_coeffs[:, [1]])
    A0B2 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:, [0]],
                                              second_polynomial_coeffs[:, [2]])
    A1B0 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:, [1]],
                                              second_polynomial_coeffs[:, [0]])
    A1B2 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:, [1]],
                                              second_polynomial_coeffs[:, [2]])
    A2B0 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:, [2]],
                                              second_polynomial_coeffs[:, [0]])
    A2B1 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:, [2]],
                                              second_polynomial_coeffs[:, [1]])

    # Assemble the cross product from the terms
    # NOTE: must reshape for broadcasting to convert (3, 1) to (3,)
    product_polynomial_coeffs: MatrixNx3f = np.zeros(
        shape=(first_degree + second_degree + 1, 3), dtype=np.float64)
    product_polynomial_coeffs[:, 0] = A1B2.flatten() - A2B1.flatten()
    product_polynomial_coeffs[:, 1] = A2B0.flatten() - A0B2.flatten()
    product_polynomial_coeffs[:, 2] = A0B1.flatten() - A1B0.flatten()

    return product_polynomial_coeffs


def compute_polynomial_mapping_dot_product() -> None:
    """
    Method not used.
    """
    unimplemented()


def compute_polynomial_mapping_derivative(degree: int, dimension: int,
                                          polynomial_coeffs: MatrixXf) -> MatrixXf:
    """ 
    Generate the polynomial coefficients for the derivative of a polynomial mapping.

    NOTE: used by rational_function.py

    :param degree: [in] degree of the polynomial
    :param dimension: [in] polynomial mapping dimension.
    :param polynomial_coeffs: [in] coefficients of the polynomial mapping. 
    polynomial_coeffs shape (degree + 1, dimension)

    :return derivative_polynomial_coeffs: derivative polynomial mapping coefficients.
    derivative_polynomial_coeffs shape (degree, dimension)
    """
    assert polynomial_coeffs.shape == (degree + 1, dimension)

    # TODO: there may be a problem with shape...
    # TODO: Though, maybe this could be fixed with NumPy's vectorization!

    # Wait, isn't this just the same as NumPy's derivative thing?
    # for i in range(1, degree + 1):
    #     for j in range(dimension):
    #         derivative_polynomial_coeffs[i - 1,
    #                                      j] = i * polynomial_coeffs[i, j]

    # np_p_deriv_coeffs = np.polynomial.polynomial.polyder(
    # polynomial_coeffs.flatten())
    # nustuff = np.gradient(polynomial_coeffs, axis=0)
    # assert np.array_equal(derivative_polynomial_coeffs,
    #   nustuff)
    # print("what")

    # Below should be equivalent to whatever is happening above.
    derivative_polynomial_coeffs: MatrixXf = np.apply_along_axis(
        np.polynomial.polynomial.polyder, axis=0, arr=polynomial_coeffs)
    assert derivative_polynomial_coeffs.shape == (degree, dimension)

    return derivative_polynomial_coeffs


def quadratic_real_roots(quadratic_coeffs: Vector3f,
                         eps: float = 1e-10) -> tuple[Vector2f, int]:
    """ 
    Compute the real roots of a quadratic polynomial.

    :param quadratic_coeffs: [in] coefficients of the polynomial
    :param eps: [in] threshold for zero comparisons

    Return:
        solutions (list): [out] real roots of the polynomial
        num_solutions (int): [out] solution count
    """
    assert quadratic_coeffs.shape == (3, )

    discriminant: float
    solutions: Vector2f = np.ndarray(shape=(2, ))
    num_solutions: int

    if eps <= abs(quadratic_coeffs[2]):
        discriminant = (-4 * quadratic_coeffs[0] * quadratic_coeffs[2] +
                        quadratic_coeffs[1] * quadratic_coeffs[1])
        if eps * eps <= discriminant:
            if 0.0 < quadratic_coeffs[1]:
                solutions[0] = (2.0 * quadratic_coeffs[0] /
                                (-quadratic_coeffs[1] - math.sqrt(discriminant)))
                solutions[1] = ((-quadratic_coeffs[1] - math.sqrt(discriminant)) /
                                (2.0 * quadratic_coeffs[2]))
            else:
                solutions[0] = ((-quadratic_coeffs[1] + math.sqrt(discriminant)) /
                                (2.0 * quadratic_coeffs[2]))
                solutions[1] = (2.0 * quadratic_coeffs[0] /
                                (-quadratic_coeffs[1] + math.sqrt(discriminant)))
            num_solutions = 2
        elif 0.0 <= discriminant:
            solutions[0] = -quadratic_coeffs[1] / (2.0 * quadratic_coeffs[2])
            num_solutions = 1
        else:
            num_solutions = 0
    elif eps <= abs(quadratic_coeffs[1]):
        solutions[0] = -quadratic_coeffs[0] / quadratic_coeffs[1]
        num_solutions = 1
    else:
        num_solutions = 0

    # TODO: solutions should be size 2
    return (solutions, num_solutions)


def polynomial_real_roots(A_coeffs: Vector1D) -> Vector1D:
    """ Compute the real roots of a polynomial.

    :param A_coeffs: [in] coefficients of the polynomial
    :return: real roots of the polynomial
    """
    # Ensuring that A_coeffs is (n, ) shape array
    assert A_coeffs.ndim == 1
    logger.info("Full coefficient vector: %s", A_coeffs)
    reduced_coeffs: Vector1D = remove_polynomial_trailing_coefficients(A_coeffs)
    logger.info("Reduced coefficient vector: %s", reduced_coeffs)

    # Check if reduced coeff is 0
    if reduced_coeffs.size == 1 and float_equal_zero(reduced_coeffs[0]):
        roots: Vector1D = np.ndarray(shape=(0, 0))
        return roots

    # Compute the complex roots
    solver_roots: Vector1D = np.roots(reduced_coeffs)

    # Find the real roots (in the style of the C++ version)
    # XXX: Involves a floating point threshold test... well... the C++ version did.
    # Should only grab the true parts...
    # https://stackoverflow.com/questions/28081247/print-real-roots-only-in-numpy
    # TODO: utilize the tolerance value inside common.py
    real_roots: Vector1D = solver_roots.real[abs(solver_roots.imag) < 1e-10]
    logger.info("Real roots: %s", real_roots)

    return real_roots


def formatted_monomial(variable: str, degree: int) -> str:
    """
    Construct a formatted string for a variable raised to some power

    :param variable: [in] variable
    :param degree: [in] power to raise the variable to
    :return: formatted monomial string
    """
    # Handle degree 0 case
    if degree < 1:
        return ""

    # Format as "<variable>^<degree>"
    monomial_string: str = variable + "^" + str(degree)
    return monomial_string


def formatted_term(coefficient: float, variable: str, precision: int = 16) -> str:
    """ 
    Construct a formatted string for a term with given coefficient and variable.

    :param coefficient: [in] coefficient of the term
    :param variable: [in] variable of the term
    :param precision: [in] floating point precision
    :return: formatted term string
    """
    term_string: str = ""

    # Zero case
    if float_equal(coefficient, 0.0):
        return ""
    # Negative case
    elif coefficient < 0:
        term_string += f" - {coefficient:.{precision}f} {variable}"
    # Positive case
    else:
        term_string += f" + {coefficient:.{precision}f} {variable}"

    return term_string


def formatted_polynomial(degree: int, dimension: int,
                         polynomial_coeffs: MatrixXf,
                         precision: int = 16) -> str:
    """ 
    Construct a formatted string for a polynomial with given coefficients
    TODO: Implement separate method for polynomial mappings

    :param degree: TODO
    :param dimension: TODO
    :param P_coeffs: [in] coefficients of the polynomial
    :param precision: [in] floating point precision

    :return: formatted polynomial string
    """
    assert np.shape(polynomial_coeffs) == (degree + 1, dimension)

    # Handle trivial case
    if polynomial_coeffs.shape[1] == 0:
        return ""

    polynomial_string: str = ""

    # Going through polynomial_coeffs columns
    for i in range(polynomial_coeffs.shape[COLS]):
        polynomial_string += f"{polynomial_coeffs[0, i]}:.{precision}f"
        for j in range(polynomial_coeffs.shape[ROWS]):
            monomial_string: str = formatted_monomial("t", j)
            polynomial_string += formatted_term(
                polynomial_coeffs[j, i], monomial_string, precision)
        polynomial_string += "\n"

    return polynomial_string


def substitute_polynomial() -> None:
    """ 
    Method not used.
    """
    unimplemented()
