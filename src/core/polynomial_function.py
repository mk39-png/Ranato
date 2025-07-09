
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
"""

# TODO: only import the things needed from numpy... that being ndarray.
import numpy as np
from ..core.common import *
import logging

# TODO: decide between SciPy and NumPy for implementing polynomials...
logger = logging.getLogger(__name__)


def remove_polynomial_trailing_coefficients(A_coeffs: np.ndarray):
    """ Remove any zeros at the end of the polynomial coefficient vector

    Args:
        A_coeffs [in]
        reduced_coeffs [out]

    Return: 
        reference to trimmed A_coeffs with trailing zeros removed
    """

    # TODO: assert that A_coeffs is indeed a vector and NOT a matrix.
    assert A_coeffs.ndim == 1
    return np.trim_zeros(A_coeffs, 'b')


def generate_monomials(degree: int, t: float, T: np.ndarray) -> None:
    """ Generate the row vector T of n + 1 monomials 1, t, ... , t^n.

    Args:
        degree (int): maximum monomial degree.
        t (float): [in] evaluation point for the monomials.
        T (np.ndarray): [out] row vector of monomials.

    Returns:
        None
    """
    assert np.shape(T) == (degree + 1,)

    T[0] = 1.0

    for i in range(1, degree + 1):
        # XXX: again, issue with T being (1, 3) rather than (, 3) and that causing a bunch of issues.
        # XXX: As in, T not being a vector with 3 elements but rather some funky matrix.
        T[i] = T[i - 1] * t


def evaluate_polynomial(degree: int, polynomial_coeffs: np.ndarray, t: float) -> np.ndarray:
    """Evaluate the polynomial with given coefficients at t.

    Args:
        degree: maximum monomial degree.
        polynomial_coeffs: [in, const, reference] coefficients of the polynomial.
        t: [in] evaluation point for the polynomial.

    Return:
        evaluation of the polynomial
    """
    assert np.shape(polynomial_coeffs) == (degree + 1, 1)

    T = np.ndarray(shape=(1, degree + 1))
    generate_monomials(degree, t, T)

    # XXX: the below is supposed to do matrix multiplication, FYI
    return T @ polynomial_coeffs


def evaluate_polynomial_mapping(degree: int, dimension: int, polynomial_coeffs: np.ndarray, t: float, polynomial_evaluation: np.ndarray) -> None:
    """Evaluate the polynomial with given coefficients at t.

    Args:
        degree (int): maximum monomial degree.
        dimension (int): polynomial dimension.
        polynomial_coeffs (np.ndarray): [in] coefficients of the polynomial.
        t (float): [in] evaluation point for the polynomial.
        polynomial_evaluation (np.ndarray): [out] evaluation of the polynomial.

    Returns:
        None
    """
    assert polynomial_coeffs.shape == (degree + 1, dimension)
    assert polynomial_evaluation.shape == (dimension,)

    T = np.ndarray(shape=(1, degree + 1))
    generate_monomials(degree, t, T)

    polynomial_evaluation = T @ polynomial_coeffs


def compute_polynomial_mapping_product(first_degree: int, second_degree: int, dimension: int, first_polynomial_coeffs: np.ndarray, second_polynomial_coeffs: np.ndarray, product_polynomial_coeffs: np.ndarray) -> None:
    """ 
    Generate the polynomial coefficients for the kronecker product of two
    polynomials of the same dimension.

    Args:
        first_degree: maximum monomial degree of the first polynomial.
        second_degree: maximum monomial degree of the second polynomial.
        dimension: polynomial dimension.
        first_polynomial_coeffs: [in] coefficients of the first polynomial.
        second_polynomial_coeffs: [in] coefficients of the second polynomial.
        product_polynomial_coeffs: [out] product polynomial coefficients.

    Returns:
        None

    """
    # TODO: ask about the shape b/c originally I had something like the pseudocode below
    # assert first_polynomial_coeffs.shape == (first_degree + 1, dimension) OR (first_degree + 1,)
    assert first_polynomial_coeffs.shape == (first_degree + 1, dimension)
    assert second_polynomial_coeffs.shape == (second_degree + 1, dimension)
    assert product_polynomial_coeffs.shape == (
        first_degree + second_degree + 1, dimension)

    # Compute the new polynomial coefficients by convolution
    # NOTE: must set all elements of original NumPy array to 0 rather than creating .zeros_like because Python handles references differently from C++
    product_polynomial_coeffs[:, :] = 0

    # XXX: This differs from C++ code in that it is not <= first_degree...
    for i in range(first_degree+1):
        for j in range(second_degree+1):
            for k in range(dimension):
                product_polynomial_coeffs[i + j, k] += first_polynomial_coeffs[i,
                                                                               k] * second_polynomial_coeffs[j, k]


def compute_polynomial_mapping_scalar_product(first_degree: int, second_degree: int, dimension: int, scalar_polynomial_coeffs: np.ndarray, polynomial_coeffs: np.ndarray, product_polynomial_coeffs: np.ndarray) -> None:
    """ 
    Generate the polynomial coefficients for the product of a
    scalar polynomial and a vector valued polynomial mapping.

    Args:
        first_degree: maximum monomial degree of the first polynomial.
        second_degree: maximum monomial degree of the second polynomial.
        dimension: polynomial mapping dimension.
        scalar_polynomial_coeffs: [in] coefficients of the scalar polynomial.
        polynomial_coeffs: [in] coefficients of the vector valued polynomial.
        product_polynomial_coeffs: [out] product vector valued polynomial mapping coefficients.

    Returns:
        None

    """
    assert np.shape(scalar_polynomial_coeffs) == (first_degree + 1, 1)
    assert np.shape(polynomial_coeffs) == (second_degree + 1, dimension)
    assert np.shape(product_polynomial_coeffs) == (
        first_degree + second_degree + 1, dimension)

    # Compute the new polynomial mapping coefficients by convolution
    product_polynomial_coeffs[:, :] = 0

    for i in range(first_degree + 1):
        for j in range(second_degree + 1):
            for k in range(dimension):
                # XXX: there may be problem with scalar_polynomial_coeffs accessing b/c of shape
                product_polynomial_coeffs[i + j,
                                          k] += scalar_polynomial_coeffs[i] * polynomial_coeffs[j, k]


def compute_polynomial_mapping_cross_product(first_degree: int, second_degree: int, first_polynomial_coeffs: np.ndarray, second_polynomial_coeffs: np.ndarray, product_polynomial_coeffs: np.ndarray):
    """
    Generate the polynomial coefficients for the cross product of two
    vector valued polynomial mappings with range R^3.

    Args:
        first_degree: maximum monomial degree of the first polynomial.
        second_degree: maximum monomial degree of the second polynomial.
        first_polynomial_coeffs: coefficients of the first polynomial.
        second_polynomial_coeffs: coefficients of the second polynomial.
        product_polynomial_coeffs [out]: product vector valued polynomial
        mapping coefficients.

    Returns:
        None
    """
    assert np.shape(first_polynomial_coeffs) == (first_degree + 1, 3)
    assert np.shape(second_polynomial_coeffs) == (second_degree + 1, 3)
    assert np.shape(product_polynomial_coeffs) == (
        first_degree + second_degree + 1, 3)

    shape = (first_degree + second_degree + 1, 1)

    A0B1 = np.ndarray(shape)
    A0B2 = np.ndarray(shape)
    A1B0 = np.ndarray(shape)
    A1B2 = np.ndarray(shape)
    A2B0 = np.ndarray(shape)
    A2B1 = np.ndarray(shape)

    # Below lines of code retrieving particular columns of first_polynomial_coeffs and second_polynomial_coeffs
    compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                       first_polynomial_coeffs[:, 0].reshape(-1, 1), second_polynomial_coeffs[:, 1].reshape(-1, 1), A0B1)
    compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                       first_polynomial_coeffs[:, 0].reshape(-1, 1), second_polynomial_coeffs[:, 2].reshape(-1, 1), A0B2)
    compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                       first_polynomial_coeffs[:, 1].reshape(-1, 1), second_polynomial_coeffs[:, 0].reshape(-1, 1), A1B0)
    compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                       first_polynomial_coeffs[:, 1].reshape(-1, 1), second_polynomial_coeffs[:, 2].reshape(-1, 1), A1B2)
    compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                       first_polynomial_coeffs[:, 2].reshape(-1, 1), second_polynomial_coeffs[:, 0].reshape(-1, 1), A2B0)
    compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                       first_polynomial_coeffs[:, 2].reshape(-1, 1), second_polynomial_coeffs[:, 1].reshape(-1, 1), A2B1)

    # Assemble the cross product from the terms
    # NOTE: must reshape for broadcasting to convert (3, 1) to (3,)
    # TODO: or just use 3,1
    # TODO: bpy wrapper class perhaps?"
    # TODO: Add a wrapper!! Like, for .col() and whatnot.
    product_polynomial_coeffs[:, 0] = A1B2.reshape(-1) - A2B1.reshape(-1)
    product_polynomial_coeffs[:, 1] = A2B0.reshape(-1) - A0B2.reshape(-1)
    product_polynomial_coeffs[:, 2] = A0B1.reshape(-1) - A1B0.reshape(-1)

    # TODO: return product_polynomial_coeffs to be more pythonic! (and update type sig)


def compute_polynomial_mapping_dot_product(dimension: int, first_degree: int, second_degree: int, first_polynomial_coeffs: np.ndarray, second_polynomial_coeffs: np.ndarray, product_polynomial_coeffs: np.ndarray) -> None:
    """
    Generate the polynomial coefficients for the cross product of two
    vector valued polynomial mappings with the same range.

    Args:
        dimension (int): polynomial mapping dimension.
        first_degree (int): maximum monomial degree of the first polynomial.
        second_degree (int): maximum monomial degree of the second polynomial.
        first_polynomial_coeffs (np.ndarray): coefficients of the first polynomial.
        second_polynomial_coeffs (np.ndarray): coefficients of the second polynomial.
        product_polynomial_coeffs (np.ndarray): [out] product vector valued polynomial mapping coefficients.

    Return:
        None
    """
    assert np.shape(first_polynomial_coeffs) == (first_degree + 1, dimension)
    assert np.shape(second_polynomial_coeffs) == (second_degree + 1, dimension)
    assert np.shape(product_polynomial_coeffs) == (
        first_degree + second_degree + 1, 1)

    pass


# NOTE: this is used by rational_function.py
def compute_polynomial_mapping_derivative(degree: int, dimension: int, polynomial_coeffs: np.ndarray, derivative_polynomial_coeffs: np.ndarray) -> None:
    """ 
    Generate the polynomial coefficients for the derivative of a
    polynomial mapping.

    Args:
        degree: PLACEHOLDER.
        dimension: polynomial mapping dimension.
        polynomial_coeffs: coefficients of the polynomial mapping.
        derivative_polynomial_coeffs: [out] derivative polynomial mapping coefficients.

    Returns:
        None
    """
    assert polynomial_coeffs.shape == (degree + 1, dimension)
    assert derivative_polynomial_coeffs.shape == (degree, dimension)

    # TODO: there may be a problem with shape...
    for i in range(1, degree + 1):
        for j in range(dimension):
            derivative_polynomial_coeffs[i - 1,
                                         j] = i * polynomial_coeffs[i, j]


def quadratic_real_roots(quadratic_coeffs: np.ndarray, solutions: list, num_solutions: int, eps: float = 1e-10) -> None:
    """ Compute the real roots of a quadratic polynomial.

    Args:
        quadratic_coeffs: coefficients of the polynomial
        solutions [out]: real roots of the polynomial
        num_solutions [out]: solution count
        eps: threshold for zero comparisons

    Return:
        None
    """
    assert np.shape(quadratic_coeffs) == (3, 1)

    pass


# TODO: maybe just add a type alias for A_coeffs to be... if that's supported in Python 3.11
def polynomial_real_roots(A_coeffs: np.ndarray) -> np.ndarray:
    """ Compute the real roots of a polynomial.

    Args:
        A_coeffs: coefficients of the polynomial

    Return:
        real roots of the polynomial
    """
    # TODO: check that A_coeffs is 1D vector

    logger.info("Full coefficient vector: %s", A_coeffs)

    # TODO: removes trailing coefficints by refernece... so A_coeffs is the one that has been modified!
    reduced_coeffs: np.ndarray = remove_polynomial_trailing_coefficients(
        A_coeffs)
    logger.info("Reduced coefficient vector: %s", reduced_coeffs)

    # TODO: whenever vector<double> used in C++ code... use NumPy ndarray???
    # roots should be the same size as reduced_coeffs... or at least allocated to have the same space...
    roots: np.ndarray

    # TODO: fix this by returning NumPy ndarray
    # check if reduced coeff is 0
    if (reduced_coeffs.size == 1 and float_equal_zero(reduced_coeffs[0])):
        return roots

    # Compute the complex roots
    solver_roots = np.roots(reduced_coeffs)

    # Find the real roots (in the style of the C++ version)
    # XXX: Involves a floating point threshold test... well... the C++ version did.
    # Should only grab the true parts...
    # https://stackoverflow.com/questions/28081247/print-real-roots-only-in-numpy
    # TODO: double check w/ prof if it's OK to go about using NumPy's float checker rather than using the float checker in common.py...
    # Because that would be redundant systems that would add bloat?
    # TODO: utilize the tolerance value inside common.py
    real_roots = solver_roots.real[abs(solver_roots.imag) < 1e-10]

    logger.info("Real roots: %s", real_roots)
    return real_roots


def formatted_monomial(variable: str, degree: int) -> str:
    """ Construct a formatted string for a variable raised to some power

    Args: 
        variable: variable
        degree: power to raise the variable to

    Return:
        formatted monomial string
    """
    pass


def formatted_term(coefficient: float, variable: str, precision: int = 16) -> str:
    """ Construct a formatted string for a term with given coefficient
        and variable.

    Args:
        coefficient: coefficient of the term
        variable: variable of the term
        precision: floating point precision

    Return:
        formatted term string
    """
    pass


# TODO: can't I put the datatypes inside polynomial_coeffs as well?
def formatted_polynomial(degree: int, dimension: int, polynomial_coeffs: np.ndarray, precision: int = 16) -> str:
    """ Construct a formatted string for a polynomial with given coefficients
        TODO: Implement separate method for polynomial mappings

    Args:
        degree:
        dimension:
        P_coeffs: coefficients of the polynomial
        precision: floating point precision

    Return:
        formatted polynomial string
    """

    assert np.shape(polynomial_coeffs) == (degree + 1, dimension)
    pass


def substitute_polynomial(A_coeffs: np.ndarray, t) -> any:
    """ Substitute some variable value that supports addition,
        multiplication, and double multiplication into a polynomial.

        Abstractly, this represents the polynomial evaluation homomorphism between
        R[X] and some R-Algebra S.

    Args:  
        VariableType: any sort of input... TODO: fix the translation from C++ to Python w/ this variable...
        A_coeffs: coefficients of the polynomial function
        t: evaluation point

    Return:
        evaluated polynomial value
    """
    # TODO: assert that A_coeffs is a vector (i.e. is 1D)

    pass
