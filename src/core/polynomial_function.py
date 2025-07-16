
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

# TODO: only import the things needed from numpy... that being ndarray.
# import numpy as np

# https://medium.com/@goldengrisha/using-numpy-typing-for-type-safe-list-handling-in-python-35f8c99c76ac
# import numpy.typing as npt
from ..core.common import *


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


def generate_monomials(degree: int, t: float) -> npt.NDArray[np.float64]:
    """ Generate the row vector T of n + 1 monomials 1, t, ... , t^n.

    Args:
        degree (int): maximum monomial degree.
        t (float): [in] evaluation point for the monomials.


    Returns:
        T (np.ndarray): [out] row vector of monomials of shape (1, degree + 1).
    """
    # assert T.shape == (degree + 1,)
    T: np.ndarray = np.ndarray(shape=(1, degree + 1))

    T[0] = 1.0

    # NOTE: In the original ASOC code, T is a Matrix with 1 row and degree + 1 columns.
    # FIXME: deal with 1D things... You know?
    for i in range(1, degree + 1):
        T[0, i] = T[0, i - 1] * t

    return T


def evaluate_polynomial(degree: int, dimension: int, polynomial_coeffs: np.ndarray, t: float) -> np.ndarray:
    """
    Evaluate the polynomial with given coefficients at t. 
    NOTE: this has been modified from the ASOC code to support any dimension.

    Args:
        degree: maximum monomial degree.
        dimension (int): polynomial dimension.
        polynomial_coeffs: [in, const, reference] coefficients of the polynomial.
        t: [in] evaluation point for the polynomial.

    Returns:
        polynomial_evaluation: evaluation of the polynomial of shape (1, dimension)
    """

    # TODO: change assertions to do NumPy-esque checks.
    assert polynomial_coeffs.shape == (
        degree + 1, dimension), "polynomial_coeffs supposed to be matrix (i.e. ndim > 1 OR shape != (degree+1,))"

    T = generate_monomials(degree, t)
    assert T.shape == (1, degree + 1)

    polynomial_evaluation = T @ polynomial_coeffs

    # HACK: I removed this assertion since it was messing up F_derivative(-1.0)[0] in "test_zero_function()" case...
    # assert polynomial_evaluation.shape == (1, dimension)

    # XXX: the below is supposed to do matrix multiplication, FYI
    return polynomial_evaluation


def evaluate_polynomial_mapping(degree: int, dimension: int, polynomial_coeffs: np.ndarray, t: float, polynomial_evaluation: np.ndarray) -> None:
    # NOTE: This function is just here because the C++ has it.
    # NOTE: But, this should be replaced with the more Pythonic version evaluate_polynomial()
    raise Exception("Deprecated. Use evalute_polynomial() instead.")
    pass


def compute_polynomial_mapping_product(first_degree: int, second_degree: int, dimension: int,
                                       first_polynomial_coeffs: np.ndarray,
                                       second_polynomial_coeffs: np.ndarray) -> np.ndarray:
    """ 
    Generate the polynomial coefficients for the kronecker product of two
    polynomials of the same dimension.

    Args:
        first_degree: maximum monomial degree of the first polynomial.
        second_degree: maximum monomial degree of the second polynomial.
        dimension: polynomial dimension.
        first_polynomial_coeffs: [in] coefficients of the first polynomial.
        second_polynomial_coeffs: [in] coefficients of the second polynomial.

    Returns:
        product_polynomial_coeffs: [out] product polynomial coefficients. shape = (first_degree + second_degree + 1, dimension)
    """
    # TODO: ask about the shape b/c originally I had something like the pseudocode below
    # assert first_polynomial_coeffs.shape == (first_degree + 1, dimension) OR (first_degree + 1,)
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
    # product_polynomial_coeffs = np.convolve(
    #     first_polynomial_coeffs, second_polynomial_coeffs)

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
                product_polynomial_coeffs[[i + j],
                                          k] += scalar_polynomial_coeffs[i] * polynomial_coeffs[j, k]

    return product_polynomial_coeffs


def compute_polynomial_mapping_cross_product(first_degree: int, second_degree: int,
                                             first_polynomial_coeffs: np.ndarray,
                                             second_polynomial_coeffs: np.ndarray) -> np.ndarray:
    """
    Generate the polynomial coefficients for the cross product of two
    vector valued polynomial mappings with range R^3.

    Args:
        first_degree: maximum monomial degree of the first polynomial.
        second_degree: maximum monomial degree of the second polynomial.
        first_polynomial_coeffs: coefficients of the first polynomial.
        second_polynomial_coeffs: coefficients of the second polynomial.

    Returns:
        product_polynomial_coeffs [out]: product vector valued polynomial
        mapping coefficients. Shape = (first_degree + second_degree + 1, 3)
    """
    assert np.shape(first_polynomial_coeffs) == (first_degree + 1, 3)
    assert np.shape(second_polynomial_coeffs) == (second_degree + 1, 3)

    # NOTE: reshape(-1, 1) is to wrap the slicing inside an array so that they are not Vectors and
    # remain Matrices of shape (degree, 1).
    # Below lines of code retrieving particular columns of first_polynomial_coeffs and
    # second_polynomial_coeffs.

    # TODO: change reshape to use []
    A0B1 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:,
                                                                      0].reshape(-1, 1),
                                              second_polynomial_coeffs[:, 1].reshape(-1, 1))
    A0B2 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:,
                                                                      0].reshape(-1, 1),
                                              second_polynomial_coeffs[:, 2].reshape(-1, 1))
    A1B0 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:,
                                                                      1].reshape(-1, 1),
                                              second_polynomial_coeffs[:, 0].reshape(-1, 1))
    A1B2 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:,
                                                                      1].reshape(-1, 1),
                                              second_polynomial_coeffs[:, 2].reshape(-1, 1))
    A2B0 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:,
                                                                      2].reshape(-1, 1),
                                              second_polynomial_coeffs[:, 0].reshape(-1, 1))
    A2B1 = compute_polynomial_mapping_product(first_degree, second_degree, 1,
                                              first_polynomial_coeffs[:,
                                                                      2].reshape(-1, 1),
                                              second_polynomial_coeffs[:, 1].reshape(-1, 1))

    # assert A0B1.shape == (first_degree + second_degree + 1, 1)
    # assert A0B1.shape == (first_degree + second_degree + 1, 1)
    # assert A0B2.shape == (first_degree + second_degree + 1, 1)
    # assert A1B0.shape == (first_degree + second_degree + 1, 1)
    # assert A1B2.shape == (first_degree + second_degree + 1, 1)
    # assert A2B0.shape == (first_degree + second_degree + 1, 1)
    # assert A2B1.shape == (first_degree + second_degree + 1, 1)

    # Assemble the cross product from the terms
    # NOTE: must reshape for broadcasting to convert (3, 1) to (3,)
    # TODO: or just use 3,1
    # TODO: bpy wrapper class perhaps?"
    # TODO: Add a wrapper!! Like, for .col() and whatnot.
    product_polynomial_coeffs = np.ndarray(
        shape=(first_degree + second_degree + 1, 3))
    product_polynomial_coeffs[:, 0] = A1B2.reshape(-1) - A2B1.reshape(-1)
    product_polynomial_coeffs[:, 1] = A2B0.reshape(-1) - A0B2.reshape(-1)
    product_polynomial_coeffs[:, 2] = A0B1.reshape(-1) - A1B0.reshape(-1)

    return product_polynomial_coeffs


def compute_polynomial_mapping_dot_product(dimension: int, first_degree: int, second_degree: int,
                                           first_polynomial_coeffs: np.ndarray,
                                           second_polynomial_coeffs: np.ndarray) -> np.ndarray:
    """
    Generate the polynomial coefficients for the cross product of two
    vector valued polynomial mappings with the same range.

    Args:
        dimension (int): polynomial mapping dimension.
        first_degree (int): maximum monomial degree of the first polynomial.
        second_degree (int): maximum monomial degree of the second polynomial.
        first_polynomial_coeffs (np.ndarray): coefficients of the first polynomial.
        second_polynomial_coeffs (np.ndarray): coefficients of the second polynomial.

    Returns:
        product_polynomial_coeffs (np.ndarray): [out] product vector valued polynomial 
                                                mapping coefficients.
    """
    assert np.shape(first_polynomial_coeffs) == (first_degree + 1, dimension)
    assert np.shape(second_polynomial_coeffs) == (second_degree + 1, dimension)
    # assert np.shape(product_polynomial_coeffs) == (
    #     first_degree + second_degree + 1, 1)

    pass


# NOTE: this is used by rational_function.py
def compute_polynomial_mapping_derivative(degree: int, dimension: int,
                                          polynomial_coeffs: np.ndarray) -> np.ndarray:
    """ 
    Generate the polynomial coefficients for the derivative of a
    polynomial mapping.

    Args:
        degree: PLACEHOLDER.
        dimension: polynomial mapping dimension.
        polynomial_coeffs: coefficients of the polynomial mapping.

    Returns:
        derivative_polynomial_coeffs: [out] derivative polynomial mapping coefficients.

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
    derivative_polynomial_coeffs = np.apply_along_axis(
        np.polynomial.polynomial.polyder, axis=0, arr=polynomial_coeffs)
    assert derivative_polynomial_coeffs.shape == (degree, dimension)

    return derivative_polynomial_coeffs


def quadratic_real_roots(quadratic_coeffs: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, int]:
    """ Compute the real roots of a quadratic polynomial.

    Args:
        quadratic_coeffs: coefficients of the polynomial
        eps: threshold for zero comparisons

    Return:
        solutions (list): [out] real roots of the polynomial
        num_solutions (int): [out] solution count
    """
    assert quadratic_coeffs.shape == (3,)

    discriminant: float
    solutions: np.ndarray = np.ndarray(shape=(2,))
    num_solutions: int

    if (eps <= abs(quadratic_coeffs[2])):
        discriminant = -4 * \
            quadratic_coeffs[0] * quadratic_coeffs[2] + \
            quadratic_coeffs[1] * quadratic_coeffs[1]
        if (eps * eps <= discriminant):
            if (0.0 < quadratic_coeffs[1]):
                solutions[0] = 2.0 * quadratic_coeffs[0] / \
                    (-quadratic_coeffs[1] - math.sqrt(discriminant))
                solutions[1] = (-quadratic_coeffs[1] -
                                math.sqrt(discriminant)) / (2.0 * quadratic_coeffs[2])
            else:
                solutions[0] = (-quadratic_coeffs[1] +
                                math.sqrt(discriminant)) / (2.0 * quadratic_coeffs[2])
                solutions[1] = 2.0 * quadratic_coeffs[0] / \
                    (-quadratic_coeffs[1] + math.sqrt(discriminant))
            num_solutions = 2
        elif (0.0 <= discriminant):
            solutions[0] = -quadratic_coeffs[1] / (2.0 * quadratic_coeffs[2])
            num_solutions = 1
        else:
            num_solutions = 0
    elif (eps <= abs(quadratic_coeffs[1])):
        solutions[0] = -quadratic_coeffs[0] / quadratic_coeffs[1]
        num_solutions = 1
    else:
        num_solutions = 0

    # TODO: solutions should be size 2
    return (solutions, num_solutions)


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
    # Handle degree 0 case
    if (degree < 1):
        return ""

    # Format as "<variable>^<degree>"
    monomial_string = (variable + "^" + str(degree))
    return monomial_string


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


# TODO: can't I put the datatypes inside polynomial_coeffs as well?
# TODO: just use degree and dimension from polynomial_coeffs shape... where (n, m) degree = n and dimension = m
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
    # assert polynomial_coeffs.dtype = np.float64

    # Handle trivial case
    if polynomial_coeffs.shape[1] == 0:
        return ""

    polynomial_string: str = ""

    # Going through polynomial_coeffs columns
    for i in polynomial_coeffs.shape[1]:
        polynomial_string += f"{polynomial_coeffs[0, i]}:.{precision}f"
        for j in polynomial_coeffs.shape[0]:
            monomial_string = formatted_monomial("t", j)
            polynomial_string += formatted_term(
                polynomial_coeffs[j, i], monomial_string, precision)
        polynomial_string += "\n"

    return polynomial_string


# TODO: this doesn't appear anywhere else...
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
