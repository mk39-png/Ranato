
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


# TODO: decide between SciPy and NumPy for implementing polynomials...


def generate_monomials(degree: int, t: float, T: np.ndarray) -> None:
    """Generate the row vector T of n + 1 monomials 1, t, ... , t^n.

    Args: 
        degree: maximum monomial degree
        t [in]: evaluation point for the monomials
        T [out]: row vector of monomials

    Return:
        None
    """
    assert np.shape(T) == (1, degree + 1)
    pass


def evaluate_polynomial(degree: int, polynomial_coeffs: np.ndarray, t: float) -> float:
    """Evaluate the polynomial with given coefficients at t.

    Args:
        degree: maximum monomial degree
        polynomial_coeffs [in]: coefficients of the polynomial
        t [in]: evaluation point for the polynomial

    Return:
        evaluation of the polynomial

    """
    assert np.shape(polynomial_coeffs) == (degree + 1, 1)
    pass


def evaluate_polynomial_mapping(degree: int, dimension: int, polynomial_coeffs: np.ndarray, t: float, polynomial_evaluation: np.ndarray) -> None:
    """Evaluate the polynomial with given coefficients at t.

    Args:
        degree: maximum monomial degree
        dimension: polynomial dimension
        polynomial_coeffs [in]: coefficients of the polynomial
        t [in]: evaluation point for the polynomial
        polynomial_evaluation [out]: evaluation of the polynomial

    Return:
        None
    """

    assert np.shape(polynomial_coeffs) == (degree + 1, dimension)
    assert np.shape(polynomial_evaluation) == (1, dimension)
    pass


def compute_polynomial_mapping_product(dimension: int, first_degree: int, second_degree: int, first_polynomial_coeffs: np.ndarray, second_polynomial_coeffs: np.ndarray, product_polynomial_coeffs: np.ndarray) -> None:
    """ Generate the polynomial coefficients for the kronecker product of two
        polynomials of the same dimension.

    Args:
        dimension: polynomial dimension
        first_degree: maximum monomial degree of the first polynomial
        second_degree: maximum monomial degree of the second polynomial
        first_polynomial_coeffs [in]: coefficients of the first polynomial
        second_polynomial_coeffs [in]: coefficients of the second polynomial
        product_polynomial_coeffs [out]: product polynomial coefficients

    Return:
        None

    """
    assert np.shape(first_polynomial_coeffs) == (first_degree + 1, dimension)
    assert np.shape(second_polynomial_coeffs) == (second_degree + 1, dimension)
    assert np.shape(product_polynomial_coeffs) == (
        first_degree + second_degree + 1, dimension)
    pass


def compute_polynomial_mapping_scalar_product(dimension: int, first_degree: int, second_degree: int, scalar_polynomial_coeffs: np.ndarray, polynomial_coeffs: np.ndarray, product_polynomial_coeffs: np.ndarray) -> None:
    """ Generate the polynomial coefficients for the product of a
        scalar polynomial and a vector valued polynomial mapping.

    Args:
        dimension: polynomial mapping dimension
        first_degree: maximum monomial degree of the first polynomial
        second_degree: maximum monomial degree of the second polynomial
        scalar_polynomial_coeffs [in]: coefficients of the scalar polynomial
        polynomial_coeffs [in]: coefficients of the vector valued polynomial
        product_polynomial_coeffs [out]: product vector valued polynomial mapping coefficients

    Return:
        None

    """
    assert np.shape(scalar_polynomial_coeffs) == (first_degree + 1, 1)
    assert np.shape(polynomial_coeffs) == (second_degree + 1, dimension)
    assert np.shape(product_polynomial_coeffs) == (
        first_degree + second_degree + 1, dimension)
    pass


def compute_polynomial_mapping_cross_product(first_degree: int, second_degree: int, scalar_polynomial_coeffs: np.ndarray, polynomial_coeffs: np.ndarray, product_polynomial_coeffs: np.ndarray):
    """ Generate the polynomial coefficients for the cross product of two
        vector valued polynomial mappings with range R^3.

    first_degree: maximum monomial degree of the first polynomial
    second_degree: maximum monomial degree of the second polynomial
    first_polynomial_coeffs: coefficients of the first polynomial
    second_polynomial_coeffs: coefficients of the second polynomial
    product_polynomial_coeffs [out]: product vector valued polynomial
    mapping coefficients
    """
    assert np.shape(scalar_polynomial_coeffs) == (first_degree + 1, 3)
    assert np.shape(polynomial_coeffs) == (second_degree + 1, 3)
    assert np.shape(product_polynomial_coeffs) == (
        first_degree + second_degree + 1, 3)
    pass


def compute_polynomial_mapping_dot_product(dimension: int, first_degree: int, second_degree: int, first_polynomial_coeffs: np.ndarray, second_polynomial_coeffs: np.ndarray, product_polynomial_coeffs: np.ndarray) -> None:
    """ Generate the polynomial coefficients for the cross product of two
        vector valued polynomial mappings with the same range.

    Args: 
        dimension: polynomial mapping dimension
        first_degree: maximum monomial degree of the first polynomial
        second_degree: maximum monomial degree of the second polynomial
        first_polynomial_coeffs: coefficients of the first polynomial
        second_polynomial_coeffs: coefficients of the second polynomial
        product_polynomial_coeffs [out]: product vector valued polynomial
        mapping coefficients

    Return:
        None
    """
    assert np.shape(first_polynomial_coeffs) == (first_degree + 1, dimension)
    assert np.shape(second_polynomial_coeffs) == (second_degree + 1, dimension)
    assert np.shape(product_polynomial_coeffs) == (
        first_degree + second_degree + 1, 1)

    pass


def compute_polynomial_mapping_derivative(degree: int, dimension: int, polynomial_coeffs: np.ndarray) -> None:
    """ Generate the polynomial coefficients for the derivative of a
        polynomial mapping.

    Args:
        degree:
        dimension:
        polynomial_coeffs: coefficients of the polynomial mapping
        derivative-polynomial_coeffs [out]: derivative polynomial mapping
        coefficients

    Return:
        None
    """
    assert np.shape(polynomial_coeffs) == (degree + 1, dimension)

    pass


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
def polynomial_real_roots(A_coeffs: np.ndarray) -> list:
    """ Compute the real roots of a polynomial.

    Args:
        A_coeffs: coefficients of the polynomial

    Return:
        real roots of the polynomial
    """
    # TODO: check that A_coeffs is 1D vector

    pass


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
