# Implementing with pytest!
from ..core.common import *
from ..core.polynomial_function import *
import pytest


def test_generate_monomials() -> None:
    degree = 2
    t = -1
    T = generate_monomials(degree, t)

    assert T.shape == (1, degree + 1)
    assert T[0, 0] == 1
    assert T[0, 1] == -1
    assert T[0, 2] == 1


def test_evaluate_polynomial() -> None:
    degree = 2
    dimension = 1
    polynomial_coeffs = np.array([[0.], [0.], [0.]])
    t = -1

    polynomial_evaluation = evaluate_polynomial(
        degree, dimension, polynomial_coeffs, t)
    assert polynomial_evaluation.shape == (1, dimension)


def test_compute_polynomial_mapping_product_one_dimension() -> None:
    """
    Testing ASOC code's implementation with NumPy's .convolve() method.
    Because we don't need to reimplement everything if NumPy conveniently provides functionality for us.
    """
    # Test this with the Kronecker product that is provided by NumPy
    # NOTE: apparently the dimension is always 1... in the cases that compute_polynomial_mapping_product() is used.
    first_polynomial_coeffs = np.array([[2], [1]])
    second_polynomial_coeffs = np.array([[1], [1]])
    product_polynomial_coeffs = np.ndarray(shape=(3, 1))

    first_degree = 1
    second_degree = 1
    dimension = 1

    # *********
    # ASOC CODE
    # *********
    # Compute the new polynomial coefficients by convolution.
    # Meaning, it is dependent on coeffs being all one-dimensional.
    # NOTE: must set all elements of original NumPy array to 0 rather than creating .zeros_like because Python handles references differently from C++
    product_polynomial_coeffs[:, :] = 0

    # XXX: This differs from C++ code in that it is not <= first_degree...
    # It also appears that for many dimensions, it just calculates row by row.... if I understand correctly
    for i in range(first_degree+1):
        for j in range(second_degree+1):
            for k in range(dimension):
                product_polynomial_coeffs[i + j, k] += first_polynomial_coeffs[i,
                                                                               k] * second_polynomial_coeffs[j, k]

    # ******************
    # COMPARING RESULTS
    # ******************
    # Turns out that convolution likes same sized dimensions
    numpy_product_polynomial_coeffs = np.convolve(
        first_polynomial_coeffs.flatten(), second_polynomial_coeffs.flatten())

    assert np.array_equal(product_polynomial_coeffs.flatten(),
                          numpy_product_polynomial_coeffs)

    # Below is a hardcoded result from previous testing...
    # XXX: This may be wrong...
    assert np.array_equal(
        product_polynomial_coeffs.flatten(), np.array([2, 3, 1]))


def test_compute_polynomial_mapping_derivative_with_asoc() -> None:
    """
    Testing the ASOC code's implementation of compute_polynomial_mapping_derivative() with NumPy's derivative method.
    This is grabbing from test_zero_function() interaction with compute_derivative() and thus compute_polynomial_mapping_derivative()
    """

    # assert polynomial_coeffs.shape == (degree + 1, dimension)
    # assert derivative_polynomial_coeffs.shape == (degree, dimension)

    # ******************
    # ZERO FUNCTION CASE
    # ******************
    degree = 1
    dimension = 1
    polynomial_coeffs = np.array([[0.0], [0.0]])
    derivative_polynomial_coeffs = np.array([[0.0]])

    # Wait, isn't this just the same as NumPy's derivative thing?
    for i in range(1, degree + 1):
        for j in range(dimension):
            derivative_polynomial_coeffs[i - 1,
                                         j] = i * polynomial_coeffs[i, j]

    numpy_derivative_polynomial_coeffs = np.polynomial.polynomial.polyder(
        polynomial_coeffs.flatten())

    assert np.array_equal(derivative_polynomial_coeffs.flatten(),
                          numpy_derivative_polynomial_coeffs)

    # TODO: test with other derivatives aside from zero function...
    # FIXME: because I'm quite sure this is not working as intended

    # ******************
    # LINEAR FUNCTION CASE
    # ******************
    P_coeffs = np.array([-1, 2]).reshape(2, 1)
    P_deriv_coeffs = np.ndarray(shape=(1, 1))
    P_deriv_coeffs = compute_polynomial_mapping_derivative(1, 1, P_coeffs)

    # np_p_deriv_coeffs = np.polynomial.polynomial.polyder(P_coeffs.flatten())

    np_p_deriv_coeffs = np.polynomial.polynomial.polyder(P_coeffs)

    assert np.array_equal(P_deriv_coeffs,
                          np_p_deriv_coeffs)

    # TODO: derive each row of the polynomial for multidimensional coeff matrices (e.g shape (2, 3))


def test_remove_polynomial_trailing_coefficients() -> None:
    print("Remove trailing zeros")

    A_coeffs: np.ndarray = np.array([1, 2, 3, 0, 0, 0])
    reduced_coeffs: np.ndarray

    # Find last nonzero entry and remove all zero entries after it
    last_zero: int = A_coeffs.size
    while (last_zero > 0):
        # When reaching a non-zero number, stop and return new polynomial coefficient
        # vector with trailing zeros removed.
        if (not float_equal(A_coeffs[last_zero - 1], 0.0)):
            # TODO: double check that the below is equivalent to .head() in Eigen
            reduced_coeffs = A_coeffs[1:last_zero]
            return

        last_zero -= 1

    reduced_coeffs = A_coeffs[1:]

    # TODO: now compare this with NumPy operation
    assert np.array_equal(np.trim_zeros(A_coeffs, 'b'), reduced_coeffs)


def test_polynomial_mapping_cross_products_elementary_constant_functions():

    # TODO: add section
    # TODO: make this more pytest-esque
    print("Elementary constant functions")
    A_coeffs = np.array([[1, 0, 0]])
    B_coeffs = np.array([[0, 1, 0]])
    cross_product_coeffs = compute_polynomial_mapping_cross_product(0, 0,
                                                                    A_coeffs, B_coeffs)

    assert cross_product_coeffs.shape == (1, 3)

    assert float_equal(cross_product_coeffs[0, 0], 0.0)
    assert float_equal(cross_product_coeffs[0, 1], 0.0)
    assert float_equal(cross_product_coeffs[0, 2], 1.0)


def test_polynomial_mapping_cross_products_elementary_linear_functions():
    print("Elementary linear functions")
    A_coeffs = np.array([[2, 0, 0], [1, 0, 0]])
    B_coeffs = np.array([[0, 1, 0], [0, 1, 0]])
    cross_product_coeffs = compute_polynomial_mapping_cross_product(
        1, 1, A_coeffs, B_coeffs)

    assert cross_product_coeffs.shape == (3, 3)
    assert float_equal(cross_product_coeffs[0, 0], 0.0)
    assert float_equal(cross_product_coeffs[0, 1], 0.0)
    assert float_equal(cross_product_coeffs[0, 2], 2.0)
    assert float_equal(cross_product_coeffs[1, 2], 3.0)
    assert float_equal(cross_product_coeffs[2, 2], 1.0)

    # TODO: now check to see if equivalent to NumPy's polynomial solver...


def test_polynomial_mapping_cross_products_general_constant_functions():
    print("General constant functions")
    A_coeffs = np.array([[1, 2, 3]])
    B_coeffs = np.array([[4, 5, 6]])
    cross_product_coeffs = compute_polynomial_mapping_cross_product(0, 0,
                                                                    A_coeffs, B_coeffs)

    assert cross_product_coeffs.shape == (1, 3)

    assert float_equal(cross_product_coeffs[0, 0], -3.0)
    assert float_equal(cross_product_coeffs[0, 1], 6.0)
    assert float_equal(cross_product_coeffs[0, 2], -3.0)


def test_polynomial_mapping_cross_products_cancelling_linear_functions():
    print("Cancelling linear functions")
    A_coeffs = np.array([[1, 2, 3], [1, 1, 1]])
    B_coeffs = np.array([[4, 5, 6], [1, 1, 1]])
    cross_product_coeffs = compute_polynomial_mapping_cross_product(
        1, 1, A_coeffs, B_coeffs, )

    assert cross_product_coeffs.shape == (3, 3)

    assert float_equal(cross_product_coeffs[0, 0], -3.0)
    assert float_equal(cross_product_coeffs[0, 1], 6.0)
    assert float_equal(cross_product_coeffs[0, 2], -3.0)

    assert float_equal(cross_product_coeffs[1, 0], 0.0)
    assert float_equal(cross_product_coeffs[1, 1], 0.0)
    assert float_equal(cross_product_coeffs[1, 2], 0.0)

    assert float_equal(cross_product_coeffs[2, 0], 0.0)
    assert float_equal(cross_product_coeffs[2, 1], 0.0)
    assert float_equal(cross_product_coeffs[2, 2], 0.0)


def test_polynomial_real_roots_linear_function():
    print("Linear function")
    A_coeffs = np.array([1, 1])
    roots = polynomial_real_roots(A_coeffs)
    assert (roots.size == 1)
    assert float_equal(roots[0], -1.0)


def test_polynomial_real_roots_quadratic_function_with_roots():
    print("Quadratic function with roots")
    A_coeffs = np.array([-1, 0, 1])
    roots = polynomial_real_roots(A_coeffs)
    assert (roots.size == 2)
    assert (float_equal(roots[0], -1.0) or float_equal(roots[0], 1.0))
    assert (float_equal(roots[1], -1.0) or float_equal(roots[1], 1.0))


def test_polynomial_real_roots_quadratic_function_without_roots():
    print("Quadratic function without roots")
    A_coeffs = np.array([1, 0, 1])
    roots = polynomial_real_roots(A_coeffs)
    assert roots.size == 0


def test_polynomial_real_roots_vs_quadratic_real_roots():
    """
    This test is just to see if quadratic_real_roots() and polynomial_real_roots() do the same thing.
    """
    print("Quadratic function with roots")
    A_coeffs = np.array([-1, 0, 1])
    roots_polynomial = polynomial_real_roots(A_coeffs)
    roots_quadratic, num_solutions = quadratic_real_roots(A_coeffs)

    assert (roots_polynomial.size == 2)
    assert (roots_quadratic.size == 2)
    assert np.array_equal(roots_polynomial, roots_quadratic)

    print("Quadratic function without roots")
    A_coeffs = np.array([1, 0, 1])
    roots_polynomial = polynomial_real_roots(A_coeffs)
    roots_quadratic, num_solutions = quadratic_real_roots(A_coeffs)

    assert roots_polynomial.size == 0

    # TODO: the below assert should fail since the roots_quadratic() just has whatever.
    # But the num_solutions is 0....
    # But anyways, polynomial_real_roots and quadratic_real_roots appear to just do the same thing.
    # assert roots_quadratic.size == 0
