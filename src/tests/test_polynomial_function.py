# Implementing with pytest!
from ..core.common import *
from ..core.polynomial_function import *
import pytest


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
    cross_product_coeffs = np.ndarray(shape=(1, 3))
    # TODO: below has the template <0, 0>... whatever that's supposed to look like...
    compute_polynomial_mapping_cross_product(0, 0,
                                             A_coeffs, B_coeffs, cross_product_coeffs)

    assert float_equal(cross_product_coeffs[0, 0], 0.0)
    assert float_equal(cross_product_coeffs[0, 1], 0.0)
    assert float_equal(cross_product_coeffs[0, 2], 1.0)


def test_polynomial_mapping_cross_products_elementary_linear_functions():
    print("Elementary linear functions")
    A_coeffs = np.array([[2, 0, 0], [1, 0, 0]])
    B_coeffs = np.array([[0, 1, 0], [0, 1, 0]])
    cross_product_coeffs = np.ndarray(shape=(3, 3))
    compute_polynomial_mapping_cross_product(
        1, 1, A_coeffs, B_coeffs, cross_product_coeffs)

    assert float_equal(cross_product_coeffs[0, 0], 0.0)
    assert float_equal(cross_product_coeffs[0, 1], 0.0)
    assert float_equal(cross_product_coeffs[0, 2], 2.0)
    assert float_equal(cross_product_coeffs[1, 2], 3.0)
    assert float_equal(cross_product_coeffs[2, 2], 1.0)


def test_polynomial_mapping_cross_products_general_constant_functions():
    print("General constant functions")
    A_coeffs = np.array([[1, 2, 3]])
    B_coeffs = np.array([[4, 5, 6]])
    cross_product_coeffs = np.ndarray(shape=(1, 3))
    compute_polynomial_mapping_cross_product(0, 0,
                                             A_coeffs, B_coeffs, cross_product_coeffs)

    assert float_equal(cross_product_coeffs[0, 0], -3.0)
    assert float_equal(cross_product_coeffs[0, 1], 6.0)
    assert float_equal(cross_product_coeffs[0, 2], -3.0)


def test_polynomial_mapping_cross_products_cancelling_linear_functions():
    print("Cancelling linear functions")
    A_coeffs = np.array([[1, 2, 3], [1, 1, 1]])
    B_coeffs = np.array([[4, 5, 6], [1, 1, 1]])
    cross_product_coeffs = np.ndarray(shape=(3, 3))
    compute_polynomial_mapping_cross_product(
        1, 1, A_coeffs, B_coeffs, cross_product_coeffs)

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
    assert (roots.size == 0)
