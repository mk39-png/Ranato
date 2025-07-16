"""
Tests to find that the derivative of a rational function can be found.
"""

from src.core.rational_function import RationalFunction
from src.core.common import *
from src.core.rational_function import *

import pytest


def test_zero_function() -> None:
    # TODO: change up from_real_line so that it works with (n,) shape rather than whatever funky thing is here.
    # P_coeffs = np.array([0, 0]).reshape(2, 1)
    # Q_coeffs = np.array([1, 0]).reshape(2, 1)
    # Meanwhile, denominator is ALWAYS going to be a vector.
    P_coeffs = np.array([[0], [0]])
    Q_coeffs = np.array([[1], [0]])
    F = RationalFunction(1, 1, P_coeffs, Q_coeffs)

    # TODO: problem is the below since the denom and numerator are NOT (n,) shaped....
    # F_derivative = RationalFunction.from_zero_function(2, 1)

    F_derivative = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], 0.0)
    assert float_equal(F_derivative(0.0)[0], 0.0)
    assert float_equal(F_derivative(1.0)[0], 0.0)


def test_constant_function() -> None:
    P_coeffs = np.array([[1], [0]])
    Q_coeffs = np.array([[1], [0]])
    F = RationalFunction(1, 1, P_coeffs, Q_coeffs)
    # F_derivative = RationalFunction.from_zero_function(2, 1)

    F_derivative = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], 0.0)
    assert float_equal(F_derivative(0.0)[0],  0.0)
    assert float_equal(F_derivative(1.0)[0],  0.0)


def test_linear_function() -> None:
    P_coeffs = np.array([-1, 2]).reshape(2, 1)
    Q_coeffs = np.array([1, 0]).reshape(2, 1)
    F = RationalFunction(1, 1, P_coeffs, Q_coeffs)

    # TODO: fix "from_zero_function"
    # F_derivative = RationalFunction.from_zero_function(2, 1)
    # F.compute_derivative(F_derivative)
    F_derivative = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], 2.0)
    assert float_equal(F_derivative(0.0)[0],  2.0)
    assert float_equal(F_derivative(1.0)[0],  2.0)


def test_quadratic_function() -> None:
    # P_coeffs = np.array([1, -2, 1])
    # Q_coeffs = np.array([1, 0])
    # NOTE: P_coeffs and Q_coeffs must have the same degree (i.e. same number of rows)
    P_coeffs = np.array([[1], [-2], [1]])
    Q_coeffs = np.array([[1], [0], [0]])

    F = RationalFunction(2, 1, P_coeffs, Q_coeffs)
    # F_derivative = RationalFunction.from_zero_function(4, 1)
    F_derivative = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], -4.0)
    assert float_equal(F_derivative(0.0)[0],  -2.0)
    assert float_equal(F_derivative(1.0)[0],  0.0)


def test_inverse_monomial_function() -> None:
    P_coeffs = np.array([[1], [0], [0]])
    Q_coeffs = np.array([[0], [0], [1]])
    F = RationalFunction(2, 1, P_coeffs, Q_coeffs)
    # F_derivative = RationalFunction.from_zero_function(4, 1)
    F_derivative = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], 2.0)
    assert float_equal(F_derivative(1.0)[0], -2.0)
    assert float_equal(F_derivative(2.0)[0], -0.25)


def test_inverse_quadratic_function() -> None:
    P_coeffs = np.array([[1], [0], [0]])
    Q_coeffs = np.array([[1], [0], [1]])
    F = RationalFunction(2, 1, P_coeffs, Q_coeffs)
    # F_derivative = RationalFunction.from_zero_function(4, 1)
    F_derivative = F.compute_derivative()

    # -2t / (1 + t ^ 2) ^ 2
    assert float_equal(F_derivative(-1.0)[0], 0.5)
    assert float_equal(F_derivative(0.0)[0], 0.0)
    assert float_equal(F_derivative(1.0)[0], -0.5)
    assert float_equal(F_derivative(2.0)[0], -0.16)


def test_rational_function() -> None:
    P_coeffs = np.array([[1], [1], [0]])
    Q_coeffs = np.array([[1], [0], [1]])
    F = RationalFunction(2, 1, P_coeffs, Q_coeffs)
    # F_derivative = RationalFunction.from_zero_function(4, 1)
    F_derivative = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], 0.5)
    assert float_equal(F_derivative(0.0)[0], 1.0)
    assert float_equal(F_derivative(1.0)[0], -0.5)
    assert float_equal(F_derivative(2.0)[0], -0.28)


def test_planar_rational_function() -> None:
    P_coeffs = np.array([[1, 1], [0, 1], [0, 0]])
    assert P_coeffs.shape == (3, 2)

    Q_coeffs = np.array([[1], [0], [1]])
    F = RationalFunction(2, 2, P_coeffs, Q_coeffs)

    # F_derivative = RationalFunction.from_zero_function(4, 2)
    F_derivative: RationalFunction = F.compute_derivative()

    # FIXME: the whole non-scalar, non-vector F_derivative results might cause a lot of problems
    # -2t / (1 + t ^ 2) ^ 2
    # (1 - 2t - t ^ 2) / (1 + t ^ 2) ^ 2
    assert float_equal(F_derivative(-1.0)[0][0], 0.5)
    assert float_equal(F_derivative(0.0)[0][0], 0.0)
    assert float_equal(F_derivative(1.0)[0][0], -0.5)
    assert float_equal(F_derivative(2.0)[0][0], -0.16)

    assert float_equal(F_derivative(-1.0)[0][1], 0.5)
    assert float_equal(F_derivative(0.0)[0][1], 1.0)
    assert float_equal(F_derivative(1.0)[0][1], -0.5)
    assert float_equal(F_derivative(2.0)[0][1], -0.28)
