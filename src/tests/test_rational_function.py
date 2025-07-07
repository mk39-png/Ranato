"""
Tests to find that the derivative of a rational function can be found.
"""

from ..core.common import *
from ..core.rational_function import *

import numpy as np
import pytest


def test_zero_function() -> None:
    P_coeffs = np.array([0, 0]).reshape(2, 1)
    Q_coeffs = np.array([1, 0]).reshape(2, 1)

    assert P_coeffs.shape == (2, 1)
    assert Q_coeffs.shape == (2, 1)

    F = RationalFunction.from_real_line(1, 1, P_coeffs, Q_coeffs)
    F_derivative = RationalFunction.from_zero_function(2, 1)

    F.compute_derivative(F_derivative)

    assert float_equal(F_derivative(-1.0)[0], 0.0)
    assert float_equal(F_derivative(0.0)[0], 0.0)
    assert float_equal(F_derivative(1.0)[0], 0.0)


def test_constant_function() -> None:
    # self.assertEqual('foo'.upper(), 'FOO')
    pass


def test_linear_function() -> None:
    # self.assertTrue('FOO'.isupper())
    # self.assertFalse('Foo'.isupper())
    pass


def test_quadratic_function() -> None:
    # s = 'hello world'
    # self.assertEqual(s.split(), ['hello', 'world'])
    # # check that s.split fails when the separator is not a string
    # with self.assertRaises(TypeError):
    #     s.split(2)
    pass


def test_inverse_monomial_function() -> None:
    pass


def test_inverse_quadratic_function() -> None:
    pass


def test_rational_function() -> None:
    pass


def test_planar_rational_function() -> None:
    pass
