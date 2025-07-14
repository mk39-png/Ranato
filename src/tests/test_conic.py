from ..core.common import *
from ..core.conic import *
import numpy as np
import pytest


def test_zero_case():
    P_coeffs = np.array([[0, 2], [0, 3], [0, 1]])
    Q_coeffs = np.array([[1], [0], [0]])
    F_coeffs = np.array([[0], [0], [0], [0], [0], [0]])
    assert P_coeffs.shape == (3, 2)
    assert Q_coeffs.shape == (3, 1)
    assert F_coeffs.shape == (6, 1)

    conic = Conic.from_numerator_denominator(P_coeffs, Q_coeffs)
    pullback = RationalFunction.from_zero_function(4, 1)
    conic.pullback_quadratic_function(1, F_coeffs, pullback)

    assert (float_equal(pullback(-1.0)[0], 0.0))
    assert (float_equal(pullback(0.0)[0], 0.0))
    assert (float_equal(pullback(1.0)[0], 0.0))


def test_unit_pullback_case():
    P_coeffs = np.array([[0, 2], [0, 3], [0, 1]])
    Q_coeffs = np.array([[1], [0], [0]])
    # NOTE: first element in F_coeffs different from zero case (for those looking for anything different between this case and zero case)
    F_coeffs = np.array([[1], [0], [0], [0], [0], [0]])
    assert P_coeffs.shape == (3, 2)
    assert Q_coeffs.shape == (3, 1)
    assert F_coeffs.shape == (6, 1)

    conic = Conic.from_numerator_denominator(P_coeffs, Q_coeffs)
    pullback = RationalFunction.from_zero_function(4, 1)
    conic.pullback_quadratic_function(1, F_coeffs, pullback)

    assert (float_equal(pullback(-1.0)[0], 1.0))
    assert (float_equal(pullback(0.0)[0], 1.0))
    assert (float_equal(pullback(1.0)[0], 1.0))


def test_u_projection_case():
    P_coeffs = np.array([[1, 1], [2, -2], [1, 1]])
    Q_coeffs = np.array([[1], [0], [1]])
    F_coeffs = np.array([[0], [1], [0], [0], [0], [0]])
    assert P_coeffs.shape == (3, 2)
    assert Q_coeffs.shape == (3, 1)
    assert F_coeffs.shape == (6, 1)

    conic = Conic.from_numerator_denominator(P_coeffs, Q_coeffs)
    pullback = RationalFunction.from_zero_function(4, 1)
    conic.pullback_quadratic_function(1, F_coeffs, pullback)

    assert (float_equal(pullback(-1.0)[0], 0.0))
    assert (float_equal(pullback(0.0)[0], 1.0))
    assert (float_equal(pullback(1.0)[0], 2.0))
