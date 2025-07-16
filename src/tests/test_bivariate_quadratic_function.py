from ..core.bivariate_quadratic_function import *
import numpy as np
import pytest


def test_compute_quadratic_cross_product():
    V_coeffs = np.array([[0, 1, 2], [3, 4, 5], [7, 8, 9]])
    W_coeffs = np.array([[0, 1, 2], [3, 4, 5], [7, 8, 9]])
    assert V_coeffs.shape == (3, 3)
    assert W_coeffs.shape == (3, 3)

    N_coeffs = compute_quadratic_cross_product(V_coeffs, W_coeffs)
    assert N_coeffs.shape == (6, 3)
