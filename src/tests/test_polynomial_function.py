# Implementing with pytest!
from ..core.common import *
from ..core.polynomial_function import *
import pytest


def test_polynomial_mapping_cross_products():

    # TODO: add section
    # TODO: make this more pytest-esque
    print("Elementary constant functions")
    A_coeffs = np.ndarray(shape=(1, 3), buffer=np.array([1, 0, 0]))
    B_coeffs = np.ndarray(shape=(1, 3), buffer=np.array([0, 1, 0]))
    cross_product_coeffs = np.ndarray(shape=(1, 3))

    # TODO: below has the template <0, 0>... whatever that's supposed to look like...
    compute_polynomial_mapping_cross_product(0, 0,
                                             A_coeffs, B_coeffs, cross_product_coeffs)

    assert (flo)
