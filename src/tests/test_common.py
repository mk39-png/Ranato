from ..core.common import *
import numpy as np
import pytest


def test_cross_product():
    v = np.array([[1], [2], [3]])
    w = np.array([[4], [5], [6]])
    assert v.shape == (3, 1)
    assert w.shape == (3, 1)

    n = cross_product(v, w,)
    n_numpy = np.cross(v, w, axis=0)

    assert np.array_equal(n, n_numpy)
