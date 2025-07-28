from src.quadratic_spline_surface.optimize_spline_surface import *
import pytest
import numpy as np
import numpy.testing as npt


def test_shift_array() -> None:
    # NOTE: I checked this behavior with the ASOC code.
    # In list[np.ndarray] = [0, 1, 2]
    # change such that it shifts to [2, 0, 1]
    # or like [10, 20, 30] becomes [30, 10, 20]
    # Basically, moving elements to the left by "shift" amount.

    # Simple array
    int_list = [0, 1, 2]

    shift_array(int_list, 2)
    assert int_list[0] == 2
    assert int_list[1] == 0
    assert int_list[2] == 1

    # list of NumPy array
    numpy_list: list[ndarray[tuple[int, int], dtype[Any]]] = [
        np.full(shape=(2, 2),  fill_value=0),
        np.full(shape=(2, 2),  fill_value=1),
        np.full(shape=(2, 2),  fill_value=2)]

    shift_array(numpy_list, 1)
    npt.assert_array_equal(numpy_list[0], np.full(shape=(2, 2), fill_value=2))
    npt.assert_array_equal(numpy_list[1], np.full(shape=(2, 2), fill_value=0))
    npt.assert_array_equal(numpy_list[2], np.full(shape=(2, 2), fill_value=1))
