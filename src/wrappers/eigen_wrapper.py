"""
Basically, writing it so that Vectors are represented in NumPy a lot better
Also, writing it so that Matrices are represented in NumPy a lot better
That way, this would be more 1-to-1 to the ASOC C++ code.
Also, I find the Eigen code to be a lot more readable than the NumPy code.
Like, [:, n] vs .col(n) and whatnot.

Insanity pursues because (1, n) or (n, 1) Matrices in Eigen are accessed and treated like vectors. 
In other words, they should be treated like (n, ) in NumPy with ndim == 1.
"""

# Wrappers for the following:
from ..core.common import logger
import numpy as np


# TODO: all other size classes like Matrix2x2r should be subclasses of Matrix()
# https://numpy.org/doc/stable/user/basics.subclassing.html
class Matrix(np.ndarray):
    """
    Matrix class exist to reduce friction in translating Eigen matrices into NumPy arrays.
    The primary issue is how (n, 1) or (1, n) matrices in Eigen are treated like 1D vectors while
    NumPy treats them like 2D arrays, which causes various issues, especially for calculations.
    """
    # TODO: should get shape as input.
    # if 1D, then yeah... create a vector.

    # Gives option to either put shape or input_array to initialzie the array...
    # Either that or create numpy array into it to construct from...
    def __new__(cls, n, m, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # add the new attribute to the created instance
        obj.info = info

        # Finally, we must return the newly created object:
        return obj

    def from_numpy_array():
        """
        Constructs Matrix from passed-in NumPy array.
        Does checking to see if 1D array is passed in.
        But... what if something is passed in... something like 
        """

        pass

    def __new__(cls, n, m, fill_value=0.0, dtype=float):
        # Create an instance using np.full to fill with default value
        obj = np.full((n, m), fill_value, dtype=dtype).view(cls)
        obj._n = n
        obj._m = m
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._n = getattr(obj, '_n', None)
        self._m = getattr(obj, '_m', None)

    def rows(self):
        """Return list of rows as 1D arrays"""
        return [self[i, :] for i in range(self.shape[0])]

    def cols(self):
        """Return list of columns as 1D arrays"""
        return [self[:, j] for j in range(self.shape[1])]


# https://stackoverflow.com/questions/26821169/python-how-to-wrap-the-np-ndarray-class
class Matrix2x2r(np.ndarray):
    # TODO: subclass Matrix(), actuallly
    def __init__(self, array: np.ndarray):
        assert array.shape == (2, 2)
        self.data = array

    def col(self, i: int):
        column = self.data[:, i]
        # This should return a
        return self.data[:, i]

    def set_col(self, i: int, values):
        self.data[:, i] = values

    # https://omkarpathak.in/2018/04/11/python-getitem-and-setitem/
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return repr(self.data)
