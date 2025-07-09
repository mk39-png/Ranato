"""
Basically, writing it so that Vectors are represented in NumPy a lot better
Also, writing it so that Matrices are represented in NumPy a lot better
That way, this would be more 1-to-1 to the ASOC C++ code.
Also, I find the Eigen code to be a lot more readable than the NumPy code.
Like, [:, n] vs .col(n) and whatnot.
"""

# Wrappers for the following:
import numpy as np


# https://stackoverflow.com/questions/26821169/python-how-to-wrap-the-np-ndarray-class
class EigenMatrix(np.ndarray):
    pass
    #     def __init__(self, array: np.ndarray):
    #         self.data = array

    #     def col(self, i: int):
    #         return self.data[:, i]

    #     def set_col(self, i: int, values):
    #         self.data[:, i] = values

    #     # https://omkarpathak.in/2018/04/11/python-getitem-and-setitem/
    #     def __getitem__(self, key):
    #         return self.data[key]

    #     def __setitem__(self, i, value):
    #         self.data[key] = value

    #     def __repr__(self):
    #         return repr(self.data)
    # # Matrix
    # .col
    # .


class EigenVector(np.ndarray):
    pass
