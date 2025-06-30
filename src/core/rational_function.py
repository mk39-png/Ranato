import numpy as np


class RationalFunction:
    # NOTE: degree and dimension ALWAYS passed into RationalFunction upon construction...
    def __init__(self, degree, dimension):
        # ****************
        # Member variables
        # ****************
        # Instead of Eigen matrix... now using numpy matrices.
        # TODO: double check that the shape is as it should be with the Eigen matrix.
        self.m_numerator_coeffs = np.ndarray(
            shape=(degree+1, dimension), dtype='float64')
        # Type interval variable...

    @classmethod
    # TODO: maybe swap out cls for a better name
    def from_polynomial_function():

    def __init__(self, degree: int, dimension: int):
        self.degree: int = degree
        self.dimension: int = dimension

    #
    # Constructors
    #
    # Default constructor for 0 function
