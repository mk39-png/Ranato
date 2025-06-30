"""@package docstring
    Quotient of scalar or vector valued polynomial functions over an interval.
"""

from dataclasses import dataclass
import numpy as np
from interval import Interval


@dataclass
class CurveDiscretizationParameters:
    num_samples: int = 5
    num_tangents_per_segment: int = 5


class RationalFunction:
    # NOTE: degree and dimension ALWAYS passed into RationalFunction upon construction...
    # ************
    # Constructors
    # ************

    # TODO: maybe have the arguments have default values... that are then dependent on degree and dimension...
    # TODO: so that would entail some sort of lambda function?
    def __init__(self, degree: int, dimension: int, numerator_coeffs: np.ndarray, denominator_coeffs: np.ndarray, domain: Interval):
        """Default constructor"""
        # ****************
        # Member variables
        # ****************
        # TODO: double check that the shape is as it should be with the Eigen matrix.
        self.m_degree = degree
        self.m_dimension = dimension
        self.m_numerator_coeffs = numerator_coeffs
        self.m_denominator_coeffs = denominator_coeffs
        self.m_domain = domain
        assert (self.is_valid())

    @classmethod
    def from_zero_function(cls, degree: int, dimension: int):
        """Default numerator set to constant in R^n"""
        numerator_coeffs = np.zeros(
            shape=(degree+1, dimension), dtype='float64')
        denominator_coeffs = np.zeros(
            shape=(degree+1, 1), dtype='float64')
        denominator_coeffs[0] = 1.0
        domain = Interval()
        domain.reset_bounds()

        return cls(degree, dimension, numerator_coeffs, denominator_coeffs, domain)

    @classmethod
    def from_polynomial_function(cls, degree: int, dimension: int, numerator_coeffs: np.ndarray):
        """ Constructor for a vector polynomial

        Args:
            numerator_coeffs (np.ndarray): coefficients of the polynomial functions

        Return:
            TODO: fill in the return value
        """
        denominator_coeffs = np.zeros(
            shape=(degree+1, 1), dtype='float64')
        denominator_coeffs[0] = 1.0
        domain = Interval()
        domain.reset_bounds()

        return cls(degree, dimension, numerator_coeffs, denominator_coeffs, domain)

    @classmethod
    def from_real_line(cls, degree: int, dimension: int, numerator_coeffs: np.ndarray, denominator_coeffs: np.ndarray):
        # TODO: assert the shape for numerator_coeffs and denominator_coeffs
        """ General constructor over entire real line.

        Args:
            numerator_coeffs [in] (np.ndarray): coefficients of the numerator polynomial
            denominator_coeffs [in] (np.ndarray): coefficients of the denominator polynomial

        Return:
            TODO: fill in return value
        """
        domain = Interval()
        domain.reset_bounds()

        return cls(degree, dimension, numerator_coeffs, denominator_coeffs, domain)

    @classmethod
    def from_interval(cls, degree: int, dimension: int, numerator_coeffs: np.ndarray, denominator_coeffs: np.ndarray, domain: Interval):
        # TODO: assert the shape for numerator_coeffs and denominator_coeffs
        """ General constructor over given interval.

        Args:
            numerator_coeffs [in] (np.ndarray): coefficients of the numerator polynomial
            denominator_coeffs [in] (np.ndarray): coefficients of the denominator polynomial
            domain [in] (Interval): domain interval for the mapping

        Return:
            TODO: fill in return value
        """

        return cls(degree, dimension, numerator_coeffs, denominator_coeffs, domain)

    # *******
    # Methods
    # *******
    # NOTE: this is never used in the ASOC code
    @property
    def degree(self):
        pass

    # NOTE: this is never used in the ASOC code
    def get_dimension(self):
        pass

    def compute_derivative(self):
        pass

    def apply_one_form(self):
        pass

    def split_at_knot(self):
        pass

    def sample_points(self):
        pass

    def start_point(self):
        pass

    def mid_point(self):
        pass

    def end_point(self):
        pass

    def evaluate(self):
        pass

    def evaluate_normalized_coordinate(self):
        pass

    def is_in_domain(self):
        pass

    def is_in_domain_interior(self):
        pass

    def discretize(self):
        pass

    # TODO: this is where I need to interact with the Blender API since *that* is now my viewer.
    def add_curve_to_viewer(self):
        pass

    def finite_difference_derivative(self):
        pass

    # *******************
    # Getters and setters
    # *******************
    def set_numerators(self):
        pass

    def set_denominator(self):
        pass

    def get_numerators(self):
        pass

    def get_denominator(self):
        pass

    # TODO: then have the domain accessible.
    # TODO: do equivalent to "friend class Conic;"
    def __is_valid(self):
        # Checks columns of m_numerator_coeffs
        if (self.m_numerator_coeffs.shape[1] == 0):
            return False
        if (self.m_denominator_coeffs.size == 0):
            return False

        return True

    # ******************************
    # Helper functions for operators
    # ******************************

    # TODO: turn "formatted_rational_function" into a __repr__ for when the rational function is printed in the interpreter
    # TODO: finish a lot of these things for PolynomialFunction and Interval classes
    def __repr__(self):
        rational_function_string: str = "RationalFunction 1/()"
        # rational_function_string += formatted_polynomial < degree, 1 > (
        #     m_denominator_coeffs, 17)
        rational_function_string += ") [\n  "

        # for each column in m_numerator_coeffs
        for i in range(self.m_numerator_coeffs.shape[1]):
            # rational_function_string += formatted_polynomial < degree, 1 > (
            #     m_numerator_coeffs.col(i), 17)
            rational_function_string += ",\n  "

        # rational_function_string += "], t in " + self.m_domain.formatted_interval()

        return rational_function_string
