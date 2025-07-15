"""@package docstring
    Quotient of scalar or vector valued polynomial functions over an interval.
"""

from dataclasses import dataclass
import logging
import numpy as np

# from polynomial_function import PolynomialFunction
from .polynomial_function import *
from .interval import Interval

logger = logging.getLogger(__name__)


# TODO: instead of a class... why not just make this a dict?
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
    # NOTE: RationalFunction is never called as RationalFunction()... always constructed with the classmethods below.
    # TODO: Maybe construct RationalFunction with one single constructor and not class methods?
    # But... class methods make life easier, no?
    # But then again... we're trying to implment a C++-like system...
    # Some systems work... like that one file I was working on.
    # But other systems... not so much
    def __init__(self, degree: int, dimension: int,
                 numerator_coeffs: np.ndarray = None, denominator_coeffs: np.ndarray = None, domain: Interval = None) -> None:
        # TODO: assert the shape for numerator_coeffs and denominator_coeffs
        """ General constructor over given interval.
            --- Possible combinations include ---
            Default constructor for 0 function R^n: numerator_coeffs == None, denominator_coeffs = None, domain == None
            Constructor for vector polynomial: denominator_coeffs == None, domain == None
            General constructor over entire real line: domain == None
            General constructor over given interval: all arguments are NOT None

        Args:
            degree: [in]
            dimension: [in]
            numerator_coeffs (np.ndarray): [in] coefficients of the numerator polynomial
            denominator_coeffs (np.ndarray): [in] coefficients of the denominator polynomial
            domain (Interval): [in] domain interval for the mapping

        Returns:
            TODO: fill in return value
        """

        # """Default constructor"""
        self.m_degree = degree
        self.m_dimension = dimension

        if (degree is None) or (dimension is None):
            raise Exception("degree and dimension cannot be None.")

        if numerator_coeffs is None:
            self.m_numerator_coeffs = np.zeros(
                shape=(degree+1, dimension), dtype='float64')
        else:
            self.m_numerator_coeffs = numerator_coeffs

        if denominator_coeffs is None:
            self.m_denominator_coeffs = np.zeros(
                shape=(degree+1, 1), dtype='float64')
            self.m_denominator_coeffs[0][0] = 1.0
        else:
            self.m_denominator_coeffs = denominator_coeffs

        if domain is None:
            self.m_domain = Interval()
        else:
            self.m_domain = domain

        assert self.m_numerator_coeffs.shape == (degree + 1, dimension)
        assert self.m_denominator_coeffs.shape == (degree + 1, 1)
        assert self.__is_valid()

    # *******
    # Methods
    # *******
    @property
    def get_degree(self) -> int:
        """
        Compute the degree of the polynomial mapping as the max of the degrees
        of the numerator and denominator degrees.
        @return degree of the rational mapping
        """
        return self.m_degree

    @property
    def get_dimension(self) -> int:
        """
        Compute the dimension of the rational mapping.
        @return dimension of the rational mapping
        """
        return self.m_dimension

    def compute_derivative(self) -> "RationalFunction":
        """
        Compute the derivative of the rational function, which is also a rational function, using the quotient rule.

        Args:
            None

        Returns:
            derivative (RationalFunction<2*degree, dimension>): [in] derivative rational function.
        """
        # TODO: is this right?
        # assert derivative.m_degree == 2 * self.m_degree
        # assert derivative.m_dimension == self.m_dimension

        # Compute the derivatives of the numerator and denominator polynomials
        logger.info("Taking derivative of rational function")
        logger.info("Numerator:\n%s", self.m_denominator_coeffs)
        logger.info("Denominator:\n%s", self.m_denominator_coeffs)

        # TODO: deal with the whole <degree, dimension> and <degree, 1> being passed in...
        numerator_deriv_coeffs = compute_polynomial_mapping_derivative(
            self.m_degree, self.m_dimension, self.m_numerator_coeffs)
        assert numerator_deriv_coeffs.shape == (
            self.m_degree, self.m_dimension)

        # HACK: denominator_deriv_coeffs must be shape (self.m_degree, 1) rather than (self.m_degree,) because compute_polynomial_mapping_derivative() is not designed to work with vectors.
        # denominator_deriv_coeffs = np.ndarray(shape=(self.m_degree, 1))

        denominator_deriv_coeffs = compute_polynomial_mapping_derivative(
            self.m_degree, 1, self.m_denominator_coeffs)
        assert denominator_deriv_coeffs.shape == (self.m_degree, 1)

        logger.info("Numerator derivative:\n%s", numerator_deriv_coeffs)
        logger.info("Denominator derivative:\n%s", denominator_deriv_coeffs)

        # TODO: 0 degree case?
        #  Compute the derivative numerator and denominator from the quotient rule
        # XXX: there may be an issue between this and the mapping_product function....
        term_0 = compute_polynomial_mapping_scalar_product(
            self.m_degree, self.m_degree - 1, self.m_dimension, self.m_denominator_coeffs, numerator_deriv_coeffs)
        term_1 = compute_polynomial_mapping_scalar_product(
            self.m_degree - 1, self.m_degree, self.m_dimension, denominator_deriv_coeffs, self.m_numerator_coeffs)

        assert term_0.shape == (2 * self.m_degree, self.m_dimension)
        assert term_1.shape == (2 * self.m_degree, self.m_dimension)

        logger.info("First term: \n%s", term_0)
        logger.info("Second term: \n%s", term_1)

        # TODO: is this supposed to be using self.m_degree? Or is it some other degree? Look at the C++ code to double check.
        num_coeffs = np.zeros(shape=(2 * self.m_degree + 1, self.m_dimension))

        # XXX: something might go wrong with the slicing...
        num_coeffs[0:2*self.m_degree, 0:self.m_dimension] = term_0 - term_1

        # denom_coeffs = np.ndarray(shape=(2 * self.m_degree + 1, 1))
        denom_coeffs = compute_polynomial_mapping_product(
            self.m_degree, self.m_degree, 1, self.m_denominator_coeffs, self.m_denominator_coeffs)

        assert denom_coeffs.shape == (2 * self.m_degree + 1, 1)

        # TODO: this should then change the derivative argument to reference a new RationalFunction
        derivative = RationalFunction(2 * self.m_degree, self.m_dimension,
                                      num_coeffs, denom_coeffs, self.m_domain)

        return derivative

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
    @property
    def get_numerators(self):
        assert self.m_numerator_coeffs.shape == (
            self.m_degree + 1, self.m_dimension)
        return self.m_numerator_coeffs

    @property
    def get_denominator(self):
        assert self.m_denominator_coeffs.shape == (self.m_degree + 1, 1)
        return self.m_denominator_coeffs

    @get_numerators.setter
    def set_numerators(self, numerator: np.ndarray):
        assert numerator.shape == (self.m_degree + 1, self.m_dimension)
        self.m_numerator_coeffs = numerator

    @get_numerators.setter
    def set_denominator(self, denominator: np.ndarray):
        assert denominator.shape == (self.m_degree + 1, 1)
        self.m_denominator_coeffs = denominator

    # TODO: then have the domain accessible.
    # TODO: do equivalent to "friend class Conic;"

    def __is_valid(self):
        # Making sure that numerator is shape (n,) array
        # NOTE: m_numerator_coeffs can have multiple dimensions...
        # It's just the denominator that must be 1 dimensional....
        # if (self.m_numerator_coeffs.ndim != 1 or self.m_numerator_coeffs.ndim != 1):
        # return False

        # This ensures that we're still dealing with matrices.
        # Because numerator can be shape (n, m) and not (n, )
        if (self.m_numerator_coeffs.shape[1] == 0):
            return False

        # Making sure that denominator is NOT empty.
        if (self.m_denominator_coeffs.size == 0):
            return False

        return True

    # ******************************
    # Helper functions for operators
    # ******************************
    # NOTE: this is equivalent to operator() in C++ code
    def __call__(self, t: float) -> np.ndarray:
        """
        Evaluate the rational mapping at domain point t.

        Args:
            t (float): [in] domain point to evaluate at.

        Returns:
            evaluated point.
        """
        return self.__evaluate(t)

    def __evaluate(self, t: float) -> np.ndarray:
        # Pt = np.ndarray(shape=(1, self.m_dimension))
        # Qt = np.ndarray(shape=(1,))

        # NOTE: using evaluate_polynomial_mapping() rather than evaluate_polynomial() for cases where m_dimension > 1
        # NOTE: keep the modification by reference since that helps showcase what shape Pt and Qt should be.

        # FIXME: Wait a minute... why is numerator all 0s with test_unit_pullback_case?
        Pt = evaluate_polynomial(degree=self.m_degree,
                                 dimension=self.m_dimension,
                                 polynomial_coeffs=self.m_numerator_coeffs,
                                 t=t)

        Qt = evaluate_polynomial(degree=self.m_degree,
                                 dimension=1,
                                 polynomial_coeffs=self.m_denominator_coeffs,
                                 t=t)

        assert Pt.shape == (1, self.m_dimension)
        assert Qt.shape == (1, 1)

        return Pt / Qt[0]

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
