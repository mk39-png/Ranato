"""@package docstring
    Quotient of scalar or vector valued polynomial functions over an interval.
"""


from dataclasses import dataclass

import numpy as np

from src.core.common import Vector1D, Vector2D, interval_lerp, logger, todo
# from src.core.polynomial_function import
from src.core.interval import Interval
from src.core.polynomial_function import (
    compute_polynomial_mapping_derivative, compute_polynomial_mapping_product,
    compute_polynomial_mapping_scalar_product, evaluate_polynomial)


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
    def __init__(self,
                 degree: int,
                 dimension: int,
                 numerator_coeffs: np.ndarray | None = None,
                 denominator_coeffs: np.ndarray | None = None,
                 domain: Interval | None = None) -> None:
        # TODO: assert the shape for numerator_coeffs and denominator_coeffs
        """ 
        General constructor over given interval.
        ### Possible combinations include 
        Default constructor for 0 function R^n: 
        numerator_coeffs == None, denominator_coeffs = None, domain == None \n
        Constructor for vector polynomial: denominator_coeffs == None, domain == None \n
        General constructor over entire real line: domain == None \n
        General constructor over given interval: all arguments are NOT None \n

        :param degree (int): [in]
        :param dimension (int): [in]
        :param numerator_coeffs (np.ndarray): [in] coefficients of the numerator polynomial
        :param denominator_coeffs (np.ndarray): [in] coefficients of the denominator polynomial
        :param domain (Interval): [in] domain interval for the mapping
        """
        self.m_degree: int = degree
        self.m_dimension: int = dimension
        self.m_numerator_coeffs: Vector2D
        self.m_denominator_coeffs: Vector2D
        self.m_domain: Interval

        if (degree is None) or (dimension is None):
            raise ValueError("degree and dimension cannot be None.")

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

    # *******************
    # Getters and setters
    # *******************
    @property
    def degree(self) -> int:
        """
        Compute the degree of the polynomial mapping as the max of the degrees
        of the numerator and denominator degrees.
        :return: degree of the rational mapping
        """
        return self.m_degree

    @property
    def dimension(self) -> int:
        """
        Compute the dimension of the rational mapping.
        :return: dimension of the rational mapping
        """
        return self.m_dimension

    @property
    def domain(self) -> Interval:
        """Retrives domain of rational function"""
        return self.m_domain

    @property
    def numerator_coeffs(self) -> np.ndarray:
        """Retrives numerator coefficients"""
        return self.m_numerator_coeffs

    @property
    def denominator_coeffs(self) -> np.ndarray:
        """Retrieves denominator coefficients"""
        return self.m_denominator_coeffs

    # ***************
    # Utility Methods
    # ***************

    def compute_derivative(self) -> "RationalFunction":
        """
        Compute the derivative of the rational function, which is also a rational function,
        using the quotient rule.

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
        logger.info("Numerator:\n%s", self.m_numerator_coeffs)
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

    def split_at_knot(self, knot: float) -> tuple["RationalFunction", "RationalFunction"]:
        """
        Split the rational function into two rational function at some knot
        in the domain.

        Used by contour network.

        :param knot: [in] point in the domain to split the function at
        :return lower_segment: [out] rational function with lower domain
        :return upper_segment: [out] rational function with upper domain
        """
        #  Build lower segment
        t0: float = self.domain.lower_bound
        assert t0 <= knot
        lower_domain = Interval(t0, knot)
        lower_segment = RationalFunction(self.degree,
                                         self.dimension,
                                         self.numerator_coeffs,
                                         self.denominator_coeffs,
                                         lower_domain)

        # Build upper segment
        t1: float = self.domain.upper_bound
        assert knot <= t1
        upper_domain = Interval(knot, t1)
        upper_segment = RationalFunction(self.degree,
                                         self.dimension,
                                         self.numerator_coeffs,
                                         self.denominator_coeffs,
                                         upper_domain)

        return lower_segment, upper_segment

    def sample_points(self, num_points: int) -> list[Vector2D]:
        """
        @brief Sample points in the rational function. 

        sample_points() seen with PlanarPoint and SpatialVector.
        @param[in] num_points: number of points to sample
        @param[out] points: vector of sampled points. list of matrices of shape (1, 2). dtype of elements == float
        """
        # Get sample of the domain
        t_samples: list[float] = self.m_domain.sample_points(num_points)
        points: list[Vector2D] = []

        for i in range(num_points):
            evaluated_rational_function: Vector2D = self.__evaluate(t_samples[i])
            points.append(evaluated_rational_function)

        assert len(points) == num_points
        return points

    def start_point(self) -> np.ndarray:
        """
        Get the point at the start of the rational mapping curve.
        Used in projected_curve_network.py

        :return: curve start point in R^n.
        :rtype: np.ndarray of dimension (1, self.dimension)
        """
        # Return the default constructor point if the domain is not bounded below
        if self.domain.is_bounded_below():
            return np.zeros(shape=(1, self.dimension), dtype=np.float64)

        t0: float = self.domain.lower_bound
        return self.__evaluate(t0)

    def mid_point(self):
        pass

    def end_point(self) -> np.ndarray:
        """
        Get the point at the end of the rational mapping curve.
        Used in projected_curve_network.py

        :return: curve end point in R^n
        :rtype: np.ndarray of dimension (1, self.dimension)
        """
        # Return the default constructor point if the domain is not bounded below
        if self.domain.is_bounded_above():
            return np.zeros(shape=(1, self.dimension), dtype=np.float64)

        t1: float = self.domain.upper_bound
        return self.__evaluate(t1)

    def evaluate_normalized_coordinate(self, t: float) -> Vector1D:
        """
        Evaluate the function at an normalized parameter in [0, 1]

        :param t: [in] normalized coordinate
        :return point: rational function evaluated at normalized coordinate t
        """
        # Check if domain is bounded
        # FIXME: potential issue returning (0, 0) shape array...
        if not self.domain.is_bounded_below():
            return np.ndarray(shape=(0, 0))
        if not self.domain.is_bounded_above():
            return np.ndarray(shape=(0, 0))

        # Linearly interpolate the coordinate
        t0: float = self.domain.lower_bound
        t1: float = self.domain.upper_bound
        s: float = interval_lerp(0.0, 1.0, t0, t1, t)

        # Evaluate at the given domain coordinate
        point: Vector2D = self.__evaluate(s)

        # HACK: flattening to 1D when evaluate should natively return 1D
        return point.flatten()

    def is_in_domain(self, t: float) -> bool:
        """
        Determine if a point is in the domain of the rational mapping.

        :return: true iff t is in the domain
        """
        return self.m_domain.contains(t)

    def is_in_domain_interior(self):
        todo()

    def discretize(self):
        todo()

    # TODO: this is where I need to interact with the Blender API since *that* is now my viewer.
    def add_curve_to_viewer(self):
        todo()

    def finite_difference_derivative(self):
        todo()

    # *******************
    # Getters and setters
    # *******************
    @property
    def numerators(self) -> Vector2D:
        """Retrieves numerator of RationalFunction"""

        assert self.m_numerator_coeffs.shape == (
            self.m_degree + 1, self.m_dimension)
        return self.m_numerator_coeffs

    @numerators.setter
    def numerators(self, numerator: np.ndarray) -> None:
        """Sets numerator of RationalFunction"""

        assert numerator.shape == (self.m_degree + 1, self.m_dimension)
        self.m_numerator_coeffs = numerator

    @property
    def denominator(self) -> Vector2D:
        """Retrieves denominator of RationalFunction"""
        assert self.m_denominator_coeffs.shape == (self.m_degree + 1, 1)
        return self.m_denominator_coeffs

    @denominator.setter
    def denominator(self, denominator: np.ndarray) -> None:
        """Sets denominator of RationalFunction"""
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
        """
        Evaluate the function at a domain coordinate

        NOTE: Pt = np.ndarray(shape=(1, self.m_dimension)) \n
        NOTE: Qt = np.ndarray(shape=(1,1))

        :param t: [in] coordinate
        :return point: rational function evaluated at coordinate t. Shape = (1, self.dimension)
        """
        # NOTE: using evaluate_polynomial_mapping() rather than evaluate_polynomial() for cases where m_dimension > 1
        # NOTE: keep the modification by reference since that helps showcase what shape Pt and Qt should be.

        # FIXME: Wait a minute... why is numerator all 0s with test_unit_pullback_case?
        # FIXME: inheriting degree from
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
