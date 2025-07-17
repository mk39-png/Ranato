"""
Representation of a line segment, which is a rational function
with degree 1 numerator and degree 0 denominator.
"""

# TODO: address the whole : public Conic part of the C++ code
from src.core.common import *
from src.core.bivariate_quadratic_function import *
from src.core.conic import *
from src.core.polynomial_function import *
from src.core.rational_function import *


class LineSegment(Conic):
    # ************
    # Constructors
    # ************

    def __init__(self, numerator_coeffs: np.ndarray = None, input_domain: Interval = None):
        # There's not much logic going on for the constructor in the C++ code, so just going to put them all here for simplicity.
        if (numerator_coeffs is None) and (input_domain is None):
            raise Exception(
                "Attempted LineSegment constructions with no parameters.")

        if (numerator_coeffs):
            self.__init_conic_coefficients(numerator_coeffs)

        if (input_domain):
            self.m_domain = input_domain

    def pullback_linear_function(self, dimension: int, F_coeffs: np.ndarray, pullback_function:     RationalFunction):
        logger.info("Pulling back line segment by linear function %s",
                    formatted_bivariate_linear_mapping(dimension, F_coeffs))

        # Separate the individual polynomial coefficients from the rational
        # function
        # TODO: typedef Matrix2x2r
        P_coeffs = self.get_numerators
        u_coeffs = P_coeffs[:, 0]
        assert u_coeffs.shape == (3, 1)
        v_coeffs = P_coeffs[:, 1]
        assert v_coeffs.shape == (3, 1)
        Q_coeffs = np.array([1.0, 0.0])

        logger.info("u function before pullback: %s",
                    formatted_polynomial(3, 1, u_coeffs))
        logger.info("v function before pullback: %s",
                    formatted_polynomial(3, 1, v_coeffs))

        # Combine quadratic monomial functions into a matrix
        monomial_coeffs = np.zeros(shape=(2, 3))
        monomial_coeffs[0, 0] = 1.0
        monomial_coeffs[0, 1] = u_coeffs[0]
        monomial_coeffs[1, 1] = u_coeffs[1]
        monomial_coeffs[0, 2] = v_coeffs[0]
        monomial_coeffs[1, 2] = v_coeffs[1]
        logger.info("Monomial coefficient matrix:\n%s", monomial_coeffs)

        # Compute the pulled back rational function numerator
        logger.info("Linear coefficient matrix:\n%s", F_coeffs)
        pullback_coeffs = monomial_coeffs * F_coeffs
        assert pullback_coeffs.shape == (2, self.m_dimension)
        logger.info("Pullback function: %s",
                    formatted_polynomial(2, self.m_dimension, pullback_coeffs))

        pullback_function = RationalFunction(
            1, self.m_dimension, pullback_coeffs, Q_coeffs, self.m_domain)

    def __init_conic_coefficients(self, numerator_coeffs: np.ndarray):
        # Build conic numerator with trivial quadratic term
        # TODO: typedef matrix3x2r
        conic_numerator_coeffs = np.ndarray(shape=(3, 2))
        # FIXME: blocking operators need correct translation
        conic_numerator_coeffs[0:2, 0:2] = numerator_coeffs
        conic_numerator_coeffs[2:1, 0:2] = 0

        # Build constant 1 denominator
        conic_denominator_coeffs = np.array([1.0, 0.0, 0.0])

        # Set conic coefficients
        self.set_numerators(conic_numerator_coeffs)
        self.set_denominator(conic_denominator_coeffs)
