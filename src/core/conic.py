

from enum import Enum
from ..core.rational_function import *


class ConicType(Enum):
    ELLIPSE = 1
    HYPERBOLA = 2
    PARABOLA = 3
    PARALLEL_LINES = 4
    INTERSECTING_LINES = 5
    LINE = 6
    POINT = 7
    EMPTY = 8
    PLANE = 9
    ERROR = 10
    UNKNOWN = 11


class Conic(RationalFunction):
    """
    Explicit representation of a conic segment
    """
    __m_type: ConicType

    def __init__(self, numerator_coeffs, denominator_coeffs, domain, type=None):

        if type != None:
            self.__m_type = type

        super().__init__(2, 2, numerator_coeffs, denominator_coeffs, domain)
        # TODO: ask prof is this is good/correct syntax
        assert self.__is_valid()
        # TODO: use class method or logic inside constructor.
        pass

    # @classmethod

    def get_type(self):
        return self.__m_type

    def transform(self, rotation, translation):
        """
        NOTE: assumes row vector points
        """
        P_rot_coeffs = self.get_numerators() * rotation + \
            self.get_denominator() * translation

        self.set_numerators(P_rot_coeffs)

    # TODO: get the type of the conic.

    def pullback_quadratic_function(self, F_coeffs: np.ndarray, pullback_function: RationalFunction):
        assert F_coeffs.shape == (6, self.m_dimension)
        logger.info("Pulling back conic by quadratic function %s",
                    formatted_bivariate_quadratic_mapping(F_coeffs))

        # Separate the individual polynomial coefficients from the rational
        # function
        P_coeffs = self.get_numerators()
        u_coeffs = P_coeffs[:, 0]
        assert u_coeffs.shape == (3, 1)
        v_coeffs = P_coeffs[:, 1]
        assert v_coeffs.shape == (3, 1)
        Q_coeffs = self.get_denominator()
        assert Q_coeffs.shape == (3, 1)

        logger.info("u function before pullback: (%s)/(%s)",
                    formatted_polynomial(u_coeffs),
                    formatted_polynomial(Q_coeffs))

        # Compute (homogenized) polynomial coefficients for the quadratic terms
        QQ_coeffs, Qu_coeffs, Qv_coeffs, uv_coeffs, uu_coeffs, vv_coeffs = np.ndarray(
            shape=(5, 1))

        QQ_coeffs = compute_polynomial_mapping_product(
            2, 2, 1, Q_coeffs, Q_coeffs)
        Qu_coeffs = compute_polynomial_mapping_product(
            2, 2, 1, Q_coeffs, u_coeffs)
        Qv_coeffs = compute_polynomial_mapping_product(
            2, 2, 1, Q_coeffs, v_coeffs)
        uv_coeffs = compute_polynomial_mapping_product(
            2, 2, 1, u_coeffs, v_coeffs)
        uu_coeffs = compute_polynomial_mapping_product(
            2, 2, 1, u_coeffs, u_coeffs)
        vv_coeffs = compute_polynomial_mapping_product(
            2, 2, 1, v_coeffs, v_coeffs)

        # Combine quadratic monomial functions into a matrix
        monomial_coeffs = np.zeros(shape=(5, 6))

        # NOTE: below was generated. Fix.
        monomial_coeffs[1:QQ_coeffs.size, 0] = QQ_coeffs
        monomial_coeffs[1:Qu_coeffs.size, 1] = Qu_coeffs
        monomial_coeffs[1:Qv_coeffs.size, 2] = Qv_coeffs
        monomial_coeffs[1:uv_coeffs.size, 3] = uv_coeffs
        monomial_coeffs[1:uu_coeffs.size, 4] = uu_coeffs
        monomial_coeffs[1:vv_coeffs.size, 5] = vv_coeffs

        logger.info("Monomial coefficients matrix:\n%s", monomial_coeffs)

        # Compute the pulled back rational function numerator
        pullback_coeffs = monomial_coeffs * F_coeffs
        assert pullback_coeffs.shape == (5, self.m_dimension)

        logger.info("Pullback numerator: %s",
                    formatted_polynomial(pullback_coeffs))
        logger.info("Pullback denominator: %s",
                    formatted_polynomial(QQ_coeffs))

        # XXX: domain() is in the C++. Meanwhile, Python code here uses self.m_domain... may cause problems
        pullback_function = RationalFunction(
            4, self.m_dimension, pullback_coeffs, QQ_coeffs, self.m_domain)

    def __is_valid(self):
        if (self.m_numerator_coeffs.shape[1] == 0):
            return False
        if (self.m_denominator_coeffs.size == 0):
            return False

        return True

    def __repr__(self):
        conic_string: str = "1/("
        conic_string += formatted_polynomial(2, 1, self.m_denominator_coeffs)
        conic_string += ") [\n "

        for i in self.m_numerator_coeffs.shape[1]:
            conic_string += formatted_polynomial(2,
                                                 1, self.m_numerator_coeffs[:, i])
            conic_string += ", \n "

        conic_string += "], t in "
        conic_string += self.m_domain

        return conic_string
