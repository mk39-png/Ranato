

from enum import Enum
from ..core.rational_function import *
from ..core.bivariate_quadratic_function import formatted_bivariate_quadratic_mapping


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

    # ************
    # Constructors
    # ************
    def __init__(self, type=ConicType.UNKNOWN):
        self.m_type = type
        assert self.__is_valid()

    # TODO: fix this whole parent class intitialization thing
    @classmethod
    def from_numerator_denominator(cls, numerator_coeffs: np.ndarray, denominator_coeffs: np.ndarray):
        super().__init__(cls.from_real_line(2, 2, numerator_coeffs, denominator_coeffs))

        # TODO: double check that this is in fact passing in value UNKNOWN
        return cls(ConicType.UNKNOWN)

    @classmethod
    def from_numerator_denominator_type(cls, numerator_coeffs: np.ndarray, denominator_coeffs: np.ndarray, m_type: ConicType):
        super().from_real_line(2, 2, numerator_coeffs, denominator_coeffs)
        return cls(m_type)

    @classmethod
    def from_numerator_denominator_domain(cls, numerator_coeffs: np.ndarray, denominator_coeffs: np.ndarray, domain: Interval):
        super().from_interval(2, 2, numerator_coeffs, denominator_coeffs, domain)
        return cls(ConicType.UNKNOWN)

    @classmethod
    def from_numerator_denominator_domain_type(cls, numerator_coeffs: np.ndarray, denominator_coeffs: np.ndarray, domain: Interval, m_type: ConicType):
        super().from_interval(2, 2, numerator_coeffs, denominator_coeffs, domain)
        return cls(m_type)

    # ******
    # PUBLIC
    # ******
    def get_type(self) -> ConicType:
        """
        @brief Get the type (e.g. hyperbola, line, etc.) of the conic.
        @return type identifier
        """
        return self.__m_type

    def transform(self, rotation: np.ndarray, translation: np.ndarray):
        """
        NOTE: assumes row vector points
        """
        assert rotation.shape == (2, 2)
        assert translation.shape == (1, 2)
        P_rot_coeffs: np.ndarray = self.get_numerators * \
            rotation + self.get_denominator * translation
        assert P_rot_coeffs.shape == (3, 2)
        self.set_numerators(P_rot_coeffs)

    def pullback_quadratic_function(self, dimension: int, F_coeffs: np.ndarray, pullback_function: RationalFunction):
        # TODO: change function to return pullback_function?
        assert F_coeffs.shape == (6, dimension)
        assert pullback_function.get_degree == 4
        assert pullback_function.get_dimension == dimension

        logger.info("Pulling back conic by quadratic function %s",
                    formatted_bivariate_quadratic_mapping(self.m_dimension, F_coeffs))

        # Separate the individual polynomial coefficients from the rational function
        P_coeffs = self.get_numerators

        # TODO: make sure that this is getting the shape we want...
        u_coeffs = P_coeffs[:, [0]]
        v_coeffs = P_coeffs[:, [1]]
        Q_coeffs = self.get_denominator
        assert u_coeffs.shape == (3, 1)
        assert v_coeffs.shape == (3, 1)
        assert Q_coeffs.shape == (3, 1)

        logger.info("u function before pullback: (%s)/(%s)",
                    formatted_polynomial(self.get_degree, self.get_dimension, u_coeffs), formatted_polynomial(self.get_degree, self.get_dimension, Q_coeffs))

        # Compute (homogenized) polynomial coefficients for the quadratic terms
        QQ_coeffs = np.ndarray(shape=(5, 1))
        Qu_coeffs = np.ndarray(shape=(5, 1))
        Qv_coeffs = np.ndarray(shape=(5, 1))
        uv_coeffs = np.ndarray(shape=(5, 1))
        uu_coeffs = np.ndarray(shape=(5, 1))
        vv_coeffs = np.ndarray(shape=(5, 1))
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
        # NOTE: need to flatten the NP matrices into vectors from (5,1) to (5,) shape for broadcasting to work
        monomial_coeffs = np.array([QQ_coeffs.flatten(),
                                    Qu_coeffs.flatten(),
                                    Qv_coeffs.flatten(),
                                    uv_coeffs.flatten(),
                                    uu_coeffs.flatten(),
                                    vv_coeffs.flatten()])
        assert monomial_coeffs.shape == (5, 6)
        logger.info("Monomial coefficients matrix:\n%s", monomial_coeffs)

        # Compute the pulled back rational function numerator
        logger.info("Quadratic coefficient matrix:\n%s", F_coeffs)
        pullback_coeffs = monomial_coeffs * F_coeffs
        assert pullback_coeffs.shape == (5, self.m_dimension)

        logger.info("Pullback numerator: %s",
                    formatted_polynomial(self.get_degree, self.get_dimension, pullback_coeffs))
        logger.info("Pullback denominator: %s",
                    formatted_polynomial(self.get_degree, self.get_dimension, QQ_coeffs))

        # XXX: domain() is in the C++. Meanwhile, Python code here uses self.m_domain... may cause problems.
        # XXX: specifically, just be wary of the getter that's used in the C++ code.
        pullback_function = RationalFunction.from_interval(
            4, self.m_dimension, pullback_coeffs, QQ_coeffs, self.m_domain)

    def __is_valid(self):
        if (self.m_numerator_coeffs.shape[1] == 0):
            return False
        if (self.m_denominator_coeffs.size == 0):
            return False

        return True

    def formatted_conic(self) -> str:
        conic_string: str = "1/("
        conic_string += formatted_polynomial(2, 1, self.m_denominator_coeffs)
        conic_string += ") [\n "

        for i in self.m_numerator_coeffs.shape[1]:
            conic_string += formatted_polynomial(2,
                                                 1, self.m_numerator_coeffs[:, i])
            conic_string += ", \n "

        conic_string += "], t in "
        conic_string += self.m_domain.formatted_interval()

        return conic_string

    def __repr__(self) -> str:
        return self.formatted_conic()
