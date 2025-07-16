from numpy._typing._array_like import NDArray
from ..core.common import *
from ..core.conic import *


def test_zero_case():
    P_coeffs = np.array([[0, 2], [0, 3], [0, 1]], dtype=np.float64)
    Q_coeffs = np.array([[1], [0], [0]], dtype=np.float64)
    F_coeffs = np.array([[0], [0], [0], [0], [0], [0]], dtype=np.float64)
    assert P_coeffs.shape == (3, 2)
    assert Q_coeffs.shape == (3, 1)
    assert F_coeffs.shape == (6, 1)

    conic = Conic(ConicType.UNKNOWN, P_coeffs, Q_coeffs)
    pullback = conic.pullback_quadratic_function(1, F_coeffs)

    assert (float_equal(pullback(-1.0)[0, 0], 0.0))
    assert (float_equal(pullback(0.0)[0, 0], 0.0))
    assert (float_equal(pullback(1.0)[0, 0], 0.0))


def test_unit_pullback_case():
    P_coeffs: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.array(
        [[0.0, 2.0], [0.0, 3.0], [0.0, 1.0]], dtype=np.float64)
    Q_coeffs: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.array(
        [[1.0], [0.0], [0.0]], dtype=np.float64)
    # NOTE: first element in F_coeffs different from zero case (for those looking for anything different between this case and zero case)
    F_coeffs: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.array([[1.0], [0.0], [0.0], [0.0], [
        0.0], [0.0]], dtype=np.float64)
    assert P_coeffs.shape == (3, 2)
    assert Q_coeffs.shape == (3, 1)
    assert F_coeffs.shape == (6, 1)

    conic = Conic(ConicType.UNKNOWN, P_coeffs, Q_coeffs)
    pullback = conic.pullback_quadratic_function(1, F_coeffs)

    assert (float_equal(pullback(-1.0)[0, 0], 1.0))
    assert (float_equal(pullback(0.0)[0, 0], 1.0))
    assert (float_equal(pullback(1.0)[0, 0], 1.0))


def test_u_projection_case():
    P_coeffs = np.array([[1.0, 1.0],
                         [2.0, -2.0],
                         [1.0, 1.0]], dtype=np.float64)
    Q_coeffs = np.array([[1.0], [0.0], [1.0]], dtype=np.float64)
    F_coeffs = np.array([[0], [1], [0], [0], [0], [0]], dtype=np.float64)
    assert P_coeffs.shape == (3, 2)
    assert Q_coeffs.shape == (3, 1)
    assert F_coeffs.shape == (6, 1)

    conic = Conic(ConicType.UNKNOWN, P_coeffs, Q_coeffs)
    pullback = conic.pullback_quadratic_function(1, F_coeffs)

    assert float_equal(pullback(-1.0)[0, 0], 0.0)
    assert float_equal(pullback(0.0)[0, 0], 1.0)
    assert float_equal(pullback(1.0)[0, 0], 2.0)
