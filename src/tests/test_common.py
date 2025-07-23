from src.core.common import *
# import pytest


def test_cross_product():
    v = np.array([[1], [2], [3]])
    w = np.array([[4], [5], [6]])
    assert v.shape == (3, 1)
    assert w.shape == (3, 1)

    n = cross_product(v, w)
    n_numpy = np.cross(v, w, axis=0)

    assert np.array_equal(n, n_numpy)


def test_convert_nested_vector_to_matrix():
    """
    Seeing if this is equivalent to numpy operation...
    """
    boundary_points: list[SpatialVector] = [np.array([[0, 1, 2]]),
                                            np.array([[3, 4, 5]]),
                                            np.array([[6, 7, 8]])]
    matrix = convert_nested_vector_to_matrix(boundary_points)

    matrix_np = np.asarray(boundary_points)
    matrix_np = matrix_np.squeeze()
    print(matrix_np)

    np.testing.assert_allclose(matrix, matrix_np)
