import pytest

from src.core.affine_manifold import AffineManifold
from src.core.common import *
from src.core.halfedge import Halfedge

# *******************
# Test Methods
# *******************


def test_index_vector_complement_spot_mesh() -> None:
    """
    Tests index_vector_complement() to initialize variable_vertices and variable_edges 
    for generate_optimized_twelve_split_position_data() for the spot_control mesh.
    NOTE: these values should be the same for the fit and full cases hessian cases.
    """
    # Get input mesh
    V, uv, F, FT = initialize_spot_control_mesh()
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    he_to_corner: list[tuple[Index, Index]] = affine_manifold.he_to_corner
    halfedge: Halfedge = affine_manifold.halfedge

    num_vertices: int = V.shape[ROWS]
    num_edges: int = halfedge.num_edges

    fixed_vertices: list[int] = []
    fixed_edges: list[int] = []
    variable_vertices: list[int] = index_vector_complement(fixed_vertices, num_vertices)
    variable_edges: list[int] = index_vector_complement(fixed_edges, num_edges)

    filepath: str = "spot_control\\optimize_spline_surface\\generate_optimized_twelve_split_position_data\\"
    compare_eigen_numpy_matrix(filepath+"variable_vertices.csv", np.array(variable_vertices))
    compare_eigen_numpy_matrix(filepath+"variable_edges.csv", np.array(variable_edges))


def test_cross_product() -> None:
    v = np.array([[1], [2], [3]])
    w = np.array([[4], [5], [6]])
    assert v.shape == (3, 1)
    assert w.shape == (3, 1)

    n = cross_product(v, w)
    n_numpy = np.cross(v, w, axis=0)

    assert np.array_equal(n, n_numpy)


def test_convert_nested_vector_to_matrix() -> None:
    """
    Seeing if this is equivalent to numpy operation...
    """
    boundary_points: list[SpatialVector] = [np.array([[0, 1, 2]], dtype=np.float64),
                                            np.array([[3, 4, 5]], dtype=np.float64),
                                            np.array([[6, 7, 8]], dtype=np.float64)]
    matrix = convert_nested_vector_to_matrix(boundary_points)

    matrix_np = np.asarray(boundary_points)
    matrix_np = matrix_np.squeeze()
    print(matrix_np)

    np.testing.assert_allclose(matrix, matrix_np)
