# import pytest

from ..core.common import *
from ..core.affine_manifold import *
from ..utils.generate_shapes import *


def test_affine_manifold_from_triangle():
    V = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    F = np.array([[0, 1, 2]])

    assert V.shape == (3, 2)
    assert F.shape == (1, 3)

    width = 3
    height = 4
    l: list[list[float]] = [[5, 4, 3]]


def test_affine_manifold_from_global_uvs():
    V = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    F = np.array([[0, 1, 2]])

    assert V.shape == (3, 2)
    assert F.shape == (1, 3)

    affine_manifold = ParametricAffineManifold(F, V)

    # Check basic manifold information
    assert affine_manifold.num_faces == 1
    assert affine_manifold.num_vertices == 3

    # Check vertex chart at vertex 0
    chart: VertexManifoldChart = affine_manifold.get_vertex_chart(0)
    assert chart.vertex_index == 0
    assert len(chart.vertex_one_ring) == 2
