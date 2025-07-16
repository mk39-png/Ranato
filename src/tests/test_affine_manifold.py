
from src.core.common import *
from src.core.affine_manifold import *
from src.utils.generate_shapes import *
import pytest

import numpy as np


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
    assert chart.vertex_one_ring[0] == 1
    assert chart.vertex_one_ring[1] == 2
    assert len(chart.face_one_ring) == 1
    assert chart.face_one_ring[0] == 0
    assert float_equal(chart.one_ring_uv_positions[0, 0], 3.0)
    assert float_equal(chart.one_ring_uv_positions[0, 1], 0.0)
    assert float_equal(chart.one_ring_uv_positions[1, 0], 0.0)
    assert float_equal(chart.one_ring_uv_positions[1, 1], 4.0)

    # Check face corner charts for face 0
    corner_uv_positions = affine_manifold.get_face_corner_charts(0)

    assert float_equal(corner_uv_positions[0][0, 0],  3.0)
    assert float_equal(corner_uv_positions[0][0, 1],  0.0)
    assert float_equal(corner_uv_positions[0][1, 0],  0.0)
    assert float_equal(corner_uv_positions[0][1, 1],  4.0)
    assert float_equal(corner_uv_positions[1][0, 0], -3.0)
    assert float_equal(corner_uv_positions[1][0, 1],  4.0)
    assert float_equal(corner_uv_positions[1][1, 0], -3.0)
    assert float_equal(corner_uv_positions[1][1, 1],  0.0)
    assert float_equal(corner_uv_positions[2][0, 0],  0.0)
    assert float_equal(corner_uv_positions[2][0, 1], -4.0)
    assert float_equal(corner_uv_positions[2][1, 0],  3.0)
    assert float_equal(corner_uv_positions[2][1, 1], -4.0)

    # Check global uv
    uv0 = affine_manifold.get_vertex_global_uv(0)
    uv1 = affine_manifold.get_vertex_global_uv(1)
    uv2 = affine_manifold.get_vertex_global_uv(2)
    assert np.array_equal(affine_manifold.get_global_uv, V)
    assert vector_equal(uv0, V[[0], :])
    assert vector_equal(uv1, V[[1], :])
    assert vector_equal(uv2, V[[2], :])
