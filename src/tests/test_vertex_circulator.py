from ..core.common import *
from ..core.vertex_circulator import *
from ..utils.generate_shapes import *

import pytest


# TODO: again! Figure out the data type before implementing...
def test_tetrahedron():
    V, F = generate_tetrahedron_VF()
    vertex_circulator = VertexCirculator(F)
    vertex_one_ring: list[int] = []
    face_one_ring: list[int] = []
    vertex_circulator.get_one_ring(0, vertex_one_ring, face_one_ring)

    assert len(vertex_one_ring) == 4
    assert len(face_one_ring) == 3
    assert vector_contains(vertex_one_ring, 1)
    assert vector_contains(vertex_one_ring, 2)
    assert vector_contains(vertex_one_ring, 3)
    assert vector_contains(face_one_ring, 0)
    assert vector_contains(face_one_ring, 1)
    assert vector_contains(face_one_ring, 2)


def test_torus():
    
    pass


def test_triangle():
    pass


def test_square():
    pass
