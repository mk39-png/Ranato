from ..core.common import *
from ..core.vertex_circulator import *
from ..utils.generate_shapes import *

import numpy as np
import pytest

# TODO: again! Figure out the data type before implementing...


def test_tetrahedron():
    V, F = generate_tetrahedron_VF()
    vertex_circulator = VertexCirculator(F)
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(0)

    assert len(vertex_one_ring) == 4
    assert len(face_one_ring) == 3
    assert vector_contains(vertex_one_ring, 1)
    assert vector_contains(vertex_one_ring, 2)
    assert vector_contains(vertex_one_ring, 3)
    assert vector_contains(face_one_ring, 0)
    assert vector_contains(face_one_ring, 1)
    assert vector_contains(face_one_ring, 2)


def test_torus():
    V, F = generate_minimal_torus_VF()
    vertex_circulator = VertexCirculator(F)
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(4)

    assert len(vertex_one_ring) == 7
    assert len(face_one_ring) == 6
    assert vector_contains(vertex_one_ring, 1)
    assert vector_contains(vertex_one_ring, 2)
    assert vector_contains(vertex_one_ring, 3)
    assert vector_contains(vertex_one_ring, 5)
    assert vector_contains(vertex_one_ring, 6)
    assert vector_contains(vertex_one_ring, 7)
    assert vector_contains(face_one_ring, 1)
    assert vector_contains(face_one_ring, 2)
    assert vector_contains(face_one_ring, 3)
    assert vector_contains(face_one_ring, 6)
    assert vector_contains(face_one_ring, 7)
    assert vector_contains(face_one_ring, 8)


def test_triangle():
    # TODO: use NumPy arrays to store faces and whatnot.
    F = np.array([[0, 1, 2]])
    # NOTE: problem with sizing of F.... an it being (3,) rather than (1, 3) or something.
    vertex_circulator = VertexCirculator(F)

    vertex_one_ring: list[int]
    face_one_ring: list[int]

    #  First vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(0)
    assert len(vertex_one_ring) == 2
    assert len(face_one_ring) == 1
    assert vertex_one_ring[0] == 1
    assert vertex_one_ring[1] == 2
    assert face_one_ring[0] == 0

    #  Second vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(1)
    assert len(vertex_one_ring) == 2
    assert len(face_one_ring) == 1
    assert (vertex_one_ring[0] == 2)
    assert (vertex_one_ring[1] == 0)
    assert (face_one_ring[0] == 0)

    #  Third vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(2)
    assert len(vertex_one_ring) == 2
    assert len(face_one_ring) == 1
    assert vertex_one_ring[0] == 0
    assert vertex_one_ring[1] == 1
    assert face_one_ring[0] == 0


def test_square():
    F = np.array([[0, 1, 2], [1, 3, 2]])

    vertex_circulator = VertexCirculator(F)
    vertex_one_ring: list[int]
    face_one_ring: list[int]

    # First vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(0)
    assert len(vertex_one_ring) == 2
    assert len(face_one_ring) == 1
    assert vertex_one_ring[0] == 1
    assert vertex_one_ring[1] == 2
    assert face_one_ring[0] == 0

    # Second vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(1)
    assert len(vertex_one_ring) == 3
    assert len(face_one_ring) == 2
    assert vertex_one_ring[0] == 3
    assert vertex_one_ring[1] == 2
    assert vertex_one_ring[2] == 0
    assert face_one_ring[0] == 1
    assert face_one_ring[1] == 0

    # Third vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(2)
    assert len(vertex_one_ring) == 3
    assert len(face_one_ring) == 2
    assert vertex_one_ring[0] == 0
    assert vertex_one_ring[1] == 1
    assert vertex_one_ring[2] == 3
    assert face_one_ring[0] == 0
    assert face_one_ring[1] == 1

    # Third vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(3)
    assert len(vertex_one_ring) == 2
    assert len(face_one_ring) == 1
    assert vertex_one_ring[0] == 2
    assert vertex_one_ring[1] == 1
    assert face_one_ring[0] == 1
