"""
Class to build circulators around vertices in VF representation
"""

import numpy as np
from ..core.common import contains_vertex

import logging

# TODO: decide between SciPy and NumPy for implementing polynomials...
logger = logging.getLogger(__name__)


def contains_edge(face: np.ndarray, vertex_0: int, vertex_1: int) -> bool:
    """
    Return true iff the face contains the edge { vertex_0, vertex_1 }
    """
    return ((contains_vertex(face, vertex_0)) and
            (contains_vertex(face, vertex_1)))


def is_left_face(face: np.ndarray, vertex_0: int, vertex_1: int) -> bool:
    """
    Return true iff the face is to the left of the given edge
    """
    if ((face[0] == vertex_0) and (face[1] == vertex_1)):
        return True
    if ((face[1] == vertex_0) and (face[2] == vertex_1)):
        return True
    if ((face[2] == vertex_0) and (face[0] == vertex_1)):
        return True

    return False


def is_right_face(face: np.ndarray, vertex_0: int, vertex_1: int) -> bool:
    """
    Return true iff the face is to the right of the given edge
    """
    if ((face[1] == vertex_0) and (face[0] == vertex_1)):
        return True
    if ((face[2] == vertex_0) and (face[1] == vertex_1)):
        return True
    if ((face[0] == vertex_0) and (face[2] == vertex_1)):
        return True

    return False


def find_next_vertex(face: np.ndarray, vertex: int) -> int:
    """
    Get the index of the vertex in the face ccw from the given vertex
    """
    if (face[0] == vertex):
        return face[1]
    if (face[1] == vertex):
        return face[2]
    if (face[2] == vertex):
        return face[0]

    return -1


def find_prev_vertex(face: np.ndarray, vertex: int) -> int:
    """
    Get the index of the vertex in the face clockwise from the given vertex
    """
    if (face[0] == vertex):
        return face[2]
    if (face[1] == vertex):
        return face[0]
    if (face[2] == vertex):
        return face[1]

    return -1


def are_adjacent(face_0: np.ndarray, face_1: np.ndarray) -> bool:
    """
    Return true iff the two faces are adjacent
    """
    if (contains_edge(face_0, face_1[0], face_1[1])):
        return True
    if (contains_edge(face_0, face_1[1], face_1[2])):
        return True
    if (contains_edge(face_0, face_1[2], face_1[0])):
        return True

    return False


# TODO: Just use list of lists for simplicity rather than trying to replace with NumPy
def compute_adjacent_faces(F: np.ndarray, all_adjacent_faces: list[list[int]]) -> None:
    """
    Get list of all faces adjacent to each vertex
    """
    # Initialize adjacent faces list
    num_vertices: int = F.max() + 1

    # This supposedly allocates space into the vector.
    all_adjacent_faces.resize(num_vertices)

    # So, this removes all elements inside the list.
    for i, face in range(len(all_adjacent_faces)):
        face.clear()

    # FIXME: try converting all_adjacent_faces to utilize NumPy arrays rather than list of lists...
    for i in range(F.shape[0]):  # rows
        for j in range(F.shape[1]):  # cols
            all_adjacent_faces[F[i, j]].push_back(i)


def compute_vertex_one_ring_first_face(F: np.ndarray, vertex_index: int, adjacent_faces: list[int]) -> int:
    """
    Compute the first face of the vertex one ring, which should be right
    boundary face for a boundary vertex.
    """
    if (len(adjacent_faces) == 0):
        return -1

    #  Get arbitrary adjacent face to start and vertex on the face
    current_face: int = adjacent_faces[0]
    current_vertex: int = find_next_vertex(F[current_face, :], vertex_index)
    logger.info("Starting search for first face from vertex %s on face %s",
                current_vertex, F[current_face, :])

    # Cycle clockwise to a starting face
    for i in range(1, len(adjacent_faces)):
        # Get previous face or return if none exists
        prev_face: int = -1

        for j in range(len(adjacent_faces)):
            f: int = adjacent_faces[j]
            if (is_right_face(F.row(f), vertex_index, current_vertex)):
                prev_face = f
                break

        # Return current face if no previous face found
        if (prev_face == -1):
            return current_face

        # Get previous face and vertex
        current_face = prev_face
        current_vertex = find_prev_vertex(F.row(current_face), current_vertex)

    # If we have not returned yet, this is an interior vertex, and we return
    # the current face as an arbitrary choice
    return current_face


def compute_vertex_one_ring(F: np.ndarray, vertex_index: int, adjacent_faces: list[int], vertex_one_ring: list[int], face_one_ring: list[int]):
    """
    Compute the vertex one ring for a vertex index using adjacent faces.
    """

    num_faces = adjacent_faces.size()
    vertex_one_ring.resize(num_faces + 1)
    face_one_ring.resize(num_faces)

    if (adjacent_faces.empty()):
        return

    # Get first face and vertex
    face_one_ring[0] = compute_vertex_one_ring_first_face(
        F, vertex_index, adjacent_faces)
    vertex_one_ring[0] = find_next_vertex(
        F.row(face_one_ring[0]), vertex_index)

    # Get remaining one ring faces and vertices
    for i in range(1, num_faces):
        # Get next vertex
        vertex_one_ring[i] = find_next_vertex(
            F.row(face_one_ring[i - 1]), vertex_one_ring[i - 1])

        # Get next face
        for j in range(num_faces):
            f: int = adjacent_faces[j]
            if (is_left_face(F.row(f), vertex_index, vertex_one_ring[i])):
                face_one_ring[i] = f

    # Get final vertex(same as first for closed loop)
    logger.info("Adding last vertex for face {} from vertex {}", F.row(
        face_one_ring[num_faces - 1]), vertex_one_ring[num_faces - 1])

    vertex_one_ring[num_faces] = find_next_vertex(
        F.row(face_one_ring[num_faces - 1]), vertex_one_ring[num_faces - 1])
    logger.info("Last vertex: {}", vertex_one_ring[num_faces])


class VertexCirculator:
    # ***************
    # Constructor
    # ***************
    def __init__(self, F: np.ndarray):
        """
        Constructor for the vertex circulator from the faces of the mesh.

        Args:
            F: [in] input mesh faces

        Returns:
            None
        """
        #  Initialize adjacent faces list
        # TODO: what is maxCoeff FYI?
        num_vertices = F.max() + 1
        self.m_all_adjacent_faces = compute_adjacent_faces(F)

        # Compute face and vertex one rings
        # TODO: wait, how does the below work?
        self.m_all_vertex_one_rings = np.empty(
            shape=num_vertices, dtype=int)
        self.m_all_face_one_rings = np.empty(
            shape=num_vertices, dtype=int)

        for i in range(num_vertices):
            compute_vertex_one_ring(F,
                                    i,
                                    self.m_all_adjacent_faces[i],
                                    self.m_all_vertex_one_rings[i],
                                    self.m_all_face_one_rings[i])

    # ***************
    # Private Members
    # ***************
    m_F:                    np.ndarray[tuple[int, int], np.dtype[np.int_]]
    m_all_adjacent_faces:   np.ndarray[tuple[int, int], np.dtype[np.int_]]
    m_all_vertex_one_rings: np.ndarray[tuple[int, int], np.dtype[np.int_]]
    m_all_face_one_rings:   np.ndarray[tuple[int, int], np.dtype[np.int_]]

    # m_all_adjacent_faces: list[list[int]]
    # m_all_vertex_one_rings: list[list[int]]
    # m_all_face_one_rings: list[list[int]]

    # ***************
    # Public Members
    # ***************
    def get_one_ring(self, vertex_index: int, vertex_one_ring: np.ndarray, face_one_ring: np.ndarray) -> None:
        """
        Get the one ring of a vertex.

        The one ring of both faces and vertices counter clockwise around the
        vertex are returned. For boundary vertices, the faces and vertices start
        at the right boundary and traverse the faces in order to the left
        boundary. For interior vertices, an arbitrary start face is chosen, and
        the vertex one ring is closed so that v_0 = v_n.

        Args:
            vertex_index:    [in]  index of the vertex to get the one ring for
            vertex_one_ring: [out] vertices ccw around the one ring
            face_one_ring:   [out] faces ccw around the one ring

        Returns:
            None
        """

        # TODO: maybe just use a list of list of ints rather than a 2D NumPy array...
        vertex_one_ring = self.m_all_vertex_one_rings[vertex_index, :]
        face_one_ring = self.m_all_vertex_one_rings[vertex_index, :]
