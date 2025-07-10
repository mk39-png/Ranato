"""
Methods to compute the boundaries of a mesh
"""

# from ..core.common import *
from ..core.halfedge import *
# from ..core.quadratic_spline_surface import *
import numpy as np


def compute_face_boundary_edges(F: np.ndarray) -> list[tuple[int, int]]:
    """
    Given a mesh, compute the edges on the boundary.

    Args:
        F: mesh faces

    Returns:
        face_boundary_edges (list[tuple[int, int]]): edges of the triangles that are boundaries, indexed by face and opposite corner local face vertex index
    """
    assert F.dtype == np.int64

    # Build halfedge for the mesh
    # TODO: use HalfEdge's definition of "Index"
    corner_to_he: list[list[Index]]
    he_to_corner: list[tuple[Index, Index]]
    halfedge = HalfEdge(F, corner_to_he, he_to_corner)

    # Get boundary halfedges
    boundary_halfedges: list[Index] = halfedge.build_boundary_halfedge_list()

    # Get boundary face corners opposite halfedge
    face_boundary_edges: list[tuple[int, int]] = [] * len(boundary_halfedges)
    for i, halfedges in enumerate(boundary_halfedges):
        face_boundary_edges[i] = he_to_corner[boundary_halfedges[i]]

    return face_boundary_edges


def compute_boundary_vertices(F: np.ndarray) -> list[int]:
    """
    Given a mesh, compute the vertices on the boundary.

    Args: 
        F: mesh_faces

    Returns:
        boundary_vertices: vertices of the mesh on the boundary

    """
    assert F.dtype == np.int64

    # Get face boundary edges
    face_boundary_edges: list[tuple[int, int]] = compute_face_boundary_edges(F)

    # Get boolean array of boundary indices
    num_vertices: int = F.max() + 1
    is_boundary_vertex: list[bool] = [False] * num_vertices

    for i, boundary_edge in enumerate(face_boundary_edges):
        # Mark boundary edge endpoints as boundary vertices
        face_index: int = face_boundary_edges[i][0]
        face_vertex_index: int = face_boundary_edges[i][1]
        is_boundary_vertex[F(face_index, (face_vertex_index + 1) % 3)] = True
        is_boundary_vertex[F(face_index, (face_vertex_index + 2) % 3)] = True

    # Convert boolean array to index vector
    unsigned_boundary_vertices = convert_boolean_array_to_index_vector(
        is_boundary_vertex)
    boundary_vertices = convert_unsigned_vector_to_signed(
        unsigned_boundary_vertices)

    return boundary_vertices
