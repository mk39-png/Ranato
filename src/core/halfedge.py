"""
Class to build halfedge from VF

TODO The cleanest way to handle this is to fill boundaries with faces
and handle boundary cases with these to avoid invalid operations. The
interface should be chosen with care to balance elegance and versatility.

Mesh halfedge representation. Supports meshes with boundary and basic
topological information. Can be initialized from face topology information.
"""

from ..core.common import *
import logging
Index = int

logger = logging.getLogger(__name__)


def build_halfedge_to_edge_maps() -> None:
    pass


class HalfEdge:
    # ******
    # PUBLIC
    # ******
    # ************
    # CONSTRUCTORS
    # ************
    INVALID_HALFEDGE_INDEX = -1
    INVALID_VERTEX_INDEX = -1
    INVALID_FACE_INDEX = -1
    INVALID_EDGE_INDEX = -1

    def __init__(self, F: np.ndarray, corner_to_he: list[list[Index]], he_to_corner: list[tuple[Index, Index]]):
        self.clear()

        # TODO: store F into m_F
        self.m_F = F
        num_faces: Index = F.shape[0]
        num_vertices: int = F.max() + 1
        num_halfedges = 3 * num_faces

        # TODO: check validity if that's something that needs to be done.
        # if CHECK_VALIDITY:
        #     if !is_manifold(F):
        #         logger.error("Input mesh is not manifold")
        #         cls.clear()
        #         return

        # Build maps between corners and halfedges
        self.build_corner_to_he_maps(num_faces, corner_to_he, he_to_corner)

        # Iterate over faces to build next, face, and to arrays
        # TODO: there's some resizing of m_next, m_face, m_to, and m_from
        for face_index in range(num_faces):
            for i in range(3):
                current_he: Index = corner_to_he[face_index][i]

        pass

    # **************
    # Element counts
    # **************

    def num_halfedges(self) -> Index:
        return self.m_num_halfedges

    def num_faces(self) -> Index:
        return self.m_num_faces

    def num_vertices(self) -> Index:
        return self.m_num_vertices

    def num_edges(self) -> Index:
        return self.m_num_edges

    # *********
    # Adjacency
    # *********
    def next_halfedge(self, he: int) -> int:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_HALFEDGE_INDEX
        return self.m_next[he]

    def opposite_halfedge(self, he: int) -> int:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_HALFEDGE_INDEX
        return self.m_opp[he]

    def halfedge_to_face(self, he: int) -> int:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_FACE_INDEX
        return self.m_face[he]

    def halfedge_to_head_vertex(self, he: int) -> int:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_VERTEX_INDEX
        return self.m_to[he]

    def halfedge_to_tail_vertex(self, he: int) -> int:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_VERTEX_INDEX
        return self.m_from[he]

    # ********************
    # Edge Representations
    # ********************

    def halfedge_to_edge(self, he: Index) -> Index:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_EDGE_INDEX
        return self.m_he2e[he]

    def edge_to_halfedge(self, e: Index) -> tuple[Index, Index]:
        return self.m_e2he[e]

    def edge_to_first_halfedge(self, e: Index) -> Index:
        return self.m_e2he[e][0]

    def edge_to_second_halfedge(self, e: Index) -> Index:
        return self.m_e2he[e][1]

    def get_halfedge_to_edge_map(self) -> list[Index]:
        return self.m_he2e

    def get_edge_to_halfedge_map(self) -> list[tuple[Index, Index]]:
        return self.m_e2he

    # ******************
    # Element predicates
    # ******************

    def is_boundary_edge(self, e: Index) -> bool:
        if (not self.is_valid_halfedge_index(self.edge_to_first_halfedge(e))):
            return True
        if (not self.is_valid_halfedge_index(self.edge_to_second_halfedge(e))):
            return True
        return False

    def is_boundary_halfedge(self, he: Index) -> bool:
        return self.is_boundary_edge(self.halfedge_to_edge(he))

    def build_boundary_edge_list(self, boundary_edges: list[Index]) -> None:
        # TODO: maybe change this to return a boundary_edges list rather than modifying by reference.
        # TODO: some reserving of space in boundary_edges
        for ei in range(self.m_num_edges):
            if (self.is_boundary_edge(ei)):
                boundary_edges.append(ei)

    def build_boundary_halfedge_list(self, boundary_halfedges: list[Index]) -> None:
        # TODO: change to return list[Index] rather than modifying by reference?
        for hi in range(self.m_num_halfedges):
            if (self.is_boundary_halfedge(hi)):
                boundary_halfedges.append(hi)

    def clear(self) -> None:
        self.m_next.clear()
        self.m_opp.clear()
        self.m_he2e.clear()
        self.m_e2he.clear()
        self.m_to.clear()
        self.m_from.clear()
        self.m_face.clear()
        self.m_out.clear()
        self.m_f2he.clear()

        # TODO: do some clearing thing for m_F
        # self.m_F.resize(0, 0)

    # *******
    # PRIVATE
    # *******
    m_next: list[Index]
    m_opp:  list[Index]
    m_he2e: list[Index]
    m_e2he: list[tuple[Index, Index]]
    m_to:   list[Index]
    m_from: list[Index]
    m_face: list[Index]
    m_out:  list[Index]
    m_f2he: list[Index]
    m_F: np.ndarray
    m_num_vertices: Index
    m_num_faces: Index
    m_num_halfedges: Index
    m_num_edges: Index

    # TODO: decide between list of int or NumPy thing...
    def build_corner_to_he_maps(self, num_faces: Index, corner_to_he: list[list[Index]], he_to_corner: list[tuple[Index, Index]]) -> None:
        # FIXME: resizing of corner_to_he and whatnot... maybe not right?
        corner_to_he = [] * num_faces
        he_to_corner = [] * 3 * num_faces

        # he_to_corner = [] * 3 * num_faces

        # Iterate over faces to build corner to he maps
        he_index: Index = 0

        for face_index in range(self.num_faces):
            # FIXME: maybe not correct resizing?
            corner_to_he[face_index] = [None] * 3

            for i in range(3):
                # Assign indices
                corner_to_he[face_index][i] = he_index
                he_to_corner[he_index] = (face_index, i)

                # Update current face index
                he_index += 1

    # *********************
    # Index validity checks
    # *********************
    def is_valid_halfedge_index(self, he: Index) -> bool:
        if (he < 0):
            return False
        if (he >= self.m_num_halfedges):
            return False
        return True

    def is_valid_vertex_index(self, vertex_index: Index) -> bool:
        if (vertex_index < 0):
            return False
        if (vertex_index >= self.m_num_vertices):
            return False
        return True

    def is_valid_face_index(self, face_index: Index) -> bool:
        if (face_index < 0):
            return False
        if (face_index >= self.m_num_faces):
            return False
        return True

    def is_valid_edge_index(self, edge_index: Index) -> bool:
        if (edge_index < 0):
            return False
        if (edge_index >= self.m_num_edges):
            return False
        return True

    # **************************
    # Invalid index constructors
    # **************************
    # def __invalid_halfedge_index(self) -> int:
    #     return -1
    # def __invalid_vertex_index(self) -> int:
    #     return -1
    # def __invalid_face_index(self) -> int:
    #     return -1
    # def __invalid_edge_index(self) -> int:
    #     return -1

    def is_valid(self) -> bool:
        if len(self.m_next) != self.num_halfedges():
            logger.error("next domain not in bijection with halfedges")
            return False
        if len(self.m_opp) != self.num_halfedges():
            logger.error("opp domain not in bijection with halfedges")
            return False
        if len(self.m_he2e) != self.num_halfedges():
            logger.error("he2e domain not in bijection with halfedges")
            return False
        if len(self.m_to) != self.num_halfedges():
            logger.error("to domain not in bijection with halfedges")
            return False
        if len(self.m_from) != self.num_halfedges():
            logger.error("from domain not in bijection with halfedges")
            return False
        if len(self.m_face) != self.num_halfedges():
            logger.error("face domain not in bijection with halfedges")
            return False
        if len(self.m_e2he) != self.num_edges():
            logger.error("e2he domain not in bijection with edges")
            return False
        if len(self.m_out) != self.num_vertices():
            logger.error("out domain not in bijection with vertices")
            return False
        if len(self.m_f2he) != self.num_faces():
            logger.error("f2he domain not in bijection with faces")
            return False
        if self.m_F.shape[0] != self.num_faces():
            logger.error("F rows not in bijection with faces")
            return False

        return True
