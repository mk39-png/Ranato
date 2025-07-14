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
from ..core.vertex_circulator import *
HalfedgeIndex = int
Index = int


def build_halfedge_to_edge_maps(opp: list[Index]) -> tuple[list[Index], list[tuple[Index, Index]]]:
    """
    Builds lists of halfedges to edges maps.

    Args:
        opp (list[Index]): in.

    Returns:
        he2e (list[Index]): halfedges to edge.
        e2he (list[tuple[Index, Index]]): edges to halfedges.
    """

    num_he: Index = len(opp)
    he2e: list[Index] = [None] * num_he
    e2he: list[tuple[Index, Index]] = []

    # Iterate over halfedges to build maps between halfedges and edges
    e_index: Index = 0
    for he in range(num_he):
        # Check if the halfedge is on the boundary
        is_boundary: bool = ((opp[he] < 0) or (opp[he] >= num_he))

        # Skip interior halfedges with lower index, but always process a boundary
        # halfedge
        if ((he >= opp[he]) or (is_boundary)):
            e2he.append((he, opp[he]))
            he2e[he] = e_index

            # Only valid for interior edges
            if (not is_boundary):
                he2e[opp[he]] = e_index

            # Update current edge index
            e_index += 1

    return he2e, e2he


class Halfedge:

    # ************
    # CONSTRUCTORS
    # ************
    # TODO: do corner_to_he and he_to_corner need to be used OUTSIDE of the class or not?
    # Because if yes and if there is some funky referencing going on, then make that whole process more Pythonic
    def __init__(self, F: np.ndarray = None,
                 corner_to_he: list[list[HalfedgeIndex]] = None,
                 he_to_corner: list[tuple[Index, Index]] = None) -> None:
        """
        TODO: deal with actually default constructor where Default trivial halfedge is made.
        Build halfedge mesh from mesh faces F with.
        """

        # Why does this start off with clear when nothing exists yet?
        # self.clear()
        # NOTE: ensuring that F is a matrix rather than a vector...
        assert F.ndim > 1

        # TODO: store F into m_F
        self.m_F = F
        num_faces: Index = F.shape[0]
        num_vertices: int = F.max() + 1
        num_halfedges: int = 3 * num_faces

        # TODO: check validity if that's something that needs to be done.
        # if CHECK_VALIDITY:
        #     if !is_manifold(F):
        #         logger.error("Input mesh is not manifold")
        #         cls.clear()
        #         return

        # Build maps between corners and halfedges
        # TODO: save these below into the actual class itself...
        self.corner_to_he, self.he_to_corner = self.build_corner_to_he_maps(
            num_faces)

        # Iterate over faces to build next, face, and to arrays
        self.m_next = [self.INVALID_HALFEDGE_INDEX] * num_halfedges
        self.m_face = [self.INVALID_FACE_INDEX] * num_halfedges
        self.m_to = [self.INVALID_VERTEX_INDEX] * num_halfedges
        self.m_from = [self.INVALID_VERTEX_INDEX] * num_halfedges
        for face_index in range(num_faces):
            for i in range(3):
                current_he: Index = self.corner_to_he[face_index][i]
                next_he: Index = self.corner_to_he[face_index][(i + 1) % 3]
                self.m_next[current_he] = next_he
                self.m_face[current_he] = face_index
                self.m_to[current_he] = F[face_index, (i + 2) % 3]
                self.m_from[current_he] = F[face_index, (i + 1) % 3]

        # Build out and f2he arrays
        self.m_out = [-1] * num_vertices
        self.m_f2he = [-1] * num_faces
        for he_index in range(num_halfedges):
            self.m_out[self.m_to[he_index]] = self.m_next[he_index]
            self.m_f2he[self.m_face[he_index]] = he_index

        # Iterate over vertices to build opp using a vertex circulator
        # Note that this is the main difficulty in constructing halfedge from VF
        vertex_circulator = VertexCirculator(F)
        self.m_opp = [-1] * 3 * num_faces
        for vertex_index in range(num_vertices):
            # Get vertex one ring
            vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(
                vertex_index)

            # Determine if we are in a boundary case
            is_boundary: bool = (vertex_one_ring[0] != vertex_one_ring[-1])
            num_adjacent_faces: Index = len(face_one_ring)

            # TODO: below code was generated. Confirm it works/is equivalent
            num_interior_edges: Index = num_adjacent_faces - \
                1 if is_boundary else num_adjacent_faces

            # Build opposite arrays
            for i in range(num_interior_edges):
                # Get current face prev (cw) halfedge from the vertex
                fi: Index = face_one_ring[i]
                # TODO: confirm I'm doing correct slicing like Eigen .row()
                fi_vertex_index: Index = find_face_vertex_index(
                    F[fi, :], vertex_index)
                current_he: Index = self.corner_to_he[fi][(
                    fi_vertex_index + 1) % 3]

                # Get next (ccw) face next (ccw) halfedge from the vertex
                fj: Index = face_one_ring[(i + 1) % num_adjacent_faces]
                fj_vertex_index: Index = find_face_vertex_index(
                    F[fj, :], vertex_index)
                opposite_he: Index = self.corner_to_he[fj][(
                    fj_vertex_index + 2) % 3]

                # Assign opposite halfedge
                self.m_opp[current_he] = opposite_he

        # Build maps between edges and halfedges
        self.m_he2e, self.m_e2he = build_halfedge_to_edge_maps(self.m_opp)

        # Set sizes
        self.m_num_halfedges = num_halfedges
        self.m_num_faces = num_faces
        self.m_num_vertices = num_vertices
        self.m_num_edges = len(self.m_e2he)

        #  Check validity
        # #if CHECK_VALIDITY
        #   if (!is_valid()) {
        #     spdlog::error("Could not build halfedge");
        #     clear();
        #     return;
        #   }
        # #endif

    # ******
    # PUBLIC
    # ******

    # **************
    # Element counts
    # **************

    @property
    def num_halfedges(self) -> Index:
        return self.m_num_halfedges

    @property
    def num_faces(self) -> Index:
        return self.m_num_faces

    @property
    def num_vertices(self) -> Index:
        return self.m_num_vertices

    @property
    def num_edges(self) -> Index:
        return self.m_num_edges

    # *********
    # Adjacency
    # *********
    def next_halfedge(self, he: Index) -> Index:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_HALFEDGE_INDEX
        return self.m_next[he]

    def opposite_halfedge(self, he: Index) -> Index:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_HALFEDGE_INDEX
        return self.m_opp[he]

    def halfedge_to_face(self, he: Index) -> Index:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_FACE_INDEX
        return self.m_face[he]

    def halfedge_to_head_vertex(self, he: Index) -> Index:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_VERTEX_INDEX
        return self.m_to[he]

    def halfedge_to_tail_vertex(self, he: Index) -> Index:
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
        # NOTE: equivalent to C++ pair .first
        return self.m_e2he[e][0]

    def edge_to_second_halfedge(self, e: Index) -> Index:
        # NOTE: equivalent to C++ pair .second
        return self.m_e2he[e][1]

    @property
    def get_halfedge_to_edge_map(self) -> list[Index]:
        return self.m_he2e

    @property
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

    @property
    def build_boundary_edge_list(self) -> list[Index]:
        """
            Args:
                None
            Returns:
                boundary_halfedges (list[Index]): out
        """
        boundary_edges: list[Index] = []

        for ei in range(self.m_num_edges):
            if (self.is_boundary_edge(ei)):
                boundary_edges.append(ei)

        return boundary_edges

    @property
    def build_boundary_halfedge_list(self) -> list[Index]:
        """
            Args:
                None
            Returns:
                boundary_halfedges (list[Index]): out
        """
        boundary_halfedges: list[Index] = []

        for hi in range(self.m_num_halfedges):
            if (self.is_boundary_halfedge(hi)):
                boundary_halfedges.append(hi)

        return boundary_halfedges

    @property
    def get_corner_to_he(self) -> list[list[int]]:
        return self.corner_to_he

    @property
    def get_he_to_corner(self) -> list[tuple[int, int]]:
        return self.he_to_corner

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

    def build_corner_to_he_maps(self, num_faces: Index) -> tuple[list[list[Index]], list[tuple[Index, Index]]]:
        """
        Builds list for corner to he and he to corner maps.

        Args:
            num_faces: in.

            # TODO: write this whole docstring in markdown notation or whatnot
        Returns:
            corner_to_he (list[list[Index]]): output.
            he_to_corner (list[tuple[Index, Index]]): out
        """

        # FIXME: resizing probably all works, but find a way that's neater than whatever is below
        corner_to_he: list[list[Index]] = [None] * num_faces
        he_to_corner: list[tuple[Index, Index]] = [None] * 3 * num_faces

        # Iterate over faces to build corner to he maps
        he_index: Index = 0

        for face_index in range(num_faces):
            corner_to_he[face_index] = [None] * 3

            for i in range(3):
                # Assign indices
                corner_to_he[face_index][i] = he_index
                he_to_corner[he_index] = (face_index, i)

                # Update current face index
                he_index += 1

        return corner_to_he, he_to_corner

    # *********************
    # Index validity checks
    # *********************
    def is_valid_halfedge_index(self, he: Index) -> bool:
        if (he < 0):
            return False
        if (he >= self.num_halfedges):
            return False
        return True

    def is_valid_vertex_index(self, vertex_index: Index) -> bool:
        if (vertex_index < 0):
            return False
        if (vertex_index >= self.num_vertices):
            return False
        return True

    def is_valid_face_index(self, face_index: Index) -> bool:
        if (face_index < 0):
            return False
        if (face_index >= self.num_faces):
            return False
        return True

    def is_valid_edge_index(self, edge_index: Index) -> bool:
        if (edge_index < 0):
            return False
        if (edge_index >= self.num_edges):
            return False
        return True

    # **************************
    # Invalid index constructors
    # **************************
    INVALID_HALFEDGE_INDEX = -1
    INVALID_VERTEX_INDEX = -1
    INVALID_FACE_INDEX = -1
    INVALID_EDGE_INDEX = -1

    def is_valid(self) -> bool:
        if len(self.m_next) != self.num_halfedges:
            logger.error("next domain not in bijection with halfedges")
            return False
        if len(self.m_opp) != self.num_halfedges:
            logger.error("opp domain not in bijection with halfedges")
            return False
        if len(self.m_he2e) != self.num_halfedges:
            logger.error("he2e domain not in bijection with halfedges")
            return False
        if len(self.m_to) != self.num_halfedges:
            logger.error("to domain not in bijection with halfedges")
            return False
        if len(self.m_from) != self.num_halfedges:
            logger.error("from domain not in bijection with halfedges")
            return False
        if len(self.m_face) != self.num_halfedges:
            logger.error("face domain not in bijection with halfedges")
            return False
        if len(self.m_e2he) != self.num_edges:
            logger.error("e2he domain not in bijection with edges")
            return False
        if len(self.m_out) != self.num_vertices:
            logger.error("out domain not in bijection with vertices")
            return False
        if len(self.m_f2he) != self.num_faces:
            logger.error("f2he domain not in bijection with faces")
            return False
        if self.m_F.shape[0] != self.num_faces:
            logger.error("F rows not in bijection with faces")
            return False

        return True
