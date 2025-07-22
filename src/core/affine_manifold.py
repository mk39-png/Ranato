"""
Representation of an affine manifold.
"""

from src.core.common import *
from src.core.halfedge import Halfedge
from src.core.vertex_circulator import VertexCirculator

import polyscope as ps
import mathutils

from dataclasses import dataclass


@dataclass
class VertexManifoldChart:
    """
    Local layout manifold chart in R2 of the one ring around a central vertex.

    The one ring is represented as a sequential list of the n faces and n + 1
    vertices around the central vertex along with the corresponding vertex uv
    positions. For i = 0,...,n-1, the ith vertex corresponds to to the first
    vertex in face i ccw from the central vertex.

    For an interior vertex, the one ring begins at an arbitrary face, and the
    nth vertex is the same as the first. For a boundary vertex, the one ring
    begins as the boundary face to the right of the vertex and ends at the left
    boundary face, and the nth vertex is generally different from the 0th.
    """

    # Index of the vertex in the affine manifold
    vertex_index: int

    # List of manifold vertex indices in the one ring
    vertex_one_ring: list[int]
    # List of manifold face indices in the one ring
    face_one_ring: list[int]

    # Local uv coordinates of the one ring vertices
    one_ring_uv_positions: np.ndarray

    # Mark boundary vertices
    is_boundary: bool = False

    # Mark cone vertices
    is_cone: bool = False

    # Mark vertices adjacent to a cone
    is_cone_adjacent: bool = False


@dataclass
class EdgeManifoldChart:
    """
    Local layout manifold chart in R2 of the triangles around an edge.
    An orientation is specified on the chart by a choice of top and bottom face.
    For an interior edge, there are always two adjacent faces. For a boundary
    edge, there is only one adjacent face. The bottom vertex and face indices
    are set to an out of range value (e.g., -1), and the bottom vertex is set to
    the empty vector.
    """

    # -- Face indices --
    top_face_index: Index
    bottom_face_index: Index

    # -- Vertex indices --
    left_vertex_index: Index
    right_vertex_index: Index
    top_vertex_index: Index
    bottom_vertex_index: Index

    # -- Vertex positions --
    left_vertex_uv_position: PlanarPoint
    right_vertex_uv_position: PlanarPoint
    top_vertex_uv_position: PlanarPoint
    bottom_vertex_uv_position: PlanarPoint

    # True iff the edge is on the boundary
    is_boundary: bool


# @dataclass
class FaceManifoldChart:
    """
    Local layout manifold chart in R2 of a triangle.
    This is the same as global uv positions when these are provided.
    """
    # # -- Face indices --
    # face_index: Index

    # # -- Vertex positions --
    # # NOTE: self_uv_position must be a size 3 list with PlanarPoint elements
    # # assert len(face_uv_positions) == 3
    # # NOTE: placeholder values below to denote that face_uv_positions is list of length 3
    # face_uv_positions: list[PlanarPoint] = [PlanarPoint(shape=(1, 2)),
    #                                         PlanarPoint(shape=(1, 2)),
    #                                         PlanarPoint(shape=(1, 2))]

    # # -- Global information --
    # # True iff the edge is on the boundary
    # is_boundary: bool = False
    # # Mark faces adjacent to a cone
    # is_cone_adjacent: bool = False
    # # Mark individual corners adjacent to a cone
    # is_cone_corner: list[bool] = [False, False, False]

    def __init__(self, face_index: Index, face_uv_positions: list[PlanarPoint],
                 is_boundary: bool = False, is_cone_adjacent: bool = False,
                 is_cone_corner: list[bool] = [False, False, False]) -> None:

        # -- Face indices --
        self.face_index: Index = face_index

        # -- Vertex positions --
        # NOTE: self_uv_position must be a size 3 list with PlanarPoint elements
        assert len(face_uv_positions) == 3
        self.face_uv_positions: list[PlanarPoint] = face_uv_positions

        # -- Global information --
        # True iff the edge is on the boundary
        self.is_boundary: bool = is_boundary
        # Mark faces adjacent to a cone
        self.is_cone_adjacent: bool = is_cone_adjacent
        # Mark individual corners adjacent to a cone
        self.is_cone_corner: list[bool] = is_cone_corner


class AffineManifold:
    """
    Representation for an affine manifold, which is a topological manifold F
    equipped with a discrete metric l that satisfies the triangle inequality.
    """

    def __init__(self, F: np.ndarray, global_uv: np.ndarray, F_uv: np.ndarray) -> None:
        """Default constructor for a trivial manifold

        Constructor for a cone manifold from a global parametrization.
        @param[in] F: faces of the cone manifold
        @param[in] global_uv: global layout of the manifold
        @param[in] F_uv: faces of the global layout
        """

        # Check the input
        # if CHECK_VALIDITY
        if not is_manifold(F):
            logger.error("Input mesh is not manifold")
            self.clear()
            return

        if not is_manifold(F_uv):
            logger.error("Input mesh is not manifold")
            self.clear()
            return

        # Comparing row sizes
        if F_uv.shape[ROWS] != F.shape[ROWS]:
            logger.error("Input mesh and uv mesh have different sizes")
            self.clear()
            return
        # endif

        # *** Topology information ***
        # TODO (from ASOC): The faces are duplicated in the halfedge. Our halfedge alway retains
        # the original VF topology, so there is no need to maintain both separately
        # TODO: check if below is actually retrieving what we need properly
        self.m_F: np.ndarray = F
        self.m_halfedge = Halfedge(F)  # Build halfedge
        he_to_edge: list[Index] = self.m_halfedge.get_halfedge_to_edge_map
        self.m_corner_to_he: list[list[Index]] = self.m_halfedge.get_corner_to_he
        self.m_corner_to_edge: list[list[Index]] = self._build_corner_to_edge_map(
            self.m_corner_to_he, he_to_edge)
        self.m_he_to_corner: list[tuple[Index, Index]] = self.m_halfedge.get_he_to_corner

        # *** Global metric information ***
        self.m_global_uv: np.ndarray = global_uv
        self.m_F_uv: np.ndarray = F_uv

        # ** Build edge lengths and charts from the global uv **
        self.m_l: list[list[float]] = self._build_lengths_from_global_uv(F_uv, global_uv)
        # * Local metric information *
        self.m_vertex_charts: list[VertexManifoldChart] = self._build_vertex_charts_from_lengths(
            F, self.m_l)
        self.m_edge_charts: list[EdgeManifoldChart] = self._build_edge_charts_from_lengths(
            F, self.m_halfedge, self.m_l)
        self.m_face_charts: list[FaceManifoldChart] = self._build_face_charts(F, global_uv, F_uv)

        # Align charts with the input parameterization
        self._align_local_charts(global_uv, F_uv)

        # Mark vertices, edges, and faces adjacent to cones
        self._mark_cones()

        # Check validity
        if not self._is_valid_affine_manifold():
            logger.error("Could not build a cone manifold")
            self.clear()

    @property
    def num_faces(self) -> Index:
        """
        Get the number of faces in the manifold

        @return number of faces in the manifold
        """
        return self.m_F.shape[0]

    @property
    def num_vertices(self) -> int:
        """
        Get the number of vertices in the manifold

        @return number of vertices in the manifold
        """
        return len(self.m_vertex_charts)

    @property
    def get_faces(self) -> np.ndarray:
        """
        Get faces for the manifold

        @return faces of the manifold
        """
        assert self.m_F.dtype == np.int64
        return self.m_F

    @property
    def get_halfedge(self) -> Halfedge:
        """
        Get halfedge for the manifold

        @return halfedge of the manifold
        """
        return self.m_halfedge

    @property
    def get_he_to_corner(self) -> list[tuple[Index, Index]]:
        """
        Get halfedge to corner map for the manifold

        @return halfedge to corner map of the manifold
        """
        return self.m_he_to_corner

    @property
    def get_F_uv(self) -> np.ndarray:
        """
        Get faces for the manifold parametrization

        @return faces of the manifold layout
        """
        return self.m_F_uv

    def get_vertex_chart(self, vertex_index: Index) -> VertexManifoldChart:
        """
        Get an isometric chart for the vertex with the given index.

        Note that this will be a homeomorphism about the vertex if and only if the
        metric is flat there. However, any such chart may be made into a
        (nonisometric) homeomorphism about the vertex by composition with a
        suitable angle normalization map (r, theta) -> (r, 2 * pi * theta / (2 *
        pi - K)), where K is the Gaussian curvature at the vertex.

        @param[in] vertex_index: index of the vertex for the chart
        @return chart for the given vertex
        """
        return self.m_vertex_charts[vertex_index]

    def get_edge_chart(self, face_index: Index, face_vertex_index: Index) -> EdgeManifoldChart:
        """
        Get an isometric chart for the edge opposite the corner with the given
        face index and vertex index within the face.

        @param[in] face_index: index of a face containing the target edge
        @param[in] face_vertex_index: index of the corner opposite the edge in the
        face
        @return chart for the given edge
        """
        edge_index: Index = self.m_corner_to_edge[face_index][face_vertex_index]
        return self.m_edge_charts[edge_index]

    def get_face_chart(self, face_index: Index) -> FaceManifoldChart:
        """
        Get an isometric chart for the given face.

        @param[in] face_index: index of a face
        @return chart for the given face
        """
        return self.m_face_charts[face_index]

    def get_face_corner_charts(self, face_index: Index) -> list:
        """
        Get the portions of the isometric vertex charts corresponding to the
        corners of a given face.

        In particular, for a face ijk, the local chart layouts are given for:
            [0] vertices j and k in the vertex chart for vertex i
            [1] vertices k and i in the vertex chart for vertex j
            [2] vertices i and j in the vertex chart for vertex k

        @param[in] face_index: index of the face for the chart segments
        @param[out] corner_uv_positions: chart uv positions as enumerated above
        """
        corner_uv_positions: list[np.ndarray] = [np.ndarray(shape=(2, 2)),
                                                 np.ndarray(shape=(2, 2)),
                                                 np.ndarray(shape=(2, 2))]

        for i in range(3):
            # Get the chart for vertex i in the given face
            vertex_index = self.m_F[face_index, i]
            chart: VertexManifoldChart = self.m_vertex_charts[vertex_index]

            # Iterate over the one ring of face vertex i
            for j, face in enumerate(chart.face_one_ring):
                # NOTE: face equivalent to chart.face_one_ring[j]

                # Skip faces until the input face is found
                if (face != face_index):
                    continue

                # Get the uv coordinates of the other two face vertices in the chart of face vertex i
                first_edge: int = j
                second_edge: int = j + 1

                # TODO: check slicing with Eigen .row()
                corner_uv_positions[i][0,
                                       :] = chart.one_ring_uv_positions[first_edge, :]
                corner_uv_positions[i][1,
                                       :] = chart.one_ring_uv_positions[second_edge, :]
                break

        assert len(corner_uv_positions) == 3
        return corner_uv_positions

    def get_face_edge_charts(self, face_index: Index) -> list:
        """
        Get the portion of the edge charts contained in the interior of the given
        face.

        In particular, for a face ijk, the local chart layouts are given for:
            [0] vertices j, k, i in the vertex chart for edge ij
            [1] vertices k, i, j in the vertex chart for edge jk
            [2] vertices i, j, k in the vertex chart for edge ki

        @param[in] face_index: index of the face for the charts
        @param[out] face_edge_uv_positions: uv positions contained in the given
        face
        """
        face_edge_uv_positions: list[PlanarPoint] = [PlanarPoint(shape=(1, 2)),
                                                     PlanarPoint(shape=(1, 2)),
                                                     PlanarPoint(shape=(1, 2))]

        # Iterate over edges
        for i in range(3):
            # Get the chart for edge jk opposite vertex i in the given face
            chart: EdgeManifoldChart = self.get_edge_chart(face_index, i)

            # Break into cases depending on if the face is the top or bottom face in the chart
            if chart.top_face_index == face_index:
                # TODO: double check slicing with Eigen .row()
                face_edge_uv_positions[i][0,
                                          :] = chart.right_vertex_uv_position
                face_edge_uv_positions[i][1, :] = chart.top_vertex_uv_position
                face_edge_uv_positions[i][2, :] = chart.left_vertex_uv_position
            elif chart.bottom_face_index == face_index:
                face_edge_uv_positions[i][0, :] = chart.left_vertex_uv_position
                face_edge_uv_positions[i][1,
                                          :] = chart.bottom_vertex_uv_position
                face_edge_uv_positions[i][2,
                                          :] = chart.right_vertex_uv_position
            else:
                logger.error(
                    "Face %s not found in the given edge chart", face_index)

    def get_face_global_uv(self, face_index: Index) -> list:
        """
        @brief Get the uv coordinates of the face.

        @param[in] face_index: index of the face for the chart
        @param[out] face_edge_uv_positions: global uv positions of the face
        """
        return self.get_face_chart(face_index).face_uv_positions

    def compute_curvature(self, vertex_index: Index) -> float:
        """
        Compute the curvature curvature at the given vertex.

        Gaussian curvature is used for interior vertices and geodesic curvature
        for boundary vertices.

        @param[in] vertex_index: index of the vertex
        @return curvature at the given vertex
        """
        chart: VertexManifoldChart = self.m_vertex_charts[vertex_index]

        # Get zero uv coordinate (location of the central vertex)
        zero: PlanarPoint = np.zeros(shape=(1, 2))

        # Compute cone angle
        cone_angle: float = 0.0

        for j in range(len(chart.face_one_ring)):
            cone_angle += angle_from_positions(2,
                                               zero,
                                               chart.one_ring_uv_positions[[j], :],
                                               chart.one_ring_uv_positions[[j + 1], :])

        # Compute geodesic curvature for boundary vertices and Gaussian curvature for
        # interior vertices
        if chart.is_boundary:
            return math.pi - cone_angle
        else:
            return (2 * math.pi) - cone_angle

    def is_boundary(self, vertex_index: Index) -> bool:
        """
        Determine if the vertex is on the boundary

        @param[in] vertex_index: index of the vertex
        @return true iff the vertex is on the boundary
        """
        return self.get_vertex_chart(vertex_index).is_boundary

    def is_flat(self, vertex_index: Index) -> bool:
        """
        Determine if the manifold is flat at the given vertex, i.e. has zero
        Gaussian curvature or is a boundary vertex.

        @param[in] vertex_index: index of the vertex
        @return true iff the manifold is flat at the vertex
        """
        # All vertices with zero curvature are flat
        if float_equal_zero(self.compute_curvature(vertex_index), 1e-5):
            return True

        # All vertices on the boundary are flat
        if self.is_boundary(vertex_index):
            return True

        return False

    def compute_flat_vertices(self) -> list[Index]:
        """
        Get list of all flat vertices in the manifold

        @param[out] flat_vertices: list of flat vertices
        """
        flat_vertices: list[Index] = []

        for vertex_index in range(self.num_vertices):
            if self.is_flat(vertex_index):
                flat_vertices.append(vertex_index)

        return flat_vertices

    def compute_cones(self) -> list[Index]:
        """
        Get list of all cones in the manifold

        @param[out] cones: list of cone vertices
        """
        cones: list[Index] = []

        for vertex_index in range(self.num_vertices):
            if not self.is_flat(vertex_index):
                logger.debug("Getting cone %s of curvature %s",
                             vertex_index, self.compute_curvature(vertex_index))
                cones.append(vertex_index)
        return cones

    def compute_cone_corners(self) -> list[list[bool]]:
        """
        Get boolean mask of all cones corners in the manifold.
        NOTE: returns a list of list with 3 bool elements.

        @param[out] is_cone_corner: true iff corner i, j is a cone
        """
        # NOTE: initializing list with False values, which does not mean anything since the ASOC code just has an array space allocation.
        # Simply there because it needed a bool type. Otherwise, I would put None, but the typing system complains.
        is_cone_corner: list[list[bool]] = [[False, False, False]
                                            for _ in range(self.num_faces)]

        for fi in range(self.num_faces):
            for k in range(3):
                is_cone_corner[fi][k] = self.get_face_chart(
                    fi).is_cone_corner[k]

        return is_cone_corner

    def compute_cone_points(self, V: np.ndarray) -> np.ndarray:
        """
        Compute a matrix of cone point positions from mesh vertex.

        @param[in] V: mesh vertex positions
        @param[out] cone_points: cone positions w.r.t. V
        """
        # Compute the cone indices
        cones: list[Index] = self.compute_cones()

        # Build the cone points from the vertex set
        num_cones: Index = len(cones)

        # shape is (num_cones, V.cols())
        cone_points: np.ndarray = np.ndarray(shape=(num_cones, V.shape[1]))

        for i in range(num_cones):
            ci: Index = cones[i]
            # FIXME: likely big issue with slicing and broadcasting below
            cone_points[i, :] = V[ci, :]

        return cone_points

    # TODO: remove the function below since it's redundant in the Python version
    def generate_cones(self) -> list[int]:
        """
        Return list of all cones in the manifold

        @return list of cone vertices
        """
        return self.compute_cones()

    def compute_boundary_vertices(self) -> list[Index]:
        """
        Get list of all boundary vertices in the manifold

        @param[out] boundary_vertices: list of boundary vertices
        """

        boundary_vertices: list[Index] = []

        for vertex_index in range(self.num_vertices):
            if self.is_boundary(vertex_index):
                boundary_vertices.append(vertex_index)

        return boundary_vertices

    # TODO: remove the function below since it's redundant?
    def generate_boundary_vertices(self) -> list[Index]:
        """
        Return list of all boundary vertices in the manifold

        @return list of boundary vertices
        """
        return self.compute_boundary_vertices()

    def mark_cone_adjacent_vertex(self, vertex_index: Index) -> None:
        """
        @brief Mark a vertex as adjacent to a cone

        @param[in] vertex_index: vertex to mark
        """
        self.m_vertex_charts[vertex_index].is_cone_adjacent = True

    def mark_cone_adjacent_face(self, face_index: Index) -> None:
        """
        @brief Mark a face as adjacent to a cone

        @param[in] face_index: face to mark
        """
        self.m_face_charts[face_index].is_cone_adjacent = True

    @property
    def get_global_uv(self) -> np.ndarray:
        """
        Get global uv coordinates

        @return global uv coordinates, or the empty matrix if they do not exist
        """
        return self.m_global_uv

    def cut_cone_edges(self) -> None:
        """
        Cut edges adjacent to cones so that a planar layout is possible around
        them.
        """
        # TODO: fix the names of these attribute getters...
        F: np.ndarray = self.get_faces

        # Get cone vertices
        cones: list[Index] = self.compute_cones()

        # For each cone, choose an edge and make it a boundary edge
        for i, _ in enumerate(cones):
            # Get cone vertex chart
            vi: Index = cones[i]
            vertex_chart: VertexManifoldChart = self.get_vertex_chart(vi)

            # Get edge chart adjacent to the cone edge
            face_index: Index = vertex_chart.face_one_ring[0]
            # FIXME: mostly likely going to be a problem with slicing below.., maybe.
            # Though, probably not since find_face_vertex_index() expects 1D arrays, luckily.
            # And slicing like below gives 1D arrays of shape (n, )
            face_vertex_index: Index = find_face_vertex_index(
                F[face_index, :], vi)

            edge_chart: EdgeManifoldChart = self.get_edge_chart(
                face_index, face_vertex_index)

            # Mark edge and endpoints as boundaries
            edge_index: Index = self.m_corner_to_edge[face_index][face_vertex_index]
            v0: Index = edge_chart.left_vertex_index
            v1: Index = edge_chart.right_vertex_index
            self.m_edge_charts[edge_index].is_boundary = True
            self.m_vertex_charts[v0].is_boundary = True
            self.m_vertex_charts[v1].is_boundary = True

    def add_to_viewer(self, V: np.ndarray, color: np.ndarray = GOLD_YELLOW) -> None:
        """
        Add the cone manifold and its data to the polyscope viewer with name
        'cone_manifold'

        @param[in] V: mesh vertex positions
        @param[in] color: color for the affine manifold in the viewer
        """
        ps.init()

        # Add manifold
        F: np.ndarray = self.get_faces
        cone_manifold: ps.SurfaceMesh = ps.register_surface_mesh("cone_manifold", V, F)
        cone_manifold.set_edge_width(1)
        # TODO: probably going to be a problem interacting with NumPy arrays...
        cone_manifold.set_color(color)

        # Add cone points
        cone_points: np.ndarray = self.compute_cone_points(V)
        cones: ps.PointCloud = ps.register_point_cloud("cones", cone_points)
        # TODO: might have problem with tuples being passed in rather than glm3 vector
        cones.set_color((0.5, 0.0, 0.0))

    def view(self, V: np.ndarray) -> None:
        """
        View the cone manifold and its data

        @param[in] V: mesh vertex positions
        """
        self.add_to_viewer(V)
        ps.show()

    def screenshot(self,
                   filename: str,
                   V: np.ndarray,
                   camera_position: np.ndarray,
                   camera_target: np.ndarray,
                   use_orthographic: bool) -> None:
        """
        Save an image of the cone manifold and its data to file.

        @param[in] filename: file to save the screenshot to
        @param[in] V: mesh vertex positions
        @param[in] camera_position: camera position for the screenshot
        @param[in] camera_target: camera target for the screenshot
        @param[in] use_orthographic: use orthographic perspective if true
        """
        # Add the contour network to the surface
        self.add_to_viewer(V)

        # Build the cameras for the viewer
        # TODO: try just passing in these numpy vectors

        # Set up the cameras
        ps.look_at(camera_position.flatten(), camera_target.flatten())

        if use_orthographic:
            # TODO: is this the right interaction?
            ps.set_view_projection_mode("Orthographic")

        # Take the screenshot
        ps.screenshot(filename)
        logger.info("Screenshot saved to %s", filename)
        ps.remove_all_structures()

    def clear(self) -> None:
        """
        Clear all internal data for a trivial cone manifold
        """
        self.m_F[:, :] = 0.0
        self.m_corner_to_he.clear()
        self.m_corner_to_edge.clear()
        self.m_he_to_corner.clear()
        self.m_halfedge.clear()

        self.m_l.clear()
        self.m_global_uv[:, :] = 0.0
        self.m_F_uv[:, :] = 0.0

        self.m_vertex_charts.clear()
        self.m_edge_charts.clear()
        self.m_face_charts.clear()

    # *****************
    # Protected Methods
    # *****************
    def _build_vertex_charts_from_lengths(self,
                                          F: np.ndarray,
                                          l: list[list[float]]) -> list[VertexManifoldChart]:
        """
        Build isometric charts for a surface with a flat metric
        """
        num_vertices: Index = F.max() + 1

        # Build vertex circulator
        vertex_circulator = VertexCirculator(F)

        # Iterate over vertices
        vertex_charts: list[VertexManifoldChart] = []

        for vertex_index in range(num_vertices):
            # Record the given vertex index in the chart
            __vertex_index: int = vertex_index

            # Build one ring in the original surface for the vertex chart
            __vertex_one_ring: list[int]
            __face_one_ring: list[int]
            __vertex_one_ring, __face_one_ring = vertex_circulator.get_one_ring(
                vertex_index)

            # Layout the vertices according to the given flat metric
            __one_ring_uv_positions: np.ndarray = self._layout_one_ring(F,
                                                                        l,
                                                                        vertex_index,
                                                                        __vertex_one_ring,
                                                                        __face_one_ring)

            # A vertex is on the boundary iff the vertex one ring is not a closed loop
            v0: Index = __vertex_one_ring[0]
            vn: Index = __vertex_one_ring[-1]
            __is_boundary: bool = (v0 != vn)

            #  By default, assume not cone adjacent (Changed later)
            #  FIXME This is dangerous... pretty it was dangerous becauase the object construction already sets is_cone_adjacent to False by default.
            # __is_cone_adjacent = False

            # Construct the VertexManifoldChart object and append to vertex_charts
            vertex_chart = VertexManifoldChart(
                __vertex_index,
                __vertex_one_ring,
                __face_one_ring,
                __one_ring_uv_positions,
                __is_boundary
            )

            vertex_charts.append(vertex_chart)

        return vertex_charts

    def _build_edge_charts_from_lengths(self,
                                        F: np.ndarray,
                                        halfedge: Halfedge,
                                        l: list[list[float]]) -> list[EdgeManifoldChart]:
        """
        """
        # Build edge charts
        num_edges: Index = halfedge.num_edges
        edge_charts: list[EdgeManifoldChart] = []  # length num_edges

        for edge_index in range(num_edges):
            # Get relevant halfedges for the face
            he_top: Index = halfedge.edge_to_first_halfedge(edge_index)
            he_bottom: Index = halfedge.edge_to_second_halfedge(edge_index)
            he_top_next: Index = halfedge.next_halfedge(he_top)
            he_bottom_next: int = halfedge.next_halfedge(he_bottom)

            # Get indices for the faces and vertices around the edge
            # chart = EdgeManifoldChart()
            __top_face_index: Index = halfedge.halfedge_to_face(he_top)
            __bottom_face_index: Index = halfedge.halfedge_to_face(he_bottom)
            __left_vertex_index: Index = halfedge.halfedge_to_tail_vertex(he_top)
            __right_vertex_index: Index = halfedge.halfedge_to_head_vertex(he_top)
            __top_vertex_index: Index = halfedge.halfedge_to_head_vertex(
                he_top_next)
            __bottom_vertex_index: Index = halfedge.halfedge_to_head_vertex(
                he_bottom_next)
            __is_boundary: bool = halfedge.is_boundary_edge(edge_index)

            # Get lengths of the edges of the top triangle
            # TODO: indexing issue perhaps below
            j_top: int = find_face_vertex_index(F[__top_face_index, :],
                                                __left_vertex_index)
            assert j_top >= 0
            lij: float = l[__top_face_index][(j_top + 2) % 3]
            ljk: float = l[__top_face_index][(j_top + 0) % 3]
            lki: float = l[__top_face_index][(j_top + 1) % 3]

            # Layout vertices starting at vi. Note that here, unlike in the vertex
            # chart case, we start from axis aligned edge with length 1
            __left_vertex_uv_position: PlanarPoint = np.array([[0.0, 0.0]])
            __right_vertex_uv_position: PlanarPoint = np.array([[1.0, 0.0]])
            assert lij > 0
            __top_vertex_uv_position: PlanarPoint = self._layout_next_vertex(
                __right_vertex_uv_position,
                ljk / lij,
                lki / lij)

            # Get center of the target edge for a later shift
            center: PlanarPoint = np.ndarray(shape=(1, 2))
            center[:, :] = 0.5 * __right_vertex_uv_position

            # If the edge is not on the boundary, build the bottom triangle
            if (not __is_boundary):
                # Get lengths of the edges of the bottom triangle if it
                # TODO: might be problem with slicing
                j_bottom: int = find_face_vertex_index(
                    F[__bottom_face_index, :], __left_vertex_index)
                assert j_bottom >= 0
                lil: float = l[__bottom_face_index][(j_bottom + 2) % 3]
                llj: float = l[__bottom_face_index][(j_bottom + 0) % 3]
                assert (float_equal(
                    lij, l[__bottom_face_index][(j_bottom + 1) % 3]))

                # Construct the last vertex counterclockwise and then reflect it
                uvl_reflected: PlanarPoint = self._layout_next_vertex(
                    __right_vertex_uv_position,
                    llj / lij,
                    lil / lij)
                __bottom_vertex_uv_position: PlanarPoint = reflect_across_x_axis(
                    uvl_reflected)

                # Shift all vertices so the midpoint is at the origin
                __left_vertex_uv_position -= center
                __right_vertex_uv_position -= center
                __top_vertex_uv_position -= center
                __bottom_vertex_uv_position -= center
            else:
                # Shift all constructed vertices so the midpoint is at the origin
                __left_vertex_uv_position -= center
                __right_vertex_uv_position -= center
                __top_vertex_uv_position -= center

                # Set the bottom uv position to the zero vector
                # TODO: make this some sort of PlanarPoint class constructor full of 0s
                __bottom_vertex_uv_position = np.zeros(shape=(1, 2))

            # Set chart
            chart = EdgeManifoldChart(
                __top_face_index,
                __bottom_face_index,
                __left_vertex_index,
                __right_vertex_index,
                __top_vertex_index,
                __bottom_vertex_index,
                __left_vertex_uv_position,
                __right_vertex_uv_position,
                __top_vertex_uv_position,
                __bottom_vertex_uv_position,
                __is_boundary
            )

            edge_charts.append(chart)

        # Done creating chart, return.
        return edge_charts

    def _build_face_charts(self,
                           F: np.ndarray,
                           global_uv: np.ndarray,
                           F_uv: np.ndarray) -> list[FaceManifoldChart]:
        """
        """
        num_faces: Index = F.shape[0]

        # FIXME: type issue below when allocating space in list
        face_charts: list[FaceManifoldChart] = []
        # [FaceManifoldChart] * num_faces

        for face_index in range(num_faces):
            __face_index: int = face_index
            __face_uv_positions: list[PlanarPoint] = [PlanarPoint(shape=(1, 2)),
                                                      PlanarPoint(shape=(1, 2)),
                                                      PlanarPoint(shape=(1, 2))]

            for face_vertex_index in range(3):
                uv_vertex_index: Index = F_uv[face_index, face_vertex_index]

                # FIXME: may be  issue with translation from Eigen:row() to NP slicin
                assert global_uv[[uv_vertex_index], :].shape == (1, 2)  # checking to see if same shape as PlanarPoint

                __face_uv_positions[face_vertex_index] = global_uv[[uv_vertex_index], :]

            # append to face_charts
            face_charts.append(FaceManifoldChart(__face_index,
                                                 __face_uv_positions))
        assert len(face_charts) == num_faces
        return face_charts

    def _build_corner_to_edge_map(self, corner_to_he: list[list[Index]], he_to_edge: list[Index]) -> list[list[int]]:
        """
        Compose corner to halfedge and halfedge to edge maps

        Returns:
            corner_to_edge
        """

        num_faces: Index = len(corner_to_he)

        PLACEHOLDER_INDEX = -1
        corner_to_edge: list[list[Index]] = [
            [PLACEHOLDER_INDEX, PLACEHOLDER_INDEX, PLACEHOLDER_INDEX] for _ in range(num_faces)
        ]

        for face_index in range(num_faces):
            for face_vertex_index in range(3):
                he_index: Index = corner_to_he[face_index][face_vertex_index]
                corner_to_edge[face_index][face_vertex_index] = he_to_edge[he_index]

        return corner_to_edge

    def _layout_next_vertex(self, current_point: PlanarPoint, next_edge_length: float, prev_edge_length: float) -> PlanarPoint:
        """
        Layout the next vertex in a triangle with given lengths and the current point
        position
        """
        # Get the current point and its rotation
        p0: PlanarPoint = current_point

        # NOTE: getting the perpendicular part?
        p0_perp = np.array([[-p0[0, 1], p0[0, 0]]])
        assert p0_perp.shape == (1, 2)

        # Get ratios of edge lengths
        # TODO: double check numpy norm with Eigen norm
        current_edge_length: float = LA.norm(current_point)
        assert isinstance(current_edge_length, float)

        l1: float = next_edge_length / current_edge_length
        l2: float = prev_edge_length / current_edge_length

        # Compute parallel and perpendicular components of the next point
        # construction
        a: float = 0.5 * (1 + l2 * l2 - l1 * l1)
        b: float = 2 * area_from_length(1.0, l1, l2)

        # Build the next point
        next_point: PlanarPoint = (a * p0) + (b * p0_perp)
        assert (not vector_contains_nan(next_point))

        # FIXME: maybe problem with float type?
        assert (float_equal(LA.norm(next_point), prev_edge_length))

        return next_point

    def _layout_one_ring(self, F: np.ndarray, l: list[list[float]], vertex_index: Index, vertex_one_ring: list[Index], face_one_ring: list[Index]) -> np.ndarray:
        """
        Layout the one ring around the given vertex from the lengths
        """
        one_ring_uv_positions = np.ndarray(shape=(len(vertex_one_ring), 2))

        # TODO: logging levels and making sure I'm doing it correctly
        if logger.getEffectiveLevel() != logging.INFO:
            logger.info("Building layout for one ring: %s", vertex_one_ring)
            # formatted_vector(vertex_one_ring))

        #  Initialize first vertex to position (l0, 0), where l0 is the length of the
        #  edge
        f0: int = face_one_ring[0]
        j0: int = find_face_vertex_index(F[f0, :], vertex_index)
        l0: float = l[f0][(j0 + 2) % 3]
        one_ring_uv_positions[0, :] = np.array([[l0, 0]])
        # assert one_ring_uv_positions.shape == (1, 2)

        # Layout remaining vertices
        for i, f in enumerate(face_one_ring):

            # TODO: see if getting matrix thing or not...
            j: int = find_face_vertex_index(F[f, :], vertex_index)
            if logger.getEffectiveLevel() != logging.INFO:
                logger.info("Laying out vertex for face %s", F[f, :])
                # logger.info("Face lengths are %s", formatted_vector(l[f]))
                logger.info("Face lengths are %s", l[f])

            # Get the lengths of the triangle edges
            next_edge_length: float = l[f][j]
            prev_edge_length: float = l[f][(j + 1) % 3]
            assert float_equal(l[f][(j + 2) % 3],
                               LA.norm(one_ring_uv_positions[[i], :]))

            # Layout the next vertex
            # TODO: fix the whole row grabbing thing...
            one_ring_uv_positions[[i + 1], :] = self._layout_next_vertex(
                one_ring_uv_positions[[i], :], next_edge_length, prev_edge_length)
            if logger.getEffectiveLevel() != logging.INFO:
                logger.info("Next vertex is %s",
                            one_ring_uv_positions[i + 1, :])

            assert float_equal(next_edge_length, LA.norm(
                one_ring_uv_positions[i + 1, :] - one_ring_uv_positions[i, :]))
            assert float_equal(prev_edge_length, LA.norm(
                one_ring_uv_positions[i + 1, :]))

        logger.info("Final layout:\n%s", one_ring_uv_positions)

        assert not matrix_contains_nan(one_ring_uv_positions)
        return one_ring_uv_positions

    def _build_lengths_from_global_uv(self, F: np.ndarray, global_uv: np.ndarray) -> list[list[float]]:
        """
        Build a corner-indexed metric for a surface with a global parametrization
        """

        num_faces: Index = F.shape[0]
        face_size: Index = F.shape[1]
        assert face_size == 3

        PLACEHOLDER_VALUE = -1
        l: list[list[float]] = [
            [PLACEHOLDER_VALUE, PLACEHOLDER_VALUE, PLACEHOLDER_VALUE] for _ in range(num_faces)
        ]

        # Iterate over faces
        for i in range(num_faces):
            # Iterate over vertices in face i
            for j in range(face_size):
                # Get the length of the edge opposite face vertex j
                prev_uv: PlanarPoint = global_uv[F[i, (j + 2) % face_size], :]
                next_uv: PlanarPoint = global_uv[F[i, (j + 1) % face_size], :]
                edge_vector: PlanarPoint = prev_uv - next_uv
                l[i][j] = LA.norm(edge_vector)

        return l

    def _align_local_charts(self, uv: np.ndarray, F_uv: np.ndarray) -> None:
        """
        Align local uv charts with the global parametrization
        """
        # Rotate and scale local layouts to align with the global layout
        for vertex_index in range(self.num_vertices):
            # Get the (transposed) similarity map that maps [1, 0]^T to the first local
            # uv edge
            local_layout: MatrixXr = self.get_vertex_chart(
                vertex_index).one_ring_uv_positions

            # FIXME: problem with shape below because of slicing... maybe
            local_edge: PlanarPoint = local_layout[[0], :]
            assert local_edge.shape == (1, 2)

            # TODO: confirm that the elements in this matrix matched position of elements in ASOC code
            # Documentation confirms that "comma intialization" in Eigen inserts row by row.
            # https://eigen.tuxfamily.org/dox-devel/group__TutorialAdvancedInitialization.html
            local_similarity_map: Matrix2x2r = np.array(
                [[local_edge[0][0], local_edge[0][1]], [-local_edge[0][1], local_edge[0][0]]])
            assert local_similarity_map.shape == (2, 2)

            # Get the global uv values corresponding the edge of the face
            edge_face_index: Index = self.get_vertex_chart(
                vertex_index).face_one_ring[0]
            edge_face_vertex_index: Index = find_face_vertex_index(
                self.m_F[edge_face_index, :], vertex_index)
            uv_vertex_index: Index = F_uv[edge_face_index,
                                          edge_face_vertex_index]
            uv_edge_vertex_index: Index = F_uv[edge_face_index,
                                               (edge_face_vertex_index + 1) % 3]

            # Get (transposed) similarity map that maps [1, 0]^T to the first global uv
            # edge
            global_edge = uv[[uv_edge_vertex_index], :] - \
                uv[[uv_vertex_index], :]
            assert global_edge.shape == (1, 2)

            # TODO: confirm that the elements in this matrix matched position of elements in ASOC code
            global_similarity_map: Matrix2x2r = np.array([[global_edge[0, 0], global_edge[0, 1]],
                                                          [-global_edge[0, 1], global_edge[0, 0]]])
            assert global_similarity_map.shape == (2, 2)

            # Apply composite similarity maps to the local uv positions
            # TODO: double check that this is doing matmul as we wanted
            similarity_map = global_similarity_map @ LA.inv(
                local_similarity_map)
            self.m_vertex_charts[vertex_index].one_ring_uv_positions = self.m_vertex_charts[
                vertex_index].one_ring_uv_positions @ similarity_map

        # Check validity after direct member variable manipulation
        is_valid: bool = self._is_valid_affine_manifold()
        assert is_valid

    def _mark_cones(self) -> None:
        """
        Mark cones and surrounding elements in the vertex and face charts
        """
        F: np.ndarray = self.get_faces
        cones: list[Index] = self.compute_cones()

        for _, ci in enumerate(cones):
            self.m_vertex_charts[ci].is_cone = True
            chart: VertexManifoldChart = self.get_vertex_chart(ci)
            logger.debug("Marking cone at %s", ci)

            # Mark vertices adjacent to cones
            logger.debug("Marking cone at adjacent vertices at %s",
                         formatted_vector(chart.vertex_one_ring, ", "))
            for j, vj in enumerate(chart.vertex_one_ring):
                self.m_vertex_charts[vj].is_cone_adjacent = True

            # Mark faces adjacent to cones
            logger.debug("Marking cone adjacent faces at %s",
                         formatted_vector(chart.face_one_ring, ", "))
            for j, fj in enumerate(chart.face_one_ring):
                self.m_face_charts[fj].is_cone_adjacent = True

                # Mark individual corners
                for k in range(3):
                    vk: Index = F[fj, k]
                    if self.m_vertex_charts[vk].is_cone:
                        self.m_face_charts[fj].is_cone_corner[k] = True

    def _compute_corner_uv_length(self, face_index: Index, face_vertex_index: Index) -> float:
        """
        """
        vn: Index = self.m_F_uv[face_index, (face_vertex_index + 1) % 3]
        vp: Index = self.m_F_uv[face_index, (face_vertex_index + 2) % 3]
        next_uv: PlanarPoint = self.m_global_uv[[vn], :]
        prev_uv: PlanarPoint = self.m_global_uv[[vp], :]
        assert next_uv.shape == (1, 2)
        assert prev_uv.shape == (1, 2)
        edge_vector = next_uv - prev_uv
        assert edge_vector.shape == (1, 2)

        result: float = LA.norm(edge_vector)
        assert isinstance(result, float)
        return result

    def _is_valid_affine_manifold(self) -> bool:
        """
        Check that the manifold is valid and self-consistent
        """
        # Threshold for length comparisons
        length_threshold: float = 1e-6

        # Zero uv coordinate
        zero: PlanarPoint = np.zeros(shape=(1, 2))

        # Face containment helper lambda
        def contains_vertex(face: np.ndarray, index: Index) -> bool:
            assert face.ndim == 1
            # TODO: double check that this works and is correct
            return np.any(face == index)

        # Edge length check helper function
        def edge_has_length(v0: np.ndarray, v1: np.ndarray, length: float) -> bool:
            assert v0.shape == (1, 2)
            assert v1.shape == (1, 2)
            edge = v1 - v0
            return float_equal(LA.norm(edge), length, length_threshold)

        # Check that the sizes of the member variables are consistent
        if self.m_F.shape[0] != len(self.m_l):
            return False

        # Check that the global metric is consistent
        for fi in range(self.num_faces):
            for j in range(3):
                # Check the length of the edge is the same as the uv length
                edge_length: float = self.m_l[fi][j]
                edge_uv_length: float = self._compute_corner_uv_length(fi, j)

                if not float_equal(edge_length, edge_uv_length, length_threshold):
                    logger.error("Inconsistent edge length %s and uv length %s for corner %s, %s",
                                 edge_length, edge_uv_length, fi, j)
                    return False

                # Get opposite halfedge and corner if it exists
                he: Index = self.m_corner_to_he[fi][j]
                if self.m_halfedge.is_boundary_halfedge(he):
                    continue
                he_opp: Index = self.m_halfedge.opposite_halfedge(he)
                fi_opp: Index = self.m_he_to_corner[he_opp][0]
                j_opp: Index = self.m_he_to_corner[he_opp][1]

                # Check uvs are the same for the opposite corners
                opposite_edge_uv_length: float = self._compute_corner_uv_length(
                    fi_opp, j_opp)
                if not float_equal(edge_length, opposite_edge_uv_length, length_threshold):
                    logger.error(
                        "Inconsistent opposite uv length for corners %s, %s and %s, %s", fi, j, fi_opp, j_opp)
                    return False

        # Check that each vertex chart is valid
        for vertex_index in range(self.num_vertices):
            chart: VertexManifoldChart = self.m_vertex_charts[vertex_index]

            # Check basic chart indexing and size validity
            if chart.vertex_index != vertex_index:
                return False
            if len(chart.vertex_one_ring) != (len(chart.face_one_ring) + 1):
                return False
            if chart.one_ring_uv_positions.shape[0] != len(chart.vertex_one_ring):
                return False

            # Check that each one ring face contains the central vertex, the vertex with
            # the same index in the vertex one ring, and the vertex with one larger index
            for i, face_index in enumerate(chart.face_one_ring):
                face_vertex_index: Index = find_face_vertex_index(
                    self.m_F[face_index, :], vertex_index)
                vi: Index = chart.vertex_one_ring[i]
                vj: Index = chart.vertex_one_ring[i + 1]

                # Check that the one ring indexing is valid
                if face_index > self.m_F.shape[0]:
                    return False
                if not contains_vertex(self.m_F[face_index, :], vertex_index):
                    return False
                if not contains_vertex(self.m_F[face_index, :], vi):
                    return False
                if not contains_vertex(self.m_F[face_index, :], vj):
                    return False

                # Check that each local uv length is compatible with the given metric
                # FIXME: pretty sure the below is not the way to go for logging with a set level
                if logger.getEffectiveLevel != logger.level:
                    logger.info("Face lengths: %s",
                                formatted_vector(self.m_l[face_index]))

                if not edge_has_length(zero, chart.one_ring_uv_positions[[i], :], self.m_l[face_index][(face_vertex_index + 2) % 3]):
                    logger.error("uv position %s in chart %s does not expect norm %s",
                                 chart.one_ring_uv_positions[i, :], vertex_index, self.m_l[face_index][(face_vertex_index + 2) % 3])
                    return False

                if not edge_has_length(chart.one_ring_uv_positions[[i + 1], :], chart.one_ring_uv_positions[[i], :], self.m_l[face_index][(face_vertex_index + 0) % 3]):
                    logger.error("uv positions %s and %s in chart %s do not have expected length %s", chart.one_ring_uv_positions[
                                 [i + 1], :], chart.one_ring_uv_positions[[i], :], vertex_index, self.m_l[face_index][(face_vertex_index + 0) % 3])
                    return False

                if not edge_has_length(zero, chart.one_ring_uv_positions[[i + 1], :], self.m_l[face_index][(face_vertex_index + 1) % 3]):
                    logger.error("uv position %s in chart %s does not have the expected norm %s",
                                 chart.one_ring_uv_positions[[i + 1], :], vertex_index, self.m_l[face_index][(face_vertex_index + 1) % 3])
                    return False

        # Return true if no issues found
        return True


# **************************
# Parametric Affine Manifold
# **************************


class ParametricAffineManifold(AffineManifold):
    """
    Representation for an affine manifold with a global parametrization, which
    yields a flat metric and thus an affine manifold structure.
    """

    def __init__(self, F: np.ndarray, global_uv: np.ndarray):
        """
        Constructor for a parametric affine manifold from a global
        parametrization.

        @param[in] F: faces of the affine manifold
        @param[in] global_uv: affine global layout of the manifold
        """
        super().__init__(F, global_uv, F)
        assert self.__is_valid_parametric_affine_manifold()

    # **************
    # Public Methods
    # **************

    def get_vertex_global_uv(self, vertex_index: Index) -> PlanarPoint:
        """
        Get global uv coordinates for a given vertex.

        param[in] vertex_index: index of the vertex for the uv position
        param[out] uv_coords: global uv position for the given vertex
        """
        # TODO: adjust shape or something
        uv_coords: PlanarPoint = self.m_global_uv[[vertex_index], :]
        assert uv_coords.shape == (1, 2)

        return uv_coords

    # ***************
    # Private Methods
    # ***************

    def __is_valid_parametric_affine_manifold(self) -> bool:
        """
        """
        if not np.array_equal(self.m_F_uv, self.m_F):
            return False

        for vertex_index in range(self.num_vertices):
            chart: VertexManifoldChart = self.m_vertex_charts[vertex_index]

            for i, _ in enumerate(chart.vertex_one_ring):
                vi: Index = chart.vertex_one_ring[i]
                # TODO: maybe some issue with slicing below
                local_uv_difference: PlanarPoint = chart.one_ring_uv_positions[[
                    i], :]
                assert local_uv_difference.shape == (1, 2)
                global_uv_difference: PlanarPoint = self.m_global_uv[[vi],
                                                                     :] - self.m_global_uv[[vertex_index], :]
                assert global_uv_difference.shape == (1, 2)

                if not vector_equal(global_uv_difference, local_uv_difference):
                    logger.error("Global uv coordinates %s and %s do not have expected difference %s",
                                 self.m_global_uv[[vi], :],
                                 self.m_global_uv[[vertex_index], :],
                                 local_uv_difference)
                    return False
        # Return true if no issues found
        return True


def remove_cones(V: np.ndarray,
                 affine_manifold: AffineManifold,
                 #  Really, only need the top two parameters.
                 pruned_V: np.ndarray,
                 pruned_affine_manifold: AffineManifold,
                 cones: list[Index],
                 removed_faces: list[Index]) -> None:
    """
    Generate an affine manifold with the cone faces removed but cone adjaceny
    information retained.

    NOTE: this method is not used anywhere.
    """
    unimplemented("This method is not used anywhere else.")

    # Compute the cones
    # TODO: why do we even need to pass in cones if they're just going to be removed anyways?
    cones = affine_manifold.compute_cones()
    # TODO: implement with proper formatted_vector() function
    logger.debug("Remove cones at %s", cones)

    # Create boolean arrays of cone adjacent vertices
    PLACEHOLDER_BOOL: bool = False
    is_cone_adjacent_vertex: list[bool] = [PLACEHOLDER_BOOL] * affine_manifold.num_vertices
    for vi in range(affine_manifold.num_vertices):
        is_cone_adjacent_vertex[vi] = affine_manifold.get_vertex_chart(
            vi).is_cone_adjacent

    # Create boolean arrays of cone adjacent faces
    is_cone_adjacent_face: list[bool] = [PLACEHOLDER_BOOL] * affine_manifold.num_faces
    for fi in range(affine_manifold.num_faces):
        is_cone_adjacent_face[fi] = affine_manifold.get_face_chart(
            fi).is_cone_adjacent

    # Remove faces from VF meshes
    F_orig: np.ndarray = affine_manifold.get_faces
    global_uv_orig: np.ndarray = affine_manifold.get_global_uv
    F_uv_orig: np.ndarray = affine_manifold.get_F_uv
    F: np.ndarray
    global_uv: np.ndarray
    F_uv: np.ndarray

    # TODO: finish implementation
    global_uv, F_uv, removed_faces = remove_mesh_vertices(global_uv_orig, F_uv_orig, cones)
    pruned_V, F = remove_mesh_faces(V, F_orig, removed_faces)

    # Remove faces from the cone adjacent arrays
    is_cone_adjacent_face_reindexed: list[bool] = remove_vector_values(
        removed_faces,
        is_cone_adjacent_face)
    is_cone_adjacent_vertex_reindexed: list[bool] = remove_vector_values(
        cones,
        is_cone_adjacent_vertex)

    # Make new affine manifold with cones removed
    pruned_affine_manifold = AffineManifold(F, global_uv, F_uv)

    # Mark cone adjacent faces
    for fi, _ in enumerate(is_cone_adjacent_face_reindexed):
        if (is_cone_adjacent_face_reindexed[fi]):
            pruned_affine_manifold.mark_cone_adjacent_face(fi)

    # Mark cone adjacent vertices
    for vi, _ in enumerate(is_cone_adjacent_vertex_reindexed):
        if (is_cone_adjacent_vertex_reindexed[vi]):
            pruned_affine_manifold.mark_cone_adjacent_vertex(vi)
